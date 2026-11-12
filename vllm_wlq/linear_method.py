import os
import re
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from safetensors import safe_open
from . import _C
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    RowParallelLinear,
    ColumnParallelLinear,
)

_DORA_WEIGHTS_CACHE: Optional[Dict[str, torch.Tensor]] = None
_DORA_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_DORA_INDEXED: Optional[Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]] = None

def get_dora_adapter(model_id_or_path: Optional[str] = None):
    global _DORA_WEIGHTS_CACHE, _DORA_CONFIG_CACHE, _DORA_INDEXED
    if _DORA_INDEXED is not None:
        return _DORA_INDEXED, _DORA_CONFIG_CACHE

    adapter_weights = {}
    adapter_config = {}

    candidate_dirs = [
        "lora_adapter_distilled",
        "./lora_adapter_distilled",
        "../lora_adapter_distilled",
        "mtg_0.01_pct_relMSE_shot/lora_adapter_distilled",
        "mtg_0.01_pct_relMSE_shot_ef/lora_adapter_distilled",
    ]

    if model_id_or_path and os.path.isdir(model_id_or_path):
        candidate_dirs.insert(0, os.path.join(model_id_or_path, "lora_adapter_distilled"))
        candidate_dirs.insert(1, model_id_or_path)

    adapter_file = None
    config_file = None

    for d in candidate_dirs:
        af = os.path.join(d, "adapter_model.safetensors")
        cf = os.path.join(d, "adapter_config.json")
        if os.path.isfile(af):
            adapter_file = af
            if os.path.isfile(cf):
                config_file = cf
            break

    if adapter_file is None and model_id_or_path:
        try:
            from huggingface_hub import hf_hub_download
            try:
                adapter_file = hf_hub_download(
                    repo_id=model_id_or_path,
                    filename="adapter_model.safetensors",
                    subfolder="lora_adapter_distilled"
                )
            except Exception:
                adapter_file = hf_hub_download(
                    repo_id=model_id_or_path,
                    filename="lora_adapter_distilled/adapter_model.safetensors"
                )
            try:
                config_file = hf_hub_download(
                    repo_id=model_id_or_path,
                    filename="adapter_config.json",
                    subfolder="lora_adapter_distilled"
                )
            except Exception:
                config_file = hf_hub_download(
                    repo_id=model_id_or_path,
                    filename="lora_adapter_distilled/adapter_config.json"
                )
        except Exception:
            pass

    if adapter_file and os.path.isfile(adapter_file):
        with safe_open(adapter_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                adapter_weights[k] = f.get_tensor(k)

        if config_file and os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)

    indexed = {}
    layer_re = re.compile(r"layers\.(\d+)\.")
    proj_names = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj_qkvz", "in_proj_qkv", "in_proj_z",
        "in_proj_ba", "in_proj_b", "in_proj_a", "out_proj",
    ]

    for k, v in adapter_weights.items():
        lm = layer_re.search(k)
        if not lm:
            continue
        l_idx = int(lm.group(1))
        for p in proj_names:
            if f".{p}." in k or k.endswith(f".{p}"):
                key = (l_idx, p)
                if key not in indexed:
                    indexed[key] = [None, None, None]  
                k_low = k.lower()
                if "lora_a" in k_low:
                    indexed[key][0] = v
                elif "lora_b" in k_low:
                    indexed[key][1] = v
                elif any(s in k_low for s in ["dora_scale", "magnitude_vector"]):
                    indexed[key][2] = v

    _DORA_WEIGHTS_CACHE = adapter_weights
    _DORA_CONFIG_CACHE = adapter_config
    _DORA_INDEXED = indexed
    return _DORA_INDEXED, _DORA_CONFIG_CACHE


def flexible_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
    shard_id = args[0] if len(args) > 0 else kwargs.get("shard_id", None)
    if not hasattr(param, "_shard_dict"):
        param._shard_dict = {}
    if not hasattr(param, "_shard_list"):
        param._shard_list = []

    if hasattr(torch, "float8_e5m2") and loaded_weight.dtype == torch.float8_e5m2:
        loaded_weight = loaded_weight.view(torch.uint8)

    if shard_id is not None:
        param._shard_dict[shard_id] = loaded_weight
    param._shard_list.append(loaded_weight)
    param.data = loaded_weight
    param._is_loaded = True


def create_layer_weight_param(layer: torch.nn.Module, data: torch.Tensor):
    is_row_parallel = isinstance(layer, RowParallelLinear) or getattr(layer, "is_row_parallel", False)
    try:
        from vllm.model_executor.parameter import ModelWeightParameter
        try:
            if is_row_parallel:
                param = ModelWeightParameter(data=data, input_dim=0)
            else:
                param = ModelWeightParameter(data=data, output_dim=0)
        except TypeError:
            try:
                param = ModelWeightParameter(data=data)
            except Exception:
                param = torch.nn.Parameter(data, requires_grad=False)
    except ImportError:
        param = torch.nn.Parameter(data, requires_grad=False)

    param.weight_loader = flexible_weight_loader
    param._shard_dict = {}
    param._is_loaded = False
    return param


def _shard_sort_key(k):
    if isinstance(k, int):
        return (0, k)
    if isinstance(k, (tuple, list)) and len(k) > 0:
        return (0, int(k[0]))
    order = {"q": 0, "k": 1, "v": 2, "gate": 0, "up": 1}
    return (1, order.get(str(k), 99))


def get_shards_in_order(param, layer):
    if param is None or not getattr(param, "_is_loaded", False):
        return []
    d = getattr(param, "_shard_dict", {})
    lst = getattr(param, "_shard_list", [])

    if d:
        return [d[k] for k in sorted(d.keys(), key=_shard_sort_key)]

    if lst:
        return lst
    return [param.data] if (hasattr(param, "data") and param.data.numel() > 0) else []


try:
    import numba
    @numba.njit(fastmath=True, nogil=True)
    def _read_bits_jit(packed_u32, bit_pos, n_bits):
        if n_bits == 0:
            return 0
        bit_pos, n_bits = int(bit_pos), int(n_bits)
        w_idx = bit_pos >> 5
        offset = bit_pos & 31
        val = int(packed_u32[w_idx]) >> offset
        bits_read = 32 - offset
        if n_bits > bits_read and (w_idx + 1) < len(packed_u32):
            val |= (int(packed_u32[w_idx + 1]) << bits_read)
        return val & ((1 << n_bits) - 1)

    @numba.njit(fastmath=True, nogil=True)
    def _decode_elias_fano_jit(ef_tile_meta, ef_bitstream_u32, num_tiles_y, num_tiles_x, tile_size):
        total_tiles = num_tiles_y * num_tiles_x
        U = tile_size * tile_size
        reconstructed = np.zeros((total_tiles, U), dtype=np.bool_)

        for k in range(total_tiles):
            start_bit = int(ef_tile_meta[k, 0])
            m = int(ef_tile_meta[k, 1])
            if m == 0:
                continue
            L = max(0, int(math.floor(math.log2(U / m))))

            low = np.zeros(m, dtype=np.int32)
            if L > 0:
                for i in range(m):
                    low[i] = _read_bits_jit(ef_bitstream_u32, start_bit + (i * L), L)

            upper_start = start_bit + (m * L)
            high = np.zeros(m, dtype=np.int32)
            cur_bit, cur_high, found = upper_start, 0, 0
            while found < m:
                w_idx = cur_bit >> 5
                offset = cur_bit & 31
                bit = (int(ef_bitstream_u32[w_idx]) >> offset) & 1
                if bit == 0:
                    cur_high += 1
                else:
                    high[found] = cur_high
                    found += 1
                cur_bit += 1

            indices = (high << L) | low
            for idx in indices:
                reconstructed[k, idx] = True

        return reconstructed
    _USE_JIT = True
except Exception:
    _USE_JIT = False


def decode_elias_fano(ef_tile_meta: np.ndarray, ef_bitstream_u32: np.ndarray, orig_shape, tile_size=32):
    H, W = orig_shape
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    H_pad, W_pad = H + pad_h, W + pad_w
    num_tiles_y, num_tiles_x = H_pad // tile_size, W_pad // tile_size

    if _USE_JIT:
        tiles = _decode_elias_fano_jit(ef_tile_meta, ef_bitstream_u32, num_tiles_y, num_tiles_x, tile_size)
    else:
        U = tile_size * tile_size
        total_tiles = num_tiles_y * num_tiles_x
        tiles = np.zeros((total_tiles, U), dtype=bool)
        bitstream_bits = np.unpackbits(ef_bitstream_u32.view(np.uint8), bitorder="little")

        for k in range(total_tiles):
            start_bit = int(ef_tile_meta[k, 0])
            m = int(ef_tile_meta[k, 1])
            if m == 0:
                continue
            L = max(0, int(math.floor(math.log2(U / m))))

            if L > 0:
                low_bits = bitstream_bits[start_bit : start_bit + m * L].reshape(m, L).astype(np.int32)
                low = (low_bits << np.arange(L, dtype=np.int32)).sum(axis=1)
            else:
                low = np.zeros(m, dtype=np.int32)

            upper_start = start_bit + m * L
            high_max = (U - 1) >> L
            unary_slice = bitstream_bits[upper_start : upper_start + m + high_max + 1]
            one_positions = np.flatnonzero(unary_slice)[:m]
            high = (one_positions - np.arange(m, dtype=np.int64)).astype(np.int32)

            indices = (high << L) | low
            tiles[k, indices] = True

    mask_padded = tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size).swapaxes(1, 2).reshape(H_pad, W_pad)
    return mask_padded[:H, :W]


def build_chunk_csr(mask_2d: np.ndarray, fp8_values_raw: torch.Tensor, chunk_m: int):
    H, W = mask_2d.shape
    num_chunks = (W + chunk_m - 1) // chunk_m
    rows, cols = np.nonzero(mask_2d)
    chunk_idx = cols // chunk_m
    c_local = (cols % chunk_m).astype(np.uint16)

    flat_chunk_ids = rows * num_chunks + chunk_idx
    counts = np.bincount(flat_chunk_ids, minlength=H * num_chunks)
    row_counts = counts.reshape(H, num_chunks)

    chunk_offsets = np.zeros((H, num_chunks + 1), dtype=np.int32)
    row_starts = np.zeros(H, dtype=np.int32)
    row_starts[1:] = np.cumsum(row_counts.sum(axis=1)[:-1])
    chunk_offsets[:, 0] = row_starts
    chunk_offsets[:, 1:] = row_starts[:, None] + np.cumsum(row_counts, axis=1)

    chunk_cols_tensor = torch.from_numpy(c_local)
    chunk_offsets_tensor = torch.from_numpy(chunk_offsets)

    if fp8_values_raw.numel() == 0:
        chunk_vals_tensor = torch.empty(0, dtype=torch.uint8)
    elif fp8_values_raw.dtype == torch.float16:
        u16 = fp8_values_raw.view(torch.uint16).to(torch.int32)
        chunk_vals_tensor = ((u16 + 0x80) >> 8).clamp(0, 255).to(torch.uint8)
    else:
        chunk_vals_tensor = fp8_values_raw.contiguous().view(torch.uint8)

    return chunk_offsets_tensor, chunk_cols_tensor, chunk_vals_tensor


def unpack_indices_continuous(q_packed_tensor: torch.Tensor, N: int, M: int, bits: int) -> np.ndarray:
    symbols_per_word = 10 if bits == 3 else (32 // bits)
    mask = (1 << bits) - 1
    shifts = (np.arange(symbols_per_word, dtype=np.uint32) * bits)
    q_u32 = q_packed_tensor.cpu().numpy().view(np.uint32)
    unpacked = (q_u32[:, None] >> shifts[None, :]) & mask
    flat = unpacked.flatten()
    total = N * M
    return flat[:total].reshape(N, M).astype(np.uint32)


def pack_indices_row_aligned(q_idx_2d: np.ndarray, bits: int):
    N, M = q_idx_2d.shape
    symbols_per_word = 10 if bits == 3 else (32 // bits)
    raw_words_per_row = (M + symbols_per_word - 1) // symbols_per_word
    words_per_row = ((raw_words_per_row + 3) // 4) * 4
    total_symbols_per_row = words_per_row * symbols_per_word
    pad_m = total_symbols_per_row - M

    q_padded = np.pad(q_idx_2d, ((0, 0), (0, pad_m)), mode='constant') if pad_m > 0 else q_idx_2d
    q_matrix = q_padded.reshape(N, words_per_row, symbols_per_word).astype(np.uint32)
    shifts = (np.arange(symbols_per_word, dtype=np.uint32) * bits)
    packed_u32 = np.bitwise_or.reduce(q_matrix << shifts[None, None, :], axis=2)
    return packed_u32.flatten(), words_per_row


def get_tp_info():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except Exception:
        return 0, 1


def slice_outliers_for_tp(
    mask_2d: np.ndarray,
    fp8_vals: torch.Tensor,
    row_slice: Optional[Tuple[int, int]] = None,
    col_slice: Optional[Tuple[int, int]] = None,
):
    """Schneidet 2D-Outlier-Maske und zugehörige 1D-FP8-Werte exakt für TP-Shards zu."""
    if row_slice is not None:
        r0, r1 = row_slice
        start_val = int(mask_2d[:r0].sum())
        end_val = start_val + int(mask_2d[r0:r1].sum())
        return mask_2d[r0:r1], fp8_vals[start_val:end_val]

    if col_slice is not None:
        c0, c1 = col_slice
        keep_grid = np.zeros_like(mask_2d, dtype=bool)
        keep_grid[:, c0:c1] = True
        keep_flat_outliers = keep_grid[mask_2d]
        return mask_2d[:, c0:c1], fp8_vals[torch.from_numpy(keep_flat_outliers)]

    return mask_2d, fp8_vals


class WLQLinearMethod(LinearMethodBase):
    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_kwargs
    ):
        layer.output_partition_sizes = list(output_partition_sizes)
        N = sum(output_partition_sizes)
        K = input_size_per_partition
        layer.N = N
        layer.K = K

        fp8_t = getattr(torch, "float8_e5m2", torch.uint8)

        layer.register_parameter("weight", create_layer_weight_param(layer, torch.empty(0, dtype=torch.float16)))
        layer.register_parameter("q_packed", create_layer_weight_param(layer, torch.empty(0, dtype=torch.int32)))
        layer.register_parameter("lut_c", create_layer_weight_param(layer, torch.empty(0, dtype=torch.float16)))
        layer.register_parameter("bits", create_layer_weight_param(layer, torch.empty(0, dtype=torch.int32)))
        layer.register_parameter("orig_shape", create_layer_weight_param(layer, torch.empty(0, dtype=torch.int32)))
        layer.register_parameter("is_quantized", create_layer_weight_param(layer, torch.empty(0, dtype=torch.uint8)))

        layer.register_parameter("ef_tile_size", create_layer_weight_param(layer, torch.empty(0, dtype=torch.int32)))
        layer.register_parameter("ef_tile_meta", create_layer_weight_param(layer, torch.empty(0, dtype=torch.int64)))
        layer.register_parameter("ef_bitstream", create_layer_weight_param(layer, torch.empty(0, dtype=torch.int32)))
        layer.register_parameter("outlier_mask_1bit", create_layer_weight_param(layer, torch.empty(0, dtype=torch.uint8)))
        layer.register_parameter("outlier_mask", create_layer_weight_param(layer, torch.empty(0, dtype=torch.uint8)))
        layer.register_parameter("outlier_values", create_layer_weight_param(layer, torch.empty(0, dtype=fp8_t)))
        layer.register_parameter("fp8_outliers", create_layer_weight_param(layer, torch.empty(0, dtype=fp8_t)))
        layer.register_parameter("outliers", create_layer_weight_param(layer, torch.empty(0, dtype=fp8_t)))

    _MAX_N: int = 0
    _MAX_K: int = 0

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = layer.lut_c.device if hasattr(layer, "lut_c") and layer.lut_c.numel() > 0 else torch.device("cpu")

        tp_rank, tp_size = get_tp_info()
        is_row_parallel = isinstance(layer, RowParallelLinear) or getattr(layer, "is_row_parallel", False)

        output_partition_sizes = getattr(layer, "output_partition_sizes", [layer.N])
        num_shards = len(output_partition_sizes)
        K_part = layer.K
        K_full = K_part * tp_size if (is_row_parallel and tp_size > 1) else K_part

        q_packed_shards = get_shards_in_order(layer.q_packed, layer)
        lut_c_shards = get_shards_in_order(layer.lut_c, layer)
        bits_shards = get_shards_in_order(layer.bits, layer) if hasattr(layer, "bits") else []
        meta_shards = get_shards_in_order(layer.ef_tile_meta, layer) if hasattr(layer, "ef_tile_meta") else []
        stream_shards = get_shards_in_order(layer.ef_bitstream, layer) if hasattr(layer, "ef_bitstream") else []
        tile_size_shards = get_shards_in_order(layer.ef_tile_size, layer) if hasattr(layer, "ef_tile_size") else []
        mask1b_shards = get_shards_in_order(layer.outlier_mask_1bit, layer) if hasattr(layer, "outlier_mask_1bit") else []
        vals_shards = get_shards_in_order(layer.outlier_values, layer) if hasattr(layer, "outlier_values") else []

        if len(lut_c_shards) > 0 and all(l.dim() == 2 and l.shape[0] > 0 for l in lut_c_shards):
            if not is_row_parallel and tp_size > 1 and sum(l.shape[0] for l in lut_c_shards) == layer.N * tp_size:
                effective_partition_sizes = [l.shape[0] // tp_size for l in lut_c_shards]
            else:
                effective_partition_sizes = [l.shape[0] for l in lut_c_shards]
        elif len(q_packed_shards) == 1 and num_shards > 1:
            effective_partition_sizes = [sum(output_partition_sizes)]
        else:
            effective_partition_sizes = output_partition_sizes

        if tp_size == 1 and hasattr(_C, "decode_ef_to_chunk_csr") and hasattr(_C, "repack_q_indices_row_aligned"):
            shard_bits_list = []
            for s in range(len(effective_partition_sizes)):
                if len(bits_shards) > 0:
                    b_t = bits_shards[s] if s < len(bits_shards) else bits_shards[0]
                    bits_s = int(b_t.item()) if b_t.numel() > 0 else self.quant_config.bits
                else:
                    bits_s = self.quant_config.bits
                shard_bits_list.append(bits_s)

            bits = max(shard_bits_list) if shard_bits_list else self.quant_config.bits
            target_k_cb = 1 << bits
            layer_chunk_m = 640 if bits == 3 else 512
            num_chunks = (K_part + layer_chunk_m - 1) // layer_chunk_m

            shard_q_aligned = []
            shard_luts = []
            shard_offsets = []
            shard_cols = []
            shard_fp8s_u8 = []
            running_outlier_offset = 0
            words_per_row = 0

            for s, N_s in enumerate(effective_partition_sizes):
                bits_s = shard_bits_list[s]
                q_p = q_packed_shards[s] if s < len(q_packed_shards) else layer.q_packed
                lut_s = (lut_c_shards[s] if s < len(lut_c_shards) else layer.lut_c).cpu()

                if bits_s == bits and hasattr(_C, "repack_q_indices_row_aligned"):
                    q_al_s, words_per_row = _C.repack_q_indices_row_aligned(q_p, N_s, K_part, bits)
                else:
                    q_idx_s = unpack_indices_continuous(q_p, N_s, K_part, bits_s)
                    packed_np_s, words_per_row = pack_indices_row_aligned(q_idx_s, bits)
                    q_al_s = torch.from_numpy(packed_np_s.view(np.int32))
                shard_q_aligned.append(q_al_s)

                if lut_s.shape[1] < target_k_cb:
                    lut_s = F.pad(lut_s, (0, target_k_cb - lut_s.shape[1]), value=0.0)
                shard_luts.append(lut_s)

                tile_sz = 64
                if len(tile_size_shards) > 0:
                    t_t = tile_size_shards[s] if s < len(tile_size_shards) else tile_size_shards[0]
                    if t_t.numel() > 0:
                        tile_sz = int(t_t.item())

                has_ef = (s < len(meta_shards) and meta_shards[s].numel() > 0 and
                          s < len(stream_shards) and stream_shards[s].numel() > 0)

                if has_ef:
                    off_s, col_s = _C.decode_ef_to_chunk_csr(
                        meta_shards[s], stream_shards[s], N_s, K_part, tile_sz, layer_chunk_m
                    )
                elif s < len(mask1b_shards) and mask1b_shards[s].numel() > 0:
                    raw_m = mask1b_shards[s].cpu().numpy()
                    m_s = np.unpackbits(raw_m)[:N_s * K_part].reshape(N_s, K_part).astype(bool)
                    off_s, col_s, _ = build_chunk_csr(m_s, torch.empty(0, dtype=torch.uint8), layer_chunk_m)
                else:
                    off_s = torch.zeros((N_s, num_chunks + 1), dtype=torch.int32)
                    col_s = torch.empty(0, dtype=torch.uint16)

                if running_outlier_offset > 0:
                    off_s = off_s + running_outlier_offset
                running_outlier_offset += int(col_s.numel())

                shard_offsets.append(off_s)
                shard_cols.append(col_s)

                if s < len(vals_shards) and vals_shards[s].numel() > 0:
                    v_s = vals_shards[s].cpu()
                else:
                    v_s = torch.zeros(int(col_s.numel()), dtype=torch.uint8, device="cpu")

                if v_s.dtype == torch.float16:
                    u16 = v_s.view(torch.uint16).to(torch.int32)
                    v_s = ((u16 + 0x80) >> 8).clamp(0, 255).to(torch.uint8)
                else:
                    v_s = v_s.contiguous().view(torch.uint8)
                shard_fp8s_u8.append(v_s)

            q_packed_aligned_cpu = torch.cat(shard_q_aligned, dim=0) if len(shard_q_aligned) > 1 else shard_q_aligned[0]
            lut_c_all = torch.cat(shard_luts, dim=0) if len(shard_luts) > 1 else shard_luts[0]
            offsets = torch.cat(shard_offsets, dim=0) if len(shard_offsets) > 1 else shard_offsets[0]
            cols = torch.cat(shard_cols, dim=0) if len(shard_cols) > 1 else shard_cols[0]
            fp8_raw = torch.cat(shard_fp8s_u8, dim=0) if len(shard_fp8s_u8) > 1 else shard_fp8s_u8[0]
            vals = fp8_raw

            dora_index, adapter_config = get_dora_adapter(getattr(self.quant_config, "model_id", None))
            has_dora = False
            lora_A_f16, lora_B_final, dora_gamma = None, None, None
            lora_scale = 1.0

            if dora_index:
                vllm_prefix = getattr(layer, "vllm_prefix", "")
                lm = re.search(r"layers\.(\d+)\.", vllm_prefix)
                l_idx = int(lm.group(1)) if lm else None

                if l_idx is not None:
                    if "in_proj_qkvz" in vllm_prefix:
                        sub_projs = ["in_proj_qkvz"] if (l_idx, "in_proj_qkvz") in dora_index else ["in_proj_qkv", "in_proj_z"]
                    elif "in_proj_ba" in vllm_prefix:
                        sub_projs = ["in_proj_ba"] if (l_idx, "in_proj_ba") in dora_index else ["in_proj_b", "in_proj_a"]
                    elif "in_proj_qkv" in vllm_prefix:
                        sub_projs = ["in_proj_qkv"]
                    elif "qkv_proj" in vllm_prefix:
                        sub_projs = ["q_proj", "k_proj", "v_proj"]
                    elif "gate_up_proj" in vllm_prefix:
                        sub_projs = ["gate_proj", "up_proj"]
                    elif "out_proj" in vllm_prefix:
                        sub_projs = ["out_proj"]
                    elif "o_proj" in vllm_prefix:
                        sub_projs = ["o_proj"]
                    elif "down_proj" in vllm_prefix:
                        sub_projs = ["down_proj"]
                    else:
                        sub_projs = []

                    collected_A, collected_B, collected_scale = [], [], []
                    for p in sub_projs:
                        entry = dora_index.get((l_idx, p))
                        if entry and entry[0] is not None and entry[1] is not None:
                            collected_A.append(entry[0].float())
                            collected_B.append(entry[1].float())
                            collected_scale.append(entry[2].float() if entry[2] is not None else None)

                    if len(collected_A) == len(sub_projs) and len(sub_projs) > 0:
                        has_dora = True
                        r_base = collected_B[0].shape[1]
                        lora_alpha = adapter_config.get("lora_alpha", r_base)
                        lora_scale = float(lora_alpha) / float(r_base)

                        if len(sub_projs) == 1:
                            lora_A_final = collected_A[0]
                            lora_B_final = collected_B[0]
                            m_vec = collected_scale[0]
                        else:
                            lora_A_final = torch.cat(collected_A, dim=0)
                            tot_N = sum(b.shape[0] for b in collected_B)
                            tot_R = sum(b.shape[1] for b in collected_B)
                            lora_B_final = torch.zeros((tot_N, tot_R), dtype=torch.float32)
                            r_off, c_off = 0, 0
                            for b in collected_B:
                                nr, nc = b.shape
                                lora_B_final[r_off : r_off + nr, c_off : c_off + nc] = b
                                r_off += nr
                                c_off += nc
                            m_vec = torch.cat(collected_scale, dim=0) if None not in collected_scale else None

                        if lora_B_final.shape[0] != layer.N or lora_A_final.shape[1] != layer.K:
                            has_dora = False
                            lora_A_f16, lora_B_final, dora_gamma = None, None, None
                        else:
                            q_dev = q_packed_aligned_cpu.to(device)
                            lut_dev = lut_c_all.to(device)
                            off_dev = offsets.to(device)
                            col_dev = cols.to(device)
                            val_dev = vals.to(device)

                            W0 = torch.empty((layer.N, layer.K), dtype=torch.float16, device=device)
                            _C.dequant_w0(q_dev, lut_dev, W0, layer.N, layer.K, bits, words_per_row, off_dev, col_dev, val_dev) if hasattr(_C, "dequant_w0") else None

                            with torch.no_grad():
                                if not hasattr(_C, "dequant_w0"):
                                    q_idx_2d = unpack_indices_continuous(q_packed_aligned_cpu, layer.N, words_per_row * (10 if bits == 3 else 32 // bits), bits)[:, :layer.K]
                                    W0 = torch.gather(lut_dev, 1, torch.from_numpy(q_idx_2d.astype(np.int64)).to(device))
                                    if val_dev.numel() > 0:
                                        off_cpu = offsets.numpy()
                                        col_cpu = cols.numpy()
                                        val_fp16 = (val_dev.to(torch.int32) << 8).to(torch.int16).view(torch.float16)
                                        rows_list, cols_list = [], []
                                        for r in range(layer.N):
                                            for c_i in range(num_chunks):
                                                s0, s1 = off_cpu[r, c_i], off_cpu[r, c_i + 1]
                                                if s1 > s0:
                                                    rows_list.append(np.full(s1 - s0, r, dtype=np.int64))
                                                    cols_list.append(col_cpu[s0:s1].astype(np.int64) + c_i * layer_chunk_m)
                                        if rows_list:
                                            W0[torch.from_numpy(np.concatenate(rows_list)).to(device), torch.from_numpy(np.concatenate(cols_list)).to(device)] = val_fp16

                                W0_f32 = W0.float()
                                A_dev = lora_A_final.to(device)
                                B_dev = lora_B_final.to(device)
                                delta_W = lora_scale * (B_dev @ A_dev)
                                V = W0_f32 + delta_W
                                norm_V = torch.norm(V, p=2, dim=1).clamp(min=1e-12)
                                if m_vec is not None and m_vec.numel() == layer.N:
                                    m_vec = m_vec.flatten().to(device)
                                else:
                                    m_vec = torch.norm(W0_f32, p=2, dim=1)
                                dora_gamma = (m_vec / norm_V).to(torch.float16)
                                del W0, W0_f32, delta_W, V

                            lora_A_f16 = lora_A_final.to(device=device, dtype=torch.float16).contiguous()
                            lora_B_final = lora_B_final.to(device=device, dtype=torch.float16).contiguous()

            params_to_clear = [
                "weight", "q_packed", "lut_c", "outlier_mask_1bit", "outlier_mask", "outlier_values",
                "fp8_outliers", "outliers", "ef_tile_meta", "ef_bitstream",
                "ef_tile_size", "orig_shape", "is_quantized", "dora_scale", "bits"
            ]
            for name in params_to_clear:
                if hasattr(layer, "_parameters") and name in layer._parameters:
                    del layer._parameters[name]
                if hasattr(layer, name):
                    try:
                        delattr(layer, name)
                    except Exception:
                        pass

            layer.q_packed_aligned = q_packed_aligned_cpu.to(device)
            layer.words_per_row = words_per_row
            layer.lut_c = lut_c_all.to(device)
            layer.quant_bits = bits
            layer.chunk_offsets = offsets.to(device)
            layer.chunk_cols = cols.to(device)
            layer.chunk_vals = vals.to(device)

            layer.has_dora = has_dora
            layer.lora_A_f16 = lora_A_f16 if has_dora else None
            layer.lora_B = lora_B_final
            layer.dora_gamma = dora_gamma
            layer.lora_scale = lora_scale

            if device.type == "cuda" and (layer.N > WLQLinearMethod._MAX_N or layer.K > WLQLinearMethod._MAX_K):
                WLQLinearMethod._MAX_N = max(WLQLinearMethod._MAX_N, layer.N)
                WLQLinearMethod._MAX_K = max(WLQLinearMethod._MAX_K, layer.K)
                from . import init_workspaces
                init_workspaces(WLQLinearMethod._MAX_N, WLQLinearMethod._MAX_K, device)
            return

        shard_q_indices = []
        shard_luts = []
        shard_masks = []
        shard_fp8s = []
        shard_bits_list = []

        q_packed_shards = get_shards_in_order(layer.q_packed, layer)
        lut_c_shards = get_shards_in_order(layer.lut_c, layer)
        bits_shards = get_shards_in_order(layer.bits, layer) if hasattr(layer, "bits") else []
        meta_shards = get_shards_in_order(layer.ef_tile_meta, layer) if hasattr(layer, "ef_tile_meta") else []
        stream_shards = get_shards_in_order(layer.ef_bitstream, layer) if hasattr(layer, "ef_bitstream") else []
        tile_size_shards = get_shards_in_order(layer.ef_tile_size, layer) if hasattr(layer, "ef_tile_size") else []
        mask1b_shards = get_shards_in_order(layer.outlier_mask_1bit, layer) if hasattr(layer, "outlier_mask_1bit") else []
        vals_shards = get_shards_in_order(layer.outlier_values, layer) if hasattr(layer, "outlier_values") else []

        if len(lut_c_shards) > 0 and all(l.dim() == 2 and l.shape[0] > 0 for l in lut_c_shards):
            if not is_row_parallel and tp_size > 1 and sum(l.shape[0] for l in lut_c_shards) == layer.N * tp_size:
                effective_partition_sizes = [l.shape[0] // tp_size for l in lut_c_shards]
            else:
                effective_partition_sizes = [l.shape[0] for l in lut_c_shards]
        elif len(q_packed_shards) == 1 and num_shards > 1:
            effective_partition_sizes = [sum(output_partition_sizes)]
        else:
            effective_partition_sizes = output_partition_sizes

        for s in range(len(effective_partition_sizes)):
            N_s_part = effective_partition_sizes[s]
            N_s_full = N_s_part * tp_size if (not is_row_parallel and tp_size > 1) else N_s_part

            if len(bits_shards) > 0:
                b_t = bits_shards[s] if s < len(bits_shards) else bits_shards[0]
                bits_s = int(b_t.item()) if b_t.numel() > 0 else self.quant_config.bits
            else:
                bits_s = self.quant_config.bits
            shard_bits_list.append(bits_s)

            q_p = q_packed_shards[s] if s < len(q_packed_shards) else layer.q_packed
            lut_s = (lut_c_shards[s] if s < len(lut_c_shards) else layer.lut_c).cpu()

            loaded_rows = lut_s.shape[0]
            need_row_slice = (not is_row_parallel and tp_size > 1 and loaded_rows == N_s_full)
            N_decode = N_s_full if need_row_slice else N_s_part

            symbols_per_word = 10 if bits_s == 3 else (32 // bits_s)
            min_words = (N_decode * K_full + symbols_per_word - 1) // symbols_per_word
            need_col_slice = (is_row_parallel and tp_size > 1 and q_p.numel() >= min_words)
            K_decode = K_full if need_col_slice else K_part

            q_idx_s = unpack_indices_continuous(q_p, N_decode, K_decode, bits_s)

            tile_sz = 64
            if len(tile_size_shards) > 0:
                t_t = tile_size_shards[s] if s < len(tile_size_shards) else tile_size_shards[0]
                if t_t.numel() > 0:
                    tile_sz = int(t_t.item())

            has_ef = (s < len(meta_shards) and meta_shards[s].numel() > 0 and
                      s < len(stream_shards) and stream_shards[s].numel() > 0)

            if has_ef:
                m_s = decode_elias_fano(
                    meta_shards[s].cpu().numpy(),
                    stream_shards[s].cpu().numpy().view(np.uint32),
                    (N_decode, K_decode),
                    tile_size=tile_sz
                )
            elif s < len(mask1b_shards) and mask1b_shards[s].numel() > 0:
                raw_m = mask1b_shards[s].cpu().numpy()
                m_s = np.unpackbits(raw_m)[:N_decode * K_decode].reshape(N_decode, K_decode).astype(bool)
            else:
                m_s = np.zeros((N_decode, K_decode), dtype=bool)

            if s < len(vals_shards) and vals_shards[s].numel() > 0:
                v_s = vals_shards[s].cpu()
            else:
                v_s = torch.zeros(int(m_s.sum()), dtype=torch.uint8, device="cpu")

            if need_row_slice:
                r0, r1 = tp_rank * N_s_part, (tp_rank + 1) * N_s_part
                q_idx_s = q_idx_s[r0:r1]
                lut_s = lut_s[r0:r1]
                m_s, v_s = slice_outliers_for_tp(m_s, v_s, row_slice=(r0, r1))
            elif need_col_slice:
                c0, c1 = tp_rank * K_part, (tp_rank + 1) * K_part
                q_idx_s = q_idx_s[:, c0:c1]
                m_s, v_s = slice_outliers_for_tp(m_s, v_s, col_slice=(c0, c1))

            shard_q_indices.append(q_idx_s)
            shard_luts.append(lut_s)
            shard_masks.append(m_s)
            shard_fp8s.append(v_s)

        bits = max(shard_bits_list) if shard_bits_list else self.quant_config.bits
        target_k_cb = 1 << bits

        padded_luts_cpu = []
        for lut_s in shard_luts:
            cur_k_cb = lut_s.shape[1]
            if cur_k_cb < target_k_cb:
                lut_s = F.pad(lut_s, (0, target_k_cb - cur_k_cb), value=0.0)
            padded_luts_cpu.append(lut_s)

        q_idx_2d = np.vstack(shard_q_indices) if len(shard_q_indices) > 1 else shard_q_indices[0]
        mask_2d = np.vstack(shard_masks) if len(shard_masks) > 1 else shard_masks[0]
        lut_c_all = torch.cat(padded_luts_cpu, dim=0) if len(padded_luts_cpu) > 1 else padded_luts_cpu[0]

        shard_fp8s_u8 = []
        for v_s in shard_fp8s:
            if v_s.dtype == torch.float16:
                u16 = v_s.view(torch.uint16).to(torch.int32)
                v_s = ((u16 + 0x80) >> 8).clamp(0, 255).to(torch.uint8)
            else:
                v_s = v_s.contiguous().view(torch.uint8)
            shard_fp8s_u8.append(v_s)
        fp8_raw = torch.cat(shard_fp8s_u8) if len(shard_fp8s_u8) > 1 else shard_fp8s_u8[0]

        packed_np, words_per_row = pack_indices_row_aligned(q_idx_2d, bits)

        layer_chunk_m = 640 if bits == 3 else 512
        offsets, cols, vals = build_chunk_csr(mask_2d, fp8_raw, layer_chunk_m)

        dora_index, adapter_config = get_dora_adapter(getattr(self.quant_config, "model_id", None))
        has_dora = False
        lora_A_f32_t, lora_B_final, dora_gamma = None, None, None
        lora_scale = 1.0

        if dora_index:
            vllm_prefix = getattr(layer, "vllm_prefix", "")
            lm = re.search(r"layers\.(\d+)\.", vllm_prefix)
            l_idx = int(lm.group(1)) if lm else None

            if l_idx is not None:
                if "in_proj_qkvz" in vllm_prefix:
                    if (l_idx, "in_proj_qkvz") in dora_index:
                        sub_projs = ["in_proj_qkvz"]
                    else:
                        sub_projs = ["in_proj_qkv", "in_proj_z"]
                elif "in_proj_ba" in vllm_prefix:
                    if (l_idx, "in_proj_ba") in dora_index:
                        sub_projs = ["in_proj_ba"]
                    else:
                        sub_projs = ["in_proj_b", "in_proj_a"]
                elif "in_proj_qkv" in vllm_prefix:
                    sub_projs = ["in_proj_qkv"]
                elif "qkv_proj" in vllm_prefix:
                    sub_projs = ["q_proj", "k_proj", "v_proj"]
                elif "gate_up_proj" in vllm_prefix:
                    sub_projs = ["gate_proj", "up_proj"]
                elif "out_proj" in vllm_prefix:
                    sub_projs = ["out_proj"]
                elif "o_proj" in vllm_prefix:
                    sub_projs = ["o_proj"]
                elif "down_proj" in vllm_prefix:
                    sub_projs = ["down_proj"]
                else:
                    sub_projs = []

                collected_A, collected_B, collected_scale = [], [], []
                for p in sub_projs:
                    entry = dora_index.get((l_idx, p))
                    if entry and entry[0] is not None and entry[1] is not None:
                        A_p = entry[0].float()
                        B_p = entry[1].float()
                        S_p = entry[2].float() if entry[2] is not None else None

                        if tp_size > 1:
                            if is_row_parallel:
                                c0, c1 = tp_rank * K_part, (tp_rank + 1) * K_part
                                A_p = A_p[:, c0:c1]
                            else:
                                n_part_p = B_p.shape[0] // tp_size
                                r0, r1 = tp_rank * n_part_p, (tp_rank + 1) * n_part_p
                                B_p = B_p[r0:r1, :]
                                if S_p is not None:
                                    S_p = S_p.flatten()[r0:r1]

                        collected_A.append(A_p)
                        collected_B.append(B_p)
                        collected_scale.append(S_p)

                if len(collected_A) == len(sub_projs) and len(sub_projs) > 0:
                    has_dora = True
                    r_base = collected_B[0].shape[1]
                    lora_alpha = adapter_config.get("lora_alpha", r_base)
                    lora_scale = float(lora_alpha) / float(r_base)

                    if len(sub_projs) == 1:
                        lora_A_final = collected_A[0]
                        lora_B_final = collected_B[0]
                        m_vec = collected_scale[0]
                    else:
                        lora_A_final = torch.cat(collected_A, dim=0)
                        tot_N = sum(b.shape[0] for b in collected_B)
                        tot_R = sum(b.shape[1] for b in collected_B)
                        lora_B_final = torch.zeros((tot_N, tot_R), dtype=torch.float32)
                        r_off, c_off = 0, 0
                        for b in collected_B:
                            nr, nc = b.shape
                            lora_B_final[r_off : r_off + nr, c_off : c_off + nc] = b
                            r_off += nr
                            c_off += nc
                        m_vec = torch.cat(collected_scale, dim=0) if None not in collected_scale else None

                    with torch.no_grad():
                        q_idx_t = torch.from_numpy(q_idx_2d.astype(np.int64)).to(device)
                        W0 = torch.gather(lut_c_all.to(device), 1, q_idx_t).clone()
                        if fp8_raw.numel() > 0:
                            outlier_mask_dev = torch.from_numpy(mask_2d).to(device)
                            if fp8_raw.dtype == torch.float16:
                                outlier_vals_fp16 = fp8_raw.to(device)
                            else:
                                outlier_vals_fp16 = (fp8_raw.to(device=device, dtype=torch.int32) << 8).to(torch.int16).view(torch.float16)
                            W0[outlier_mask_dev] = outlier_vals_fp16

                        W0_f32 = W0.float()
                        A_dev = lora_A_final.to(device)
                        B_dev = lora_B_final.to(device)
                        delta_W = lora_scale * (B_dev @ A_dev)
                        V = W0_f32 + delta_W
                        norm_V = torch.norm(V, p=2, dim=1).clamp(min=1e-12)
                        m_vec = m_vec.flatten().to(device) if m_vec is not None else torch.norm(W0_f32, p=2, dim=1)
                        dora_gamma = (m_vec / norm_V).to(torch.float16)
                        del W0, W0_f32, delta_W, V

                    lora_A_f16 = lora_A_final.to(device=device, dtype=torch.float16).contiguous()
                    lora_B_final = lora_B_final.to(device=device, dtype=torch.float16).contiguous()

        params_to_clear = [
            "weight", "q_packed", "lut_c", "outlier_mask_1bit", "outlier_mask", "outlier_values",
            "fp8_outliers", "outliers", "ef_tile_meta", "ef_bitstream",
            "ef_tile_size", "orig_shape", "is_quantized", "dora_scale", "bits"
        ]
        for name in params_to_clear:
            if hasattr(layer, "_parameters") and name in layer._parameters:
                del layer._parameters[name]
            if hasattr(layer, name):
                try:
                    delattr(layer, name)
                except Exception:
                    pass

        layer.q_packed_aligned = torch.from_numpy(packed_np.view(np.int32)).to(device)
        layer.words_per_row = words_per_row
        layer.lut_c = lut_c_all.to(device)
        layer.quant_bits = bits
        layer.chunk_offsets = offsets.to(device)
        layer.chunk_cols = cols.to(device)
        layer.chunk_vals = vals.to(device)

        layer.has_dora = has_dora
        layer.lora_A_f16 = lora_A_f16 if has_dora else None
        layer.lora_B = lora_B_final
        layer.dora_gamma = dora_gamma
        layer.lora_scale = lora_scale

        if device.type == "cuda" and (layer.N > WLQLinearMethod._MAX_N or layer.K > WLQLinearMethod._MAX_K):
            WLQLinearMethod._MAX_N = max(WLQLinearMethod._MAX_N, layer.N)
            WLQLinearMethod._MAX_K = max(WLQLinearMethod._MAX_K, layer.K)
            from . import init_workspaces
            init_workspaces(WLQLinearMethod._MAX_N, WLQLinearMethod._MAX_K, device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype

        x_2d = x.reshape(-1, layer.K)
        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        M_batch = x_2d.shape[0]
        out_fp16 = torch.empty((M_batch, layer.N), dtype=torch.float16, device=x.device)

        has_dora = getattr(layer, "has_dora", False)
        lora_B = layer.lora_B if has_dora else None
        dora_gamma = layer.dora_gamma if has_dora else None
        bits = getattr(layer, "quant_bits", self.quant_config.bits)

        if M_batch == 1:
            lora_h = None
            if has_dora and getattr(layer, "lora_A_f16", None) is not None:
                lora_h = F.linear(x_2d.squeeze(0), layer.lora_A_f16).float()

            torch.ops.wlq_fastShot.gemv(
                layer.q_packed_aligned,
                layer.lut_c,
                x_2d.squeeze(0),
                out_fp16.squeeze(0),
                layer.N,
                layer.K,
                bits,
                layer.chunk_offsets,
                layer.chunk_cols,
                layer.chunk_vals,
                lora_B,
                lora_h,
                dora_gamma,
                getattr(layer, "lora_scale", 1.0),
            )
        else:
            lora_H = None
            if has_dora and getattr(layer, "lora_A_f16", None) is not None:
                lora_H = F.linear(x_2d, layer.lora_A_f16).float()

            torch.ops.wlq_fastShot.gemm(
                x_2d,
                layer.q_packed_aligned,
                layer.lut_c,
                out_fp16,
                layer.N,
                layer.K,
                bits,
                layer.words_per_row,
                layer.chunk_offsets,
                layer.chunk_cols,
                layer.chunk_vals,
                lora_H,
                lora_B,
                dora_gamma,
                getattr(layer, "lora_scale", 1.0),
            )

        out = out_fp16 if orig_dtype == torch.float16 else out_fp16.to(orig_dtype)

        if bias is not None:
            out.add_(bias)

        return out.view(*orig_shape[:-1], layer.N)