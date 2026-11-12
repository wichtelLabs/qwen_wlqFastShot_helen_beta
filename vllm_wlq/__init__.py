"""
vLLM Metalog Quantization Plugin: wlq_fastShot
"""
import torch
from . import _C
from typing import Optional

try:
    @torch.library.custom_op("wlq_fastShot::gemv", mutates_args={"out"})
    def _gemv_op(
        q_packed: torch.Tensor,
        lut_c: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        N: int,
        K: int,
        bits: int,
        chunk_offsets: Optional[torch.Tensor] = None,
        chunk_cols: Optional[torch.Tensor] = None,
        chunk_vals: Optional[torch.Tensor] = None,
        lora_B: Optional[torch.Tensor] = None,
        lora_h: Optional[torch.Tensor] = None,
        dora_gamma: Optional[torch.Tensor] = None,
        lora_scale: float = 1.0,
    ) -> None:
        _C.gemv_forward(
            q_packed, lut_c, x, out,
            int(N), int(K), int(bits),
            chunk_offsets, chunk_cols, chunk_vals,
            lora_B, lora_h, dora_gamma, float(lora_scale)
        )

    # Für CUDA Graphs: Teilt PyTorch mit, dass die Operation "out" in-place mutiert
    @_gemv_op.register_fake
    def _gemv_fake(
        q_packed, lut_c, x, out, N, K, bits,
        chunk_offsets=None, chunk_cols=None, chunk_vals=None,
        lora_B=None, lora_h=None, dora_gamma=None, lora_scale=1.0
    ):
        return None

    @torch.library.custom_op("wlq_fastShot::gemm", mutates_args={"out"})
    def _gemm_op(
        X: torch.Tensor,
        q_packed: torch.Tensor,
        lut_c: torch.Tensor,
        out: torch.Tensor,
        N: int,
        K: int,
        bits: int,
        words_per_row: int,
        chunk_offsets: Optional[torch.Tensor] = None,
        chunk_cols: Optional[torch.Tensor] = None,
        chunk_vals: Optional[torch.Tensor] = None,
        lora_H: Optional[torch.Tensor] = None,
        lora_B: Optional[torch.Tensor] = None,
        dora_gamma: Optional[torch.Tensor] = None,
        lora_scale: float = 1.0,
    ) -> None:
        _C.gemm_forward(
            X, q_packed, lut_c, out,
            int(N), int(K), int(bits), int(words_per_row),
            chunk_offsets, chunk_cols, chunk_vals,
            lora_H, lora_B, dora_gamma, float(lora_scale)
        )

    # Für CUDA Graphs: Fake Implementation für GEMM
    @_gemm_op.register_fake
    def _gemm_fake(
        X, q_packed, lut_c, out, N, K, bits, words_per_row,
        chunk_offsets=None, chunk_cols=None, chunk_vals=None,
        lora_H=None, lora_B=None, dora_gamma=None, lora_scale=1.0
    ):
        return None

except Exception:
    pass

def init_workspaces(max_N: int, max_K: int, device: torch.device):
    if hasattr(_C, "init_workspaces"):
        _C.init_workspaces(int(max_N), int(max_K), device)

def register():
    """vLLM General Plugin Entrypoint"""
    pass

from .quant_config import WLQFastShotConfig
from .linear_method import WLQLinearMethod

__all__ = ["WLQFastShotConfig", "WLQLinearMethod"]
