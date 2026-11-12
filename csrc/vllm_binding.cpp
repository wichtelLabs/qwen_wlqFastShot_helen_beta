#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

extern "C" void launch_gemv_chunk_csr(
    const int32_t* q_packed_ptr,
    const void* lut_c_ptr,
    const void* x_ptr,
    void* out_ptr,
    float* partial_accum_ptr,
    const int32_t* chunk_offsets_ptr,
    const uint16_t* chunk_cols_ptr,
    const uint8_t* chunk_vals_ptr,
    const void* lora_B_ptr,
    const float* lora_h_ptr,
    const void* dora_gamma_ptr,
    float lora_scale,
    int r,
    int N,
    int M,
    int bits,
    int num_chunks,
    cudaStream_t stream
);

extern "C" void launch_gemm_metalog_dora(
    const void* X_ptr,
    const int32_t* q_packed_ptr,
    const void* lut_c_ptr,
    void* out_ptr,
    const int32_t* chunk_offsets_ptr,
    const uint16_t* chunk_cols_ptr,
    const uint8_t* chunk_vals_ptr,
    const float* lora_H_ptr,
    const void* lora_B_ptr,
    const void* dora_gamma_ptr,
    float lora_scale,
    int r_rank,
    int M,
    int N,
    int K,
    int bits,
    int words_per_row,
    int num_chunks,
    int chunk_m,
    cudaStream_t stream
);

extern "C" void launch_dequant_patch_metalog(
    const int32_t* q_packed_ptr,
    const void* lut_c_ptr,
    void* W_out_ptr,
    const int32_t* chunk_offsets_ptr,
    const uint16_t* chunk_cols_ptr,
    const uint8_t* chunk_vals_ptr,
    int N,
    int K,
    int bits,
    int words_per_row,
    int num_chunks,
    int chunk_m,
    cudaStream_t stream
);

extern "C" void launch_fused_dora_epilogue(
    void* out_ptr,
    const float* lora_H_ptr,
    const void* lora_B_ptr,
    const void* dora_gamma_ptr,
    float lora_scale,
    int r_rank,
    int M,
    int N,
    cudaStream_t stream
);

static inline uint32_t read_bits_u32(const uint32_t* __restrict__ data, int64_t total_words, int64_t bit_pos, int n_bits) {
    if (n_bits <= 0) return 0;
    const int64_t w_idx = bit_pos >> 5;
    const int offset = static_cast<int>(bit_pos & 31);
    uint64_t val = static_cast<uint64_t>(data[w_idx]) >> offset;
    const int bits_read = 32 - offset;
    if (n_bits > bits_read && (w_idx + 1) < total_words) {
        val |= static_cast<uint64_t>(data[w_idx + 1]) << bits_read;
    }
    return static_cast<uint32_t>(val & ((1ULL << n_bits) - 1ULL));
}

std::tuple<torch::Tensor, torch::Tensor> decode_ef_to_chunk_csr_cpp(
    torch::Tensor ef_tile_meta,   
    torch::Tensor ef_bitstream,   
    int64_t N,
    int64_t K,
    int64_t tile_size,
    int64_t chunk_m
) {
    auto meta_cpu = ef_tile_meta.to(torch::kCPU, torch::kInt64).contiguous();
    auto stream_cpu = ef_bitstream.to(torch::kCPU).contiguous();

    const int64_t* meta_ptr = meta_cpu.data_ptr<int64_t>();
    const uint32_t* bs_ptr = reinterpret_cast<const uint32_t*>(stream_cpu.data_ptr());
    const int64_t total_words = stream_cpu.numel();
    const int64_t total_tiles = meta_cpu.size(0);

    const int64_t pad_w = (tile_size - (K % tile_size)) % tile_size;
    const int64_t W_pad = K + pad_w;
    const int64_t num_tiles_x = W_pad / tile_size;
    const int64_t U = tile_size * tile_size;
    const int64_t num_chunks = (K + chunk_m - 1) / chunk_m;

    int64_t total_outliers = 0;
    for (int64_t t = 0; t < total_tiles; ++t) {
        total_outliers += meta_ptr[t * 2 + 1];
    }

    auto chunk_offsets = torch::zeros({N, num_chunks + 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    if (total_outliers == 0) {
        auto empty_cols = torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCPU));
        return std::make_tuple(chunk_offsets, empty_cols);
    }

    std::vector<uint64_t> coords;
    coords.reserve(static_cast<size_t>(total_outliers));

    for (int64_t k = 0; k < total_tiles; ++k) {
        const int64_t start_bit = meta_ptr[k * 2];
        const int64_t m = meta_ptr[k * 2 + 1];
        if (m <= 0) continue;

        const int64_t tile_y = k / num_tiles_x;
        const int64_t tile_x = k % num_tiles_x;
        const int64_t base_r = tile_y * tile_size;
        const int64_t base_c = tile_x * tile_size;

        int L = 0;
        if (U > m) {
            L = static_cast<int>(std::floor(std::log2(static_cast<double>(U) / static_cast<double>(m))));
            if (L < 0) L = 0;
        }

        int64_t cur_bit = start_bit + m * L;
        int32_t cur_high = 0;

        for (int64_t i = 0; i < m; ++i) {
            const uint32_t low = (L > 0) ? read_bits_u32(bs_ptr, total_words, start_bit + i * L, L) : 0U;
            while (cur_bit < total_words * 32) {
                const uint32_t bit = (bs_ptr[cur_bit >> 5] >> (cur_bit & 31)) & 1U;
                ++cur_bit;
                if (bit == 0U) {
                    ++cur_high;
                } else {
                    break;
                }
            }
            const int64_t local_idx = (static_cast<int64_t>(cur_high) << L) | static_cast<int64_t>(low);
            const int64_t r = base_r + (local_idx / tile_size);
            const int64_t c = base_c + (local_idx % tile_size);
            if (r < N && c < K) {
                coords.push_back((static_cast<uint64_t>(r) << 32) | static_cast<uint64_t>(c));
            }
        }
    }

    std::sort(coords.begin(), coords.end());

    const int64_t valid_outliers = static_cast<int64_t>(coords.size());
    auto chunk_cols = torch::empty({valid_outliers}, torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCPU));
    int32_t* off_ptr = chunk_offsets.data_ptr<int32_t>();
    uint16_t* cols_ptr = reinterpret_cast<uint16_t*>(chunk_cols.data_ptr());

    for (int64_t i = 0; i < valid_outliers; ++i) {
        const uint64_t rc = coords[i];
        const int64_t r = static_cast<int64_t>(rc >> 32);
        const int64_t c = static_cast<int64_t>(rc & 0xFFFFFFFFULL);
        const int64_t c_idx = c / chunk_m;
        cols_ptr[i] = static_cast<uint16_t>(c % chunk_m);
        off_ptr[r * (num_chunks + 1) + c_idx + 1] += 1;
    }

    int32_t running = 0;
    for (int64_t r = 0; r < N; ++r) {
        int32_t* row_off = off_ptr + r * (num_chunks + 1);
        row_off[0] = running;
        for (int64_t c = 0; c < num_chunks; ++c) {
            running += row_off[c + 1];
            row_off[c + 1] = running;
        }
    }

    return std::make_tuple(chunk_offsets, chunk_cols);
}

std::tuple<torch::Tensor, int64_t> repack_q_indices_row_aligned_cpp(
    torch::Tensor q_packed_raw,
    int64_t N,
    int64_t K,
    int64_t bits
) {
    auto q_cpu = q_packed_raw.to(torch::kCPU).contiguous();
    const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(q_cpu.data_ptr());
    const int64_t src_words = q_cpu.numel();

    const int64_t symbols_per_word = (bits == 3) ? 10 : (32 / bits);
    const int64_t raw_words_per_row = (K + symbols_per_word - 1) / symbols_per_word;
    const int64_t words_per_row = ((raw_words_per_row + 3) / 4) * 4;
    const uint32_t sym_mask = (1U << bits) - 1U;

    auto out_tensor = torch::zeros({N * words_per_row}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    uint32_t* dst_u32 = reinterpret_cast<uint32_t*>(out_tensor.data_ptr());

    int64_t global_sym_idx = 0;
    for (int64_t r = 0; r < N; ++r) {
        uint32_t* dst_row = dst_u32 + r * words_per_row;
        for (int64_t w = 0; w < raw_words_per_row; ++w) {
            uint32_t packed_word = 0U;
            const int64_t syms_in_this_word = std::min<int64_t>(symbols_per_word, K - w * symbols_per_word);
            for (int64_t s = 0; s < syms_in_this_word; ++s) {
                const int64_t src_w = global_sym_idx / symbols_per_word;
                const int64_t src_s = global_sym_idx % symbols_per_word;
                const uint32_t sym = (src_w < src_words)
                    ? ((src_u32[src_w] >> (src_s * bits)) & sym_mask)
                    : 0U;
                packed_word |= (sym << (s * bits));
                ++global_sym_idx;
            }
            dst_row[w] = packed_word;
        }
    }

    return std::make_tuple(out_tensor, words_per_row);
}

static torch::Tensor g_split_k_workspace;
static torch::Tensor g_w_workspace;

void init_workspaces(int64_t max_N, int64_t max_K, torch::Device device) {
    constexpr int MAX_SPLITS = 32;
    const int64_t req_floats = static_cast<int64_t>(MAX_SPLITS) * max_N;
    if (!g_split_k_workspace.defined() || g_split_k_workspace.device() != device || g_split_k_workspace.numel() < req_floats) {
        g_split_k_workspace = torch::empty({MAX_SPLITS, max_N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }
    const int64_t req_w_elems = max_N * max_K;
    if (!g_w_workspace.defined() || g_w_workspace.device() != device || g_w_workspace.numel() < req_w_elems) {
        g_w_workspace = torch::empty({req_w_elems}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    }
}

void gemv_forward(
    torch::Tensor q_packed,
    torch::Tensor lut_c,
    torch::Tensor x,
    torch::Tensor out,
    int64_t N,
    int64_t K,
    int64_t bits,
    c10::optional<torch::Tensor> chunk_offsets,
    c10::optional<torch::Tensor> chunk_cols,
    c10::optional<torch::Tensor> chunk_vals,
    c10::optional<torch::Tensor> lora_B,
    c10::optional<torch::Tensor> lora_h,
    c10::optional<torch::Tensor> dora_gamma,
    double lora_scale
) {
    TORCH_CHECK(q_packed.is_cuda() && lut_c.is_cuda() && x.is_cuda() && out.is_cuda(), "Tensoren muessen auf CUDA liegen");
    TORCH_CHECK(q_packed.is_contiguous() && lut_c.is_contiguous() && x.is_contiguous() && out.is_contiguous(), "Tensoren muessen contiguous sein");

    const int32_t* chunk_offsets_ptr = (chunk_offsets.has_value() && chunk_offsets->defined() && chunk_offsets->numel() > 0)
        ? chunk_offsets->data_ptr<int32_t>() : nullptr;
    const uint16_t* chunk_cols_ptr = (chunk_cols.has_value() && chunk_cols->defined() && chunk_cols->numel() > 0)
        ? reinterpret_cast<const uint16_t*>(chunk_cols->data_ptr()) : nullptr;
    const uint8_t* chunk_vals_ptr = (chunk_vals.has_value() && chunk_vals->defined() && chunk_vals->numel() > 0)
        ? reinterpret_cast<const uint8_t*>(chunk_vals->data_ptr()) : nullptr;

    const int chunk_m = (bits == 3) ? 640 : 512;
    const int num_chunks = static_cast<int>((K + chunk_m - 1) / chunk_m);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* lora_h_ptr = (lora_h.has_value() && lora_h->defined() && lora_h->numel() > 0) ? lora_h->data_ptr<float>() : nullptr;
    const void* lora_B_ptr = (lora_B.has_value() && lora_B->defined() && lora_B->numel() > 0) ? lora_B->data_ptr() : nullptr;
    const void* dora_gamma_ptr = (dora_gamma.has_value() && dora_gamma->defined() && dora_gamma->numel() > 0) ? dora_gamma->data_ptr() : nullptr;
    int r_rank = (lora_B.has_value() && lora_B->defined() && lora_B->dim() > 1) ? static_cast<int>(lora_B->size(1)) : 0;

    constexpr int MAX_SPLITS = 32;
    const int64_t req_floats = static_cast<int64_t>(MAX_SPLITS) * N;
    if (!g_split_k_workspace.defined() || g_split_k_workspace.device() != x.device() || g_split_k_workspace.numel() < req_floats) {
        g_split_k_workspace = torch::empty({MAX_SPLITS, N}, x.options().dtype(torch::kFloat32));
    }

    launch_gemv_chunk_csr(
        q_packed.data_ptr<int32_t>(),
        lut_c.data_ptr(),
        x.data_ptr(),
        out.data_ptr(),
        g_split_k_workspace.data_ptr<float>(),
        chunk_offsets_ptr,
        chunk_cols_ptr,
        chunk_vals_ptr,
        lora_B_ptr,
        lora_h_ptr,
        dora_gamma_ptr,
        static_cast<float>(lora_scale),
        r_rank,
        static_cast<int>(N),
        static_cast<int>(K),
        static_cast<int>(bits),
        num_chunks,
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void gemm_forward(
    torch::Tensor X,
    torch::Tensor q_packed,
    torch::Tensor lut_c,
    torch::Tensor out,
    int64_t N,
    int64_t K,
    int64_t bits,
    int64_t words_per_row,
    c10::optional<torch::Tensor> chunk_offsets,
    c10::optional<torch::Tensor> chunk_cols,
    c10::optional<torch::Tensor> chunk_vals,
    c10::optional<torch::Tensor> lora_H,
    c10::optional<torch::Tensor> lora_B,
    c10::optional<torch::Tensor> dora_gamma,
    double lora_scale
) {
    TORCH_CHECK(X.is_cuda() && q_packed.is_cuda() && lut_c.is_cuda() && out.is_cuda(), "Tensoren muessen auf CUDA liegen");
    TORCH_CHECK(X.is_contiguous() && q_packed.is_contiguous() && lut_c.is_contiguous() && out.is_contiguous(), "Tensoren muessen contiguous sein");

    const int64_t M = X.size(0);
    const int32_t* chunk_offsets_ptr = (chunk_offsets.has_value() && chunk_offsets->defined() && chunk_offsets->numel() > 0)
        ? chunk_offsets->data_ptr<int32_t>() : nullptr;
    const uint16_t* chunk_cols_ptr = (chunk_cols.has_value() && chunk_cols->defined() && chunk_cols->numel() > 0)
        ? reinterpret_cast<const uint16_t*>(chunk_cols->data_ptr()) : nullptr;
    const uint8_t* chunk_vals_ptr = (chunk_vals.has_value() && chunk_vals->defined() && chunk_vals->numel() > 0)
        ? reinterpret_cast<const uint8_t*>(chunk_vals->data_ptr()) : nullptr;

    const float* lora_H_ptr = (lora_H.has_value() && lora_H->defined() && lora_H->numel() > 0) ? lora_H->data_ptr<float>() : nullptr;
    const void* lora_B_ptr = (lora_B.has_value() && lora_B->defined() && lora_B->numel() > 0) ? lora_B->data_ptr() : nullptr;
    const void* dora_gamma_ptr = (dora_gamma.has_value() && dora_gamma->defined() && dora_gamma->numel() > 0) ? dora_gamma->data_ptr() : nullptr;
    int r_rank = (lora_B.has_value() && lora_B->defined() && lora_B->dim() > 1) ? static_cast<int>(lora_B->size(1)) : 0;

    const int chunk_m = (bits == 3) ? 640 : 512;
    const int num_chunks = static_cast<int>((K + chunk_m - 1) / chunk_m);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (M >= 16) {
        const int64_t req_w_elems = N * K;
        if (!g_w_workspace.defined() || g_w_workspace.device() != X.device() ||
            g_w_workspace.numel() < req_w_elems) {
            g_w_workspace = torch::empty({req_w_elems}, X.options());
        }
        torch::Tensor W_deq = g_w_workspace.narrow(0, 0, req_w_elems).view({N, K});

        launch_dequant_patch_metalog(
            q_packed.data_ptr<int32_t>(),
            lut_c.data_ptr(),
            W_deq.data_ptr(),
            chunk_offsets_ptr,
            chunk_cols_ptr,
            chunk_vals_ptr,
            static_cast<int>(N),
            static_cast<int>(K),
            static_cast<int>(bits),
            static_cast<int>(words_per_row),
            num_chunks,
            chunk_m,
            stream
        );

        at::mm_out(out, X, W_deq.t());

        if (lora_H_ptr != nullptr && lora_B_ptr != nullptr && dora_gamma_ptr != nullptr && r_rank > 0) {
            launch_fused_dora_epilogue(
                out.data_ptr(),
                lora_H_ptr,
                lora_B_ptr,
                dora_gamma_ptr,
                static_cast<float>(lora_scale),
                r_rank,
                static_cast<int>(M),
                static_cast<int>(N),
                stream
            );
        }
    } else {
        launch_gemm_metalog_dora(
            X.data_ptr(),
            q_packed.data_ptr<int32_t>(),
            lut_c.data_ptr(),
            out.data_ptr(),
            chunk_offsets_ptr,
            chunk_cols_ptr,
            chunk_vals_ptr,
            lora_H_ptr,
            lora_B_ptr,
            dora_gamma_ptr,
            static_cast<float>(lora_scale),
            r_rank,
            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K),
            static_cast<int>(bits),
            static_cast<int>(words_per_row),
            num_chunks,
            chunk_m,
            stream
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dequant_w0_forward(
    torch::Tensor q_packed,
    torch::Tensor lut_c,
    torch::Tensor W_out,
    int64_t N,
    int64_t K,
    int64_t bits,
    int64_t words_per_row,
    torch::Tensor chunk_offsets,
    torch::Tensor chunk_cols,
    torch::Tensor chunk_vals
) {
    const int chunk_m = (bits == 3) ? 640 : 512;
    const int num_chunks = static_cast<int>((K + chunk_m - 1) / chunk_m);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_dequant_patch_metalog(
        q_packed.data_ptr<int32_t>(),
        lut_c.data_ptr(),
        W_out.data_ptr(),
        chunk_offsets.numel() > 0 ? chunk_offsets.data_ptr<int32_t>() : nullptr,
        chunk_cols.numel() > 0 ? reinterpret_cast<const uint16_t*>(chunk_cols.data_ptr()) : nullptr,
        chunk_vals.numel() > 0 ? reinterpret_cast<const uint8_t*>(chunk_vals.data_ptr()) : nullptr,
        static_cast<int>(N),
        static_cast<int>(K),
        static_cast<int>(bits),
        static_cast<int>(words_per_row),
        num_chunks,
        chunk_m,
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dequant_w0",
        &dequant_w0_forward,
        "Fast GPU W0 reconstruction for DoRA gamma calibration"
    );

    m.def(
        "decode_ef_to_chunk_csr",
        &decode_ef_to_chunk_csr_cpp,
        "Fast C++ Elias-Fano to Chunk-CSR decoder",
        py::arg("ef_tile_meta"),
        py::arg("ef_bitstream"),
        py::arg("N"),
        py::arg("K"),
        py::arg("tile_size"),
        py::arg("chunk_m")
    );

    m.def(
        "repack_q_indices_row_aligned",
        &repack_q_indices_row_aligned_cpp,
        "Fast C++ continuous-to-row-aligned q_packed repacker",
        py::arg("q_packed_raw"),
        py::arg("N"),
        py::arg("K"),
        py::arg("bits")
    );

    m.def(
        "init_workspaces",
        &init_workspaces,
        "Pre-allocate CUDA workspaces for CUDA Graph safety",
        py::arg("max_N"),
        py::arg("max_K"),
        py::arg("device")
    );

    m.def(
        "gemv_forward",
        &gemv_forward,
        "WLQ FastShot GEMV (Decode M=1)",
        py::arg("q_packed"),
        py::arg("lut_c"),
        py::arg("x"),
        py::arg("out"),
        py::arg("N"),
        py::arg("K"),
        py::arg("bits"),
        py::arg("chunk_offsets") = py::none(),
        py::arg("chunk_cols") = py::none(),
        py::arg("chunk_vals") = py::none(),
        py::arg("lora_B") = py::none(),
        py::arg("lora_h") = py::none(),
        py::arg("dora_gamma") = py::none(),
        py::arg("lora_scale") = 1.0
    );

    m.def(
        "gemm_forward",
        &gemm_forward,
        "WLQ FastShot GEMM (Prefill M>1)",
        py::arg("X"),
        py::arg("q_packed"),
        py::arg("lut_c"),
        py::arg("out"),
        py::arg("N"),
        py::arg("K"),
        py::arg("bits"),
        py::arg("words_per_row"),
        py::arg("chunk_offsets") = py::none(),
        py::arg("chunk_cols") = py::none(),
        py::arg("chunk_vals") = py::none(),
        py::arg("lora_H") = py::none(),
        py::arg("lora_B") = py::none(),
        py::arg("dora_gamma") = py::none(),
        py::arg("lora_scale") = 1.0
    );
}