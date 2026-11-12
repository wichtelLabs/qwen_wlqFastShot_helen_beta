#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

using namespace nvcuda;

constexpr int GEMM_BLOCK_K = 128;
constexpr int GEMM_PAD_K   = 8;
constexpr int GEMM_STRIDE_K = GEMM_BLOCK_K + GEMM_PAD_K; 

constexpr int GEMM_OUT_PAD_N = 8;
constexpr int GEMM_STRIDE_N  = GEMM_BLOCK_N + GEMM_OUT_PAD_N; 


template <int BLOCK_M>
struct GemmSharedStorage {
    alignas(16) half s_lut[GEMM_BLOCK_N][16]; 

    union {
        struct {
            alignas(16) half smem_W[GEMM_BLOCK_N][GEMM_STRIDE_K];
            alignas(16) half smem_X[BLOCK_M][GEMM_STRIDE_K];    
        };
        alignas(16) float smem_out[BLOCK_M][GEMM_STRIDE_N];       
    };
};


template <int BITS, int BLOCK_M, bool HAS_DORA>
__global__ void gemm_metalog_dora_kernel(
    const half* __restrict__ X,               
    const uint32_t* __restrict__ q_packed,     
    const half* __restrict__ lut_c,           
    half* __restrict__ out,                    // [M, N]
    const int32_t* __restrict__ chunk_offsets,
    const uint16_t* __restrict__ chunk_cols,   
    const uint8_t* __restrict__ chunk_vals,   
    const float* __restrict__ lora_H,          
    const half* __restrict__ lora_B,           
    const half* __restrict__ dora_gamma,       // [N]
    const float lora_scale,
    const int r_rank,
    const int M,
    const int N,
    const int K,
    const int words_per_row,
    const int num_chunks,
    const int chunk_m
) {
    constexpr int K_CB = 1 << BITS;
    extern __shared__ uint8_t raw_smem[];
    auto& s = *reinterpret_cast<GemmSharedStorage<BLOCK_M>*>(raw_smem);

    const int tid = threadIdx.x;
    const int wid = tid >> 5; 

    constexpr int WARP_M_STEPS = (BLOCK_M == 64) ? 2 : 1;
    constexpr int WARP_N_STEPS = (BLOCK_M == 64) ? 2 : 1;

    int warp_m_offset = 0;
    int warp_n_offset = 0;
    if constexpr (BLOCK_M == 16) {
        warp_m_offset = 0;
        warp_n_offset = wid * 16;
    } else {
        const int w_row = wid >> 1; // 0..1
        const int w_col = wid & 1;  // 0..1
        warp_m_offset = w_row * 32;
        warp_n_offset = w_col * 32;
    }

    const int tile_n = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int block_n_base = tile_n * GEMM_BLOCK_N;
    const int block_m_base = tile_m * BLOCK_M;

    if (block_n_base >= N || block_m_base >= M) return;

    for (int idx = tid; idx < GEMM_BLOCK_N * K_CB; idx += blockDim.x) {
        const int r_n = idx / K_CB;
        const int sym = idx % K_CB;
        const int global_n = block_n_base + r_n;
        s.s_lut[r_n][sym] = (global_n < N) ? lut_c[static_cast<int64_t>(global_n) * K_CB + sym] : __float2half(0.0f);
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[WARP_M_STEPS][WARP_N_STEPS];
    #pragma unroll
    for (int mi = 0; mi < WARP_M_STEPS; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_N_STEPS; ++ni) {
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    for (int k_base = 0; k_base < K; k_base += GEMM_BLOCK_K) {
        constexpr int U4_PER_ROW_X = GEMM_BLOCK_K / 8; 
        constexpr int TOTAL_U4_X   = BLOCK_M * U4_PER_ROW_X;

        for (int idx = tid; idx < TOTAL_U4_X; idx += blockDim.x) {
            const int r_m = idx / U4_PER_ROW_X;
            const int v_k = idx % U4_PER_ROW_X;
            const int gm  = block_m_base + r_m;
            const int gk  = k_base + v_k * 8;

            uint4* smem_row_u4 = reinterpret_cast<uint4*>(&s.smem_X[r_m][0]);
            if (gm < M && gk + 8 <= K) {
                const uint4* g_ptr = reinterpret_cast<const uint4*>(&X[static_cast<int64_t>(gm) * K + gk]);
                smem_row_u4[v_k] = *g_ptr;
            } else {
                #pragma unroll
                for (int sub = 0; sub < 8; ++sub) {
                    s.smem_X[r_m][v_k * 8 + sub] = (gm < M && gk + sub < K)
                        ? X[static_cast<int64_t>(gm) * K + gk + sub]
                        : __float2half(0.0f);
                }
            }
        }

        constexpr int U4_PER_ROW_Q = (BITS == 4) ? 4 : ((BITS == 2) ? 2 : ((BITS == 1) ? 1 : 4));
        constexpr int TOTAL_U4_Q   = GEMM_BLOCK_N * U4_PER_ROW_Q;

        for (int v = tid; v < TOTAL_U4_Q; v += blockDim.x) {
            const int r_n   = v / U4_PER_ROW_Q;
            const int v_sub = v % U4_PER_ROW_Q;
            const int gn    = block_n_base + r_n;

            int k_word_start = 0;
            if constexpr (BITS == 4) {
                k_word_start = (k_base >> 3) + (v_sub << 2);
            } else if constexpr (BITS == 2) {
                k_word_start = (k_base >> 4) + (v_sub << 2);
            } else if constexpr (BITS == 1) {
                k_word_start = (k_base >> 5) + (v_sub << 2);
            } else if constexpr (BITS == 3) {
                k_word_start = (k_base / 10) + (v_sub << 2);
            }

            uint4 q4 = make_uint4(0, 0, 0, 0);
            if (gn < N && k_word_start < words_per_row) {
                const int64_t g_word_idx = static_cast<int64_t>(gn) * words_per_row + k_word_start;
                if (k_word_start + 4 <= words_per_row) {
                    q4 = *reinterpret_cast<const uint4*>(&q_packed[g_word_idx]);
                } else {
                    uint32_t* q4_arr = reinterpret_cast<uint32_t*>(&q4);
                    for (int w = 0; w < 4 && k_word_start + w < words_per_row; ++w) {
                        q4_arr[w] = q_packed[g_word_idx + w];
                    }
                }
            }

            const uint32_t words[4] = {q4.x, q4.y, q4.z, q4.w};
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t word = words[w];
                if constexpr (BITS == 4) {
                    const int c_base_k = (v_sub * 4 + w) * 8;
                    #pragma unroll
                    for (int s_idx = 0; s_idx < 8; ++s_idx) {
                        const int c_k = c_base_k + s_idx;
                        if (c_k < GEMM_BLOCK_K) {
                            const uint32_t sym = (word >> (s_idx * 4)) & 15;
                            s.smem_W[r_n][c_k] = (gn < N && k_base + c_k < K) ? s.s_lut[r_n][sym] : __float2half(0.0f);
                        }
                    }
                } else if constexpr (BITS == 2) {
                    const int c_base_k = (v_sub * 4 + w) * 16;
                    #pragma unroll
                    for (int s_idx = 0; s_idx < 16; ++s_idx) {
                        const int c_k = c_base_k + s_idx;
                        if (c_k < GEMM_BLOCK_K) {
                            const uint32_t sym = (word >> (s_idx * 2)) & 3;
                            s.smem_W[r_n][c_k] = (gn < N && k_base + c_k < K) ? s.s_lut[r_n][sym] : __float2half(0.0f);
                        }
                    }
                } else if constexpr (BITS == 1) {
                    const int c_base_k = (v_sub * 4 + w) * 32;
                    #pragma unroll
                    for (int s_idx = 0; s_idx < 32; ++s_idx) {
                        const int c_k = c_base_k + s_idx;
                        if (c_k < GEMM_BLOCK_K) {
                            const uint32_t sym = (word >> s_idx) & 1;
                            s.smem_W[r_n][c_k] = (gn < N && k_base + c_k < K) ? s.s_lut[r_n][sym] : __float2half(0.0f);
                        }
                    }
                } else if constexpr (BITS == 3) {
                    const int c_base_k = (v_sub * 4 + w) * 10;
                    #pragma unroll
                    for (int s_idx = 0; s_idx < 10; ++s_idx) {
                        const int c_k = c_base_k + s_idx;
                        if (c_k < GEMM_BLOCK_K) {
                            const uint32_t sym = (word >> (s_idx * 3)) & 7;
                            s.smem_W[r_n][c_k] = (gn < N && k_base + c_k < K) ? s.s_lut[r_n][sym] : __float2half(0.0f);
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (chunk_offsets != nullptr && chunk_cols != nullptr && chunk_vals != nullptr) {
            const int c_idx = k_base / chunk_m;
            const int k_sub = k_base % chunk_m;

            const int r_n      = tid >> 1; // 0 .. 63
            const int sub_lane = tid & 1;  // 0 .. 1

            if (r_n < GEMM_BLOCK_N) {
                const int gn = block_n_base + r_n;
                if (gn < N) {
                    const int64_t off_base = static_cast<int64_t>(gn) * (num_chunks + 1) + c_idx;
                    const int start_off = chunk_offsets[off_base];
                    const int end_off   = chunk_offsets[off_base + 1];

                    for (int i = start_off + sub_lane; i < end_off; i += 2) {
                        const int c_local = chunk_cols[i];
                        if (c_local >= k_sub && c_local < k_sub + GEMM_BLOCK_K) {
                            const int c_k = c_local - k_sub;
                            const uint8_t val_u8 = chunk_vals[i];
                            s.smem_W[r_n][c_k] = __ushort_as_half(static_cast<uint16_t>(val_u8) << 8);
                        }
                    }
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < GEMM_BLOCK_K; k_step += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[WARP_M_STEPS];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[WARP_N_STEPS];

            #pragma unroll
            for (int mi = 0; mi < WARP_M_STEPS; ++mi) {
                wmma::load_matrix_sync(a_frag[mi], &s.smem_X[warp_m_offset + mi * 16][k_step], GEMM_STRIDE_K);
            }

            #pragma unroll
            for (int ni = 0; ni < WARP_N_STEPS; ++ni) {
                wmma::load_matrix_sync(b_frag[ni], &s.smem_W[warp_n_offset + ni * 16][k_step], GEMM_STRIDE_K);
            }

            #pragma unroll
            for (int mi = 0; mi < WARP_M_STEPS; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < WARP_N_STEPS; ++ni) {
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < WARP_M_STEPS; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_N_STEPS; ++ni) {
            wmma::store_matrix_sync(
                &s.smem_out[warp_m_offset + mi * 16][warp_n_offset + ni * 16],
                c_frag[mi][ni],
                GEMM_STRIDE_N,
                wmma::mem_row_major
            );
        }
    }
    __syncthreads();

    for (int idx = tid; idx < BLOCK_M * GEMM_BLOCK_N; idx += blockDim.x) {
        const int r_m = idx / GEMM_BLOCK_N;
        const int r_n = idx % GEMM_BLOCK_N;
        const int gm = block_m_base + r_m;
        const int gn = block_n_base + r_n;

        if (gm < M && gn < N) {
            float val = s.smem_out[r_m][r_n];

            if constexpr (HAS_DORA) {
                float lora_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < r_rank; ++k) {
                    const float h_val = lora_H[static_cast<int64_t>(gm) * r_rank + k];
                    const half  b_val = lora_B[static_cast<int64_t>(gn) * r_rank + k];
                    lora_sum += h_val * __half2float(b_val);
                }
                val = (val + lora_scale * lora_sum) * __half2float(dora_gamma[gn]);
            }

            out[static_cast<int64_t>(gm) * N + gn] = __float2half_rn(val);
        }
    }
}


template <int BITS, bool HAS_DORA>
inline void dispatch_gemm_m(
    int M, int N, int K,
    const half* X, const uint32_t* q_packed, const half* lut_c, half* out,
    const int32_t* chunk_offsets, const uint16_t* chunk_cols, const uint8_t* chunk_vals,
    const float* lora_H, const half* lora_B, const half* dora_gamma,
    float lora_scale, int r_rank, int words_per_row, int num_chunks, int chunk_m,
    cudaStream_t stream
) {
    constexpr int BM = 16;
    constexpr int THREADS = 128; 
    const dim3 block(THREADS);
    const int num_tiles_n = (N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N;
    const int num_tiles_m = (M + BM - 1) / BM;
    const dim3 grid(num_tiles_n, num_tiles_m);

    const size_t smem = sizeof(GemmSharedStorage<BM>); // ~23.8 KB (< 48 KB Standard-Limit)
    gemm_metalog_dora_kernel<BITS, BM, HAS_DORA><<<grid, block, smem, stream>>>(
        X, q_packed, lut_c, out, chunk_offsets, chunk_cols, chunk_vals,
        lora_H, lora_B, dora_gamma, lora_scale, r_rank, M, N, K,
        words_per_row, num_chunks, chunk_m
    );
}


template <int BITS>
__global__ void dequant_patch_metalog_kernel(
    const uint32_t* __restrict__ q_packed,     
    const half* __restrict__ lut_c,            // [N, K_cb]
    half* __restrict__ W_out,                  // [N, K]
    const int32_t* __restrict__ chunk_offsets, 
    const uint16_t* __restrict__ chunk_cols,   
    const uint8_t* __restrict__ chunk_vals,   
    const int N,
    const int K,
    const int words_per_row,
    const int num_chunks,
    const int chunk_m
) {
    constexpr int K_CB = 1 << BITS;
    __shared__ half s_lut[K_CB];

    const int row = blockIdx.x;
    if (row >= N) return;

    const int tid = threadIdx.x;

    if (tid < K_CB) {
        s_lut[tid] = lut_c[static_cast<int64_t>(row) * K_CB + tid];
    }
    __syncthreads();

    const uint32_t* q_row = q_packed + static_cast<int64_t>(row) * words_per_row;
    half* w_row           = W_out + static_cast<int64_t>(row) * K;

    if constexpr (BITS == 4) {
        const int num_words = (K + 7) >> 3;
        for (int w = tid; w < num_words; w += blockDim.x) {
            const uint32_t word = q_row[w];
            const int col_base  = w << 3;
            if (col_base + 8 <= K) {
                half out_buf[8];
                #pragma unroll
                for (int s = 0; s < 8; ++s) {
                    out_buf[s] = s_lut[(word >> (s * 4)) & 15];
                }
                *reinterpret_cast<uint4*>(&w_row[col_base]) = *reinterpret_cast<uint4*>(out_buf);
            } else {
                for (int s = 0; s < 8 && col_base + s < K; ++s) {
                    w_row[col_base + s] = s_lut[(word >> (s * 4)) & 15];
                }
            }
        }
    } else if constexpr (BITS == 2) {
        const int num_words = (K + 15) >> 4;
        for (int w = tid; w < num_words; w += blockDim.x) {
            const uint32_t word = q_row[w];
            const int col_base  = w << 4;
            if (col_base + 16 <= K) {
                half out_buf[16];
                #pragma unroll
                for (int s = 0; s < 16; ++s) {
                    out_buf[s] = s_lut[(word >> (s * 2)) & 3];
                }
                uint4* dst = reinterpret_cast<uint4*>(&w_row[col_base]);
                uint4* src = reinterpret_cast<uint4*>(out_buf);
                dst[0] = src[0];
                dst[1] = src[1];
            } else {
                for (int s = 0; s < 16 && col_base + s < K; ++s) {
                    w_row[col_base + s] = s_lut[(word >> (s * 2)) & 3];
                }
            }
        }
    } else if constexpr (BITS == 1) {
        const int num_words = (K + 31) >> 5;
        for (int w = tid; w < num_words; w += blockDim.x) {
            const uint32_t word = q_row[w];
            const int col_base  = w << 5;
            if (col_base + 32 <= K) {
                half out_buf[32];
                #pragma unroll
                for (int s = 0; s < 32; ++s) {
                    out_buf[s] = s_lut[(word >> s) & 1];
                }
                uint4* dst = reinterpret_cast<uint4*>(&w_row[col_base]);
                uint4* src = reinterpret_cast<uint4*>(out_buf);
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            } else {
                for (int s = 0; s < 32 && col_base + s < K; ++s) {
                    w_row[col_base + s] = s_lut[(word >> s) & 1];
                }
            }
        }
    } else if constexpr (BITS == 3) {
        const int num_words = (K + 9) / 10;
        for (int w = tid; w < num_words; w += blockDim.x) {
            const uint32_t word = q_row[w];
            const int col_base  = w * 10;
            #pragma unroll
            for (int s = 0; s < 10; ++s) {
                if (col_base + s < K) {
                    w_row[col_base + s] = s_lut[(word >> (s * 3)) & 7];
                }
            }
        }
    }
    __syncthreads();

    if (chunk_offsets != nullptr && chunk_cols != nullptr && chunk_vals != nullptr) {
        const int64_t row_off_base = static_cast<int64_t>(row) * (num_chunks + 1);
        for (int c_idx = 0; c_idx < num_chunks; ++c_idx) {
            const int start_off = chunk_offsets[row_off_base + c_idx];
            const int end_off   = chunk_offsets[row_off_base + c_idx + 1];
            const int col_chunk_base = c_idx * chunk_m;

            for (int i = start_off + tid; i < end_off; i += blockDim.x) {
                const int col = col_chunk_base + chunk_cols[i];
                if (col < K) {
                    const uint8_t val_u8 = chunk_vals[i];
                    w_row[col] = __ushort_as_half(static_cast<uint16_t>(val_u8) << 8);
                }
            }
        }
    }
}

__global__ void fused_dora_epilogue_kernel(
    half* __restrict__ out,              // [M, N]
    const float* __restrict__ lora_H,    // [M, r]
    const half* __restrict__ lora_B,     // [N, r]
    const half* __restrict__ dora_gamma, // [N]
    const float lora_scale,
    const int r_rank,
    const int M,
    const int N
) {
    const int gn = blockIdx.x * blockDim.x + threadIdx.x;
    const int gm = blockIdx.y;

    if (gm < M && gn < N) {
        const int64_t idx = static_cast<int64_t>(gm) * N + gn;
        float val = __half2float(out[idx]);

        float lora_sum = 0.0f;
        for (int k = 0; k < r_rank; ++k) {
            lora_sum += lora_H[static_cast<int64_t>(gm) * r_rank + k] *
                        __half2float(lora_B[static_cast<int64_t>(gn) * r_rank + k]);
        }

        val = (val + lora_scale * lora_sum) * __half2float(dora_gamma[gn]);
        out[idx] = __float2half_rn(val);
    }
}

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
) {
    const uint32_t* q_packed = reinterpret_cast<const uint32_t*>(q_packed_ptr);
    const half* lut_c = reinterpret_cast<const half*>(lut_c_ptr);
    half* W_out = reinterpret_cast<half*>(W_out_ptr);

    const dim3 grid(N);
    const dim3 block(256);

    switch (bits) {
        case 1: dequant_patch_metalog_kernel<1><<<grid, block, 0, stream>>>(q_packed, lut_c, W_out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, N, K, words_per_row, num_chunks, chunk_m); break;
        case 2: dequant_patch_metalog_kernel<2><<<grid, block, 0, stream>>>(q_packed, lut_c, W_out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, N, K, words_per_row, num_chunks, chunk_m); break;
        case 3: dequant_patch_metalog_kernel<3><<<grid, block, 0, stream>>>(q_packed, lut_c, W_out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, N, K, words_per_row, num_chunks, chunk_m); break;
        case 4: dequant_patch_metalog_kernel<4><<<grid, block, 0, stream>>>(q_packed, lut_c, W_out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, N, K, words_per_row, num_chunks, chunk_m); break;
    }
}

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
) {
    half* out = reinterpret_cast<half*>(out_ptr);
    const half* lora_B = reinterpret_cast<const half*>(lora_B_ptr);
    const half* dora_gamma = reinterpret_cast<const half*>(dora_gamma_ptr);

    const dim3 block(256);
    const dim3 grid((N + 255) / 256, M);
    fused_dora_epilogue_kernel<<<grid, block, 0, stream>>>(
        out, lora_H_ptr, lora_B, dora_gamma, lora_scale, r_rank, M, N
    );
}

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
) {
    const half* X = reinterpret_cast<const half*>(X_ptr);
    const uint32_t* q_packed = reinterpret_cast<const uint32_t*>(q_packed_ptr);
    const half* lut_c = reinterpret_cast<const half*>(lut_c_ptr);
    half* out = reinterpret_cast<half*>(out_ptr);

    const half* lora_B = reinterpret_cast<const half*>(lora_B_ptr);
    const half* dora_gamma = reinterpret_cast<const half*>(dora_gamma_ptr);

    const bool has_dora = (lora_H_ptr != nullptr && lora_B_ptr != nullptr && dora_gamma_ptr != nullptr && r_rank > 0);

    if (has_dora) {
        switch (bits) {
            case 1:
                dispatch_gemm_m<1, true>(M, N, K, X, q_packed, lut_c, out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, lora_H_ptr, lora_B, dora_gamma, lora_scale, r_rank, words_per_row, num_chunks, chunk_m, stream);
                break;
            case 2:
                dispatch_gemm_m<2, true>(M, N, K, X, q_packed, lut_c, out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, lora_H_ptr, lora_B, dora_gamma, lora_scale, r_rank, words_per_row, num_chunks, chunk_m, stream);
                break;
            case 3:
                dispatch_gemm_m<3, true>(M, N, K, X, q_packed, lut_c, out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, lora_H_ptr, lora_B, dora_gamma, lora_scale, r_rank, words_per_row, num_chunks, chunk_m, stream);
                break;
            case 4:
                dispatch_gemm_m<4, true>(M, N, K, X, q_packed, lut_c, out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, lora_H_ptr, lora_B, dora_gamma, lora_scale, r_rank, words_per_row, num_chunks, chunk_m, stream);
                break;
            default:
                printf("[CUDA ERROR] GEMM Bits %d nicht unterstützt!\n", bits);
                break;
        }
    } else {
        switch (bits) {
            case 1:
                dispatch_gemm_m<1, false>(M, N, K, X, q_packed, lut_c, out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, nullptr, nullptr, nullptr, 0.0f, 0, words_per_row, num_chunks, chunk_m, stream);
                break;
            case 2:
                dispatch_gemm_m<2, false>(M, N, K, X, q_packed, lut_c, out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, nullptr, nullptr, nullptr, 0.0f, 0, words_per_row, num_chunks, chunk_m, stream);
                break;
            case 3:
                dispatch_gemm_m<3, false>(M, N, K, X, q_packed, lut_c, out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, nullptr, nullptr, nullptr, 0.0f, 0, words_per_row, num_chunks, chunk_m, stream);
                break;
            case 4:
                dispatch_gemm_m<4, false>(M, N, K, X, q_packed, lut_c, out, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, nullptr, nullptr, nullptr, 0.0f, 0, words_per_row, num_chunks, chunk_m, stream);
                break;
            default:
                printf("[CUDA ERROR] GEMM Bits %d nicht unterstützt!\n", bits);
                break;
        }
    }
}
