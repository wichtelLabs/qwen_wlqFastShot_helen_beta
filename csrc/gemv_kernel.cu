#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>

__device__ __forceinline__ uint32_t to_shared_addr(const void* ptr) {
    uint32_t smem_addr;
    asm volatile(
        "{ .reg .u64 u64ptr; cvta.to.shared.u64 u64ptr, %1; cvt.u32.u64 %0, u64ptr; }\n"
        : "=r"(smem_addr) : "l"(ptr)
    );
    return smem_addr;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem_addr = to_shared_addr(smem_ptr);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(glob_ptr)
        : "memory"
    );
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
#endif

__device__ __forceinline__ uint32_t ptx_bfe(uint32_t val, uint32_t start, uint32_t len) {
    uint32_t ret;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(start), "r"(len));
    return ret;
}

template <int CHUNK_M, int ROWS_PER_BLOCK>
struct SharedStorage {
    alignas(16) half s_x[2][CHUNK_M];             
    alignas(16) half s_lut[ROWS_PER_BLOCK][16];  
};

template <int CHUNK_M>
__device__ __forceinline__ void load_x_chunk_async(
    const half* __restrict__ x,
    half s_x[CHUNK_M],
    int col_chunk,
    int k_limit,
    int tid
) {
    constexpr int CHUNK_BYTES = CHUNK_M * sizeof(half);
    constexpr int U4_COUNT = CHUNK_BYTES / 16; // 16 Bytes = 8 half

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    #pragma unroll
    for (int idx = tid; idx < U4_COUNT; idx += blockDim.x) {
        const int col = col_chunk + idx * 8;
        if (col + 8 <= k_limit) {
            cp_async_16(&s_x[idx * 8], &x[col]);
        } else {
            #pragma unroll
            for (int sub = 0; sub < 8; ++sub) {
                s_x[idx * 8 + sub] = (col + sub < k_limit) ? x[col + sub] : __float2half(0.0f);
            }
        }
    }
    #else
    #pragma unroll
    for (int idx = tid; idx < U4_COUNT; idx += blockDim.x) {
        const int col = col_chunk + idx * 8;
        if (col + 8 <= k_limit) {
            *reinterpret_cast<uint4*>(&s_x[idx * 8]) = *reinterpret_cast<const uint4*>(&x[col]);
        } else {
            #pragma unroll
            for (int sub = 0; sub < 8; ++sub) {
                s_x[idx * 8 + sub] = (col + sub < k_limit) ? x[col + sub] : __float2half(0.0f);
            }
        }
    }
    #endif
}

template <int BITS, int ROWS_PER_BLOCK, int CHUNK_M, bool HAS_DORA>
__global__ void gemv_chunk_csr_kernel(
    const uint32_t* __restrict__ q_packed,
    const half* __restrict__ lut_c,
    const half* __restrict__ x,
    half* __restrict__ out,
    float* __restrict__ partial_accum,
    const int32_t* __restrict__ chunk_offsets,
    const uint16_t* __restrict__ chunk_cols,
    const uint8_t* __restrict__ chunk_vals,
    const half* __restrict__ lora_B,
    const float* __restrict__ lora_h,
    const half* __restrict__ dora_gamma,
    const float lora_scale,
    const int r_rank,
    const int N,
    const int M,
    const int num_chunks
) {
    constexpr int WARP_SIZE = 32;
    constexpr uint32_t FULL_WARP_MASK = 0xFFFFFFFFu;
    constexpr int K_CB = 1 << BITS;

    extern __shared__ uint8_t smem_raw[];
    auto& s = *reinterpret_cast<SharedStorage<CHUNK_M, ROWS_PER_BLOCK>*>(smem_raw);

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    const int block_row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int row = block_row_base + warp_id;
    const bool valid_row = (row < N);

    const int split_idx  = blockIdx.y;
    const int num_splits = gridDim.y;
    const int chunks_per_split = (num_chunks + num_splits - 1) / num_splits;
    const int chunk_start = split_idx * chunks_per_split;
    const int chunk_end   = min(num_chunks, chunk_start + chunks_per_split);

    if (lane_id < 16) {
        s.s_lut[warp_id][lane_id] = (valid_row && lane_id < K_CB)
            ? lut_c[static_cast<int64_t>(row) * K_CB + lane_id]
            : __float2half(0.0f);
    }
    __syncthreads();

    int words_per_row = 0;
    if constexpr (BITS == 1)      words_per_row = (((M + 31) >> 5) + 3) & ~3;
    else if constexpr (BITS == 2) words_per_row = (((M + 15) >> 4) + 3) & ~3;
    else if constexpr (BITS == 3) words_per_row = (((M + 9) / 10) + 3) & ~3;
    else if constexpr (BITS == 4) words_per_row = (((M + 7) >> 3) + 3) & ~3;

    const uint32_t* q_row = valid_row ? (q_packed + static_cast<int64_t>(row) * words_per_row) : nullptr;
    const uint4* q_row_u4 = reinterpret_cast<const uint4*>(q_row);

    float accum0 = 0.0f;
    int read_buf = 0;
    int write_buf = 0;

    if (chunk_start < chunk_end) {
        load_x_chunk_async<CHUNK_M>(x, s.s_x[0], chunk_start * CHUNK_M, M, tid);
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        cp_async_commit();
        cp_async_wait_group<0>();
        #endif
        __syncthreads();
    }

    for (int c_idx = chunk_start; c_idx < chunk_end; ++c_idx) {
        const int col_chunk = c_idx * CHUNK_M;
        const int next_c_idx = c_idx + 1;
        write_buf = read_buf ^ 1;

        if (next_c_idx < chunk_end) {
            load_x_chunk_async<CHUNK_M>(x, s.s_x[write_buf], next_c_idx * CHUNK_M, M, tid);
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            cp_async_commit();
            #endif
        }

        if (valid_row) {
            if constexpr (BITS == 4) {
                if (lane_id < 16) {
                    const int u4_idx = (col_chunk >> 5) + lane_id;
                    const uint4 q4 = (u4_idx < (words_per_row >> 2)) ? q_row_u4[u4_idx] : make_uint4(0, 0, 0, 0);
                    const uint32_t words[4] = {q4.x, q4.y, q4.z, q4.w};
                    const int c_base_thread = lane_id * 32;

                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t word = words[w];
                        const int c_w = c_base_thread + w * 8;
                        #pragma unroll
                        for (int s_idx = 0; s_idx < 8; ++s_idx) {
                            const int c_local = c_w + s_idx;
                            if (col_chunk + c_local < M) {
                                const uint32_t sym = (word >> (s_idx * 4)) & 15;
                                accum0 = fmaf(__half2float(s.s_lut[warp_id][sym]),
                                              __half2float(s.s_x[read_buf][c_local]),
                                              accum0);
                            }
                        }
                    }
                }
            } else if constexpr (BITS == 2) {
                if (lane_id < 8) {
                    const int u4_idx = (col_chunk >> 6) + lane_id;
                    const uint4 q4 = (u4_idx < (words_per_row >> 2)) ? q_row_u4[u4_idx] : make_uint4(0, 0, 0, 0);
                    const uint32_t words[4] = {q4.x, q4.y, q4.z, q4.w};
                    const int c_base_thread = lane_id * 64;

                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t word = words[w];
                        const int c_w = c_base_thread + w * 16;
                        #pragma unroll
                        for (int s_idx = 0; s_idx < 16; ++s_idx) {
                            const int c_local = c_w + s_idx;
                            if (col_chunk + c_local < M) {
                                const uint32_t sym = (word >> (s_idx * 2)) & 3;
                                accum0 = fmaf(__half2float(s.s_lut[warp_id][sym]),
                                              __half2float(s.s_x[read_buf][c_local]),
                                              accum0);
                            }
                        }
                    }
                }
            } else if constexpr (BITS == 1) {
                if (lane_id < 4) {
                    const int u4_idx = (col_chunk >> 7) + lane_id;
                    const uint4 q4 = (u4_idx < (words_per_row >> 2)) ? q_row_u4[u4_idx] : make_uint4(0, 0, 0, 0);
                    const uint32_t words[4] = {q4.x, q4.y, q4.z, q4.w};
                    const int c_base_thread = lane_id * 128;

                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t word = words[w];
                        const int c_w = c_base_thread + w * 32;
                        #pragma unroll
                        for (int s_idx = 0; s_idx < 32; ++s_idx) {
                            const int c_local = c_w + s_idx;
                            if (col_chunk + c_local < M) {
                                const uint32_t sym = (word >> s_idx) & 1;
                                accum0 = fmaf(__half2float(s.s_lut[warp_id][sym]),
                                              __half2float(s.s_x[read_buf][c_local]),
                                              accum0);
                            }
                        }
                    }
                }
            } else if constexpr (BITS == 3) {
                if (lane_id < 16) {
                    const int u4_idx = (col_chunk / 40) + lane_id;
                    const uint4 q4 = (u4_idx < (words_per_row >> 2)) ? q_row_u4[u4_idx] : make_uint4(0, 0, 0, 0);
                    const uint32_t words[4] = {q4.x, q4.y, q4.z, q4.w};
                    const int c_base_thread = lane_id * 40;

                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const uint32_t word = words[w];
                        const int c_w = c_base_thread + w * 10;
                        #pragma unroll
                        for (int s_idx = 0; s_idx < 10; ++s_idx) {
                            const int c_local = c_w + s_idx;
                            if (col_chunk + c_local < M) {
                                const uint32_t sym = (word >> (s_idx * 3)) & 7;
                                accum0 = fmaf(__half2float(s.s_lut[warp_id][sym]),
                                              __half2float(s.s_x[read_buf][c_local]),
                                              accum0);
                            }
                        }
                    }
                }
            }

            if (chunk_offsets != nullptr && chunk_cols != nullptr && chunk_vals != nullptr) {
                const int64_t offset_base = static_cast<int64_t>(row) * (num_chunks + 1) + c_idx;
                const int32_t start_off = chunk_offsets[offset_base];
                const int32_t end_off   = chunk_offsets[offset_base + 1];
                const int m_outliers    = end_off - start_off;

                #pragma unroll
                for (int i = lane_id; i < m_outliers; i += WARP_SIZE) {
                    const int32_t out_idx  = start_off + i;
                    const uint16_t c_local = chunk_cols[out_idx];
                    const uint8_t  val_u8  = chunk_vals[out_idx];
                    const half w_true_h = __ushort_as_half(static_cast<uint16_t>(val_u8) << 8);
                    const float w_true  = __half2float(w_true_h);

                    const int col = col_chunk + c_local;
                    if (col < M) {
                        uint32_t sym = 0;
                        if constexpr (BITS == 1)      sym = ptx_bfe(__ldg(&q_row[col >> 5]), col & 31, 1);
                        else if constexpr (BITS == 2) sym = ptx_bfe(__ldg(&q_row[col >> 4]), (col & 15) * 2, 2);
                        else if constexpr (BITS == 3) sym = ptx_bfe(__ldg(&q_row[col / 10]), (col % 10) * 3, 3);
                        else if constexpr (BITS == 4) sym = ptx_bfe(__ldg(&q_row[col >> 3]), (col & 7) * 4, 4);

                        const float delta_w = w_true - __half2float(s.s_lut[warp_id][sym]);
                        accum0 = fmaf(delta_w, __half2float(s.s_x[read_buf][c_local]), accum0);
                    }
                }
            }
        }

        if (next_c_idx < chunk_end) {
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            cp_async_wait_group<0>();
            #endif
            __syncthreads();
            read_buf = write_buf;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        accum0 += __shfl_down_sync(FULL_WARP_MASK, accum0, offset);
    }

    if (lane_id == 0 && valid_row) {
        if (num_splits > 1) {
            partial_accum[static_cast<int64_t>(split_idx) * N + row] = accum0;
        } else {
            if constexpr (HAS_DORA) {
                float lora_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < r_rank; ++k) {
                    lora_sum += lora_h[k] * __half2float(lora_B[static_cast<int64_t>(row) * r_rank + k]);
                }
                accum0 = (accum0 + lora_scale * lora_sum) * __half2float(dora_gamma[row]);
            }
            out[row] = __float2half_rn(accum0);
        }
    }
}

template <bool HAS_DORA>
__global__ void reduce_split_k_dora_kernel(
    const float* __restrict__ partial_accum,
    half* __restrict__ out,
    const half* __restrict__ lora_B,
    const float* __restrict__ lora_h,
    const half* __restrict__ dora_gamma,
    const float lora_scale,
    const int r_rank,
    const int N,
    const int num_splits
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < num_splits; ++s) {
        sum += partial_accum[static_cast<int64_t>(s) * N + row];
    }

    if constexpr (HAS_DORA) {
        float lora_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < r_rank; ++k) {
            lora_sum += lora_h[k] * __half2float(lora_B[static_cast<int64_t>(row) * r_rank + k]);
        }
        sum = (sum + lora_scale * lora_sum) * __half2float(dora_gamma[row]);
    }

    out[row] = __float2half_rn(sum);
}

inline int get_device_sm_count() {
    static int num_sms = -1;
    if (num_sms == -1) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return num_sms;
}

template <int BITS, int ROWS_PER_BLOCK, int CHUNK_M, bool HAS_DORA>
inline void launch_with_rows(
    dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
    const uint32_t* q_u32, const half* lut_half, const half* x_half, half* out_half,
    float* partial_accum, const int32_t* chunk_offsets_ptr, const uint16_t* chunk_cols_ptr,
    const uint8_t* chunk_vals_ptr, const half* lora_B, const float* lora_h, const half* dora_gamma,
    float lora_scale, int r_rank, int N, int M, int num_chunks
) {
    const size_t smem_size = sizeof(SharedStorage<CHUNK_M, ROWS_PER_BLOCK>);
    auto k_fn = gemv_chunk_csr_kernel<BITS, ROWS_PER_BLOCK, CHUNK_M, HAS_DORA>;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(k_fn), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
    }

    k_fn<<<grid_dim, block_dim, smem_size, stream>>>(
        q_u32, lut_half, x_half, out_half, partial_accum,
        chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr,
        lora_B, lora_h, dora_gamma, lora_scale, r_rank, N, M, num_chunks
    );

    if (grid_dim.y > 1) {
        const dim3 red_block(256);
        const dim3 red_grid((N + 255) / 256);
        reduce_split_k_dora_kernel<HAS_DORA><<<red_grid, red_block, 0, stream>>>(
            partial_accum, out_half, lora_B, lora_h, dora_gamma,
            lora_scale, r_rank, N, grid_dim.y
        );
    }
}

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
) {
    constexpr int THREADS_PER_BLOCK = 128;
    constexpr int ROWS_PER_BLOCK = THREADS_PER_BLOCK / 32; 

    const int tiles_n = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int num_sms = get_device_sm_count();

    int num_splits = 1;
    if (tiles_n < num_sms * 2 && partial_accum_ptr != nullptr) {
        const int target_splits = (num_sms * 2 + tiles_n - 1) / tiles_n;
        num_splits = std::min(target_splits, num_chunks);
        num_splits = std::min(num_splits, 32); 
        if (num_splits < 1) num_splits = 1;
    }

    const dim3 block_dim(THREADS_PER_BLOCK);
    const dim3 grid_dim(tiles_n, num_splits);

    const uint32_t* q_u32 = reinterpret_cast<const uint32_t*>(q_packed_ptr);
    const half* lut_half = reinterpret_cast<const half*>(lut_c_ptr);
    const half* x_half = reinterpret_cast<const half*>(x_ptr);
    half* out_half = reinterpret_cast<half*>(out_ptr);

    const half* lora_B = reinterpret_cast<const half*>(lora_B_ptr);
    const half* dora_gamma = reinterpret_cast<const half*>(dora_gamma_ptr);
    const bool has_dora = (lora_B != nullptr && lora_h_ptr != nullptr && dora_gamma != nullptr && r > 0);

    if (has_dora) {
        switch (bits) {
            case 1: launch_with_rows<1, ROWS_PER_BLOCK, 512, true>(grid_dim, block_dim, stream, q_u32, lut_half, x_half, out_half, partial_accum_ptr, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, lora_B, lora_h_ptr, dora_gamma, lora_scale, r, N, M, num_chunks); break;
            case 2: launch_with_rows<2, ROWS_PER_BLOCK, 512, true>(grid_dim, block_dim, stream, q_u32, lut_half, x_half, out_half, partial_accum_ptr, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, lora_B, lora_h_ptr, dora_gamma, lora_scale, r, N, M, num_chunks); break;
            case 3: launch_with_rows<3, ROWS_PER_BLOCK, 640, true>(grid_dim, block_dim, stream, q_u32, lut_half, x_half, out_half, partial_accum_ptr, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, lora_B, lora_h_ptr, dora_gamma, lora_scale, r, N, M, num_chunks); break;
            case 4: launch_with_rows<4, ROWS_PER_BLOCK, 512, true>(grid_dim, block_dim, stream, q_u32, lut_half, x_half, out_half, partial_accum_ptr, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, lora_B, lora_h_ptr, dora_gamma, lora_scale, r, N, M, num_chunks); break;
        }
    } else {
        switch (bits) {
            case 1: launch_with_rows<1, ROWS_PER_BLOCK, 512, false>(grid_dim, block_dim, stream, q_u32, lut_half, x_half, out_half, partial_accum_ptr, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, nullptr, nullptr, nullptr, 0.0f, 0, N, M, num_chunks); break;
            case 2: launch_with_rows<2, ROWS_PER_BLOCK, 512, false>(grid_dim, block_dim, stream, q_u32, lut_half, x_half, out_half, partial_accum_ptr, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, nullptr, nullptr, nullptr, 0.0f, 0, N, M, num_chunks); break;
            case 3: launch_with_rows<3, ROWS_PER_BLOCK, 640, false>(grid_dim, block_dim, stream, q_u32, lut_half, x_half, out_half, partial_accum_ptr, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, nullptr, nullptr, nullptr, 0.0f, 0, N, M, num_chunks); break;
            case 4: launch_with_rows<4, ROWS_PER_BLOCK, 512, false>(grid_dim, block_dim, stream, q_u32, lut_half, x_half, out_half, partial_accum_ptr, chunk_offsets_ptr, chunk_cols_ptr, chunk_vals_ptr, nullptr, nullptr, nullptr, 0.0f, 0, N, M, num_chunks); break;
        }
    }
}
