#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#if defined(_WIN32)
#define SIRT_API __declspec(dllexport)
#else
#define SIRT_API __attribute__((visibility("default")))
#endif

using SirtProgressCallback = void (*)(int, int, void*);
using SirtCancelCallback = int (*)(void*);

static int run_sirt_loop(
    int num_rays,
    int num_pixels,
    int nz,
    int iterations,
    float d_h,
    const float* b_flat,
    const float* V_target,
    const int* A_indptr,
    const int* A_indices,
    const float* A_data,
    const int* AT_indptr,
    const int* AT_indices,
    const float* AT_data,
    const float* R_norm,
    const float* C_norm,
    const bool* ray_hits_any,
    float* b_opt,
    SirtProgressCallback progress,
    SirtCancelCallback cancel,
    void* context
) {
    if (num_rays <= 0 || num_pixels <= 0 || nz <= 0 || iterations <= 0 ||
        b_flat == nullptr || V_target == nullptr || A_indptr == nullptr ||
        AT_indptr == nullptr || R_norm == nullptr || C_norm == nullptr ||
        ray_hits_any == nullptr || b_opt == nullptr) {
        return -1;
    }

    if (cancel != nullptr && cancel(context)) {
        return 1;
    }

    const std::size_t ray_values = static_cast<std::size_t>(num_rays) * nz;
    const std::size_t pixel_values = static_cast<std::size_t>(num_pixels) * nz;
    std::copy(b_flat, b_flat + ray_values, b_opt);

    std::vector<float> dose(pixel_values, 0.0f);
    std::vector<float> error(pixel_values, 0.0f);

    for (int it = 0; it < iterations; ++it) {
        if (cancel != nullptr && cancel(context)) {
            return 1;
        }

        // Reconstruct the delivered dose: dose = A^T * b_opt.
        #pragma omp parallel for schedule(dynamic)
        for (int p = 0; p < num_pixels; ++p) {
            const int start = AT_indptr[p];
            const int end = AT_indptr[p + 1];
            float* d_ptr = &dose[static_cast<std::size_t>(p) * nz];
            std::fill(d_ptr, d_ptr + nz, 0.0f);

            for (int idx = start; idx < end; ++idx) {
                const int r = AT_indices[idx];
                const float val = AT_data[idx];
                const float* b_ptr = &b_opt[static_cast<std::size_t>(r) * nz];
                #pragma omp simd
                for (int z = 0; z < nz; ++z) {
                    d_ptr[z] += val * b_ptr[z];
                }
            }
        }

        float dose_max = 0.0f;
        #pragma omp parallel
        {
            float thread_max = 0.0f;
            #pragma omp for nowait
            for (long long i = 0; i < static_cast<long long>(pixel_values); ++i) {
                const float value = dose[static_cast<std::size_t>(i)];
                if (value > thread_max) {
                    thread_max = value;
                }
            }
            #pragma omp critical
            {
                if (thread_max > dose_max) {
                    dose_max = thread_max;
                }
            }
        }
        if (dose_max <= 0.0f) {
            dose_max = 1.0f;
        }

        const float inv_dose_max = 1.0f / dose_max;
        #pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(pixel_values); ++i) {
            const std::size_t index = static_cast<std::size_t>(i);
            const float dose_norm = dose[index] * inv_dose_max;
            float dose_arg = -15.0f * (dose_norm - d_h);
            dose_arg = std::max(-80.0f, std::min(80.0f, dose_arg));
            const float dose_thresh = 1.0f / (1.0f + std::exp(dose_arg));
            error[index] = V_target[index] - dose_thresh;
        }

        // Update each ray independently. Reuse one scratch vector per worker
        // instead of allocating a vector for every ray on every iteration.
        #pragma omp parallel
        {
            std::vector<float> delta(static_cast<std::size_t>(nz));
            #pragma omp for schedule(dynamic)
            for (int r = 0; r < num_rays; ++r) {
                float* b_ptr = &b_opt[static_cast<std::size_t>(r) * nz];
                if (!ray_hits_any[r]) {
                    std::fill(b_ptr, b_ptr + nz, 0.0f);
                    continue;
                }

                std::fill(delta.begin(), delta.end(), 0.0f);
                const int start = A_indptr[r];
                const int end = A_indptr[r + 1];
                const float r_norm_val = R_norm[r];

                for (int idx = start; idx < end; ++idx) {
                    const int p = A_indices[idx];
                    const float combined_val = A_data[idx] * C_norm[p];
                    const float* err_ptr = &error[static_cast<std::size_t>(p) * nz];
                    #pragma omp simd
                    for (int z = 0; z < nz; ++z) {
                        delta[static_cast<std::size_t>(z)] += combined_val * err_ptr[z];
                    }
                }

                #pragma omp simd
                for (int z = 0; z < nz; ++z) {
                    const float new_val = b_ptr[z] + r_norm_val * delta[static_cast<std::size_t>(z)];
                    b_ptr[z] = std::max(new_val, 0.0f);
                }
            }
        }

        if (progress != nullptr) {
            progress(it + 1, iterations, context);
        }
    }
    return 0;
}

extern "C" {

SIRT_API void forward_project(
    int num_rays,
    int num_pixels,
    int nz,
    const float* V_target,
    const int* A_indptr,
    const int* A_indices,
    const float* A_data,
    float* P0
) {
    if (num_rays <= 0 || num_pixels <= 0 || nz <= 0 || V_target == nullptr ||
        A_indptr == nullptr || P0 == nullptr) {
        return;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < num_rays; ++r) {
        float* p_ptr = &P0[static_cast<std::size_t>(r) * nz];
        std::fill(p_ptr, p_ptr + nz, 0.0f);
        for (int idx = A_indptr[r]; idx < A_indptr[r + 1]; ++idx) {
            const int p = A_indices[idx];
            if (p < 0 || p >= num_pixels) {
                continue;
            }
            const float val = A_data[idx];
            const float* v_ptr = &V_target[static_cast<std::size_t>(p) * nz];
            #pragma omp simd
            for (int z = 0; z < nz; ++z) {
                p_ptr[z] += val * v_ptr[z];
            }
        }
    }
}

SIRT_API int sirt_loop_ex(
    int num_rays,
    int num_pixels,
    int nz,
    int iterations,
    float d_h,
    const float* b_flat,
    const float* V_target,
    const int* A_indptr,
    const int* A_indices,
    const float* A_data,
    const int* AT_indptr,
    const int* AT_indices,
    const float* AT_data,
    const float* R_norm,
    const float* C_norm,
    const bool* ray_hits_any,
    float* b_opt,
    SirtProgressCallback progress,
    SirtCancelCallback cancel,
    void* context
) {
    return run_sirt_loop(
        num_rays, num_pixels, nz, iterations, d_h, b_flat, V_target,
        A_indptr, A_indices, A_data, AT_indptr, AT_indices, AT_data,
        R_norm, C_norm, ray_hits_any, b_opt, progress, cancel, context
    );
}

SIRT_API void sirt_loop(
    int num_rays,
    int num_pixels,
    int nz,
    int iterations,
    float d_h,
    const float* b_flat,
    const float* V_target,
    const int* A_indptr,
    const int* A_indices,
    const float* A_data,
    const int* AT_indptr,
    const int* AT_indices,
    const float* AT_data,
    const float* R_norm,
    const float* C_norm,
    const bool* ray_hits_any,
    float* b_opt
) {
    (void)run_sirt_loop(
        num_rays, num_pixels, nz, iterations, d_h, b_flat, V_target,
        A_indptr, A_indices, A_data, AT_indptr, AT_indices, AT_data,
        R_norm, C_norm, ray_hits_any, b_opt, nullptr, nullptr, nullptr
    );
}

}
