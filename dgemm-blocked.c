const char* dgemm_desc = "Simple blocked dgemm.";
#include <immintrin.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 40
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static inline void do_block(int lda, int M, int N, int K,
                            const double* __restrict__ A,
                            const double* __restrict__ B,
                            double* __restrict__ C)
{
    A = (const double*)__builtin_assume_aligned(A, 32);
    B = (const double*)__builtin_assume_aligned(B, 32);
    C = (double*)__builtin_assume_aligned(C, 32);

    int j = 0;

    // ---- 4x8 AVX2 microkernel over columns ----
    for (; j + 7 < N; j += 8) {
        double* C0 = C + (j+0)*lda;
        double* C1 = C + (j+1)*lda;
        double* C2 = C + (j+2)*lda;
        double* C3 = C + (j+3)*lda;
        double* C4 = C + (j+4)*lda;
        double* C5 = C + (j+5)*lda;
        double* C6 = C + (j+6)*lda;
        double* C7 = C + (j+7)*lda;

        const double* B0 = B + (j+0)*lda;
        const double* B1 = B + (j+1)*lda;
        const double* B2 = B + (j+2)*lda;
        const double* B3 = B + (j+3)*lda;
        const double* B4 = B + (j+4)*lda;
        const double* B5 = B + (j+5)*lda;
        const double* B6 = B + (j+6)*lda;
        const double* B7 = B + (j+7)*lda;

        int i = 0;
        for (; i + 3 < M; i += 4) {
            // load 4x8 tile of C into registers
            __m256d c0 = _mm256_loadu_pd(C0 + i);
            __m256d c1 = _mm256_loadu_pd(C1 + i);
            __m256d c2 = _mm256_loadu_pd(C2 + i);
            __m256d c3 = _mm256_loadu_pd(C3 + i);
            __m256d c4 = _mm256_loadu_pd(C4 + i);
            __m256d c5 = _mm256_loadu_pd(C5 + i);
            __m256d c6 = _mm256_loadu_pd(C6 + i);
            __m256d c7 = _mm256_loadu_pd(C7 + i);

            int k = 0;
            for (; k + 1 < K; k += 2) {
                const double* Acol0 = A + (k+0)*lda;
                const double* Acol1 = A + (k+1)*lda;

                __m256d a0 = _mm256_loadu_pd(Acol0 + i);
                __m256d a1 = _mm256_loadu_pd(Acol1 + i);

                c0 = _mm256_fmadd_pd(a0, _mm256_broadcast_sd(B0 + (k+0)), c0);
                c1 = _mm256_fmadd_pd(a0, _mm256_broadcast_sd(B1 + (k+0)), c1);
                c2 = _mm256_fmadd_pd(a0, _mm256_broadcast_sd(B2 + (k+0)), c2);
                c3 = _mm256_fmadd_pd(a0, _mm256_broadcast_sd(B3 + (k+0)), c3);
                c4 = _mm256_fmadd_pd(a0, _mm256_broadcast_sd(B4 + (k+0)), c4);
                c5 = _mm256_fmadd_pd(a0, _mm256_broadcast_sd(B5 + (k+0)), c5);
                c6 = _mm256_fmadd_pd(a0, _mm256_broadcast_sd(B6 + (k+0)), c6);
                c7 = _mm256_fmadd_pd(a0, _mm256_broadcast_sd(B7 + (k+0)), c7);

                c0 = _mm256_fmadd_pd(a1, _mm256_broadcast_sd(B0 + (k+1)), c0);
                c1 = _mm256_fmadd_pd(a1, _mm256_broadcast_sd(B1 + (k+1)), c1);
                c2 = _mm256_fmadd_pd(a1, _mm256_broadcast_sd(B2 + (k+1)), c2);
                c3 = _mm256_fmadd_pd(a1, _mm256_broadcast_sd(B3 + (k+1)), c3);

                c4 = _mm256_fmadd_pd(a1, _mm256_broadcast_sd(B4 + (k+1)), c4);
                c5 = _mm256_fmadd_pd(a1, _mm256_broadcast_sd(B5 + (k+1)), c5);
                c6 = _mm256_fmadd_pd(a1, _mm256_broadcast_sd(B6 + (k+1)), c6);
                c7 = _mm256_fmadd_pd(a1, _mm256_broadcast_sd(B7 + (k+1)), c7);
            }


            // handle odd K
            for (; k < K; ++k) {
                const double* Acol = A + k*lda;
                __m256d a = _mm256_loadu_pd(Acol + i);

                __m256d b0 = _mm256_broadcast_sd(B0 + k);
                __m256d b1 = _mm256_broadcast_sd(B1 + k);
                __m256d b2 = _mm256_broadcast_sd(B2 + k);
                __m256d b3 = _mm256_broadcast_sd(B3 + k);
                __m256d b4 = _mm256_broadcast_sd(B4 + k);
                __m256d b5 = _mm256_broadcast_sd(B5 + k);
                __m256d b6 = _mm256_broadcast_sd(B6 + k);
                __m256d b7 = _mm256_broadcast_sd(B7 + k);

                c0 = _mm256_fmadd_pd(a, b0, c0);
                c1 = _mm256_fmadd_pd(a, b1, c1);
                c2 = _mm256_fmadd_pd(a, b2, c2);
                c3 = _mm256_fmadd_pd(a, b3, c3);
                c4 = _mm256_fmadd_pd(a, b4, c4);
                c5 = _mm256_fmadd_pd(a, b5, c5);
                c6 = _mm256_fmadd_pd(a, b6, c6);
                c7 = _mm256_fmadd_pd(a, b7, c7);
            }

            // store tile of C once
            _mm256_storeu_pd(C0 + i, c0);
            _mm256_storeu_pd(C1 + i, c1);
            _mm256_storeu_pd(C2 + i, c2);
            _mm256_storeu_pd(C3 + i, c3);
            _mm256_storeu_pd(C4 + i, c4);
            _mm256_storeu_pd(C5 + i, c5);
            _mm256_storeu_pd(C6 + i, c6);
            _mm256_storeu_pd(C7 + i, c7);
        }

        // edge rows for this 8-col panel
        for (; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                double a = A[i + k*lda];
                C0[i] += a * B0[k];
                C1[i] += a * B1[k];
                C2[i] += a * B2[k];
                C3[i] += a * B3[k];
                C4[i] += a * B4[k];
                C5[i] += a * B5[k];
                C6[i] += a * B6[k];
                C7[i] += a * B7[k];
            }
        }
    }

    for (; j < N; ++j) {
        double* Ccol = C + j * lda;
        const double* Bcol = B + j * lda;
        for (int k = 0; k < K; ++k) {
            const double bkj = Bcol[k];
            const double* Acol = A + k * lda;
            for (int i = 0; i < M; ++i) {
                Ccol[i] += Acol[i] * bkj;
            }
        }
    }
}




/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}
