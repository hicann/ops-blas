/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <complex>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "chemm_param.h"

/*!
 * \brief Read Hermitian matrix element from triangular storage.
 *
 * The stored triangle (upper or lower) contains valid complex values.
 * The missing triangle is reconstructed by conjugate symmetry: A(i,j) = conj(A(j,i)).
 * Diagonal elements: imag part is ignored (Hermitian diagonal is real; only real part used).
 *
 * \param a      Pointer to Hermitian matrix in column-major storage (lda is leading dimension)
 * \param lda    Leading dimension of A
 * \param uplo   Which triangle is stored
 * \param row    Requested row index
 * \param col    Requested column index
 * \return       Complex value at (row, col)
 */
static inline std::complex<float> ChemmGetHermValue(
    const aclblasComplex* a, int64_t lda, aclblasFillMode_t uplo, int64_t row, int64_t col)
{
    if (row == col) {
        // Diagonal: Hermitian matrix has real diagonal (imaginary part must be 0)
        float realPart = a[row * lda + col].real;
        return std::complex<float>(realPart, 0.0f);
    }
    if (uplo == ACLBLAS_UPPER) {
        if (row <= col) {
            // Stored in upper triangle
            return std::complex<float>(a[row * lda + col].real, a[row * lda + col].imag);
        }
        // Not stored: A(row,col) = conj(A(col,row))
        float r = a[col * lda + row].real;
        float i = a[col * lda + row].imag;
        return std::complex<float>(r, -i);
    } else {
        // ACLBLAS_LOWER
        if (row >= col) {
            // Stored in lower triangle
            return std::complex<float>(a[row * lda + col].real, a[row * lda + col].imag);
        }
        // Not stored: A(row,col) = conj(A(col,row))
        float r = a[col * lda + row].real;
        float i = a[col * lda + row].imag;
        return std::complex<float>(r, -i);
    }
}

inline aclblasStatus_t ChemmValidateCpuArgs(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if ((side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) ||
        (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) || m < 0 || n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr ||
        lda < std::max<int64_t>(1, aDim) || ldb < std::max<int64_t>(1, n) ||
        ldc < std::max<int64_t>(1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline void ChemmInitCpuOutput(const aclblasComplex* beta, aclblasComplex* C, int64_t m, int64_t n, int64_t ldc,
    std::vector<double>& real, std::vector<double>& imag)
{
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            size_t index = static_cast<size_t>(i) * n + j;
            double cr = C[i * ldc + j].real;
            double ci = C[i * ldc + j].imag;
            real[index] = beta->real * cr - beta->imag * ci;
            imag[index] = beta->real * ci + beta->imag * cr;
        }
    }
}

inline void ChemmComputeCpuLeft(const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda,
    const aclblasComplex* B, int64_t ldb, aclblasFillMode_t uplo, int64_t m, int64_t n,
    std::vector<double>& real, std::vector<double>& imag)
{
    const double alphaReal = static_cast<double>(alpha->real);
    const double alphaImag = static_cast<double>(alpha->imag);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t k = 0; k < m; ++k) {
            std::complex<float> value = ChemmGetHermValue(A, lda, uplo, i, k);
            const double valueReal = static_cast<double>(value.real());
            const double valueImag = static_cast<double>(value.imag());
            double ar = alphaReal * valueReal - alphaImag * valueImag;
            double ai = alphaReal * valueImag + alphaImag * valueReal;
            for (int64_t j = 0; j < n; ++j) {
                size_t index = static_cast<size_t>(i) * n + j;
                double br = B[k * ldb + j].real;
                double bi = B[k * ldb + j].imag;
                real[index] += ar * br - ai * bi;
                imag[index] += ar * bi + ai * br;
            }
        }
    }
}

inline void ChemmComputeCpuRight(const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda,
    const aclblasComplex* B, int64_t ldb, aclblasFillMode_t uplo, int64_t m, int64_t n,
    std::vector<double>& real, std::vector<double>& imag)
{
    const double alphaReal = static_cast<double>(alpha->real);
    const double alphaImag = static_cast<double>(alpha->imag);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t k = 0; k < n; ++k) {
            std::complex<float> value = ChemmGetHermValue(A, lda, uplo, k, j);
            const double valueReal = static_cast<double>(value.real());
            const double valueImag = static_cast<double>(value.imag());
            double ar = alphaReal * valueReal - alphaImag * valueImag;
            double ai = alphaReal * valueImag + alphaImag * valueReal;
            for (int64_t i = 0; i < m; ++i) {
                size_t index = static_cast<size_t>(i) * n + j;
                double br = B[i * ldb + k].real;
                double bi = B[i * ldb + k].imag;
                real[index] += br * ar - bi * ai;
                imag[index] += br * ai + bi * ar;
            }
        }
    }
}

inline void ChemmWriteCpuOutput(aclblasComplex* C, int64_t m, int64_t n, int64_t ldc,
    const std::vector<double>& real, const std::vector<double>& imag)
{
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            size_t index = static_cast<size_t>(i) * n + j;
            C[i * ldc + j].real = static_cast<float>(real[index]);
            C[i * ldc + j].imag = static_cast<float>(imag[index]);
        }
    }
}

/*!
 * \brief CPU reference implementation for aclblasChemm.
 *
 * Signature matches the proposed aclblasChemm API exactly.
 * Performs parameter validation mirroring the NPU implementation,
 * then computes the Hermitian matrix multiplication using double-precision
 * accumulation for accuracy.
 */
inline aclblasStatus_t aclblasChemm_cpu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    aclblasStatus_t status = ChemmValidateCpuArgs(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return status;
    }

    // --- computation ---
    // Use double-precision for the entire accumulation to match the precision
    // of the original triple-nested loop golden.  Loop order is reordered so
    // that Hermitian reads (ChemmGetHermValue) happen O(m*k) or O(k*n) times
    // instead of O(m*n*k) — essential for large matrices to finish in time.
    // C is large for 2048×2048 (32 MB) → allocate double-precision scratch on heap.
    const size_t cComplexCount = static_cast<size_t>(m) * static_cast<size_t>(n);
    std::vector<double> cDoubleReal(cComplexCount, 0.0);
    std::vector<double> cDoubleImag(cComplexCount, 0.0);

    ChemmInitCpuOutput(beta, C, m, n, ldc, cDoubleReal, cDoubleImag);
    if (side == ACLBLAS_SIDE_LEFT) {
        ChemmComputeCpuLeft(alpha, A, lda, B, ldb, uplo, m, n, cDoubleReal, cDoubleImag);
    } else {
        ChemmComputeCpuRight(alpha, A, lda, B, ldb, uplo, m, n, cDoubleReal, cDoubleImag);
    }
    ChemmWriteCpuOutput(C, m, n, ldc, cDoubleReal, cDoubleImag);

    return ACLBLAS_STATUS_SUCCESS;
}

// ── golden cache: avoid recomputing CPU golden every test run ──

namespace chemm_golden_cache {

constexpr uint32_t CHEMM_GOLDEN_VERSION = 2;
constexpr uint32_t CHEMM_GOLDEN_MAGIC = 0x4D474843;

inline void ensureDir(const std::string& path)
{
    std::string dir = path;
    if (!dir.empty() && dir.back() == '/') {
        dir.pop_back();
    }
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
    }
}

inline uint64_t computeParamHash(const ChemmParam& p)
{
    struct HashAccum {
        uint64_t h = 14695981039346656037ULL;
        void feed(const void* data, size_t len)
        {
            const auto* bytes = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < len; ++i) {
                h ^= static_cast<uint64_t>(bytes[i]);
                h *= 1099511628211ULL;
            }
        }
    } acc;

    acc.feed(&p.side, sizeof(p.side));
    acc.feed(&p.uplo, sizeof(p.uplo));
    acc.feed(&p.m, sizeof(p.m));
    acc.feed(&p.n, sizeof(p.n));
    acc.feed(&p.alphaReal, sizeof(p.alphaReal));
    acc.feed(&p.alphaImag, sizeof(p.alphaImag));
    acc.feed(&p.betaReal, sizeof(p.betaReal));
    acc.feed(&p.betaImag, sizeof(p.betaImag));
    acc.feed(&p.lda, sizeof(p.lda));
    acc.feed(&p.ldb, sizeof(p.ldb));
    acc.feed(&p.ldc, sizeof(p.ldc));
    acc.feed(&p.randomSeed, sizeof(p.randomSeed));
    acc.feed(&p.aFill.method, sizeof(p.aFill.method));
    acc.feed(&p.aFill.pattern, sizeof(p.aFill.pattern));
    acc.feed(&p.aFill.val1, sizeof(p.aFill.val1));
    acc.feed(&p.aFill.val2, sizeof(p.aFill.val2));
    acc.feed(&p.aFill.bandedKl, sizeof(p.aFill.bandedKl));
    acc.feed(&p.aFill.bandedKu, sizeof(p.aFill.bandedKu));
    acc.feed(&p.bFill.method, sizeof(p.bFill.method));
    acc.feed(&p.bFill.pattern, sizeof(p.bFill.pattern));
    acc.feed(&p.bFill.val1, sizeof(p.bFill.val1));
    acc.feed(&p.bFill.val2, sizeof(p.bFill.val2));
    acc.feed(&p.bFill.bandedKl, sizeof(p.bFill.bandedKl));
    acc.feed(&p.bFill.bandedKu, sizeof(p.bFill.bandedKu));
    acc.feed(&p.cFill.method, sizeof(p.cFill.method));
    acc.feed(&p.cFill.pattern, sizeof(p.cFill.pattern));
    acc.feed(&p.cFill.val1, sizeof(p.cFill.val1));
    acc.feed(&p.cFill.val2, sizeof(p.cFill.val2));
    acc.feed(&p.cFill.bandedKl, sizeof(p.cFill.bandedKl));
    acc.feed(&p.cFill.bandedKu, sizeof(p.cFill.bandedKu));

    return acc.h;
}

inline bool loadGolden(const std::string& filepath, std::vector<float>& data, uint64_t paramHash)
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t storedHash = 0;
    uint64_t storedCount = 0;

    if (!ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic)))
        return false;
    if (!ifs.read(reinterpret_cast<char*>(&version), sizeof(version)))
        return false;
    if (!ifs.read(reinterpret_cast<char*>(&storedHash), sizeof(storedHash)))
        return false;
    if (!ifs.read(reinterpret_cast<char*>(&storedCount), sizeof(storedCount)))
        return false;

    if (magic != CHEMM_GOLDEN_MAGIC || version != CHEMM_GOLDEN_VERSION || storedHash != paramHash) {
        return false;
    }

    data.resize(static_cast<size_t>(storedCount));
    if (!ifs.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(storedCount * sizeof(float)))) {
        data.clear();
        return false;
    }

    return ifs.good();
}

inline void saveGolden(const std::string& filepath, const std::vector<float>& data, uint64_t paramHash)
{
    size_t slash = filepath.rfind('/');
    if (slash != std::string::npos) {
        ensureDir(filepath.substr(0, slash));
    }

    std::ofstream ofs(filepath, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        return;
    }

    uint32_t magic = CHEMM_GOLDEN_MAGIC;
    uint32_t version = CHEMM_GOLDEN_VERSION;
    uint64_t count = static_cast<uint64_t>(data.size());

    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&paramHash), sizeof(paramHash));
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

} // namespace chemm_golden_cache

inline std::string makeGoldenDir(const std::string& filePath)
{
    size_t slash = filePath.rfind('/');
    if (slash == std::string::npos) {
        return "golden/";
    }
    return filePath.substr(0, slash + 1) + "golden/";
}
