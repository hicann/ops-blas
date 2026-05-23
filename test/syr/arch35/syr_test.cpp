/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

#define LOG_PRINT(message, ...) do { printf(message, ##__VA_ARGS__); } while (0)

// Precision standard (ops-precision-standard skill, float32 community):
//   MERE = avg(|out - gold| / (|gold| + 1e-7))
//   MARE = max(|out - gold| / (|gold| + 1e-7))
//   Pass: MERE < THRESHOLD && MARE < 10 * THRESHOLD
constexpr float MERE_THRESHOLD = 1.220703125e-4f;  // 2^-13
constexpr float MARE_RATIO = 10.0f;

// Golden: matches kernel's row-major access: A_golden[row * lda + col]
// UPPER: col >= row, LOWER: col <= row
static void GoldenUpper(const std::vector<float>& x, float alpha,
                        const std::vector<float>& A_orig, std::vector<float>& A_golden,
                        int64_t n, int64_t lda)
{
    A_golden = A_orig;
    for (int64_t row = 0; row < n; ++row)
        for (int64_t col = row; col < n; ++col)
            A_golden[row * lda + col] += alpha * x[row] * x[col];
}

static void GoldenLower(const std::vector<float>& x, float alpha,
                        const std::vector<float>& A_orig, std::vector<float>& A_golden,
                        int64_t n, int64_t lda)
{
    A_golden = A_orig;
    for (int64_t row = 0; row < n; ++row)
        for (int64_t col = 0; col <= row; ++col)
            A_golden[row * lda + col] += alpha * x[row] * x[col];
}

static uint32_t VerifyTriRegion(const float* out, const float* gold,
                                 int64_t n, int64_t lda, aclblasFillMode uplo, const char* nm)
{
    double sumRelErr = 0.0;
    float maxRelErr = 0.0f;
    uint32_t count = 0;
    for (int64_t row = 0; row < n; ++row) {
        int64_t c0 = (uplo == ACLBLAS_UPPER) ? row : 0;
        int64_t c1 = (uplo == ACLBLAS_UPPER) ? n : row + 1;
        for (int64_t col = c0; col < c1; ++col) {
            int64_t idx = row * lda + col;
            float relErr = std::abs(out[idx] - gold[idx]) / (std::abs(gold[idx]) + 1e-7f);
            sumRelErr += relErr;
            if (relErr > maxRelErr) maxRelErr = relErr;
            ++count;
        }
    }
    float mere = (count > 0) ? static_cast<float>(sumRelErr / count) : 0.0f;
    float mare = maxRelErr;
    float mareThreshold = MARE_RATIO * MERE_THRESHOLD;
    bool pass = (mere < MERE_THRESHOLD) && (mare < mareThreshold);
    if (pass)
        LOG_PRINT("  [%s] Triangular PASSED (MERE=%.6e, MARE=%.6e)\n", nm, mere, mare);
    else
        LOG_PRINT("  [%s] Triangular FAILED (MERE=%.6e, MARE=%.6e, limit=%.6e/%.6e)\n",
                  nm, mere, mare, MERE_THRESHOLD, mareThreshold);
    return pass ? 0 : 1;
}

// VerifyUnchangedRegion: bitwise exact match for the non-computed triangle.
// This is a correctness check (not a precision check) — the kernel must not touch this region.
static uint32_t VerifyUnchangedRegion(const float* out, const float* orig,
                                       int64_t n, int64_t lda, aclblasFillMode uplo, const char* nm)
{
    uint32_t err = 0;
    for (int64_t row = 0; row < n; ++row) {
        int64_t c0 = (uplo == ACLBLAS_UPPER) ? 0 : row + 1;
        int64_t c1 = (uplo == ACLBLAS_UPPER) ? row : n;
        for (int64_t col = c0; col < c1; ++col) {
            if (out[row * lda + col] != orig[row * lda + col]) ++err;
        }
    }
    if (err == 0) LOG_PRINT("  [%s] Unchanged PASSED\n", nm);
    else LOG_PRINT("  [%s] Unchanged FAILED: %u errors\n", nm, err);
    return err;
}

static uint32_t VerifyFull(const float* out, const float* gold,
                            int64_t n, int64_t lda, const char* nm)
{
    double sumRelErr = 0.0;
    float maxRelErr = 0.0f;
    int64_t total = n * lda;
    for (int64_t i = 0; i < total; ++i) {
        float relErr = std::abs(out[i] - gold[i]) / (std::abs(gold[i]) + 1e-7f);
        sumRelErr += relErr;
        if (relErr > maxRelErr) maxRelErr = relErr;
    }
    float mere = (total > 0) ? static_cast<float>(sumRelErr / total) : 0.0f;
    float mare = maxRelErr;
    float mareThreshold = MARE_RATIO * MERE_THRESHOLD;
    bool pass = (mere < MERE_THRESHOLD) && (mare < mareThreshold);
    if (pass)
        LOG_PRINT("  [%s] Full PASSED (MERE=%.6e, MARE=%.6e)\n", nm, mere, mare);
    else
        LOG_PRINT("  [%s] Full FAILED (MERE=%.6e, MARE=%.6e, limit=%.6e/%.6e)\n",
                  nm, mere, mare, MERE_THRESHOLD, mareThreshold);
    return pass ? 0 : 1;
}

static aclblasStatus_t Run(aclblasHandle h, aclrtStream s,
                            aclblasFillMode uplo, int n, float alpha,
                            const std::vector<float>& x, int incx,
                            std::vector<float>& A, int lda)
{
    size_t xb = (1 + (static_cast<size_t>(n) - 1) * static_cast<size_t>(std::abs(incx))) * sizeof(float);
    size_t ab = static_cast<size_t>(n) * static_cast<size_t>(lda) * sizeof(float);
    float *dx = nullptr, *da = nullptr;
    if (aclrtMalloc((void**)&dx, xb, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
    if (aclrtMalloc((void**)&da, ab, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { aclrtFree(dx); return ACLBLAS_STATUS_ALLOC_FAILED; }
    aclrtMemcpy(dx, xb, x.data(), xb, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(da, ab, A.data(), ab, ACL_MEMCPY_HOST_TO_DEVICE);
    auto ret = aclblasSsyr(h, uplo, n, &alpha, dx, incx, da, lda);
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclrtSynchronizeStream(s);
        aclrtMemcpy(A.data(), ab, da, ab, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    aclrtFree(da); aclrtFree(dx);
    return ret;
}

// L0
static uint32_t L001(aclblasHandle h, aclrtStream s) {
    int n = 128, lda = n; float alpha = 2.0f;
    std::vector<float> x(n), A(n*lda);
    for (int64_t i=0;i<n;++i) x[i]=i+1;
    for (int64_t i=0;i<n*lda;++i) A[i]=(i%100)-50;
    std::vector<float> gold; GoldenUpper(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L0-01")+
           VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_UPPER,"SYR-L0-01");
}
static uint32_t L002(aclblasHandle h, aclrtStream s) {
    int n=128,lda=n; float alpha=2.0f;
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=i+1;
    for(int64_t i=0;i<n*lda;++i)A[i]=(i%100)-50;
    std::vector<float> gold; GoldenLower(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_LOWER,n,alpha,x,1,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_LOWER,"SYR-L0-02")+
           VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_LOWER,"SYR-L0-02");
}
static uint32_t L003(aclblasHandle h, aclrtStream s) {
    int n=64,lda=n; float alpha=0.0f;
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=i+1;
    for(int64_t i=0;i<n*lda;++i)A[i]=i+1;
    auto gold=A; auto out=A;
    Run(h,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    uint32_t e=VerifyFull(out.data(),gold.data(),n,lda,"SYR-L0-03-UPPER");
    out=A; Run(h,s,ACLBLAS_LOWER,n,alpha,x,1,out,lda);
    return e+VerifyFull(out.data(),gold.data(),n,lda,"SYR-L0-03-LOWER");
}
static uint32_t L004(aclblasHandle h, aclrtStream s) {
    int n=1,lda=1; float alpha=2.0f;
    std::vector<float> x={3},A={5},gold;
    GoldenUpper(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    return VerifyFull(out.data(),gold.data(),n,lda,"SYR-L0-04");
}
static uint32_t L005(aclblasHandle h, aclrtStream s) {
    int n=64,lda=n; float alpha=-1.5f;
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=i%10;
    for(int64_t i=0;i<n*lda;++i)A[i]=i%7;
    std::vector<float> gold; GoldenUpper(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    uint32_t e=VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L0-05-UPPER");
    GoldenLower(x,alpha,A,gold,n,lda); out=A;
    Run(h,s,ACLBLAS_LOWER,n,alpha,x,1,out,lda);
    return e+VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_LOWER,"SYR-L0-05-LOWER");
}
static uint32_t L006(aclblasHandle h, aclrtStream) {
    float alpha=1.0f,d=0;
    auto ret=aclblasSsyr(h,ACLBLAS_UPPER,0,&alpha,&d,1,&d,1);
    if(ret==ACLBLAS_STATUS_SUCCESS){LOG_PRINT("[SYR-L0-06] PASSED\n");return 0;}
    LOG_PRINT("[SYR-L0-06] FAILED\n");return 1;
}

// L1
static uint32_t L101(aclblasHandle h, aclrtStream s) {
    int n=64,incx=2,lda=n; float alpha=2.0f;
    std::vector<float> x(1+(n-1)*incx),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i*incx]=i+1;
    for(int64_t i=0;i<n*lda;++i)A[i]=(i%50)-25;
    std::vector<float> xd(n); for(int64_t i=0;i<n;++i)xd[i]=x[i*incx];
    std::vector<float> gold; GoldenUpper(xd,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,x,incx,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L1-01")+
           VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_UPPER,"SYR-L1-01");
}
static uint32_t L102(aclblasHandle h, aclrtStream s) {
    int n=64,incx=3,lda=n; float alpha=1.0f;
    std::vector<float> x(1+(n-1)*incx),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i*incx]=(i+1)%10;
    for(int64_t i=0;i<n*lda;++i)A[i]=i%20;
    std::vector<float> xd(n); for(int64_t i=0;i<n;++i)xd[i]=x[i*incx];
    std::vector<float> gold; GoldenLower(xd,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_LOWER,n,alpha,x,incx,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_LOWER,"SYR-L1-02")+
           VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_LOWER,"SYR-L1-02");
}
static uint32_t L103(aclblasHandle h, aclrtStream s) {
    int n=64,lda=n+32; float alpha=2.0f;
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=i+1;
    for(int64_t i=0;i<n*lda;++i)A[i]=(i%30)-15;
    std::vector<float> gold; GoldenUpper(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L1-03");
}
static uint32_t L104(aclblasHandle h, aclrtStream s) {
    int n=64,lda=n+8; float alpha=1.5f;
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=(i*3)%11;
    for(int64_t i=0;i<n*lda;++i)A[i]=i%13;
    std::vector<float> gold; GoldenLower(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_LOWER,n,alpha,x,1,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_LOWER,"SYR-L1-04");
}
static uint32_t L105(aclblasHandle h, aclrtStream s) {
    int n=4,lda=n; float alpha=0.5f;
    std::vector<float> x={1,2,3,4},A={10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
    std::vector<float> gold; GoldenUpper(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    uint32_t e=VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L1-05-UPPER");
    GoldenLower(x,alpha,A,gold,n,lda); out=A;
    Run(h,s,ACLBLAS_LOWER,n,alpha,x,1,out,lda);
    return e+VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_LOWER,"SYR-L1-05-LOWER");
}
static uint32_t L106(aclblasHandle h, aclrtStream s) {
    int n=512,lda=n; float alpha=0.5f;
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=i%31;
    for(int64_t i=0;i<n*lda;++i)A[i]=i%41;
    std::vector<float> gold; GoldenUpper(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    uint32_t e=VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L1-06-UPPER");
    GoldenLower(x,alpha,A,gold,n,lda); out=A;
    Run(h,s,ACLBLAS_LOWER,n,alpha,x,1,out,lda);
    return e+VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_LOWER,"SYR-L1-06-LOWER");
}
static uint32_t L107(aclblasHandle h, aclrtStream s) {
    int n=128,lda=n; float alpha=0.5f;
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=(i%21-10)/100.0f;
    for(int64_t i=0;i<n*lda;++i)A[i]=(i%17-8)/100.0f;
    std::vector<float> gold; GoldenUpper(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L1-07")+
           VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_UPPER,"SYR-L1-07");
}
static uint32_t L108(aclblasHandle h, aclrtStream s) {
    int n=128,lda=n; float alpha=0.5f;
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=i%2001-1000;
    for(int64_t i=0;i<n*lda;++i)A[i]=i%2001-1000;
    std::vector<float> gold; GoldenLower(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_LOWER,n,alpha,x,1,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_LOWER,"SYR-L1-08")+
           VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_LOWER,"SYR-L1-08");
}
// Negative incx: BLAS convention — x[0] at memory offset (n-1)*|incx|, x[n-1] at offset 0.
static uint32_t L109(aclblasHandle h, aclrtStream s) {
    int n=64,incx=-1,absIncx=1,lda=n; float alpha=2.0f;
    std::vector<float> xLogical(n);
    for(int64_t i=0;i<n;++i)xLogical[i]=i+1;
    std::vector<float> xMem(1+(n-1)*absIncx,0.0f);
    for(int64_t i=0;i<n;++i)xMem[i*absIncx]=xLogical[n-1-i];
    std::vector<float> A(n*lda);
    for(int64_t i=0;i<n*lda;++i)A[i]=(i%100)-50;
    std::vector<float> gold; GoldenUpper(xLogical,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,xMem,incx,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L1-09")+
           VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_UPPER,"SYR-L1-09");
}
static uint32_t L110(aclblasHandle h, aclrtStream s) {
    int n=64,incx=-2,absIncx=2,lda=n; float alpha=1.5f;
    std::vector<float> xLogical(n);
    for(int64_t i=0;i<n;++i)xLogical[i]=(i*3)%11;
    std::vector<float> xMem(1+(n-1)*absIncx,0.0f);
    for(int64_t i=0;i<n;++i)xMem[i*absIncx]=xLogical[n-1-i];
    std::vector<float> A(n*lda);
    for(int64_t i=0;i<n*lda;++i)A[i]=i%13;
    std::vector<float> gold; GoldenLower(xLogical,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_LOWER,n,alpha,xMem,incx,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_LOWER,"SYR-L1-10");
}

// L2
static uint32_t L201(aclblasHandle h, aclrtStream) {
    float alpha=1.0f,d=0;
    auto ret=aclblasSsyr(h,ACLBLAS_UPPER,-1,&alpha,&d,1,&d,1);
    if(ret==ACLBLAS_STATUS_INVALID_VALUE){LOG_PRINT("[SYR-L2-01] PASSED\n");return 0;}
    LOG_PRINT("[SYR-L2-01] FAILED\n");return 1;
}
static uint32_t L202(aclblasHandle h, aclrtStream) {
    float alpha=1.0f,d=0;
    auto ret=aclblasSsyr(h,ACLBLAS_UPPER,4,&alpha,&d,0,&d,4);
    if(ret==ACLBLAS_STATUS_INVALID_VALUE){LOG_PRINT("[SYR-L2-02] PASSED\n");return 0;}
    LOG_PRINT("[SYR-L2-02] FAILED\n");return 1;
}
static uint32_t L203(aclblasHandle h, aclrtStream) {
    float alpha=1.0f,*da=nullptr;
    aclrtMalloc((void**)&da,64,ACL_MEM_MALLOC_HUGE_FIRST);
    auto ret=aclblasSsyr(h,ACLBLAS_UPPER,4,&alpha,(const float*)nullptr,1,da,4);
    aclrtFree(da);
    if(ret==ACLBLAS_STATUS_INVALID_VALUE){LOG_PRINT("[SYR-L2-03] PASSED\n");return 0;}
    LOG_PRINT("[SYR-L2-03] FAILED\n");return 1;
}
static uint32_t L204(aclblasHandle h, aclrtStream) {
    float alpha=1.0f,*dx=nullptr;
    aclrtMalloc((void**)&dx,16,ACL_MEM_MALLOC_HUGE_FIRST);
    auto ret=aclblasSsyr(h,ACLBLAS_UPPER,4,&alpha,dx,1,nullptr,4);
    aclrtFree(dx);
    if(ret==ACLBLAS_STATUS_INVALID_VALUE){LOG_PRINT("[SYR-L2-04] PASSED\n");return 0;}
    LOG_PRINT("[SYR-L2-04] FAILED\n");return 1;
}
static uint32_t L205(aclblasHandle h, aclrtStream) {
    float alpha=1.0f,d=0;
    auto ret=aclblasSsyr(h,ACLBLAS_UPPER,4,&alpha,&d,1,&d,3);
    if(ret==ACLBLAS_STATUS_INVALID_VALUE){LOG_PRINT("[SYR-L2-05] PASSED\n");return 0;}
    LOG_PRINT("[SYR-L2-05] FAILED\n");return 1;
}
static uint32_t L206(aclrtStream s) {
    int n=64,lda=n; float alpha=2.0f;
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=i+1;
    for(int64_t i=0;i<n*lda;++i)A[i]=(i%100)-50;
    std::vector<float> gold; GoldenUpper(x,alpha,A,gold,n,lda);
    auto out=A; Run(nullptr,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    return VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L2-06")+
           VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_UPPER,"SYR-L2-06");
}
static uint32_t L207(aclblasHandle h, aclrtStream s) {
    int n=4096,lda=n; float alpha=1.0f;
    LOG_PRINT("[SYR-L2-07] Allocating A: %lld MB\n",(long long)(n*lda*4/1024/1024));
    std::vector<float> x(n),A(n*lda);
    for(int64_t i=0;i<n;++i)x[i]=i%997;
    for(int64_t i=0;i<n*lda;++i)A[i]=i%503;
    LOG_PRINT("[SYR-L2-07-UPPER] Computing golden...\n");
    std::vector<float> gold; GoldenUpper(x,alpha,A,gold,n,lda);
    auto out=A; Run(h,s,ACLBLAS_UPPER,n,alpha,x,1,out,lda);
    uint32_t e=VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_UPPER,"SYR-L2-07-UPPER")+
               VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_UPPER,"SYR-L2-07-UPPER");
    LOG_PRINT("[SYR-L2-07-LOWER] Computing golden...\n");
    GoldenLower(x,alpha,A,gold,n,lda); out=A;
    Run(h,s,ACLBLAS_LOWER,n,alpha,x,1,out,lda);
    return e+VerifyTriRegion(out.data(),gold.data(),n,lda,ACLBLAS_LOWER,"SYR-L2-07-LOWER")+
             VerifyUnchangedRegion(out.data(),A.data(),n,lda,ACLBLAS_LOWER,"SYR-L2-07-LOWER");
}

#define RUN(fn) do{uint32_t e=fn(handle,stream);tc++;if(e>0)te++;}while(0)

int main()
{
    aclInit(nullptr); aclrtSetDevice(0);
    aclblasHandle handle=nullptr; aclblasCreate(&handle);
    aclrtStream stream=nullptr; aclrtCreateStream(&stream);
    uint32_t tc=0,te=0;
    LOG_PRINT("=== SYR Test Suite (float32) ===\n\n");
    LOG_PRINT("--- L0 ---\n\n");
    RUN(L001);RUN(L002);RUN(L003);RUN(L004);RUN(L005);RUN(L006);
    LOG_PRINT("\n--- L1 ---\n\n");
    RUN(L101);RUN(L102);RUN(L103);RUN(L104);RUN(L105);RUN(L106);RUN(L107);RUN(L108);RUN(L109);RUN(L110);
    LOG_PRINT("\n--- L2 ---\n\n");
    RUN(L201);RUN(L202);RUN(L203);RUN(L204);RUN(L205);
    {uint32_t e=L206(stream);tc++;if(e>0)te++;}
    RUN(L207);
    LOG_PRINT("\n========================================\n");
    LOG_PRINT("  Total: %u  Passed: %u  Failed: %u\n",tc,tc-te,te);
    LOG_PRINT("========================================\n");
    if(te==0) LOG_PRINT("  RESULT: ALL TESTS PASSED\n");
    else LOG_PRINT("  RESULT: %u FAILED\n",te);
    aclrtDestroyStream(stream); aclblasDestroy(handle);
    aclrtResetDevice(0); aclFinalize();
    return te>0?1:0;
}
