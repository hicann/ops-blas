# Trsmbatched算子

## 算子概述

批量三角矩阵线性方程组求解（Batched Triangular Solve），对多个独立的三角线性方程组进行批量求解，解矩阵 X[i] 原地覆盖 B[i]。

数学表达式：

```
side='L': op(A[i]) * X[i] = alpha * B[i]
side='R': X[i] * op(A[i]) = alpha * B[i]
```

其中 op(A) 由 trans 参数决定：trans='N' 时 op(A)=A，trans='T' 时 op(A)=A^T，trans='C' 时 op(A)=A^H（实数下等价于 A^T）。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrsmBatched | 单精度浮点批量三角求解 |

## 算子执行接口

### aclblasStrsmBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 strsmbatched 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m,
    int n,
    const float* alpha,
    const float* const A[],
    int lda,
    float* const B[],
    int ldb,
    int batchCount);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| side | 输入 | aclblasSideMode_t | 三角矩阵位置：ACLBLAS_SIDE_LEFT（A 在左侧）/ ACLBLAS_SIDE_RIGHT（A 在右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 三角存储模式：ACLBLAS_UPPER（上三角）/ ACLBLAS_LOWER（下三角），Host 内存 |
| trans | 输入 | aclblasOperation_t | 转置模式：ACLBLAS_OP_N（不转置）/ ACLBLAS_OP_T（转置）/ ACLBLAS_OP_C（共轭转置，实数下等价于 T），Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT（非单位对角线）/ ACLBLAS_UNIT（单位对角线，假定为 1 不访问），Host 内存 |
| m | 输入 | int | B 的行数，>= 0，Host 内存 |
| n | 输入 | int | B 的列数，>= 0，Host 内存 |
| alpha | 输入 | const float* | 缩放因子指针，alpha=0 时 B 置零（不访问 A），Host 内存 |
| A | 输入 | const float* const[] | 批量三角矩阵指针数组，A[i] 指向第 i 个三角矩阵 A[i]；side=LEFT 时 A[i] 为 (M,M) 矩阵，side=RIGHT 时 A[i] 为 (N,N) 矩阵，Device 内存 |
| lda | 输入 | int | A[i] 的 leading dimension，side=LEFT 时 lda >= max(1,M)，side=RIGHT 时 lda >= max(1,N)，Host 内存 |
| B | 输入/输出 | float* const[] | 批量右端/解矩阵指针数组，B[i] 指向第 i 个 (M,N) 矩阵，输出时原地覆盖为解矩阵 X[i]，Device 内存 |
| ldb | 输入 | int | B[i] 的 leading dimension，ldb >= max(1,M)，Host 内存 |
| batchCount | 输入 | int | 批量大小，>= 1，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- batchCount >= 1
- side 必须为 ACLBLAS_SIDE_LEFT 或 ACLBLAS_SIDE_RIGHT
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- diag 必须为 ACLBLAS_NON_UNIT 或 ACLBLAS_UNIT
- alpha 不为 nullptr
- B 不为 nullptr
- A 不为 nullptr（当 alpha != 0 时）
- lda >= max(1, K)，其中 K = (side==LEFT ? m : n)
- ldb >= max(1, m)
- Blocked 路径要求：NoTrans 时 lda 和 ldb 须 8 对齐，尾 panel bs 须 8 对齐；不满足时回退至 SIMT-only 路径
- 仅支持 FP32 数据类型

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

class AclContext {
public:
    explicit AclContext(int deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclblasHandleDeleter {
    void operator()(aclblasHandle_t handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
    }
};

int aclblasStrsmBatchedTest(AclContext& ctx)
{
    // 2 batches, each A is 3x3 lower triangular, B is 3x2
    constexpr int batchCount = 2;
    constexpr int m = 3;
    constexpr int n = 2;
    constexpr int lda = m;
    constexpr int ldb = m;
    float alpha = 1.0f;

    // Host data for batch 0: A0 (lower triangular, column-major)
    // A0 = [2  0  0]    B0 = [8  10]
    //      [1  3  0]        [5   7]
    //      [4  2  1]        [6   3]
    float hA0[lda * m] = {2.0f, 1.0f, 4.0f, 0.0f, 3.0f, 2.0f, 0.0f, 0.0f, 1.0f};
    float hB0[ldb * n] = {8.0f, 5.0f, 6.0f, 10.0f, 7.0f, 3.0f};

    // Host data for batch 1: A1 (lower triangular, column-major)
    // A1 = [1  0  0]    B1 = [1  2]
    //      [2  1  0]        [3  4]
    //      [3  2  1]        [5  6]
    float hA1[lda * m] = {1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f};
    float hB1[ldb * n] = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f};

    size_t aBytes = static_cast<size_t>(lda) * m * sizeof(float);
    size_t bBytes = static_cast<size_t>(ldb) * n * sizeof(float);
    size_t ptrArrayBytes = static_cast<size_t>(batchCount) * sizeof(float*);

    // Allocate Device memory for A0, A1, B0, B1
    void* rawA0 = nullptr;
    auto aclRet = aclrtMalloc(&rawA0, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA0(rawA0);

    void* rawA1 = nullptr;
    aclRet = aclrtMalloc(&rawA1, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA1(rawA1);

    void* rawB0 = nullptr;
    aclRet = aclrtMalloc(&rawB0, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dB0(rawB0);

    void* rawB1 = nullptr;
    aclRet = aclrtMalloc(&rawB1, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dB1(rawB1);

    // Copy host data to device
    aclRet = aclrtMemcpy(dA0.get(), aBytes, hA0, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dA1.get(), aBytes, hA1, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dB0.get(), bBytes, hB0, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dB1.get(), bBytes, hB1, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // Build pointer arrays on host, then copy to device
    float* hAPtrs[batchCount] = {
        static_cast<float*>(dA0.get()),
        static_cast<float*>(dA1.get())
    };
    float* hBPtrs[batchCount] = {
        static_cast<float*>(dB0.get()),
        static_cast<float*>(dB1.get())
    };

    void* rawAPtrArray = nullptr;
    aclRet = aclrtMalloc(&rawAPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAPtrArray(rawAPtrArray);

    void* rawBPtrArray = nullptr;
    aclRet = aclrtMalloc(&rawBPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dBPtrArray(rawBPtrArray);

    aclRet = aclrtMemcpy(dAPtrArray.get(), ptrArrayBytes, hAPtrs, ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dBPtrArray.get(), ptrArrayBytes, hBPtrs, ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // Create ops-blas handle
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(handle.get(), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    // Call aclblasStrsmBatched: solve op(A[i]) * X[i] = alpha * B[i]
    blasRet = aclblasStrsmBatched(
        handle.get(),
        ACLBLAS_SIDE_LEFT,
        ACLBLAS_LOWER,
        ACLBLAS_OP_N,
        ACLBLAS_NON_UNIT,
        m, n, &alpha,
        static_cast<const float* const*>(dAPtrArray.get()), lda,
        static_cast<float* const*>(dBPtrArray.get()), ldb,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // Copy results back to host
    float hResult0[ldb * n] = {};
    float hResult1[ldb * n] = {};
    aclRet = aclrtMemcpy(hResult0, bBytes, dB0.get(), bBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(hResult1, bBytes, dB1.get(), bBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // Print results
    printf("Batch 0 result:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("  X[%d,%d] = %f\n", i, j, hResult0[j * ldb + i]);
        }
    }
    printf("Batch 1 result:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("  X[%d,%d] = %f\n", i, j, hResult1[j * ldb + i]);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStrsmBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
