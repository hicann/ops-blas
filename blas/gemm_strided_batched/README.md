# GemmStridedBatched算子

## 算子概述

GemmStridedBatched（Single-precision Strided Batched General Matrix Multiplication）算子实现了单精度浮点批量矩阵乘法，对一批规格一致的矩阵三元组 (A_i, B_i, C_i) 执行 GEMM 运算并通过固定 stride 定位每个 batch 的子矩阵，核心运算为 C_i = alpha * op(A_i) * op(B_i) + beta * C_i。

数学表达式：

```
C_i = alpha * op(A_i) * op(B_i) + beta * C_i,   i = 0, 1, ..., batchCount-1

其中：
  A_i = A + i * strideA   （A 是 batch 0 的首地址，strideA 是相邻 batch 间的元素偏移量）
  B_i = B + i * strideB
  C_i = C + i * strideC
  strideA/strideB/strideC = 0 时表示所有 batch 复用同一矩阵（广播）

  op(X) = X       当 trans = ACLBLAS_OP_N
  op(X) = X^T     当 trans = ACLBLAS_OP_T
  op(X) = X^H     当 trans = ACLBLAS_OP_C （实数 FP32 下 X^H == X^T）
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| [aclblasSgemmStridedBatched](#aclblassgemmstridedbatched) | 单精度浮点批量矩阵乘法（FP32） |

## 算子执行接口

### aclblasSgemmStridedBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 aclblasSgemmStridedBatched 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemmStridedBatched(aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, const float* alpha, const float* A, int lda, int64_t strideA, const float* B, int ldb, int64_t strideB, const float* beta, float* C, int ldc, int64_t strideC, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transA | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数 FP32 下等价于 T），Host 内存 |
| transB | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（取值同 transA），Host 内存 |
| m | 输入 | int | op(A_i) 与 C_i 的行数，m >= 0，Host 内存 |
| n | 输入 | int | op(B_i) 与 C_i 的列数，n >= 0，Host 内存 |
| k | 输入 | int | op(A_i) 的列数、op(B_i) 的行数，k >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 指向标量 alpha 的 Host 指针，全批共用，不可为 nullptr，Host 内存 |
| A | 输入 | const float*（FP32） | batch 0 矩阵 A 的 Device 内存首地址，列主序存储，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维度，transA=N 时 lda >= max(1, m)，transA=T/C 时 lda >= max(1, k)，Host 内存 |
| strideA | 输入 | int64_t | 相邻两 batch 的 A 起始元素偏移（以元素个数计，非字节），允许为 0（表示 A 在所有 batch 间广播），Host 内存 |
| B | 输入 | const float*（FP32） | batch 0 矩阵 B 的 Device 内存首地址，列主序存储，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度，transB=N 时 ldb >= max(1, k)，transB=T/C 时 ldb >= max(1, n)，Host 内存 |
| strideB | 输入 | int64_t | 相邻两 batch 的 B 起始元素偏移（以元素个数计，非字节），允许为 0（表示 B 在所有 batch 间广播），Host 内存 |
| beta | 输入 | const float*（FP32） | 指向标量 beta 的 Host 指针，全批共用，不可为 nullptr，Host 内存 |
| C | 输入/输出 | float*（FP32） | batch 0 矩阵 C 的 Device 内存首地址，列主序存储，就地读写，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度，ldc >= max(1, m)，Host 内存 |
| strideC | 输入 | int64_t | 相邻两 batch 的 C 起始元素偏移（以元素个数计，非字节），Host 内存 |
| batchCount | 输入 | int | batch 数量，batchCount >= 0，Host 内存 |

#### 约束说明

- handle 不可为 nullptr，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- transA、transB 必须属于 {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}
- m >= 0、n >= 0、k >= 0、batchCount >= 0
- alpha、beta 不可为 nullptr
- transA=N 时 lda >= max(1, m)；transA=T/C 时 lda >= max(1, k)
- transB=N 时 ldb >= max(1, k)；transB=T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- k > 0 时（且 m>0、n>0、batchCount>0），A、B 不可为 nullptr
- beta != 0 时（且 m>0、n>0、batchCount>0），C 不可为 nullptr
- 所有矩阵按列主序（Column-Major）存储，lda/ldb/ldc 语义遵循 NETLIB/CBLAS BLAS 标准
- strideA/strideB 取值无上界强制校验，允许为 0（表示对应矩阵在所有 batch 间广播复用）；调用方须保证各 batch 子矩阵在所声明的 shape/ld/stride 下不越界访问显存
- strideC == 0 且 batchCount > 1 时各 batch 写入同一 C 区域，存在写覆盖/竞争，语义未定义，由调用方负责规避
- m == 0 或 n == 0 或 batchCount == 0 时无计算，直接返回 `ACLBLAS_STATUS_SUCCESS`
- k == 0 或 alpha == 0（k>0）时跳过矩阵乘，逐 batch 执行 C_i = beta * C_i（beta==0 时置零，beta==1 时不变）

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

    struct BlasHandleDeleter {
        void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
    };

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

int aclblasSgemmStridedBatchedTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(handlePtr.get(), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据：batchCount=2, m=n=k=4, transA/transB=N, alpha=1.0, beta=0.0
    constexpr int m = 4;
    constexpr int n = 4;
    constexpr int k = 4;
    constexpr int lda = 4;
    constexpr int ldb = 4;
    constexpr int ldc = 4;
    constexpr int64_t strideA = static_cast<int64_t>(m) * k;
    constexpr int64_t strideB = static_cast<int64_t>(k) * n;
    constexpr int64_t strideC = static_cast<int64_t>(m) * n;
    constexpr int batchCount = 2;
    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<float> hA(static_cast<size_t>(strideA) * batchCount, 0.0f);
    std::vector<float> hB(static_cast<size_t>(strideB) * batchCount, 0.0f);
    std::vector<float> hC(static_cast<size_t>(strideC) * batchCount, 0.0f);
    for (int b = 0; b < batchCount; ++b) {
        for (int i = 0; i < m * k; ++i) {
            hA[static_cast<size_t>(b) * strideA + i] = static_cast<float>(i + b + 1);
        }
        for (int i = 0; i < k * n; ++i) {
            hB[static_cast<size_t>(b) * strideB + i] = static_cast<float>(i + b + 1);
        }
    }

    // 3. 申请 Device 内存并拷贝数据
    size_t aBytes = hA.size() * sizeof(float);
    size_t bBytes = hB.size() * sizeof(float);
    size_t cBytes = hC.size() * sizeof(float);

    void* rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> aDevicePtr(rawA, aclrtFree);

    void* rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> bDevicePtr(rawB, aclrtFree);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> cDevicePtr(rawC, aclrtFree);

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(cDevicePtr.get(), cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSgemmStridedBatched（alpha/beta 为 Host 指针）
    blasRet = aclblasSgemmStridedBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, &alpha,
        static_cast<float*>(aDevicePtr.get()), lda, strideA, static_cast<float*>(bDevicePtr.get()), ldb, strideB,
        &beta, static_cast<float*>(cDevicePtr.get()), ldc, strideC, batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasSgemmStridedBatched failed. ERROR: %d\n", blasRet); return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    for (int b = 0; b < batchCount; ++b) {
        LOG_PRINT("batch %d result:\n", b);
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                size_t idx = static_cast<size_t>(b) * strideC + static_cast<size_t>(col) * ldc + row;
                LOG_PRINT("%.1f ", hC[idx]);
            }
            LOG_PRINT("\n");
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgemmStridedBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgemmStridedBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
