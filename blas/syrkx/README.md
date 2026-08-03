# Syrkx算子

## 算子概述

Syrkx 算子实现了对称秩k更新（A≠B），核心运算为 `C = alpha * op(A) * op(B)^T + beta * C`，其中 A 和 B 为不同矩阵，C 为对称矩阵，仅 uplo 指定的三角部分被更新。

数学表达式：

```
C = alpha * op(A) * op(B)^T + beta * C
```

其中：
- `op(A)` 和 `op(B)` 由 `trans` 参数决定：
  - `trans = ACLBLAS_OP_N`：`op(A) = A`，`op(B) = B`，A 和 B 为 (N, K) 矩阵
  - `trans = ACLBLAS_OP_T`：`op(A) = A^T`，`op(B) = B^T`，A 和 B 为 (K, N) 矩阵
- `C` 为 (N, N) 对称矩阵，仅上三角或下三角部分被更新
- `alpha` 和 `beta` 为标量缩放因子
- `uplo` 参数指定填充上三角 (`ACLBLAS_UPPER`) 或下三角 (`ACLBLAS_LOWER`)
- 存储顺序：列主序（Column-Major）

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsyrkx | 单精度浮点对称秩k更新（A≠B） |

## 算子执行接口

### aclblasSsyrkx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsyrkx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定 C 的填充模式：ACLBLAS_UPPER("U")=上三角，ACLBLAS_LOWER("L")=下三角，Host 内存 |
| trans | 输入 | aclblasOperation_t | 指定 A/B 是否转置：ACLBLAS_OP_N("N")=不转置，ACLBLAS_OP_T("T")=转置，ACLBLAS_OP_C("C")=共轭转置（实数域等价于 OP_T），Host 内存 |
| n | 输入 | int | 矩阵 C 的行列数，且为 op(A) 的行数和 op(B)^T 的列数，Host 内存 |
| k | 输入 | int | op(A) 的列数和 op(B)^T 的行数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量缩放因子，不可为 nullptr，Device 内存 |
| A | 输入 | const float*（FP32） | 输入矩阵 A，列主序存储，Device 内存 |
| lda | 输入 | int | A 的 leading dimension，Host 内存 |
| B | 输入 | const float*（FP32） | 输入矩阵 B，列主序存储，Device 内存 |
| ldb | 输入 | int | B 的 leading dimension，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量缩放因子，不可为 nullptr，Device 内存 |
| C | 输入/输出 | float*（FP32） | 输入/输出矩阵 C，(N, N) 对称矩阵，列主序存储，仅 uplo 指定的三角部分被更新，Device 内存 |
| ldc | 输入 | int | C 的 leading dimension，Host 内存 |

#### 约束说明

- n >= 0
- k >= 0
- alpha 不可为 nullptr
- beta 不可为 nullptr
- A 在 n > 0 且 k > 0 时不可为 nullptr
- B 在 n > 0 且 k > 0 时不可为 nullptr
- C 在 n > 0 时不可为 nullptr
- trans = ACLBLAS_OP_N 时：lda >= max(1, n)，ldb >= max(1, n)
- trans = ACLBLAS_OP_T 或 ACLBLAS_OP_C 时：lda >= max(1, k)，ldb >= max(1, k)
- ldc >= max(1, n)
- 当 n == 0 时，直接返回成功
- 当 k == 0 时，仅执行 C = beta * C 的缩放更新

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

int aclblasSsyrkxTest(AclContext& ctx)
{
    // 参数设置：trans=N, uplo=U, n=3, k=2
    constexpr int n = 3;
    constexpr int k = 2;
    constexpr int lda = n;  // trans=N: lda >= max(1, n)
    constexpr int ldb = n;  // trans=N: ldb >= max(1, n)
    constexpr int ldc = n;  // ldc >= max(1, n)
    float alpha = 1.0f;
    float beta = 0.0f;

    // 列主序存储矩阵 A (3x2)
    // A = | 1.0  3.0 |
    //     | 2.0  4.0 |
    //     | 5.0  6.0 |
    float hA[lda * k] = {1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f};

    // 列主序存储矩阵 B (3x2)
    // B = | 1.0  4.0 |
    //     | 2.0  5.0 |
    //     | 3.0  6.0 |
    float hB[ldb * k] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // 列主序存储矩阵 C (3x3)，beta=0 时初值不影响结果
    float hC[ldc * n] = {0.0f};

    constexpr size_t bytesA = lda * k * sizeof(float);
    constexpr size_t bytesB = ldb * k * sizeof(float);
    constexpr size_t bytesC = ldc * n * sizeof(float);
    constexpr size_t bytesScalar = sizeof(float);

    // 申请 Device 内存
    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, bytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bytesB, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dB(rawB);

    void *rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, bytesC, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dC(rawC);

    void *rawAlpha = nullptr;
    aclRet = aclrtMalloc(&rawAlpha, bytesScalar, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAlpha(rawAlpha);

    void *rawBeta = nullptr;
    aclRet = aclrtMalloc(&rawBeta, bytesScalar, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dBeta(rawBeta);

    // 拷贝数据到 Device
    aclRet = aclrtMemcpy(dA.get(), bytesA, hA, bytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dB.get(), bytesB, hB, bytesB, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dC.get(), bytesC, hC, bytesC, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dAlpha.get(), bytesScalar, &alpha, bytesScalar, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dBeta.get(), bytesScalar, &beta, bytesScalar, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    // 调用 aclblasSsyrkx: C = alpha * A * B^T + beta * C
    blasRet = aclblasSsyrkx(
        static_cast<aclblasHandle_t>(handlePtr.get()),
        ACLBLAS_UPPER,       // uplo: 上三角
        ACLBLAS_OP_N,        // trans: 不转置
        n, k,
        static_cast<const float*>(dAlpha.get()),
        static_cast<const float*>(dA.get()), lda,
        static_cast<const float*>(dB.get()), ldb,
        static_cast<const float*>(dBeta.get()),
        static_cast<float*>(dC.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    // 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC, bytesC, dC.get(), bytesC, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 打印上三角结果（列主序）
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            printf("C[%d][%d] = %f\n", i, j, hC[j * ldc + i]);
        }
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSsyrkxTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
