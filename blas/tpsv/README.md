# Tpsv算子

## 算子概述

tpsv (Triangular Packed matrix Solve) 求解三角线性系统。该算子针对三角矩阵的 packed 存储格式进行优化，支持前向和后向求解。

数学表达式：

```
op(A) * x = b
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStpsv | 单精度三角 packed 矩阵求解 |

## 算子执行接口

### aclblasStpsv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStpsv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const float *AP, float *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) — A 为上三角矩阵；ACLBLAS_LOWER(122) — A 为下三角矩阵，Host 内存 |
| trans | 输入 | aclblasOperation_t | ACLBLAS_OP_N(111) — op(A) = A；ACLBLAS_OP_T(112) — op(A) = A^T；ACLBLAS_OP_C(113) — op(A) = A^H（FP32 下与 T 等价），Host 内存 |
| diag | 输入 | aclblasDiagType_t | ACLBLAS_NON_UNIT(131) — 对角元从 AP 读取；ACLBLAS_UNIT(132) — 对角元固定为 1，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0。n == 0 时为空操作直接返回成功，Host 内存 |
| AP | 输入 | const float*（FP32） | packed 三角矩阵指针，共 n*(n+1)/2 个元素，Device 内存 |
| x | 输入/输出 | float*（FP32） | 输入时存储右端向量 b，输出时原地覆盖为解向量 x，Device 内存 |
| incx | 输入 | int | x 的存储增量，incx != 0（可正可负）。incx < 0 时 x 反向存储，Host 内存 |

#### 约束说明

- n >= 0，n == 0 时为空操作直接返回成功
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- diag 必须为 ACLBLAS_NON_UNIT 或 ACLBLAS_UNIT
- incx != 0（可正可负）
- AP、x 不可为 nullptr

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

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
int aclblasStpsvTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int n = 2;
    constexpr int incx = 1;
    constexpr int apSize = n * (n + 1) / 2;
    constexpr size_t apBytes = apSize * sizeof(float);
    constexpr size_t xBytes = n * sizeof(float);

    std::vector<float> hAP = {1.0f, 2.0f, 3.0f};
    std::vector<float> hX = {1.0f, 8.0f};

    void* rawAP = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawAP, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for AP failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawAP, aclrtFree);

    void* rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), apBytes, hAP.data(), apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for AP failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dXPtr.get(), xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasStpsv(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        n, static_cast<const float*>(dAPtr.get()), static_cast<float*>(dXPtr.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStpsv failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<float> xResult(n, 0.0f);
    aclRet = aclrtMemcpy(xResult.data(), xBytes, dXPtr.get(), xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("x[%d] = %f\n", i, xResult[i]);
    }

    LOG_PRINT("aclblasStpsv test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStpsvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasStpsvTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
