# Scal算子

## 算子概述

向量缩放算子，实现向量乘以标量的运算。包含实数向量缩放（Sscal）和复数向量缩放（Cscal）。

数学表达式：

```
x[i] = alpha * x[i]  (i = 0 .. n-1，步长为 incx)
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSscal | 实数向量乘以标量 |
| aclblasCscal | 复数向量乘以复数标量 |

## 算子执行接口

### aclblasSscal

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSscal(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量 x 中的元素个数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 指向标量乘数的指针，Host 内存 |
| x | 输入/输出 | float*（FP32） | float 向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

**Ascend 950PR / Ascend 950DT（arch35）：**

- n 为整数；n <= 0 时为 no-op（直接返回 ACLBLAS_STATUS_SUCCESS，不修改 x）
- incx 为整数；incx <= 0 时为 no-op（不修改 x，对齐参考 BLAS cblas_sscal 的 IF (INCX.LE.0) RETURN 语义）；incx > 0 时支持任意步长
- handle 不能为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- n > 0 时 alpha、x 不能为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE

**Atlas A2 / Atlas A3 系列产品（arch22）：**

- incx 参数当前实现未启用（固定按连续向量 incx=1 处理，传入的 incx 取值不生效）
- 未对 n、handle、alpha、x 做入参校验


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

int aclblasSscalTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    int n = 5;
    int incx = 1;
    float alpha = 2.0f;
    std::vector<float> xHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};  // 缩放后期望 {2,4,6,8,10}
    size_t xBytes = n * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawMem = nullptr;
    auto aclRet = aclrtMalloc(&rawMem, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> xDevicePtr(rawMem, aclrtFree);

    aclRet = aclrtMemcpy(xDevicePtr.get(), xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSscal（alpha 为 Host 指针，原地缩放）
    blasRet = aclblasSscal(
        static_cast<aclblasHandle_t>(handlePtr.get()), n, &alpha,
        static_cast<float*>(xDevicePtr.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSscal failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> resultData(n, 0);
    aclRet = aclrtMemcpy(resultData.data(), xBytes, xDevicePtr.get(), xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("result[%d] is: %f\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSscalTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSscalTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCscal

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCscal(aclblasHandle_t handle, const int64_t n, const std::complex<float> alpha, uint8_t* x, const int64_t incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int64_t | 向量 x 中的复数元素个数，Host 内存 |
| alpha | 输入 | const std::complex<float> | 用于乘法的复数标量，Host 内存 |
| x | 输入/输出 | uint8_t*（FP32 complex） | 复数向量，包含 n 个 complex<float> 元素，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
