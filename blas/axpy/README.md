# Axpy算子

## 算子概述

基础向量运算，实现 `y = alpha * x + y`。

数学表达式：

```
y[i] = alpha * x[i] + y[i]  for i = 0 to n-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSaxpy | 单精度浮点 AXPY |
| aclblasCaxpy | 复数 AXPY |

## 算子执行接口

### aclblasSaxpy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx, float* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 指向标量乘数的指针，Host 内存 |
| x | 输入 | float*（FP32） | 输入向量 x，Device 内存 |
| incx | 输入 | int | 向量 x 的步长，Host 内存 |
| y | 输入/输出 | float*（FP32） | 输入/输出向量 y，Device 内存 |
| incy | 输入 | int | 向量 y 的步长，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- incy != 0

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

```cpp
// 待补齐
```

### aclblasCaxpy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
int aclblasCaxpy(aclblasHandle handle, const std::complex<float> *x, std::complex<float> *y, const std::complex<float> alpha, const int64_t n, const int64_t incx, const int64_t incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle | ACL 流句柄，Host 内存 |
| x | 输入 | const std::complex<float>*（FP32 complex） | 输入复向量，Device 内存 |
| y | 输入/输出 | std::complex<float>*（FP32 complex） | 输入/输出复向量，Device 内存 |
| alpha | 输入 | const std::complex<float> | 复数标量系数，Host 内存 |
| n | 输入 | int64_t | 向量长度，Host 内存 |
| incx | 输入 | int64_t | x 的步长（当前未使用），Host 内存 |
| incy | 输入 | int64_t | y 的步长（当前未使用），Host 内存 |

#### 约束说明

- n >= 0

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include <complex>

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle;
    aclblasCreate(&handle);

    int64_t n = 1024;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = {2.0f, 1.0f};

    uint8_t *dX, *dY;
    aclrtMalloc((void**)&dX, n * sizeof(std::complex<float>), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dY, n * sizeof(std::complex<float>), ACL_MEM_MALLOC_HUGE_FIRST);

    aclblasCaxpy(handle, n, alpha, dX, incx, dY, incy);

    aclrtStreamSynchronize(nullptr);

    aclrtFree(dX);
    aclrtFree(dY);
    aclblasDestroy(handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```
