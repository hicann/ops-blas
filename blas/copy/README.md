# Copy算子

## 算子概述

向量拷贝算子，实现 `Y = X` 的向量数据搬移。

数学表达式：

```
Y[i] = X[i],   for i = 0, 1, ..., N-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasScopy | 单精度浮点向量拷贝 |

## 算子执行接口

### aclblasScopy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasScopy(aclblasHandle_t handle, int n, const float *x, int incx, float *y, int incy);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量长度（元素个数），Host 内存 |
| x | 输入 | const float*（FP32） | 源向量 X，只读，Device 内存 |
| incx | 输入 | int | X 元素的步长（以 float 元素为单位），不可为 0，Host 内存 |
| y | 输出 | float*（FP32） | 目标向量 Y，可写，Device 内存 |
| incy | 输入 | int | Y 元素的步长（以 float 元素为单位），不可为 0，Host 内存 |

#### 约束说明

- n >= 0（n < 0 时返回 INVALID_VALUE）
- incx != 0
- incy != 0
- x、y 不可为 nullptr

#### 调用示例

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle;
    aclblasCreate(&handle);

    int64_t n = 1024;
    int64_t incx = 1;
    int64_t incy = 1;

    uint8_t *dX, *dY;
    aclrtMalloc((void**)&dX, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dY, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    aclblasScopy(handle, dX, dY, n, incx, incy);

    aclrtStreamSynchronize(nullptr);

    aclrtFree(dX);
    aclrtFree(dY);
    aclblasDestroy(handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```
