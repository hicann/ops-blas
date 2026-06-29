# RotEx算子

## 算子概述

RotEx (Givens Rotation) 算子实现了向量的 Givens 旋转，对向量 x 和 y 的对应元素对应用旋转矩阵。算子只支持对称类型（xType==yType==csType），x/y 可为 FP32/BF16/FP16，计算在 FP32 精度下执行。根据步长模式自动选择 SIMD 连续路径或 SIMT 离散路径。

数学表达式（Fortran 1-based 索引）：

```
x[k] = c * x[k] + s * y[j]
y[j] = -s * x[k] + c * y[j]

k = 1 + (i-1) * incx
j = 1 + (i-1) * incy
  for i = 1, 2, ..., n
```

其中 `c = cos(theta)`、`s = sin(theta)` 为 Givens 旋转的标量参数。

第二个等式中 `y[j]` 的计算使用旋转前的原始 `x[k]` 值（即第一条公式更新前的值），而非更新后的值。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasRotEx | Givens 旋转，支持混合精度（S 组） |

## 算子执行接口

### aclblasRotEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasRotEx(
    aclblasHandle_t handle,
    int n,
    void *x,
    aclDataType xType,
    int incx,
    void *y,
    aclDataType yType,
    int incy,
    const void *c,
    const void *s,
    aclDataType csType,
    aclDataType executionType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量 x 和 y 中需旋转的元素数量，n >= 0，Host 内存 |
| x | 输入/输出 | void* | 输入/输出向量 x 的设备指针，按 xType 类型解释，Device 内存 |
| xType | 输入 | aclDataType | 向量 x 的数据类型枚举，S 组支持 FP32 / FP16 / BF16，Host 内存 |
| incx | 输入 | int | x 元素的步长（正数或负数，不可为零），Host 内存 |
| y | 输入/输出 | void* | 输入/输出向量 y 的设备指针，按 yType 类型解释，Device 内存 |
| yType | 输入 | aclDataType | 向量 y 的数据类型枚举，S 组支持 FP32 / FP16 / BF16，Host 内存 |
| incy | 输入 | int | y 元素的步长（正数或负数，不可为零），Host 内存 |
| c | 输入 | const void* | Givens 旋转的余弦值标量，按 csType 解释，Host 或 Device 内存 |
| s | 输入 | const void* | Givens 旋转的正弦值标量，按 csType 解释，Host 或 Device 内存 |
| csType | 输入 | aclDataType | c 和 s 的数据类型枚举，S 组支持 FP32 / FP16 / BF16，Host 内存 |
| executionType | 输入 | aclDataType | 计算精度类型枚举，arch35 仅支持 FP32，Host 内存 |

#### 支持数据类型

| executionType | xType / yType | csType |
|---------------|---------------|--------|
| FP32 | FP32 | FP32 |
| FP32 | FP16 | FP16 |
| FP32 | BF16 | BF16 |

> arch35 芯片仅支持 S 组（executionType=FP32），D 组（FP64）、C 组（C64）和 Z 组（C128）当前在 arch35 上不提供支持，传入对应 executionType 将返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

#### 约束说明

- `handle != nullptr`，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- `n >= 0`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `n == 0` 时直接返回 `ACLBLAS_STATUS_SUCCESS`，不启动 Kernel
- `n > 0` 时 `x != nullptr`、`y != nullptr`、`c != nullptr`、`s != nullptr`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `incx != 0`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `incy != 0`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `executionType` 必须为 FP32（arch35 仅支持 S 组），否则返回 `ACLBLAS_STATUS_NOT_SUPPORTED`
- `xType`、`yType` 和 `csType` 必须相同（仅对称类型），且必须为 FP32 / FP16 / BF16 之一，否则返回 `ACLBLAS_STATUS_NOT_SUPPORTED`
- c/s 标量指针可为 Host 或 Device 内存，运行时自动检测并处理

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include <cstdint>
#include <vector>

int main()
{
    // 初始化 ACL
    aclInit(nullptr);
    aclrtSetDevice(0);

    // 创建 handle
    aclblasHandle_t handle;
    aclblasCreate(&handle);

    // 参数配置
    int n = 1024;
    aclDataType xType = ACL_FLOAT;
    aclDataType yType = ACL_FLOAT;
    int incx = 1;
    int incy = 1;
    aclDataType csType = ACL_FLOAT;
    aclDataType executionType = ACL_FLOAT;

    // 分配 host 内存
    std::vector<float> hX(n, 1.0f);
    std::vector<float> hY(n, 2.0f);
    float c = 0.8f;
    float s = 0.6f;

    // 分配 device 内存
    void *dX = nullptr;
    void *dY = nullptr;
    aclrtMalloc(&dX, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dY, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    // 拷贝数据到 device
    aclrtMemcpy(dX, n * sizeof(float), hX.data(), n * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dY, n * sizeof(float), hY.data(), n * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);

    // 调用 RotEx 算子
    aclblasRotEx(handle, n, dX, xType, incx, dY, yType, incy,
                 &c, &s, csType, executionType);

    // 同步等待计算完成
    aclrtStreamSynchronize(nullptr);

    // 结果拷贝回 host
    aclrtMemcpy(hX.data(), n * sizeof(float), dX, n * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hY.data(), n * sizeof(float), dY, n * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);

    // 释放资源
    aclrtFree(dX);
    aclrtFree(dY);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```
