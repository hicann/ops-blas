# {算子名称}算子

<!--
模板使用说明：
1. 将所有 {占位符} 替换为实际内容，占位符命名规则见下表。
2. 按实际包含的接口增删子章节（### 节）。如仅有单精度接口则只保留一个 ### 节；如有更多接口（如 aclblasH{op}、aclblasC{op}）则继续添加。
3. 参数说明列必须写清楚参数含义和内存位置（Host 内存/Device 内存）。
4. 约束说明如无约束则写"无"，不允许留空。
5. 调用示例须由开发者在本地跑通后再上库。
6. 完成文档后删除本使用说明注释块。

占位符约定：

| 占位符 | 含义 | 示例 |
|--------|------|------|
| {算子名称} | 算子中文名 | 向量缩放 |
| {op} | 算子英文缩写（小写） | scal |
| {功能描述} | 算子功能的一句话描述 | 对向量进行标量缩放 |
| {运算描述} | 核心数学运算描述 | y = alpha * x |
| {单精度功能描述} | aclblasS{op} 接口功能简述 | 实数向量缩放 |
| {双精度功能描述} | aclblasD{op} 接口功能简述 | 双精度向量缩放 |
| {支持/不支持} | 各产品行是否支持，按实情填写 | 支持 |
| {n含义} | n 参数的具体含义 | 向量 x 中的元素个数 |
| {x含义} | x 参数的具体含义 | 向量 x 的数据指针 |
| {y含义} | y 参数的具体含义 | 向量 y 的数据指针 |
| {约束列表} | 参数约束条件列表，无约束则写"无" | n >= 0 |
-->

## 算子概述

{一段话描述算子的功能定位和核心运算。例如："{op} 算子实现了{功能描述}，核心运算为{运算描述}。"}

数学表达式：

```
{数学公式，使用纯文本。如需要可用 LaTeX 格式（$$...$$）}
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasS{op} | {单精度功能描述} |
| aclblasD{op} | {双精度功能描述} |

## 算子执行接口

### aclblasS{op}

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：{支持/不支持}
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：{支持/不支持}
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：{支持/不支持}

#### 函数原型

```cpp
aclblasStatus_t aclblasS{op}(aclblasHandle_t handle, int n, const float *x, float *y, ...)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | {n含义}，Host 内存 |
| x | 输入 | const float*（FP32） | {x含义}，Device 内存 |
| y | 输入/输出 | float*（FP32） | {y含义}，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0

{按算子实际参数填写完整约束列表，每项一条；如无约束则写"无"，不允许留空。}

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

    // 分配 Device 内存、初始化数据、调用 aclblasS{op}、同步并释放资源

    aclblasDestroy(handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```

### aclblasD{op}

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：{支持/不支持}
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：{支持/不支持}
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：{支持/不支持}

#### 函数原型

```cpp
aclblasStatus_t aclblasD{op}(aclblasHandle_t handle, int n, const double *x, double *y, ...)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | {n含义}，Host 内存 |
| x | 输入 | const double*（FP64） | {x含义}，Device 内存 |
| y | 输入/输出 | double*（FP64） | {y含义}，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0

{按算子实际参数填写完整约束列表，每项一条；如无约束则写"无"，不允许留空。}

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

    // 分配 Device 内存、初始化数据、调用 aclblasD{op}、同步并释放资源

    aclblasDestroy(handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```
