# Iamax算子

## 算子概述

BLAS Iamax（最大绝对值元素索引）算子实现了查找向量中绝对值最大的元素索引，是 BLAS 基础线性代数库中的核心算子之一。该算子返回 1-based 索引，遵循 BLAS 惯例，常用于主元选择和迭代算法中。

数学表达式：

```
result = argmax_i |x[i]|
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasIamax | 查找向量中绝对值最大元素的索引 |

## 算子执行接口

### aclblasIamax

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
int aclblasIamax(const float *x, int32_t *result, const int64_t n, const int64_t incx, const uint32_t dtypeFlag, void *stream);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| x | 输入 | const float*（FP32） | 向量，包含 n 个元素，Device 内存 |
| result | 输出 | int32_t* | 最大绝对值元素的索引（1-based），Device 内存 |
| n | 输入 | int64_t | 向量元素个数，Host 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |
| dtypeFlag | 输入 | uint32_t | 数据类型标志，0 表示实数 float，1 表示复数 float，Host 内存 |
| stream | 输入 | void* | ACL 流，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
