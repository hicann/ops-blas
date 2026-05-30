# sbmv 算子实现

## 概述

BLAS sbmv 算子实现。

sbmv (Symmetric Banded Matrix-Vector Multiplication) 算子实现了对称带状矩阵与向量的乘法运算：

```
y = alpha * A * x + beta * y
```

其中 A 为 n×n 对称带状矩阵，有 k 条次对角线和 k 条超对角线。A 以列主序压缩存储为 (k+1)×n 的数组，仅存储上三角或下三角部分，对称部分通过已有元素推断。

## 支持的产品

- Atlas A5 训练系列产品（ascend950）

## 目录结构介绍

```
├── sbmv
│   └── ssbmv
│       ├── README.md
│       └── arch35/
│           ├── ssbmv_host.cpp
│           ├── ssbmv_kernel.cpp
│           └── ssbmv_tiling_data.h
```

## 算子描述

对应的接口为：

```c
aclblasStatus_t aclblasSsbmv(aclblasHandle handle,
                 aclblasFillMode uplo, int n, int k, const float *alpha,
                 const float *A, int lda, const float *x, int incx, const float *beta,
                 float *y, int incy);
```

| 参数 | in/out | 设备 | 含义 |
|------|--------|------|------|
| handle | in | host | aclblas 库句柄，携带 stream |
| uplo | in | host | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122) |
| n | in | host | 方阵阶数，n ≥ 0 |
| k | in | host | 次对角线/超对角线数量，k ≥ 0 |
| alpha | in | host | 标量 alpha 的指针 |
| A | in | device | 带状对称矩阵，列主序，维度 (k+1)×n |
| lda | in | host | A 的主维数，lda ≥ k+1 |
| x | in | device | 输入向量，n 个元素 |
| incx | in | host | x 的步长，incx ≠ 0（可正可负） |
| beta | in | host | 标量 beta 的指针 |
| y | in/out | device | 输入/输出向量，n 个元素 |
| incy | in | host | y 的步长，incy ≠ 0（可正可负） |

**注意**：A、x、y 必须为 device 侧指针，由调用者在调用前通过 `aclrtMalloc` 分配并通过 `aclrtMemcpy` 拷入数据。alpha、beta 为 host 侧指针。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

## 编译运行

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=sbmv --soc=ascend950

# 编译并运行测试
bash build.sh --ops=sbmv --soc=ascend950 --run

# 直接运行已编译的测试
LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH ./build/test/sbmv/ssbmv/ssbmv_test
```
