# sspmv 算子实现

## 概述

BLAS spmv 算子 arch35 实现。

sspmv (Symmetric Packed Matrix-Vector Multiplication) 实现了对称压缩矩阵与向量的乘法运算：

```
y = alpha * A * x + beta * y
```

其中 A 为 n×n 对称矩阵，以 packed format 压缩存储，仅存上三角或下三角部分（共 n(n+1)/2 个元素），对称部分通过已有元素推断。

## 支持的产品

- Atlas A5 训练系列产品（ascend950）

## 目录结构介绍

```
├── spmv
│   ├── README.md
│   └── arch35/
│       ├── README.md
│       ├── sspmv_host.cpp
│       ├── sspmv_kernel.cpp
│       └── sspmv_tiling_data.h
```

## 算子描述

对应的接口为：

```c
aclblasStatus_t aclblasSspmv(aclblasHandle handle,
                 aclblasFillMode uplo, int n, const float *alpha,
                 const float *AP, const float *x, int incx, const float *beta,
                 float *y, int incy);
```

| 参数 | in/out | 设备 | 含义 |
|------|--------|------|------|
| handle | in | host | aclblas 库句柄，携带 stream |
| uplo | in | host | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122) |
| n | in | host | 方阵阶数，n ≥ 0 |
| alpha | in | host | 标量 alpha 的指针 |
| AP | in | device | 对称压缩矩阵，列主序，共 n(n+1)/2 个元素 |
| x | in | device | 输入向量，n 个元素 |
| incx | in | host | x 的步长，incx ≠ 0（可正可负） |
| beta | in | host | 标量 beta 的指针 |
| y | in/out | device | 输入/输出向量，n 个元素 |
| incy | in | host | y 的步长，incy ≠ 0（可正可负） |

**注意**：AP、x、y 必须为 device 侧指针，由调用者在调用前通过 `aclrtMalloc` 分配并通过 `aclrtMemcpy` 拷入数据。alpha、beta 为 host 侧指针。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

## 编译运行

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=spmv --soc=ascend950

# 编译并运行测试
bash build.sh --ops=spmv --soc=ascend950 --run
```
