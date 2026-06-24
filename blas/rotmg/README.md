# rotmg 算子实现

## 概述

BLAS rotmg 算子实现。

rotmg (Construct Modified Givens Rotation) 算子根据输入的标量值构造 modified Givens 旋转矩阵 H 的参数，使得：

```
    H^T * diag(d1, d2) * H = diag(d1_new, d2_new)
    H * [x1, y1]^T = [x1_new, 0]^T
```

其中 H 为 2×2 变换矩阵，由 param 数组按 BLAS 标准编码：

| param[0] (sflag) | H 矩阵                                    | 存储的元素          |
|------------------|--------------------------------------------|---------------------|
| -1.0             | [[h11, h12], [h21, h22]]                   | param[1..4] 全部存储 |
| 0.0              | [[1, h12], [h21, 1]]                       | param[2]=h21, param[3]=h12 |
| 1.0              | [[h11, 1], [-1, h22]]                      | param[1]=h11, param[4]=h22 |
| -2.0             | [[1, 0], [0, 1]]                           | 无（恒等变换，仅设 sflag） |

## 特点

**rotmg 是纯标量计算**：与 rotm（对长度为 n 的向量施加旋转）不同，rotmg 仅操作 4 个标量值（d1, d2, x1, y1），构造旋转参数。rotmg 的输出通常直接作为下游 `rotm` 算子的输入。

### 实现策略（双路径）

使用 `aclrtPointerGetAttributes` 判断五个指针（d1/d2/x1/y1/param）的存储位置：

- **全部 Host 指针**：直接 CPU 计算，无 kernel 开销，无数据搬运。
- **全部 Device 指针**：启动 device kernel（1 block SIMT），计算和结果都留在 device 上，直接对接下游 `rotm` 等算子。
- **混合指针**：返回 `ACLBLAS_STATUS_INVALID_VALUE`，不符合 BLAS 标准要求。

## 支持的产品

- Atlas A5 训练系列产品（ascend950 / arch35）

## 目录结构介绍

```
blas/rotmg/
├── README.md                       // 说明文档
└── arch35/
    ├── srotmg_host.cpp             // Host 侧实现（CPU 路径 + kernel 路径调度 + kernel 声明）
    ├── srotmg_kernel.cpp           // Device kernel（SIMT 标量计算）
    └── srotmg_tiling_data.h        // Tiling 数据结构
```

## 算子描述

对应的接口为：

```c
aclblasStatus_t aclblasSrotmg(aclblasHandle_t handle,
                 float *d1, float *d2, float *x1,
                 const float *y1, float *param);
```

| 参数   | in/out   | 设备          | 含义                              |
|--------|----------|---------------|-----------------------------------|
| handle | in       | host          | aclblas 库句柄，携带 stream        |
| d1     | in/out   | host / device | x 的缩放因子                       |
| d2     | in/out   | host / device | y 的缩放因子                       |
| x1     | in/out   | host / device | 向量的第一个分量                    |
| y1     | in       | host / device | 向量的第二个分量                    |
| param  | out      | host / device | 5 个元素的旋转参数数组              |

**注意**：d1, d2, x1, y1, param 必须全部位于 host 侧或全部位于 device 侧；混合 host/device 指针将返回 `ACLBLAS_STATUS_INVALID_VALUE`。

## 算法说明

算法实现参考 netlib 标准 BLAS `srotmg.f`：

1. 若 `d1 < 0`：sflag = -1，全部置零。
2. 若 `d2 * y1 == 0`：sflag = -2，恒等变换，不修改 d1/d2/x1。
3. 根据 `|d1*x1^2|` 与 `|d2*y1^2|` 的大小关系计算旋转参数：
   - `|d1*x1^2| > |d2*y1^2|`：sflag = 0，h12 和 h21 非平凡。
   - 否则：sflag = 1，h11 和 h22 非平凡。
4. 缩放检查：对 d1 和 d2 进行 range check（[RGAMSQ, GAMSQ]），必要时吸收缩放因子到矩阵中（sflag 变为 -1）。

参考论文：Lawson, Hanson, Kincaid, Krogh, "Basic Linear Algebra Subprograms for Fortran Usage", ACM TOMS Vol.5 No.3, Sep 1979, pp.308-323.

## 编译运行

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子
bash build.sh --ops=srotmg --soc=ascend950

# 编译并运行测试
bash build.sh --ops=srotmg --soc=ascend950 --run
```
