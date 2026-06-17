# rotm 算子实现

## 概述

BLAS rotm 算子实现。

rotm (Modified Givens Rotation) 算子对向量 x 和 y 应用 modified Givens 旋转：

```
[x; y] := H * [x; y]
```

其中 H 为 2×2 变换矩阵，由 param 数组按 BLAS 标准编码：

| param[0] (flag) | H 矩阵 |
|-----------------|--------|
| -1.0 | [[param[1], param[3]], [param[2], param[4]]] |
| 0.0 | [[1, param[3]], [param[2], 1]] |
| 1.0 | [[param[1], 1], [-1, param[4]]] |
| -2.0 | 无操作（恒等变换） |

## 支持的产品

- Atlas A5 训练系列产品（ascend950 / arch35）
- Atlas A2/A3 训练/推理系列产品（ascend910B / arch22）

## 目录结构介绍

```
blas/rotm/
├── README.md                       // 说明文档
├── arch22/
│   ├── srotm_host.cpp              // Host 侧实现（arch22）
│   ├── srotm_kernel.cpp            // Kernel 侧实现（arch22）
│   └── srotm_tiling_data.h         // Tiling 数据结构（arch22）
└── arch35/
    ├── srotm_host.cpp              // Host 侧实现（arch35）
    ├── srotm_kernel.cpp            // Kernel 侧实现（arch35）
    └── srotm_tiling_data.h         // Tiling 数据结构（arch35）
```

## 算子描述

对应的接口为：

```c
aclblasStatus_t aclblasSrotm(aclblasHandle_t handle,
                 int n, float *x, int incx, float *y, int incy,
                 const float *param);
```

| 参数 | in/out | 设备 | 含义 |
|------|--------|------|------|
| handle | in | host | aclblas 库句柄，携带 stream |
| n | in | host | 向量长度，n >= 0 |
| x | in/out | device | 输入/输出向量 x |
| incx | in | host | x 的步长，incx != 0（可正可负） |
| y | in/out | device | 输入/输出向量 y |
| incy | in | host | y 的步长，incy != 0（可正可负） |
| param | in | host | 5 个元素的旋转参数数组：[flag, h11, h21, h12, h22] |

**注意**：x、y 必须为 device 侧指针，由调用者在调用前通过 `aclrtMalloc` 分配并通过 `aclrtMemcpy` 拷入数据。param 为 host 侧指针。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

### arch22 实现

Host 侧首先根据 `sparam` 解析出本次变换需要使用的系数，并按向量长度选择核执行所需的分块信息。当前实现使用 by-value tiling：Host 侧将 `x/y` device 地址、实际存储长度、`incx/incy` 以及本次变换用到的 `sflag/h11/h12/h21/h22` 一并写入 tiling 结构后，直接调用一次 `srotm_kernel_do` 下发到 stream。当 `incx == 1 && incy == 1` 时，device 侧按 512B 对齐 chunk 做多工作块分配；当存在非 unit stride 时，当前实现会收敛为单工作块执行。主 kernel 根据逻辑索引直接计算带步长的 GM 地址，完成原地读写。

### arch35 双路径策略

A5 平台根据步长自动选择执行路径：

| 条件 | 路径 | 说明 |
|------|------|------|
| incx=1 且 incy=1 | DMA (SIMD membase) | 连续访存，双缓冲流水线 |
| incx=-1 且 incy=-1 | DMA (SIMD membase) | 连续反向访存，双缓冲流水线 |
| 其他步长组合 | SIMT | Grid-stride 并行，兼容任意步长 |

## 编译运行

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=srotm --soc=ascend950

# 编译并运行测试
bash build.sh --ops=srotm --soc=ascend950 --run
```
