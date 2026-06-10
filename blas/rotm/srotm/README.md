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
├── rotm
│   └── srotm
│       ├── README.md
│       ├── arch22/
│       │   ├── srotm_host.cpp
│       │   ├── srotm_kernel.cpp
│       │   └── srotm_tiling_data.h
│       └── arch35/
│           ├── srotm_common.h
│           ├── srotm_host.cpp
│           ├── srotm_kernel.cpp
│           └── srotm_tiling_data.h
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
| n | in | host | 向量长度，n ≥ 0 |
| x | in/out | device | 输入/输出向量 x |
| incx | in | host | x 的步长，incx ≠ 0（可正可负） |
| y | in/out | device | 输入/输出向量 y |
| incy | in | host | y 的步长，incy ≠ 0（可正可负） |
| param | in | host | 5 个元素的旋转参数数组：[flag, h11, h21, h12, h22] |

**注意**：x、y 必须为 device 侧指针，由调用者在调用前通过 `aclrtMalloc` 分配并通过 `aclrtMemcpy` 拷入数据。param 为 host 侧指针。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

### 双路径策略（arch35）

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

# 直接运行已编译的测试
LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH ./build/test/rotm/srotm/srotm_test
```
