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

测试代码位于 `test/rotm/`：

```
test/rotm/
└── srotm/
    ├── CMakeLists.txt              // 编译工程文件
    ├── srotm_param.h               // CSV 参数解析
    ├── srotm_golden.h              // CPU golden（调用 cblas_srotm）
    └── arch35/
        ├── srotm_test.cpp          // GTest 精度测试
        ├── srotm_test.csv          // CSV 测试用例
        └── srotm_npu_wrapper.h     // NPU 调用封装
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

Host 侧通过 `GetAivCoreCount()` 获取 AIV 核数，Tiling 数据通过值传递给 kernel（无需 GM 分配）。Host 侧为异步执行，不包含流同步操作，由调用方负责同步。

## 测试用例覆盖

测试基于 GTest + CSV 驱动框架，golden 实现调用 Netlib BLAS `cblas_srotm`。

| 分组 | 用例数 | 覆盖场景 |
|------|--------|----------|
| L0 | 6 | flag=-1/0/1/-2 四种模式、n=1 单元素、正步长 incx=2 incy=3 |
| L1 | 34 | 小 n=2~33、块边界 n=127/128/129/255/256、负步长、大 n=4097~131072、全零 H 矩阵、DMA 双负步长 |
| L2 | 2 | n=0 提前退出、n<0 提前退出 |

## 编译运行

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=srotm --soc=ascend950

# 编译并运行测试
bash build.sh --ops=srotm --soc=ascend950 --run
```
