## Srotm算子实现

## 概述

本样例展示 `aclblasSrotm` 在 Ascend 平台上的基本使用流程。

Srotm(Modified Givens Transformation) 算子对两个单精度向量 `x` 和 `y` 原地施加修正 Givens 变换，是 BLAS 基础线性代数库中的经典向量变换算子之一。

该算子根据 `sparam[0]` 指定的 `flag` 选择不同的变换矩阵形式，支持正步长、负步长以及非连续步长的向量访问方式。

## 算子描述

- 算子功能：
  Srotm 算子对向量 `x` 和 `y` 原地执行修正 Givens 变换。对应的数学表达式为：
```
[x_i']   [h11 h12] [x_i]
[y_i'] = [h21 h22] [y_i]
```

  其中，矩阵参数由 `sparam` 决定：

  - 当 `sparam[0] == -2` 时，不对输入向量做任何修改
  - 当 `sparam[0] < 0` 且不等于 `-2` 时，`h11 = sparam[1]`，`h21 = sparam[2]`，`h12 = sparam[3]`，`h22 = sparam[4]`
  - 当 `sparam[0] == 0` 时，`h11 = 1`，`h22 = 1`，`h21 = sparam[2]`，`h12 = sparam[3]`
  - 当 `sparam[0] > 0` 时，`h11 = sparam[1]`，`h12 = 1`，`h21 = -1`，`h22 = sparam[4]`

  本样例覆盖了以下场景：

  - `flag < 0`、`flag == 0`、`flag > 0` 和 `flag == -2` 四类参数模式
  - `incx`、`incy` 为正、负以及非 1 步长的向量访问
  - device pointer 输入输出路径
  - device pointer 非连续 stride 由主 kernel 直接按 `incx/incy` 访问
  - 非 unit stride 场景当前以单工作块执行，优先保证带步长原地读写的正确性
  - 从极小非对齐长度、512B 对齐边界、多核分块边界到大规模向量的多组测试样例

- 调用接口：
```
aclblasStatus_t aclblasSrotm(aclblasHandle handle, float *x, float *y, const float *sparam,
                             const int64_t n, const int64_t incx, const int64_t incy);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">srotm 参数说明</td>
   </tr>
   <tr>
      <td rowspan="8" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">aclblas 算子句柄。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">输入输出向量 x，包含 n 个逻辑元素，按 incx 指定的步长访问。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">输入输出向量 y，包含 n 个逻辑元素，按 incy 指定的步长访问。</td>
   </tr>
   <tr>
      <td align="center">sparam</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">长度为 5 的参数数组，用于描述 Modified Givens 变换矩阵。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">参与变换的向量元素个数。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量 x 中连续逻辑元素之间的步长，支持负值，但不能为 0。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量 y 中连续逻辑元素之间的步长，支持负值，但不能为 0。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Srotm</td></tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">N</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">sparam</td><td align="center">5</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td><td align="center">x</td><td align="center">N</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">N</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">srotm_kernel</td></tr>
  </table>

- 算子实现：

    Host 侧首先根据 `sparam` 解析出本次变换需要使用的系数，并按向量长度选择核执行所需的分块信息。当前实现使用 by-value tiling：Host 侧将 `x/y` device 地址、实际存储长度、`incx/incy` 以及本次变换用到的 `sflag/h11/h12/h21/h22` 一并写入 tiling 结构后，直接调用一次 `srotm_kernel_do` 下发到 stream；测试或调用侧在需要消费结果时再同步并执行 D2H 读回。当 `incx == 1 && incy == 1` 时，device 侧按 512B 对齐 chunk 做多工作块分配；当存在非 unit stride 时，当前实现会收敛为单工作块执行。主 kernel 根据逻辑索引直接计算带步长的 GM 地址，完成原地读写。

## 测试内容

本测试用于验证 `aclblasSrotm`（Single-precision Modified Givens Transformation）算子的正确性，当前样例执行 36 个 device 路径功能和精度用例。

### 阶段 1: L0 门槛级功能用例（6 个用例）

| 编号 | 描述 | n | incx | incy | sparam[0] | sparam |
|------|------|---|------|------|-----------|--------|
| L001 | 正常-通用矩阵模式 | 128 | 1 | 1 | -1.0 | {-1.0, 0.75, -0.25, 0.5, 1.5} |
| L002 | 正常-flag=0 模式 | 1024 | 1 | 1 | 0.0 | {0.0, 0.0, 0.6, -1.1, 0.0} |
| L003 | 正常-flag>0 模式 | 2048 | 1 | 1 | 1.0 | {1.0, 1.25, 0.0, 0.0, 0.8} |
| L004 | 边界-no-op 模式 | 4 | 1 | 1 | -2.0 | {-2.0, 0.0, 0.0, 0.0, 0.0} |
| L005 | 边界-单元素 | 1 | 1 | 1 | -1.0 | {-1.0, 0.75, -0.25, 0.5, 1.5} |
| L006 | 正常-非连续 stride | 3072 | 2 | 3 | -1.0 | {-1.0, 0.3, -0.8, 1.4, 0.6} |

### 阶段 2: L1 参数组合用例（28 个用例）

| 编号 | 参数特点 |
|------|---------|
| L101 | `flag < 0`、`flag == 0`、`flag > 0` 和 `flag == -2` 四类 `sparam[0]` 模式 |
| L102 | `incx=2/3/4`，稀疏 x 读取和写回 |
| L103 | `incy=2/3/5`，稀疏 y 读取和写回 |
| L104 | `incx=-1/-2/-3`，x 负 stride |
| L105 | `incy=-1/-2/-3/-5`，y 负 stride |
| L106 | `incx`、`incy` 同时为负，覆盖反向访问组合 |
| L107 | `n=31/32/33/127/129/255/256/257`，覆盖小规模、非对齐和边界长度 |
| L108 | `n=4096/8192/16384/32768/65536/131072`，覆盖中大规模 unit stride |
| L109 | `n=1048575`，覆盖大规模压力场景 |
| L110 | device pointer 非连续 stride，由 kernel 直接按 `incx/incy` 访问 |

### 阶段 3: L2 边界入参用例（2 个用例）

| 编号 | 边界条件 | 期望返回值 |
|------|---------|-----------|
| L201 | n == 0 | ACLBLAS_STATUS_SUCCESS |
| L202 | n < 0 | ACLBLAS_STATUS_SUCCESS |

说明：`n <= 0` 遵循 BLAS 约定，接口无需执行计算并返回 `ACLBLAS_STATUS_SUCCESS`。


## 编译运行

在仓库根目录下执行如下步骤，编译并执行算子。

- 配置环境变量
  请根据当前环境上 CANN 开发套件包的安装方式，选择对应配置环境变量的命令。

  - 默认路径，root 用户安装 CANN 软件包

   ```bash
   source /usr/local/Ascend/cann/set_env.sh
   ```

  - 默认路径，非 root 用户安装 CANN 软件包

   ```bash
   source $HOME/Ascend/cann/set_env.sh
   ```

  - 指定路径 install_path，安装 CANN 软件包

   ```bash
   source ${install_path}/cann/set_env.sh
   ```

- 样例执行

```bash
bash build.sh --ops=srotm --soc=ascend910b --run
```
