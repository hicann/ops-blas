## blasLtMatmul实现

## 概述

本样例展示 `blasLtMatmul` 在 Ascend 平台上的基本使用流程，包含下列步骤及对应接口的调用：

- 资源初始化（`aclInit` / `aclrtSetDevice` / `aclrtCreateStream`）
- 创建句柄（`aclblasLtCreate`）
- 创建矩阵布局（`aclblasLtMatrixLayoutCreate` + `ACLBLASLT_MATRIX_LAYOUT_ORDER`）
- 创建算子描述（`aclblasLtMatmulDescCreate`，并配置 epilogue / bias）
- 启发式算法选择（`aclblasLtMatmulAlgoGetHeuristic`）
- 执行矩阵乘操作（`aclblasLtMatmul`）


## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── blasLtMatmul
│   ├── CMakeLists.txt            // 编译工程文件
│   ├── README.md                 // 说明文档
│   └── blasLtMatmul_test.cpp     // 调用样例
```

## 算子描述

- 算子功能：  
  `blasLtMatmul` 用于完成矩阵乘法，并可通过 epilogue 融合后处理（如 bias、ReLU 等）。其基础数学表达式为：

```
D = alpha * op(A) * op(B) + beta * C
```

  其中：
  - `A`：当 `op(A)=A` 时形状为 `M x K`；当 `op(A)=A^T` 时形状为 `K x M`
  - `B`：当 `op(B)=B` 时形状为 `K x N`；当 `op(B)=B^T` 时形状为 `N x K`
  - `C/D`：形状为 `M x N`
  - `alpha`、`beta`：标量

  本样例在算子描述中配置了：
  - `ACLBLASLT_EPILOGUE_RELU_BIAS`：融合 `Bias + ReLU` 后处理
  - `ACLBLASLT_MATMUL_DESC_BIAS_POINTER`：传入 bias 指针（bias 被视为 `1 x N`）

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">M * K</td><td align="center">float16 / float32</td><td align="center">ND</td></tr>
  <tr><td align="center">B</td><td align="center">K * N</td><td align="center">float16 / float32</td><td align="center">ND</td></tr>
  <tr><td align="center">C</td><td align="center">M * N</td><td align="center">float16 / float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">D</td><td align="center">M * N</td><td align="center">float16 / float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">计算类型(ComputeType)</td><td colspan="4" align="center">ACLBLASLT_COMPUTE_32F</td></tr>
  <tr><td rowspan="1" align="center">Epilogue(可选)</td><td colspan="4" align="center">DEFAULT / RELU / BIAS / RELU_BIAS</td></tr>
  <tr><td rowspan="1" align="center">bias(可选)</td><td colspan="4" align="center">1 * N，data type 与输出一致或为 float32（以实际支持为准）</td></tr>
  </table>

  支持的数据类型（可参考 cublasLtMatmul 的常见支持组合表达方式）：
  - A/B 输入：`float16` / `float32`
  - 计算/累加：`float32`（本样例使用 `ACLBLASLT_COMPUTE_32F`）
  - C/D 输出：`float16` / `float32`

  说明：
  - 上述“算子规格”以接口层面描述为主；不同产品/版本可能存在差异，最终以实际运行时支持情况为准。
  - 本样例代码当前使用：`A/B = float16`、`C/D = float32`、`ComputeType = ACLBLASLT_COMPUTE_32F`，并启用 `Bias + ReLU` 融合。

- 算子实现：

  该算子通过启发式接口 `aclblasLtMatmulAlgoGetHeuristic` 获取候选算法，并选择其中一个算法执行 `aclblasLtMatmul`。
  可通过 `aclblasLtMatmulPreferenceSetAttribute(ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ...)` 设置 workspace 上限，以影响算法选择。

- 调用实现：

  本样例为 Host API 调用示例，不涉及 `<<<>>>` 内核调用符；直接通过 `aclblasLt*` 系列接口完成算子配置与执行。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。

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
bash build.sh --op=blasLtMatmul --run # --op=<算子名> --run 可选参数，执行测试样例
```

执行结果如下，说明精度对比成功。

```bash
[Success] Case accuracy is verification passed.
```
