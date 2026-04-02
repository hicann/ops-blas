## Sger算子实现

## 概述

本样例展示 `aclblasSger` 在 Ascend 平台上的基本使用流程。

Sger (Rank-1 Update) 实现矩阵的秩-1更新操作，数学表达式为：

```
A = A + alpha * x * y^T
```

其中：
- `x`：长度为 `m` 的列向量
- `y`：长度为 `n` 的行向量
- `alpha`：标量
- `A`：`m x n` 矩阵

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── sger
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── sger_test.cpp       // 算子调用样例
```

## 算子描述

- 算子功能：
  Sger算子实现了秩-1更新操作，将 `alpha * x * y^T` 加到矩阵 `A` 上。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Sger</td></tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">m</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">A</td><td align="center">m x n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">A</td><td align="center">m x n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sger</td></tr>
  </table>

- 调用实现：
  本样例为 Host API 调用示例，使用 `aclblasSger` 接口完成算子配置与执行。

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
bash build.sh --ops=sger --soc=ascend950 --run
```

执行结果如下，说明精度对比成功。

```bash
[Success] Case accuracy verification passed.
```
