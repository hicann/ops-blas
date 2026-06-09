## Sscal 算子实现 (arch35 / ascend950)

## 概述

BLAS Sscal(Scale)算子实现，对应接口：

```cpp
aclblasStatus_t aclblasSscal(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx);
```

数学表达式：`x[i] = alpha * x[i]`（i = 0 .. n-1，步长为 incx）。

## 支持的产品

- Atlas A5 训练系列产品 / Atlas A5 推理系列产品（ascend950）

## 目录结构

```
├── sscal
│   ├── arch35                 // arch35(ascend950) 实现
│   │   ├── sscal_host.cpp     // host 侧实现
│   │   ├── sscal_kernel.cpp   // kernel 侧实现
│   │   └── sscal_tiling_data.h // tiling 数据结构
│   └── README.md              // 说明文档
```

（编译由上层 `blas/CMakeLists.txt` 自动收集 arch35 目录源文件）

## 算子描述

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">sscal 参数说明</td>
   </tr>
   <tr>
      <td rowspan="5" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">ACL 流 handle，用于传入 stream。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量 x 中的元素个数。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">标量乘数（float 指针）。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">float 向量，包含 n 个元素。</td>
   </tr>
</table>

<table>
   <tr><td rowspan="1" align="center">算子类型 (OpType)</td><td colspan="4" align="center">Sscal</td></tr>
   <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
   <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
   <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
   <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sscal_kernel</td></tr>
</table>

- 算子实现

  使用 TPipe + TQue(VECIN/VECOUT) 双队列流水：
  1. MTE3: DataCopy / DataCopyPad 将 x 从 GM 搬入 UB（32B 对齐，tail 用 Pad 补齐）
  2. V: Muls 指令完成向量乘标量 `outLocal[i] = inLocal[i] * alpha`
  3. MTE3: DataCopy / DataCopyPad 将结果写回 GM

  多核并行：按 AIV core 数量均分 n 个元素，每个 core 处理 `perCoreN` 个（ELEMENTS_PER_BLOCK=8 对齐），
  末尾 core 吸收余数。Tile 循环避免 UB 溢出。

## 编译运行

- 配置环境变量
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 编译并执行测试
  ```bash
  bash build.sh --ops=sscal --run --soc=ascend950
  ```

  执行结果示例：
  ```bash
  [----------] Global test environment tear-down
  [==========] 38 tests from 2 test suites ran. (6826 ms total)
  [  PASSED  ] 38 tests.
  ```
