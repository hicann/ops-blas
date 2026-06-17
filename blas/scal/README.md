## Scal算子实现

## 概述

BLAS Scal算子实现，包含实数向量缩放（Sscal）和复数向量缩放（Cscal）。

Scal（Scale）算子实现了向量缩放运算，是BLAS基础线性代数库中的核心算子之一。

该算子包含以下接口：
- **aclblasSscal**：实数向量乘以标量，`x[i] = alpha * x[i]`
- **aclblasCscal**：复数向量乘以复数标量，`(a+bi)*(c+di) = (ac-bd) + (ad+bc)i`

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| Ascend 950PR / Ascend 950DT                                  |    ✓    | arch35 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    ✓    | arch22 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    ✓    | arch22 |

## 目录结构介绍

```
blas/scal/
├── README.md                       // 说明文档
├── arch22/
│   ├── cscal_host.cpp              // Cscal Host 侧实现（arch22）
│   ├── cscal_kernel.cpp            // Cscal Kernel 侧实现（arch22）
│   ├── sscal_host.cpp              // Sscal Host 侧实现（arch22）
│   └── sscal_kernel.cpp            // Sscal Kernel 侧实现（arch22）
└── arch35/
    ├── sscal_host.cpp              // Sscal Host 侧实现（arch35）
    ├── sscal_kernel.cpp            // Sscal Kernel 侧实现（arch35）
    └── sscal_tiling_data.h         // Sscal Tiling 数据结构（arch35）
```

## 算子描述

### Sscal（实数向量缩放）

- 算子功能：

sscal算子实现了实数向量x乘以标量alpha。对应的数学表达式为：
```
x[i] = alpha * x[i]  (i = 0 .. n-1，步长为 incx)
```

- 对应的接口为：
```cpp
aclblasStatus_t aclblasSscal(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx);
```

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

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型 (OpType)</td><td colspan="4" align="center">Sscal</td></tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sscal_kernel</td></tr>
  </table>

- 算子实现：

  使用 TPipe + TQue(VECIN/VECOUT) 双队列流水：
  1. MTE3: DataCopy / DataCopyPad 将 x 从 GM 搬入 UB（32B 对齐，tail 用 Pad 补齐）
  2. V: Muls 指令完成向量乘标量 `outLocal[i] = inLocal[i] * alpha`
  3. MTE3: DataCopy / DataCopyPad 将结果写回 GM

  多核并行：按 AIV core 数量均分 n 个元素，每个 core 处理 `perCoreN` 个（ELEMENTS_PER_BLOCK=8 对齐），
  末尾 core 吸收余数。Tile 循环避免 UB 溢出。

### Cscal（复数向量缩放）

- 算子功能：

cscal算子实现了复数向量x乘以复数标量alpha。对应的数学表达式为：
```
x = alpha * x
```
复数乘法公式：`(real, imag) * (alpha_r, alpha_i) = (real*alpha_r - imag*alpha_i, real*alpha_i + imag*alpha_r)`

- 对应的接口：
```cpp
int aclblasCscal(aclblasHandle handle, std::complex<float> *x, const std::complex<float> alpha,
                 const int64_t n, const int64_t incx);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">cscal 参数说明</td>
   </tr>
   <tr>
      <td rowspan="6" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">ACL流handle，用于传入stream。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">复数向量，包含n个complex<float>元素。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">用于乘法的复数标量。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量x中的复数元素个数。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x中连续元素之间的步长。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Cscal</td></tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">N</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cscal</td></tr>
  </table>

- 算子实现：

    将复数向量从GM搬运到UB，使用vreducev2进行虚实分离，分别计算实部*实部、实部*虚部、虚部*实部、虚部*虚部，再使用add_v合并结果，最后通过vgather进行虚实合并并搬运回GM。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 编译运行

在仓库根目录下执行如下步骤，编译并运行算子测试。

- 配置环境变量
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- 编译并执行测试
  ```bash
  bash build.sh --ops=sscal --run --soc=ascend950
  bash build.sh --ops=cscal --run
  ```

  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
