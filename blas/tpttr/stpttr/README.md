## stpttr算子实现

## 概述

BLAS stpttr算子实现。

stpttr(Symmetric Triangular matrix, Packed format To Triangular matrix, Regular storage)算子将 LAPACK 压缩格式（packed format）中的对称三角矩阵展开为按列主序存储的常规二维矩阵。仅写入 `uplo` 指定的三角区域，矩阵另一三角及未参与运算的元素保持原值不变。

## 产品支持情况

| 产品                                                         |  是否支持 |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ×    |

## 目录结构介绍

```
blas/tpttr/stpttr/
├── README.md                   // 说明文档
└── arch35/
    ├── stpttr_host.cpp         // Host 侧实现
    ├── stpttr_kernel.cpp       // Kernel 侧实现
    └── stpttr_tiling_data.h    // Tiling 数据结构
```

测试代码位于 `test/tpttr/stpttr/`：

```
test/tpttr/stpttr/
├── CMakeLists.txt              // 编译工程文件
├── stpttr_param.h              // 参数结构体（继承 BlasTestParamBase）
├── stpttr_golden.h             // CPU golden（签名与 BLAS API 一致）
└── arch35/
    ├── stpttr_npu_wrapper.h    // NPU wrapper（封装 aclrtMalloc/H2D/kernel/D2H/free）
    ├── stpttr_test.cpp         // 精度测试（GTest 入口）
    └── stpttr_test.csv         // 精度测试用例表
```

## 算子描述

- 算子功能：  
  将压缩格式三角矩阵 `AP` 中的元素按 `uplo` 展开到常规矩阵 `A` 的对应三角区域：

  - `uplo == ACLBLAS_LOWER`：复制到 `A` 的下三角（含对角），上三角不变  
  - `uplo == ACLBLAS_UPPER`：复制到 `A` 的上三角（含对角），下三角不变  

  `AP` 为列优先压缩存储，长度为 `n * (n + 1) / 2`；`A` 为 `lda × n` 的列主序矩阵，`lda >= max(1, n)`。`n == 0` 时直接返回成功，不访问缓冲区。

  对应的接口为：

```
aclblasStatus_t aclblasStpttr(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const float *AP,
    float *A,
    int lda);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">stpttr 参数说明</td>
   </tr>
   <tr>
      <td rowspan="7" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">handle</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">aclbLAS 库上下文句柄。</td>
   </tr>
   <tr>
      <td align="center">uplo</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">三角存储方式：ACLBLAS_UPPER(121)、ACLBLAS_LOWER(122)。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">方阵维数，须 &gt;= 0；为 0 时立即返回成功。</td>
   </tr>
   <tr>
      <td align="center">AP</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">压缩格式输入，&lt;type&gt; 数组，长度 n*(n+1)/2。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">常规输出矩阵，&lt;type&gt; 数组，维度 lda × n；非目标三角保持原值。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">A 的主维长度，须满足 lda &gt;= max(1, n)。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">stpttr</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">AP</td><td align="center">n*(n+1)/2</td><td align="center">float</td><td align="center">packed</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">A</td><td align="center">lda * n</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">stpttr_kernel</td></tr>
  </table>

- 算子实现：  

    Host 侧完成参数校验与 Tiling 计算（按 Vector Core 数切分列块），将 Tiling 数据拷贝至 Device 后，通过 `stpttr_kernel_do` 启动 Kernel。Kernel 按列从 GM 上的压缩缓冲区 `AP` 分块搬入 UB，再写回 GM 上常规矩阵 `A` 的对应三角列段；`lda > n` 时列间存在 stride 间隔。

- 调用实现  
    使用内核调用符 `<<<>>>`（`stpttr_kernel_do`）在 `aclblas` 关联的 stream 上异步执行，Host 在返回前同步 stream。

## 测试用例覆盖

| 分组 | 用例数 | 覆盖场景 |
|------|--------|----------|
| L0 参数校验 | 4 | 未初始化 handle、n&lt;0、lda 过小、非法 uplo |
| L0 功能 | 13 | n=0/1/2/4/32/128/512，LOWER/UPPER |
| L1 规模与 lda | 18 | n=8~1024、lda&gt;n（8×12、16×32 等） |
| L1 特殊数值 | 12 | 全零、大数、负数、inf、nan、极值组合 |
| L1 参数校验 | 8 | AP/A 空指针、非法 uplo、n=0 与 lda 组合 |
| L1 往返与大规模 | 4 | strttp→stpttr 往返（32×32）、n=10240 |

ST 采用 GTest 参数化 + `stpttr_test.csv`，`BlasTest<StpttrParam>` fixture，精度模式为 **EXACT**（仅比对有效三角区，其余位置为 sentinel -999）。

**注意**：`makeBlasArray` 的 size 参数为 `int64_t`，调用时需显式转换：`makeBlasArray(static_cast<int64_t>(p.lda) * p.n, p.a)`，确保负值 n 正确返回空数组。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的安装方式，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- 样例执行
  ```bash
  bash build.sh --ops=stpttr --soc=ascend950 --run
  ```

  其中`--soc` 为**可选**参数，用于指定目标硬件平台（与上文「产品支持情况」对应）。按实际硬件选用：

  | 产品 | `--soc` 取值 |
  |------|----------------|
  | Ascend 950PR / Ascend 950DT | `ascend950` |
  | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | `ascend910_93` |
  | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | `ascend910b` |

  执行结果如下，说明精度对比成功。
  ```bash
  [PASS] stpttr_test
  ```
