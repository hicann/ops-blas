## aclblasLtMatmul 接口实现

## 概述

BLAS Lt 矩阵乘法（`aclblasLtMatmul`）接口实现与精度测试。

`aclblasLtMatmul` 实现了通用矩阵乘法运算，对应的数学表达式为：

```
D = alpha * op(A) * op(B) + beta * C
```

其中 A、B 为输入矩阵，C 为累加矩阵，D 为输出矩阵，alpha 和 beta 为标量，op(A)/op(B) 支持不转置（N）和转置（T）。当前实现支持 FP32、MXFP8（E4M3FN）、MXFP4（E2M1）三种输入类型组合，输出支持 FP32 和 BF16。

## 产品支持情况

| 产品                                                         |  是否支持 |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✗    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ✗    |

> MXFP8/MXFP4 量化路径依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`）。

## 目录结构介绍

接口实现位于 `blasLt/`：

```
blasLt/
├── aclblasLt.cpp                          // aclBLASLt 库入口，含 aclblasLtMatmul 路由
├── matmul_fp32/arch35/
│   ├── matmul_fp32_host.cpp               // FP32 Host 侧 Tiling
│   └── matmul_fp32_kernel.cpp             // FP32 Kernel 侧实现
├── matmul_mxfp8/arch35/
│   ├── matmul_mxfp8_host.cpp              // MXFP8 Host 侧 Tiling
│   └── matmul_mxfp8_kernel.cpp            // MXFP8 Kernel 侧实现
├── matmul_mxfp4/arch35/
│   ├── matmul_mxfp4_host.cpp              // MXFP4 Host 侧 Tiling
│   └── matmul_mxfp4_kernel.cpp            // MXFP4 Kernel 侧实现
└── utils/
    └── kernel_utils.h                       // shared kernel helpers
```

测试代码位于 `test/blasLtMatmul/`：

```
test/blasLtMatmul/
├── README.md                              // 说明文档（本文档）
├── CMakeLists.txt                         // 编译工程文件
├── blasLtMatmul_param.h                   // 参数结构体（继承 BlasTestParamBase）
├── blasLtMatmul_golden.h                  // CPU golden（封装 aclblasLtMatmul CPU 参考）
└── arch35/
    ├── blasLtMatmul_npu_wrapper.h         // NPU wrapper（封装 aclrtMalloc/H2D/kernel/D2H/free）
    ├── blasLtMatmul_test.cpp              // 精度测试（GTest 入口）
    └── blasLtMatmul_test.csv              // 精度测试用例表
```

## 接口描述

- 接口功能：  
  执行矩阵乘法 D = alpha * op(A) * op(B) + beta * C。支持 FP32 全精度路径，以及 MXFP8/MXFP4 量化输入路径（需配合 scale factor）。

- 对应接口为：

```
aclblasStatus_t aclblasLtMatmul(
    aclblasLtHandle_t lightHandle,
    aclblasLtMatmulDesc_t computeDesc,
    const void* alpha,
    const void* A,
    aclblasLtMatrixLayout_t Adesc,
    const void* B,
    aclblasLtMatrixLayout_t Bdesc,
    const void* beta,
    const void* C,
    aclblasLtMatrixLayout_t Cdesc,
    void* D,
    aclblasLtMatrixLayout_t Ddesc,
    const aclblasLtMatmulAlgo_t* algo,
    void* workspace,
    size_t workspaceSizeInBytes,
    aclrtStream stream);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">aclblasLtMatmul 参数说明</td>
   </tr>
   <tr>
      <td rowspan="18" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">lightHandle</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">aclBLASLt 库上下文句柄，由 aclblasLtCreate 创建。不可为 NULL，否则返回 ACLBLAS_STATUS_NOT_INITIALIZED。</td>
   </tr>
   <tr>
      <td align="center">computeDesc</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵乘法描述符，设置 transA/transB、epilogue、scale 指针等属性。不可为 NULL。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">用于乘法的 float 标量。不可为 NULL。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">输入矩阵 A，数据类型由 Adesc 指定。不可为 NULL（m&gt;0 且 n&gt;0 时）。</td>
   </tr>
   <tr>
      <td align="center">Adesc</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵 A 的 layout 描述符（rows/cols/ld/order/dtype）。</td>
   </tr>
   <tr>
      <td align="center">B</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">输入矩阵 B，数据类型由 Bdesc 指定。不可为 NULL（m&gt;0 且 n&gt;0 时）。</td>
   </tr>
   <tr>
      <td align="center">Bdesc</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵 B 的 layout 描述符。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">用于累加的 float 标量。不可为 NULL。beta=0 时 C 可不参与计算。</td>
   </tr>
   <tr>
      <td align="center">C</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">累加矩阵 C。beta=0 时可为 NULL。当前测试覆盖 C=NULL 场景。</td>
   </tr>
   <tr>
      <td align="center">Cdesc</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵 C 的 layout 描述符。</td>
   </tr>
   <tr>
      <td align="center">D</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">输出矩阵 D，维度 m x n。不可为 NULL（m&gt;0 且 n&gt;0 时）。</td>
   </tr>
   <tr>
      <td align="center">Ddesc</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵 D 的 layout 描述符，指定输出数据类型（FP32 或 BF16）。</td>
   </tr>
   <tr>
      <td align="center">algo</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">算法描述符，可为 NULL（使用默认算法）。</td>
   </tr>
   <tr>
      <td align="center">workspace</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">工作空间内存，可为 NULL。非 NULL 时需 16B 对齐。</td>
   </tr>
   <tr>
      <td align="center">workspaceSizeInBytes</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">工作空间大小（字节）。</td>
   </tr>
   <tr>
      <td align="center">stream</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">AscendCL 执行流。</td>
   </tr>
</table>

### 当前支持的入参/出参范围

<table>
   <tr>
      <td align="center">参数项</td>
      <td align="center">支持范围</td>
      <td align="center">说明</td>
   </tr>
   <tr>
      <td align="center">dtypeA / dtypeB</td>
      <td align="center">FP32；MXFP8_E4M3FN；MXFP4_E2M1</td>
      <td align="center">A/B 须为同类型组合：FP32×FP32、MXFP8×MXFP8、MXFP4×MXFP4。其他组合返回 ACLBLAS_STATUS_NOT_SUPPORTED。</td>
   </tr>
   <tr>
      <td align="center">dtypeC</td>
      <td align="center">FP32</td>
      <td align="center">累加矩阵类型，当前固定为 FP32。</td>
   </tr>
   <tr>
      <td align="center">dtypeD</td>
      <td align="center">FP32；BF16</td>
      <td align="center">MXFP8/MXFP4 路径支持 FP32 或 BF16 输出；FP32 路径输出 FP32。</td>
   </tr>
   <tr>
      <td align="center">computeType</td>
      <td align="center">ACLBLAS_COMPUTE_32F</td>
      <td align="center">所有已支持路径均使用 32F 计算精度。</td>
   </tr>
   <tr>
      <td align="center">transA / transB</td>
      <td align="center">N、T</td>
      <td align="center">对应 ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）。</td>
   </tr>
   <tr>
      <td align="center">M / N / K</td>
      <td align="center">M,N,K ≥ 0</td>
      <td align="center">M=0 或 N=0 时为空操作，直接返回 SUCCESS。MXFP8/MXFP4 路径要求 K 为 32 的整数倍，否则返回 ACLBLAS_STATUS_INVALID_VALUE。</td>
   </tr>
   <tr>
      <td align="center">lda / ldb / ldc / ldd</td>
      <td align="center">ld ≥ 物理列数</td>
      <td align="center">行主序（ACLBLASLT_ORDER_ROW）存储，ld 为 leading dimension，须 ≥ 矩阵物理列数。MXFP4 的 ld 为逻辑元素 leading dim（2 个 FP4 元素打包为 1 字节）。</td>
   </tr>
   <tr>
      <td align="center">alpha / beta</td>
      <td align="center">float</td>
      <td align="center">当前测试覆盖 alpha=1.0、beta=0.0。beta=0 时 C 可为 NULL。</td>
   </tr>
   <tr>
      <td align="center">epilogue</td>
      <td align="center">ACLBLASLT_EPILOGUE_DEFAULT</td>
      <td align="center">当前仅支持默认 epilogue。</td>
   </tr>
   <tr>
      <td align="center">scaleA / scaleB</td>
      <td align="center">MXFP8/MXFP4 必填</td>
      <td align="center">通过 computeDesc 的 ACLBLASLT_MATMUL_DESC_A/B_SCALE_POINTER 设置，E8M0 格式，按 K 方向每 32 元素一组。不可为 NULL。</td>
   </tr>
   <tr>
      <td align="center">algo</td>
      <td align="center">default / NULL</td>
      <td align="center">可为 NULL，使用默认算法。</td>
   </tr>
   <tr>
      <td align="center">order</td>
      <td align="center">ACLBLASLT_ORDER_ROW</td>
      <td align="center">当前实现使用行主序。</td>
   </tr>
</table>

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">aclblasLtMatmul</td></tr>
  <tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">M×K（或转置后 K×M）</td><td align="center">FP32 / MXFP8 / MXFP4</td><td align="center">ND</td></tr>
  <tr><td align="center">B</td><td align="center">K×N（或转置后 N×K）</td><td align="center">FP32 / MXFP8 / MXFP4</td><td align="center">ND</td></tr>
  <tr><td align="center">C</td><td align="center">M×N</td><td align="center">FP32</td><td align="center">ND</td></tr>
  <tr><td align="center">scaleA</td><td align="center">按 K 分组</td><td align="center">E8M0 (uint8)</td><td align="center">ND</td></tr>
  <tr><td align="center">scaleB</td><td align="center">按 K 分组</td><td align="center">E8M0 (uint8)</td><td align="center">ND</td></tr>
  <tr><td align="center">alpha/beta</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">D</td><td align="center">M×N</td><td align="center">FP32 / BF16</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">MatmulFp32Kernel / matmul_mxfp8_kernel_do / ltmatmul_mxfp4_kernel_do</td></tr>
  </table>

- 算子实现：  
  Host 侧根据 A/B 数据类型路由至 FP32、MXFP8 或 MXFP4 对应的 Tiling 与 Kernel 实现。MXFP 路径在 Kernel 内完成量化矩阵乘加，输出经 epilogue 处理写入 D。

- 调用实现：  
  通过 aclBLASLt 标准 API 调用，内部使用内核调用符 `<<<>>>` 启动 NPU 核函数。

## 测试用例覆盖

| 分组 | 用例数 | 覆盖场景 |
|------|--------|----------|
| L0 FP32 基础 | 7 | 小/中/大规模 NN、TN/NT/TT 转置、algo=nullptr |
| L0 MXFP8 | 6 | K=32 基础/大规模、四种转置、C=null |
| L0 MXFP4 | 7 | K=32 基础/大规模、四种转置、C=null、FP32 输出 |
| L0 异常入参 | 4 | handle/desc/alpha/A 为 NULL |
| L0 边界 | 6 | M=0/N=0 空操作、K 非 32 倍数非法、algo=nullptr |
| L1 FP32 扩展 | 11 | 矩形矩阵、非方阵转置、瘦矩阵 M128×N32×K128 |
| L1 MXFP8 扩展 | 22 | 多种 K 规模、矩形/奇数维度、scale 全零、K 非法值 |
| L1 MXFP4 扩展 | 17 | 多种 K 规模、矩形/转置、scale 全零、K 非法值 |
| TEST_F 固定用例 | 4 | NullHandle、NullComputeDesc、NullAlpha、NullA |

## 编译运行

在本样例根目录下执行如下步骤，编译并执行测试。

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
  bash build.sh --ops=blasLtMatmul --soc=ascend950 --run
  ```

  其中 `--soc` 为**可选**参数，用于指定目标硬件平台（与上文「产品支持情况」对应）。按实际硬件选用：

  | 产品 | `--soc` 取值 |
  |------|----------------|
  | Ascend 950PR / Ascend 950DT | `ascend950` |

  执行结果如下，说明精度对比成功。
  ```bash
  [PASS] blasLtMatmul_test
  ```
