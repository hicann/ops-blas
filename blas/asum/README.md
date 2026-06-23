## sasum算子实现

## 概述

BLAS sasum算子实现。

sasum(Sum of Absolute Values)算子实现了计算向量元素绝对值的和，是BLAS基础线性代数库中的核心算子之一。

该算子计算L1范数（曼哈顿范数），常用于向量稀疏度度量和误差估计。针对arch22（Atlas A2/A3）和arch35（Ascend 950）分别进行了实现和优化。

## 产品支持情况

| 产品                                                         |  是否支持 | 架构 |
| :----------------------------------------------------------- |:-------:|:----:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ✓    | arch35 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ✓    | arch22 |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    ✓    | arch22 |

## 目录结构介绍

```
blas/asum/
├── README.md                       // 说明文档
├── arch35/
│   ├── sasum_tiling_data.h         // Tiling 数据结构
│   ├── sasum_host.cpp              // Host 侧实现
│   └── sasum_kernel.cpp            // Kernel 侧实现
└── arch22/
    ├── sasum_host.cpp              // Host 侧实现
    └── sasum_kernel.cpp            // Kernel 侧实现
```

测试代码位于 `test/asum/`：

```
test/asum/
└── sasum/
    ├── CMakeLists.txt              // 编译工程文件
    ├── sasum_param.h               // CSV 参数解析
    ├── sasum_golden.h              // CPU golden（调用 cblas_sasum）
    └── arch35/
        ├── sasum_test.cpp          // GTest 精度测试（arch35）
        ├── sasum_test.csv          // CSV 测试用例
        └── sasum_npu_wrapper.h     // NPU 调用封装
```

## 算子描述

- 算子功能：  
sasum算子实现了计算向量元素绝对值的和。对应的数学表达式为：  
```
result = sum(|x[i]|) for i = 0 to n-1
```

对应的接口为：
```
aclblasStatus_t aclblasSasum(
    aclblasHandle_t handle,
    int n,
    const float *x,
    int incx,
    float *result);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">sasum 参数说明</td>
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
      <td align="center">aclBLAS 库上下文句柄。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量元素个数。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">float 向量，包含 n 个元素。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x 中连续元素之间的步长，不可为0。</td>
   </tr>
   <tr>
      <td align="center">result</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">向量元素绝对值之和。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">sasum</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">n</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">核函数名</td><td colspan="6" align="center">arch22: sasum_kernel</td></tr>
  <tr><td colspan="6" align="center">arch35: sasum_aiv_kernel / sasum_simt_kernel / sasum_reduce_kernel</td></tr>
  </table>

- 算子实现：

  - **arch22**（Atlas A2/A3）：

    采用单AIV kernel实现。Host侧将向量按32B对齐粒度分块，构造Tiling数据并通过GM传递给Device。Kernel侧每个AI Core负责一块连续数据：将向量分片从GM搬运到UB，通过`Abs`计算绝对值，再通过`ReduceSum`计算局部和，最后通过`DataCopyPad`+原子加将局部结果累加到全局输出。通过元素级均分策略实现多核负载均衡。

  - **arch35**（Ascend 950）：

    采用多kernel实现，根据步长分两条路径：
    - **incx=1 路径（AIV）**：单kernel `sasum_aiv_kernel` 实现。Host侧通过`GetAivCoreCount()`获取AIV核数并均分元素到各核，Tiling数据通过值传递（无需GM分配）。每个AI Core按`maxDataCount`分片迭代：将子块从GM搬运到UB，`Abs`取绝对值后`ReduceSum`求局部和，通过`DataCopyPad`+原子加写回全局result。
    - **incx≠1 路径（SIMT + Reduce）**：两阶段实现。第一阶段`sasum_simt_kernel`通过SIMT多线程并行，每个线程按步长跨步访问向量元素，在UB上完成局部绝对值累加，再通过线程内树形归约得到每核部分和写入workspace（使用handle workspace）。第二阶段`sasum_reduce_kernel`使用1个AI Core将workspace中所有部分和累加得到最终结果。

- 调用实现  
    使用内核调用符<<<>>>调用核函数。arch35下incx=1时直接调用`sasum_aiv_kernel`，incx≠1时依次调用`sasum_simt_kernel`和`sasum_reduce_kernel`。Host侧为异步执行，不包含流同步操作，由调用方负责同步。

## 测试用例覆盖

测试基于 GTest + CSV 驱动框架，golden 实现调用 Netlib BLAS `cblas_sasum`。

| 分组 | 用例数 | 覆盖场景 |
|------|--------|----------|
| 异常 | 4 | n=-1无效入参、n=0提前退出、incx=0无效入参、x空指针 |
| L0 | 7 | AIV路径(incx=1)：n=1/8/32/128；SIMT路径(incx≠1)：n=128 incx=2/3/7 |
| L1 | 19 | AIV路径(incx=1)：n=1~256（含非对齐n=17、典型n=128等）、全零/全负/正负混合/inf特殊数据；SIMT路径(incx≠1)：n=1/3/8/17/64/128 多步长 |
| 补充 | 5 | n=1/17/128 (incx=1, RANDOM)、正负混合n=128 (MIXED)、n=128 incx=2 (SIMT) |

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
  bash build.sh --ops=sasum --soc=ascend950 --run
  ```

  其中`--soc` 为**可选**参数，用于指定目标硬件平台（与上文「产品支持情况」对应），决定编译哪个架构的实现：

  | 产品 | `--soc` 取值 | 架构 |
  |------|----------------|:----:|
  | Ascend 950PR / Ascend 950DT | `ascend950` | arch35 |
  | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | `ascend910_93` | arch22 |
  | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | `ascend910b` | arch22 |

  执行结果如下，说明精度对比成功。
  ```bash
  [PASS] sasum_test
  ```
