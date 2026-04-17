## CgemvBatched算子实现

## 概述

BLAS CgemvBatched算子实现。

CgemvBatched(批量复数矩阵-向量乘法)算子实现了批量复数矩阵与向量的乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子针对批量计算场景进行了优化，支持普通、转置和共轭转置三种矩阵操作模式，高效完成批量矩阵-向量乘法。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── cgemv_batched
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── cgemv_batched_test.cpp // 算子调用样例
```

## 算子描述

- 算子功能：  
CgemvBatched算子实现了批量复数矩阵与向量的乘法。对应的数学表达式为：  
```
y[i] = A[i] * x[i]        (trans = N)
y[i] = A[i]^T * x[i]      (trans = T)
y[i] = A[i]^H * x[i]      (trans = C)
```
其中A[i]是复数矩阵，x[i]和y[i]是复数向量，i是批次索引

复数乘法公式：(a + bi) * (c + di) = (ac - bd) + (ad + bc)i

对应的接口为：
```
int aclblasCgemvBatched(const void *A, const void *x, void *y,
                        const int64_t batchCount, const int64_t m, const int64_t n,
                        const int32_t trans, const int32_t dtype,
                        void *stream);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">cgemv_batched 参数说明</td>
   </tr>
   <tr>
      <td rowspan="9" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">batchCount</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">批次数。</td>
   </tr>
   <tr>
      <td align="center">m</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵的行数。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵的列数。</td>
   </tr>
   <tr>
      <td align="center">A</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">批量复数矩阵，batchCount个m×n矩阵。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">批量复数向量，batchCount个向量。</td>
   </tr>
   <tr>
      <td align="center">trans</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">矩阵操作类型：0=N(普通)，1=T(转置)，2=C(共轭转置)。</td>
   </tr>
   <tr>
      <td align="center">dtype</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">数据类型：0=half，1=float。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">批量复数向量，batchCount个向量。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">CgemvBatched</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A, x</td><td align="center">batchCount × m × n</td><td align="center">complex&lt;float&gt;</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">batchCount × m</td><td align="center">complex&lt;float&gt;</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cgemv_batched_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从A和x的GM地址分块搬运到UB，使用GatherMask分离实部和虚部，进行复数矩阵-向量乘法计算，最后归约并搬出到y所在的GM地址。

- 调用实现  
    使用内核调用符<<<>>>调用核函数。 

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
  bash build.sh --ops=cgemv_batched --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
