## symv算子实现

## 概述

BLAS symv算子实现。

symv(Symmetric Matrix-Vector Multiplication)算子实现了对称矩阵与向量的乘法运算，是BLAS基础线性代数库中的核心算子之一。

该算子针对对称矩阵的存储特性进行了优化，并高效完成矩阵与向量的乘加运算。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── symv
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── symv_test.cpp       // 算子调用样例
```

## 算子描述

- 算子功能：  
symv算子实现了将对称压缩矩阵乘以向量。对应的数学表达式为：  
```
z = alpha * A * x + beta * y
```
A为对称矩阵，x和y是向量，alpha和beta是标量

对称矩阵A的下三角部分元素按行储存，对应的对称部分通过已有元素推断得出。压缩对称矩阵格式仅需要`n * (n + 1) / 2`个元素储存。

对应的接口为：
```
int aclblasSymv(const float *a, const int64_t lda, const float *x, const float *y, float *z,
				const float alpha, const float beta,
				const int64_t n, const int64_t incx, const int64_t incy, void *stream);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">symv 参数说明</td>
   </tr>
   <tr>
      <td rowspan="12" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">对称矩阵 A 的行数和列数。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host/device</td>
      <td align="center">in</td>
      <td align="center">用于乘法的 &lt;type&gt; 标量。</td>
   </tr>
   <tr>
      <td align="center">aPacked</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">对称矩阵 &lt;type&gt; 数组，维度为 n x n。</td>
   </tr>
   <tr>
      <td align="center">lda</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">用于存储矩阵A的二维数组的主维。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">&lt;type&gt; 向量，包含 n 个元素。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x 中连续元素之间的步长。</td>
   </tr>
   <tr>
      <td align="center">beta</td>
      <td align="center">host/device</td>
      <td align="center">in</td>
      <td align="center">用于乘法的 &lt;type&gt; 标量。如果 beta == 0，则 y 不必是有效输入。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">&lt;type&gt; 向量，包含 n 个元素。</td>
   </tr>
   <tr>
      <td align="center">incy</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">y 中连续元素之间的步长。</td>
   </tr>
   <tr>
      <td align="center">z</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">&lt;type&gt; 向量，包含 n 个元素。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">symv</td></tr>
  </tr>
  <tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">N * (N + 1) /2</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">x</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">alpha</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">beta</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">symv_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从A,x,y的GM地址分块搬运到UB，进行计算后再搬出到z所在的GM地址。

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
  bash build.sh --ops=symv --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```