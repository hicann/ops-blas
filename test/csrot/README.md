## Csrot算子实现

## 概述

BLAS Csrot算子实现。

Csrot(复数向量旋转)算子实现了对两个复数向量的平面旋转运算，是BLAS基础线性代数库中的核心算子之一。

该算子实现了Givens旋转，常用于QR分解、求解线性方程组和特征值计算等数值算法中。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── csrot
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── csrot_test.cpp      // 算子调用样例
```

## 算子描述

- 算子功能：  
Csrot算子实现了对两个复数向量的平面旋转。对应的数学表达式为：  
```
x[i] = c * x[i] + s * y[i]
y[i] = c * y[i] - s * x[i] (使用原始x[i])
```
其中 c = cos(θ)，s = sin(θ)，θ为旋转角度

对应的接口为：
```
int aclblasCsrot(float *x, float *y, const int64_t n, const float c, const float s, void *stream);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">csrot 参数说明</td>
   </tr>
   <tr>
      <td rowspan="6" align="center">参数列表</td>
      <td align="center">Param.</td>
      <td align="center">Memory</td>
      <td align="center">in/out</td>
      <td align="center">含义</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量元素个数。</td>
   </tr>
   <tr>
      <td align="center">c</td>
      <td align="center">host/device</td>
      <td align="center">in</td>
      <td align="center">旋转角度的余弦值。</td>
   </tr>
   <tr>
      <td align="center">s</td>
      <td align="center">host/device</td>
      <td align="center">in</td>
      <td align="center">旋转角度的正弦值。</td>
   </tr>
   <tr>
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">向量，包含 n 个元素，原地修改。</td>
   </tr>
   <tr>
      <td align="center">y</td>
      <td align="center">device</td>
      <td align="center">in/out</td>
      <td align="center">向量，包含 n 个元素，原地修改。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Csrot</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x, y</td><td align="center">n</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x, y</td><td align="center">n</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">csrot_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从x和y的GM地址分块搬运到UB，进行旋转计算后再搬出到x和y所在的GM地址。

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
  bash build.sh --ops=csrot --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
