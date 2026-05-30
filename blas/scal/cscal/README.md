## Cscal算子实现

## 概述

BLAS Cscal算子实现。

Cscal(Complex Scale)算子实现了复数向量缩放运算，是BLAS基础线性代数库中的核心算子之一。

该算子实现复数向量乘以复数标量：`(a+bi)*(c+di) = (ac-bd) + (ad+bc)i`

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── cscal
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── cscal_test.cpp       // 算子调用样例
```

## 算子描述

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
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">8 * 2048</td><td align="center">complex<float></td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cscal</td></tr>
  </table>

- 算子实现： 

    将复数向量从GM搬运到UB，使用vreducev2进行虚实分离，分别计算实部*实部、实部*虚部、虚部*实部、虚部*虚部，再使用add_v合并结果，最后通过vgather进行虚实合并并搬运回GM。

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
  bash build.sh --ops=cscal --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  [PASS] cscal_test
  ```