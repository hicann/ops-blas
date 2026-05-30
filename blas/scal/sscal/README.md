## Sscal算子实现

## 概述

BLAS Sscal算子实现。

Sscal(Scale)算子实现了向量缩放运算，是BLAS基础线性代数库中的核心算子之一。

本测试同时验证两个接口：
- **aclblasSscal**: 实数向量 × 实数标量
- **aclblasCsscal**: 复数向量 × 实数标量（复数向量视为2*n个float元素，复用sscal kernel）

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── sscal
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── sscal_test.cpp       // 算子调用样例
```

## 算子描述

- 算子功能：  
sscal算子实现了向量x乘以标量alpha。对应的数学表达式为：  
```
x = alpha * x
```

- 对应的接口：
```cpp
// 实数向量缩放
int aclblasSscal(aclblasHandle handle, float *x, const float alpha, const int64_t n, const int64_t incx);

// 复数向量缩放（实数标量）
int aclblasCsscal(aclblasHandle handle, std::complex<float> *x, const float alpha, const int64_t n, const int64_t incx);
```

<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">sscal 参数说明</td>
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
      <td align="center">向量，包含n个元素（sscal为float，csscal为complex）。</td>
   </tr>
   <tr>
      <td align="center">alpha</td>
      <td align="center">host</td>
      <td align="center">in</td>
      <td align="center">用于乘法的标量。</td>
   </tr>
   <tr>
      <td align="center">n</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">向量x中的元素个数。</td>
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
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Sscal</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sscal_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从GM地址搬运到UB，使用Muls指令完成向量乘标量运算，再将结果搬运回GM地址。
    csscal接口将复数向量视为2*n个float元素，直接复用sscal kernel完成计算。

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
  bash build.sh --ops=sscal --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  ========== Testing aclblasSscal ==========
  [Success] Sscal accuracy verification passed.
  [PASS] Sscal test passed
  ========== Testing aclblasCsscal ==========
  [Success] Csscal accuracy verification passed.
  [PASS] Csscal test passed
  ========================================
  Test Summary:
    Passed: 2 - Sscal Csscal
  ========================================
  ```