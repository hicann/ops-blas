## Snrm2算子实现

## 概述

BLAS Snrm2算子实现。

Snrm2(欧几里得范数)算子实现了计算向量的欧几里得范数（2-范数），是BLAS基础线性代数库中的核心算子之一。

该算子计算L2范数，常用于向量长度计算、归一化和误差估计。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── snrm2
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── snrm2_test.cpp      // 算子调用样例
```

## 算子描述

- 算子功能：  
Snrm2算子实现了计算向量的欧几里得范数。对应的数学表达式为：  
```
result = sqrt(sum(|x[i]|^2)) for i = 0 to n-1
```

对应的接口为：
```
int aclblasSnrm2(float *x, float *result, const int64_t n, const int64_t incx, void *stream);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">snrm2 参数说明</td>
   </tr>
   <tr>
      <td rowspan="5" align="center">参数列表</td>
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
      <td align="center">x</td>
      <td align="center">device</td>
      <td align="center">in</td>
      <td align="center">向量，包含 n 个元素。</td>
   </tr>
   <tr>
      <td align="center">incx</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">x 中连续元素之间的步长。</td>
   </tr>
   <tr>
      <td align="center">result</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">向量的欧几里得范数。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Snrm2</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">n</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">1</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">snrm2_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从x的GM地址分块搬运到UB，并行计算各核的局部平方和，最后归约并计算平方根得到最终结果。

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
  bash build.sh --ops=snrm2 --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
