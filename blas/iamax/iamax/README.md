## Iamax算子实现

## 概述

BLAS Iamax算子实现。

Iamax(最大绝对值元素索引)算子实现了查找向量中绝对值最大的元素索引，是BLAS基础线性代数库中的核心算子之一。

该算子返回1-based索引，遵循BLAS惯例，常用于主元选择和迭代算法中。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── iamax
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   └── iamax_test.cpp      // 算子调用样例
```

## 算子描述

- 算子功能：  
Iamax算子实现了查找向量中绝对值最大的元素索引。对应的数学表达式为：  
```
result = argmax_i |x[i]|
```
返回1-based索引

对应的接口为：
```
int aclblasIamax(const float *x, int32_t *result, const int64_t n, const int64_t incx, 
                 const uint32_t dtypeFlag, void *stream);
```
<table>
   <tr>
      <td rowspan="1" align="center">参数</td>
      <td colspan="4" align="center">iamax 参数说明</td>
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
      <td align="center">dtypeFlag</td>
      <td align="center"></td>
      <td align="center">in</td>
      <td align="center">数据类型标志，0表示实数float，1表示复数float。</td>
   </tr>
   <tr>
      <td align="center">result</td>
      <td align="center">device</td>
      <td align="center">out</td>
      <td align="center">最大绝对值元素的索引（1-based）。</td>
   </tr>
</table>


- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Iamax</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">n</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">result</td><td align="center">1</td><td align="center">int32</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">iamax_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从x的GM地址分块搬运到UB，并行计算各核的局部最大值，最后归约得到全局最大值索引。

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
  bash build.sh --ops=iamax --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
