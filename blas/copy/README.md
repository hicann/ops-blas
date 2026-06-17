## Scopy算子实现

## 概述

BLAS Scopy算子实现，同时支持Ccopy复数向量复制。

### 支持的接口

- **aclblasScopy**: 实数向量复制，将x的数据拷贝到y
- **aclblasCcopy**: 复数向量复制，复用scopy kernel实现

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
blas/copy/
├── README.md                       // 说明文档
├── scopy_host.cpp                  // Host 侧实现
└── scopy_kernel.cpp                // Kernel 侧实现
```

## 算子描述

- 算子功能：  
scopy算子实现了将x上的数据拷贝到y。对应的数学表达式为：  
```
y = x
```

### aclblasScopy接口

实数向量复制：
```cpp
int aclblasScopy(aclblasHandle handle, uint8_t *x, uint8_t *y, const int64_t n, const int64_t incx, const int64_t incy);
```

**参数说明：**
- handle: aclblas句柄，用于管理stream和workspace
- x: 输入向量x的device侧指针（uint8_t*类型）
- y: 输出向量y的device侧指针（uint8_t*类型）
- n: 向量长度
- incx: x的步长
- incy: y的步长

### aclblasCcopy接口

复数向量复制（复用scopy kernel）：
```cpp
int aclblasCcopy(aclblasHandle handle, uint8_t *x, uint8_t *y, const int64_t n, const int64_t incx, const int64_t incy);
```

**参数说明：**
- handle: aclblas句柄，用于管理stream和workspace
- x: 输入复数向量x的device侧指针（uint8_t*类型，实际存储为连续的实部、虚部交替的float数组，2*n个float元素）
- y: 输出复数向量y的device侧指针（uint8_t*类型，实际存储为连续的实部、虚部交替的float数组，2*n个float元素）
- n: 复数向量长度
- incx: x的步长
- incy: y的步长

复数向量存储为连续的实部、虚部交替的float数组（2*n个float元素），直接调用scopy kernel处理2*n个float元素即可完成复数向量复制。

**注意：** 接口参数x和y为device侧指针，调用前需要通过aclrtMalloc分配device内存，并通过aclrtMemcpy将数据从host侧拷贝到device侧。计算完成后需要通过aclrtMemcpy将结果从device侧拷回到host侧。
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">scopy_kernel</td></tr>
  </table>

- 算子实现： 

    将输入数据从输入x的GM地址搬运到UB，再搬出到输入y所在的GM地址。

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
  bash build.sh --ops=scopy --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```