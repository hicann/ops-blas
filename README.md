# ops-blas

## 🔥Latest News
- [2026/03] ops-blas项目上线，提供BLAS计算的API以及现代灵活接口aclBLASLt，支持混合精度、融合后处理、启发式算法选择等高级特性。

## 🚀概述
ops-blas是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供高性能线性代数计算以及轻量化GEMM调用的算子库。

## 📝环境部署
### 快速安装CANN软件
本节提供快速安装CANN软件的示例命令，更多安装步骤请参考[详细安装指南](#cann详细安装指南)。

#### 安装前准备
在线安装和离线安装时，需确保已具备Python环境及pip3，当前CANN支持Python3.7.x至3.11.4版本。
离线安装时，请单击[获取链接](https://www.hiascend.com/developer/download/community/result?module=cann)下载CANN软件包，并上传到安装环境任意路径。
#### 安装CANN
```shell
chmod +x Ascend-cann-toolkit_8.5.RC1_linux-$(arch).run
./Ascend-cann-toolkit_8.5.RC1_linux-$(arch).run --install
```
#### 安装后配置
配置环境变量脚本set_env.sh，当前安装路径以${HOME}/Ascend为例。
```
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```

### CANN详细安装指南
开发者可访问[昇腾文档-昇腾社区](https://www.hiascend.com/document)->CANN社区版->软件安装，查看CANN软件安装引导，根据机器环境、操作系统和业务场景选择后阅读详细安装步骤。

### 基础工具版本要求与安装
安装CANN之后，您可安装一些工具方便后续开发，参见以下内容：

* [CANN依赖列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0045.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)
* [CANN安装后操作](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0094.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)

## ⚡️快速入门
 - 加速库下载
    ```sh
    git clone https://gitcode.com/cann/ops-blas.git
    ```
   您可自行选择需要的分支。
 - 执行编译安装
    编译加速库，设置加速库环境变量：
    ```sh
    cd ops-blas
    bash build.sh --op=scopy --run # --op=<算子名> --run可选参数，执行测试样例
    ```

## 📂目录结构
ops-blas库的目录结构如下：

```
ops-blas
├── build          //可存放构建生成的文件
├── docs           //文档文件
├── example        //算子调用示例代码，包含可直接运行的Demo
├── include        //存放公共头文件
├── scripts        //脚本文件存放目录
├── blaslt           //blaslt主体源代码目录
├── blas            //blas主体源代码目录
│   ├── utils      //公共函数
│   ├── dot        //向量点积算子实现
│   ├── gemv       //一般矩阵向量乘法算子实现
│   ├──  ...       //其他算子实现
│   └── CMakeLists.txt
├── tests          //测试代码
```

## 💬相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-linear-algebra)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-blas/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-blas/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-blas/wiki)获取技术文章，如系列化教程、优秀实践等。