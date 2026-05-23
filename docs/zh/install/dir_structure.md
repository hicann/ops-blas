# 项目目录

> 本章罗列的部分目录或文件为可选，请以实际交付件为准。另请注意：
>
> - **算子目录与源码前缀**：`blas/` 下子目录名表示一类 BLAS 能力，目录内 `*_host.cpp`、`*_kernel.cpp` 等文件名前缀可能与目录名不同（例如目录 `trsv` 内存放 `strsv_*` 实现），以目录内实际文件为准。
> - **测试子目录命名**：`test/` 下子目录名通常与测试源文件名前缀一致（如 `test/strsv/strsv_test.cpp`），可能与 `blas/` 侧目录名不完全相同，以 `test/` 下实际目录为准。
> - **平台裁剪**：部分算子会随 `SOC_VERSION` 在 `blas/CMakeLists.txt` 的排除列表中不参与编译，若本地仓中缺少对应目录或无可执行体，属工程裁剪策略所致。
> - **架构子目录**：若某算子目录下存在 `${arch_dir}`（如 `arch35`、`arch20`），表示面向特定 NPU 架构的差异化实现，由顶层 CMake 按当前 SOC 选择是否编入；若缺少该目录，通常表示当前实现未拆分架构专用源码。
> - **测试工程**：`test/` 下用例需在配置工程时开启 `BUILD_TEST` 并设置 `TEST_NAMES`（分号分隔的算子测试子目录名）后才会参与构建，详见根目录 [CMakeLists.txt](../../../CMakeLists.txt) 与 [QuickStart](../../../docs/QUICKSTART.md)。
> - 若需补充算子或文档，欢迎参考 [贡献指南](../../../CONTRIBUTING.md)。

项目全量目录层级介绍如下：

```text
├── cmake                                               # 工程辅助 CMake 脚本（版本、打包、第三方获取等）
│   ├── test.cmake                                      # 测试 target 注册
│   ├── package.cmake                                   # CPack/安装包相关逻辑
│   ├── version.cmake
│   ├── makeself.cmake
│   └── third_party                                     # 第三方工具脚本
├── blas                                                # aclBLAS 算子源码目录
│   ├── CMakeLists.txt                                  # blas 源文件收集、平台排除与架构子目录参与规则
│   ├── common                                          # 算子公共头文件与工具（类型、迭代器、内存与布局等）
│   │   └── helper                                      # Host 侧辅助实现与 kernel 工具头文件（句柄、日志、pipe/tiling 工具等）
│   └── ${op_name}                                      # 单个 BLAS 相关能力目录，${op_name} 为小写目录名
│       ├── ${routine}_host.cpp                         # Host 侧：参数检查、任务下发等，${routine} 为实际 BLAS 例程前缀
│       ├── ${routine}_kernel.cpp                       # Device 侧：Ascend C 核函数入口
│       ├── kernel                                      # 可选，核函数按场景拆分子目录（如 trans/no_trans 多个 *kernel*.cpp）
│       │   ├── ${routine}_${variant}_kernel.cpp
│       │   └── ...
│       ├── ${impl_headers}.h                           # 可选，kernel 实现或计划类头文件（如 batched 的 plan、utils）
│       └── ${arch_dir}                                 # 可选，面向特定架构的源码（如 arch35），由 SOC 配置决定是否编译
├── blasLt                                              # aclBLASLt 源码目录
│   ├── CMakeLists.txt
│   ├── aclblasLt.cpp                                   # BLASLt 侧入口与对外逻辑
│   └── ${feature_dir}                                  # 能力子目录，如 matmul（核函数、工具头文件等）
│       ├── ${feature}_kernel.cpp
│       └── ...
├── include                                             # 对外头文件
│   ├── cann_ops_blas.h                                 # aclBLAS API 声明
│   ├── cann_ops_blasLt.h                               # aclBLASLt API 声明
│   └── cann_ops_blas_common.h                          # 公共类型与声明
├── docs                                                # 项目文档
│   ├── QUICKSTART.md                                   # 快速入门
│   └── zh
│       ├── api_list.md                                 # 接口列表
│       └── install
│           ├── quick_install.md                        # 环境部署
│           └── dir_structure.md                        # 目录结构
├── scripts                                             # 辅助脚本（打包安装说明、版本信息生成等）
│   ├── package
│   └── util
├── test                                                # 算子级测试（BUILD_TEST=ON 且配置 TEST_NAMES 时参与构建）
│   ├── CMakeLists.txt                                  # 按 TEST_NAMES 批量 add_subdirectory
│   └── ${op_name}
│       ├── CMakeLists.txt
│       ├── ${test_entry}.cpp                           # 可执行测试源文件（根目录，所有 SOC 均编译）
│       ├── ${arch_dir}                                 # 可选，架构专用测试源（如 arch35），由 --soc 对应 SOC_ARCH_DIRS 决定是否编译
│       └── README.md                                   # 可选，本算子测试说明
├── CMakeLists.txt                                      # 工程入口：ops_blas / ops_blasLt 库、安装规则、可选测试与打包
├── CONTRIBUTING.md                                     # 贡献指南
├── LICENSE                                             # 开源许可证
├── README.md                                           # 项目总览
├── SECURITY.md                                         # 安全声明
├── build.sh                                            # 编译脚本
├── install_deps.sh                                     # 依赖安装脚本
└── ...
```
