# strmm 算子测试

## 算子简介

strmm（实三角矩阵乘法）算子实现三角矩阵与普通矩阵的乘法运算。

计算公式：`C = alpha * op(A) * op(B)`

其中：
- A 为三角矩阵（上三角或下三角）
- B 为普通矩阵
- op(A) 表示矩阵 A 可以选择不转置(N)、转置(T)
- op(B) 表示矩阵 B 可以选择不转置(N)、转置(T)

## 测试说明

本测试包含以下场景：

1. **基础测试**：side=L, uplo=L, transa=N, transb=N, diag=N
   - 下三角矩阵 A 与矩阵 B 相乘
   - alpha = 1.0

2. **矩阵A转置测试**：side=L, uplo=L, transa=T, transb=N, diag=N
   - 下三角矩阵 A 转置后与矩阵 B 相乘
   - alpha = 2.0

3. **矩阵B转置测试**：side=L, uplo=L, transa=N, transb=T, diag=N
   - 下三角矩阵 A 与转置后的矩阵 B 相乘
   - alpha = 0.5

## 编译运行

```bash
bash build.sh --ops=strmm --run
```

## 验证标准

精度误差阈值：1e-3