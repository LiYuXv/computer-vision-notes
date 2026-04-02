---
title: NumPy 学习笔记
date: 2026-04-02
tags:
  - Python
  - NumPy
  - 图像处理
  - 学习笔记
aliases:
  - Numpy 学习笔记
  - NumPy 基础
---

# NumPy 学习笔记

> [!info] 定位
> NumPy 是 Python 科学计算的核心库，提供了高性能的多维数组对象 `ndarray` 以及围绕数组的计算、变形、统计和读写能力。
> 在 [[opencv学习笔记]] 中，OpenCV 图像本质上也是 NumPy 数组，因此 NumPy 是图像处理的基础。

---

## 1. NumPy 是什么？

### 1.1 核心理解

NumPy 可以理解为：
- 比 Python 原生 `list` 更适合做数值计算
- 支持一维、二维、三维甚至更高维数据
- 支持“整体运算”，不需要总是自己写 `for` 循环

常见导入方式：

```python
import numpy as np
```

> [!tip] 为什么重要
> 在数据分析、机器学习、图像处理里，大量数据都天然适合表示为矩阵或张量，而 NumPy 正是处理这些结构的基础工具。

---

## 2. 数组基础

### 2.1 什么是数组（array）

- **1D array**：一维数组，像一行数字
- **2D array**：二维数组，像表格
- **3D array**：三维数组，像很多层表格叠起来

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.array([[(1.5, 2, 3), (4, 5, 6)],
              [(3, 2, 1), (4, 5, 6)]])
```

### 2.2 axis 的概念

> [!definition] axis
> `axis` 表示沿哪个轴进行运算。

- `axis=0`：沿着“行的方向”往下看，表示**按列处理**
- `axis=1`：沿着“列的方向”往右看，表示**按行处理**

例如：

```python
b.max(axis=0)   # 每一列的最大值
b.max(axis=1)   # 每一行的最大值
```

> [!tip] 记忆方式
> 看 `axis=0` 时，是把每一列聚合起来；看 `axis=1` 时，是把每一行聚合起来。

---

## 3. 创建数组（Creating Arrays）

### 3.1 从列表创建

```python
a = np.array([1, 2, 3])
b = np.array([(1.5, 2, 3), (4, 5, 6)], dtype=float)
```

### 3.2 创建特殊数组

```python
np.zeros((3, 4))                  # 全零数组
np.ones((2, 3, 4), dtype=np.int16) # 全一数组
np.empty((2, 3))                  # 未初始化数组
np.full((2, 2), 7)                # 全部填 7
np.eye(3)                         # 单位矩阵
np.arange(10, 30, 5)              # 等步长序列
np.linspace(0, 2, 9)              # 等间距序列
np.random.random((2, 3))          # 0~1 随机数数组
```

### 3.3 常用创建方式速查

| 函数 | 作用 |
|------|------|
| `np.array()` | 从列表或元组创建数组 |
| `np.zeros()` | 创建全零数组 |
| `np.ones()` | 创建全一数组 |
| `np.empty()` | 创建未初始化数组 |
| `np.full()` | 创建固定值数组 |
| `np.eye()` | 创建单位矩阵 |
| `np.arange()` | 按步长创建序列 |
| `np.linspace()` | 按等间距创建序列 |
| `np.random.random()` | 创建随机数组 |

---

## 4. 查看数组属性（Inspecting Your Array）

NumPy 数组通常要求元素类型统一。

```python
print(b.shape)       # 形状
print(b.ndim)        # 维数
print(b.size)        # 元素总数
print(b.dtype)       # 元素数据类型
print(b.dtype.name)  # 数据类型名称
print(b.astype(int)) # 类型转换
```

### 常见属性说明

| 属性 / 方法 | 含义 |
|------------|------|
| `shape` | 数组形状，如 `(2, 3)` |
| `ndim` | 数组维数 |
| `size` | 元素总数 |
| `dtype` | 元素类型 |
| `astype()` | 转换数据类型 |
| `len(a)` | 第一维长度 |

> [!warning] 注意
> `len(a)` 只返回第一维长度，不等于元素总数；元素总数应该看 `a.size`。

---

## 5. 索引、切片与取值

## 5.1 一维切片

切片遵循“左闭右开”。

```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
a[2:7]
a[::2]
a[::-1]
```

- `a[2:7]`：取索引 2 到 6
- `a[::2]`：每隔 2 个取一个
- `a[::-1]`：反向切片

### 5.2 多维切片

```python
b[0:2, 1]
b[:1]
b[0]
c[1, ...]
c[1, :, :]
```

### 5.3 `b[0]` 与 `b[0:1]` 的区别

- `b[0]`：降成一维
- `b[0:1]`：仍然保持二维

### 5.4 布尔索引

```python
a[a < 2]
a[a >= 2]
a[a != 2]
b[b > 2]
```

> [!tip] 特点
> 布尔索引对二维数组也适用，但结果通常会被拉平成一维。

### 5.5 Fancy Indexing（花式索引）

```python
b[[1, 0, 1, 0], [0, 1, 2, 0]]
b[[1, 0, 1, 0]][:, [0, 1, 2, 0]]
```

- 第一种：按指定坐标逐个取元素
- 第二种：先选行，再选列，得到子矩阵

---

## 6. 数组运算（Array Math）

> [!definition] 按元素运算
> 多数 NumPy 运算默认是按元素逐项进行，而不是线性代数中的矩阵乘法。

```python
a + b
 a - b
 a * b
 a / b
np.multiply(a, b)
np.exp(a)
np.sqrt(a)
np.sin(a)
np.cos(a)
np.log(a)
```

### 6.1 点积

```python
a.dot(b.T)
```

这里的点积遵循线性代数规则，而不是逐元素乘法。

### 6.2 数组与标量运算

```python
a + 10
a * 2
a / 2
```

> [!tip] 理解
> 标量会自动广播到数组中的每个元素。

---

## 7. 聚合函数（Aggregate Functions）

常见聚合函数包括：

```python
a.sum()
a.sum(axis=0)
a.sum(axis=1)

a.min()
a.max()
a.mean()
a.cumsum()
np.median(a)
a.std()
np.corrcoef(a, b[0])
```

### 常用聚合函数表

| 函数 | 作用 |
|------|------|
| `sum()` | 求和 |
| `min()` / `max()` | 最小值 / 最大值 |
| `mean()` | 平均值 |
| `cumsum()` | 累计和 |
| `np.median()` | 中位数 |
| `std()` | 标准差 |
| `np.corrcoef()` | 相关系数 |

> [!example] 典型用法
> - `a.sum(axis=0)`：每列求和
> - `a.sum(axis=1)`：每行求和

---

## 8. 数组操作（Array Manipulation）

### 8.1 转置与展平

```python
b.T
b.flatten()
b.ravel()
```

- `flatten()`：返回副本
- `ravel()`：通常返回视图，效率更高

### 8.2 变形

```python
b.reshape(3, 2)
b.reshape(3, -1)
x.resize(3, 2)
```

- `reshape()`：返回新视图/新数组，不直接改原数组
- `resize()`：直接修改原数组
- `-1`：让 NumPy 自动推断这一维长度

### 8.3 插入、删除、追加

```python
np.append(a, [4, 5])
np.insert(a, 2, [10, 11])
np.delete(a, 2)
```

> [!warning] 注意
> `append`、`insert`、`delete` 默认都返回**新数组**，不会原地修改原数组。

### 8.4 拼接

```python
np.concatenate((a, a))
np.vstack((a, a))
np.hstack((e, f))
np.column_stack((a, d))
np.r_[a, d]
np.c_[a, d]
```

### 8.5 拆分

```python
np.split(a, 3)
np.hsplit(b, 3)
```

---

## 9. 数组比较、复制与排序

### 9.1 数组比较

```python
a == b
a < 2
np.array_equal(a, c)
```

- `a == b`：逐元素比较
- `np.array_equal(a, c)`：整体判断两个数组是否完全相同

### 9.2 复制：view 与 copy

```python
h = a.view()
i = a.copy()
```

| 方法 | 特点 |
|------|------|
| `view()` | 浅层视图，共享底层数据 |
| `copy()` | 深拷贝，不共享数据 |

> [!warning] 易错点
> 修改 `view()` 得到的新对象，原数组也会变化；修改 `copy()` 得到的新对象，原数组不会变。

### 9.3 排序

```python
np.sort(a)
a.sort()
x.sort(axis=0)
x.sort(axis=1)
```

- `np.sort(a)`：返回新数组
- `a.sort()`：原地排序
- `axis=0`：按列排序
- `axis=1`：按行排序

---

## 10. 数据读写（I/O）

### 10.1 文本文件

```python
np.savetxt('array.txt', a, fmt='%d')
np.loadtxt('array.txt', dtype=int)
np.genfromtxt('array.txt', dtype=int)
```

### 10.2 二进制文件

```python
np.save('array.npy', a)
np.savez('arrays.npz', a=a, b=b)
data = np.load('array.npy')
data = np.load('arrays.npz')
data['a']
```

### I/O 方式对比

| 方法 | 说明 |
|------|------|
| `savetxt` / `loadtxt` | 文本格式，便于查看 |
| `genfromtxt` | 更灵活，适合缺失值 |
| `save` / `load` | `.npy` 二进制，适合单数组 |
| `savez` | `.npz` 压缩，适合保存多个数组 |

---

## 11. 数据类型（Data Types）

常见类型：

```python
np.int64
np.float32
np.complex128
np.bool_
np.object_
np.string_
np.unicode_
```

字符串类型转换：

```python
b.astype(str)
```

> [!warning] 补充
> 笔记原始内容中写到了 `np.bool`，在新版本 NumPy 中更推荐使用 `np.bool_` 或 Python 原生 `bool`。

---

## 12. 查看帮助

```python
np.info(np.ndarray)
```

这个命令可以查看 `ndarray` 的构造方式、属性和方法，是学习 API 的有效方式。

---

## 13. 常用方法速查表

| 类别 | 常用函数 / 属性 |
|------|----------------|
| 创建 | `array` `zeros` `ones` `empty` `full` `eye` `arange` `linspace` |
| 属性 | `shape` `ndim` `size` `dtype` |
| 索引 | 切片 `:`、布尔索引、花式索引 |
| 运算 | `+ - * /` `exp` `sqrt` `sin` `cos` `log` `dot` |
| 聚合 | `sum` `min` `max` `mean` `cumsum` `median` `std` |
| 变形 | `reshape` `resize` `flatten` `ravel` `T` |
| 拼接拆分 | `concatenate` `vstack` `hstack` `split` `hsplit` |
| 复制排序 | `view` `copy` `sort` `array_equal` |
| I/O | `savetxt` `loadtxt` `save` `savez` `load` |

---

## 14. 学习重点与易错点

> [!summary] 重点
> 先真正理解这 5 个核心概念：**shape、axis、切片、按元素运算、reshape**。

### 易错点总结

1. **`axis=0` 和 `axis=1` 容易混淆**
   - `axis=0`：按列聚合
   - `axis=1`：按行聚合

2. **`b[0]` 和 `b[0:1]` 不一样**
   - 前者降维，后者保留维度

3. **`flatten()` 和 `ravel()` 不一样**
   - `flatten` 更安全，`ravel` 更高效

4. **`reshape()` 与 `resize()` 不一样**
   - `reshape` 通常不改原数组，`resize` 会改原数组

5. **`view()` 与 `copy()` 不一样**
   - `view` 共享数据，`copy` 独立数据

---

## 15. 与 OpenCV 的联系

在 OpenCV 中：
- 灰度图像通常是二维 NumPy 数组
- 彩色图像通常是三维 NumPy 数组，形状一般为 `(height, width, channels)`
- 图像裁剪、通道处理、本质上都离不开 NumPy 的切片与数组运算

可继续阅读：[[opencv学习笔记]]

---

## 16. 一句话总结

> [!quote]
> NumPy 的核心就是：**用统一的数据结构 `ndarray`，高效地表示和操作多维数值数据。**

---

*整理来源：study_numpy.ipynb | 由 Claudian 整理为 Obsidian 笔记*