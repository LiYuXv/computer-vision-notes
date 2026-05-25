# CNN 图像分类讲解

**User:** Anonymous  
**Created:** 5/24/2026 19:59:07  
**Updated:** 5/24/2026 21:18:30  
**Exported:** 5/24/2026 21:20:06  
**Link:** [https://chatgpt.com/g/g-p-69ca8f8ab6e881918bdad9b692c413de-ji-suan-ji-shi-jue-xue-xi/c/6a12e7f5-fd98-83ec-b4dc-6e91a3244751](https://chatgpt.com/g/g-p-69ca8f8ab6e881918bdad9b692c413de-ji-suan-ji-shi-jue-xue-xi/c/6a12e7f5-fd98-83ec-b4dc-6e91a3244751)  

## Response:
2026/5/24 21:16:54

> **整理复习笔记：基于CNN的图像识别**
>
> 我将整理笔记，并包括核心公式、网络结构、重点区分以及可能的考试考点。不过，由于不能引用行号，我会尽力通过文件标记或网址作为来源。笔记以复习目的为主，确保内容简明详细，包含关键公式和网络架构，避免过于冗长。
>
> 已思考 27s

下面是**第十二讲：基于 CNN 的图像识别**的复习笔记版，按 PPT 顺序整理，重点放在考试/复习最容易用到的概念、公式和网络对比上。`Lecture12_CNN_for_image_recognition.pdf`

# 第十二讲 基于 CNN 的图像识别

## 1. CNN 图像分类基本流程

CNN 用于图像识别时，整体流程是：

$$
\text{输入图像} \rightarrow \text{卷积特征提取} \rightarrow \text{分类器} \rightarrow \text{类别概率}
$$

典型结构：

$$
\text{Convolution} + \text{BatchNorm} + \text{Activation}
$$

重复多次后，再经过池化、全连接层或全局平均池化，最后输出类别分数。

CNN 的特点：

- 主要用于计算机视觉；
- 也可用于语音识别等任务；
- 卷积层自动学习图像特征，不需要人工设计 Sobel、SIFT 那样的固定特征。

---

## 2. 卷积层参数

卷积层常见参数：

| 参数 | 含义 |
|---|---|
| input channel | 输入通道数，例如 RGB 图像为 3 |
| output channel | 输出通道数，也就是卷积核个数 |
| kernel size | 卷积核大小，例如 $3 \times 3$、$5 \times 5$ |
| padding | 边缘填充 |
| stride | 步长 |
| dilation rate | 空洞卷积膨胀率 |
| group number | 分组卷积组数 |

### 2.1 卷积输出尺寸公式

$$
W'=\frac{W-F+2P}{S}+1
$$

其中：

- $W$：输入尺寸；
- $F$：卷积核大小；
- $P$：padding；
- $S$：stride。

如果输入是 $W \times W \times C_{in}$，卷积核个数为 $C_{out}$，则输出尺寸为：

$$
W' \times W' \times C_{out}
$$

### 2.2 卷积层参数量

普通卷积参数量：

$$
F \times F \times C_{in} \times C_{out}
$$

如果考虑 bias，再加：

$$
C_{out}
$$

一般 PPT 里计算时忽略 bias。

---

## 3. Softmax 与交叉熵损失

CNN 最后一层通常输出的是每个类别的**原始分数 logits**，例如：

$$
s=[3.2,\ 5.1,\ -1.7]
$$

这些分数不一定为正，也不要求和为 1，所以不能直接看成概率。

---

### 3.1 Softmax

Softmax 用来把 logits 转换成概率：

$$
P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_j e^{s_j}}
$$

特点：

- 每个概率都大于 0；
- 所有类别概率之和为 1；
- 适合**类别互斥**的单标签分类任务。

例如：

$$
[3.2,\ 5.1,\ -1.7]
$$

经过 Softmax 后可能变成：

$$
[0.13,\ 0.87,\ 0.00]
$$

说明模型认为第二类概率最大。

---

### 3.2 交叉熵损失 Cross Entropy

单个样本损失：

$$
L_i=-\log P(Y=y_i|X=x_i)
$$

如果使用 one-hot 标签：

$$
L_i=-\sum_j y_j \log p_j
$$

其中：

- $y_j$：真实标签的 one-hot 编码；
- $p_j$：Softmax 输出的类别概率。

如果真实类别是 cat，但模型给 cat 的概率是 0.13，则：

$$
L_i=-\log(0.13)=2.04
$$

核心理解：

> 真实类别概率越大，损失越小；真实类别概率越小，损失越大。

---

### 3.3 Softmax + Cross Entropy 的使用场景

适用于：

- 单标签分类；
- 类别互斥；
- 例如：猫、狗、车、鸟，只能属于其中一类。

如果是**类别不互斥**的多标签分类，例如一张图同时包含“人、车、道路”，则一般不用 Softmax，而是：

$$
\text{Sigmoid} + \text{Binary Cross Entropy}
$$

---

## 4. 典型图像分类网络发展

本章主要讲：

- AlexNet
- VGG
- GoogLeNet
- ResNet
- MobileNet
- ShuffleNet
- CNN 特征解释

ImageNet 竞赛推动了 CNN 的发展。PPT 中指出，AlexNet 是第一个基于 CNN 的 ImageNet 获胜模型，之后网络逐渐变深，错误率快速下降。

---

# 5. AlexNet

AlexNet 是 2012 年 ImageNet 分类冠军，是 CNN 在大规模图像识别中取得突破的标志性网络。

## 5.1 AlexNet 基本结构

AlexNet 共 8 层，主要结构：

$$
\text{CONV1} \rightarrow \text{POOL1} \rightarrow \text{NORM1}
$$

$$
\rightarrow \text{CONV2} \rightarrow \text{POOL2} \rightarrow \text{NORM2}
$$

$$
\rightarrow \text{CONV3} \rightarrow \text{CONV4} \rightarrow \text{CONV5}
$$

$$
\rightarrow \text{POOL3} \rightarrow \text{FC6} \rightarrow \text{FC7} \rightarrow \text{FC8}
$$

输入图像尺寸：

$$
227 \times 227 \times 3
$$

---

## 5.2 AlexNet 第一层输出尺寸计算

CONV1 参数：

- 输入：$227 \times 227 \times 3$
- 卷积核：$11 \times 11$
- 卷积核个数：96
- stride = 4
- padding = 0

输出空间尺寸：

$$
W'=\frac{227-11+2\times0}{4}+1=55
$$

所以输出为：

$$
55 \times 55 \times 96
$$

---

## 5.3 AlexNet 第一层参数量

每个卷积核大小：

$$
11 \times 11 \times 3
$$

共有 96 个卷积核：

$$
11 \times 11 \times 3 \times 96=34848
$$

约等于：

$$
35K
$$

---

## 5.4 Pooling 层参数量

POOL1：

- 输入：$55 \times 55 \times 96$
- 池化核：$3 \times 3$
- stride = 2

输出尺寸：

$$
\frac{55-3}{2}+1=27
$$

输出为：

$$
27 \times 27 \times 96
$$

Pooling 层不需要学习参数，所以参数量为：

$$
0
$$

---

## 5.5 AlexNet 完整简化结构

| 层 | 输出尺寸 |
|---|---|
| INPUT | $227 \times 227 \times 3$ |
| CONV1 | $55 \times 55 \times 96$ |
| MAX POOL1 | $27 \times 27 \times 96$ |
| CONV2 | $27 \times 27 \times 256$ |
| MAX POOL2 | $13 \times 13 \times 256$ |
| CONV3 | $13 \times 13 \times 384$ |
| CONV4 | $13 \times 13 \times 384$ |
| CONV5 | $13 \times 13 \times 256$ |
| MAX POOL3 | $6 \times 6 \times 256$ |
| FC6 | 4096 |
| FC7 | 4096 |
| FC8 | 1000 |

### AlexNet 重要结论

PPT 强调：

> 参数主要分布在全连接层，卷积层参数相对较少。

例如：

$$
FC7:4096 \times 4096=16.8M
$$

而 CONV1 只有约 35K 参数。

所以早期 CNN 的一个问题是：

> 全连接层参数过多，模型较大，容易过拟合。

---

# 6. VGG

VGG 的核心思想：

> 使用小卷积核，构建更深的网络。

## 6.1 VGG 特点

- 所有卷积层使用 $3 \times 3$ 卷积核；
- 网络分为 5 个 block；
- 每个 block 中有 1 到 4 个卷积层；
- 每个 block 后接 max pooling；
- 常见版本有 VGG16、VGG19。

---

## 6.2 为什么使用 $3 \times 3$ 小卷积核？

PPT 中讲：

> 三个 $3 \times 3$ 卷积层堆叠，具有和一个 $7 \times 7$ 卷积层相同的有效感受野。

也就是说：

$$
3 \times 3 \rightarrow 5 \times 5 \rightarrow 7 \times 7
$$

连续堆叠三个 $3 \times 3$ 卷积后，最终能看到 $7 \times 7$ 范围。

---

## 6.3 小卷积核的优势

假设每层通道数都是 $C$。

一个 $7 \times 7$ 卷积参数量：

$$
7^2C^2=49C^2
$$

三个 $3 \times 3$ 卷积参数量：

$$
3 \times 3^2C^2=27C^2
$$

所以小卷积核的好处：

1. 参数更少；
2. 网络更深；
3. 中间有更多 ReLU，非线性表达能力更强；
4. 感受野可以等效扩大。

一句话记忆：

> VGG 用多个 $3 \times 3$ 小卷积替代大卷积，既减少参数，又增加非线性。

---

# 7. GoogLeNet

GoogLeNet 的核心是 **Inception 模块**。

## 7.1 GoogLeNet 特点

- 22 层深网络；
- 使用 Inception module 增加网络宽度；
- 属于 “Network in Network” 思想；
- 通过堆叠局部优秀模块构成整体网络。

---

## 7.2 Naive Inception 模块

朴素 Inception 模块对同一个输入并行执行：

- $1 \times 1$ 卷积；
- $3 \times 3$ 卷积；
- $5 \times 5$ 卷积；
- $3 \times 3$ pooling。

然后将所有输出在通道维度拼接。

作用：

> 并行分支可以提取多尺度特征。

问题：

> 计算复杂度高。

---

## 7.3 $1 \times 1$ Bottleneck 降维

为了解决计算量过大的问题，GoogLeNet 在 $3 \times 3$、$5 \times 5$ 卷积前加入 $1 \times 1$ 卷积。

作用：

1. 压缩通道数；
2. 减少参数量；
3. 降低计算量；
4. 增加非线性表达。

例如：

$$
\text{输入通道多} \rightarrow 1 \times 1 \text{降维} \rightarrow 3 \times 3/5 \times 5 \text{卷积}
$$

---

## 7.4 全局平均池化 Global Average Pooling

GoogLeNet 在最后卷积层后使用全局平均池化。

如果特征图是：

$$
H \times W \times C
$$

全局平均池化后变成：

$$
1 \times 1 \times C
$$

作用：

- 对每个通道的空间位置求平均；
- 替代多个昂贵的全连接层；
- 减少参数；
- 降低过拟合风险。

---

## 7.5 辅助分类器 Auxiliary Classifier

GoogLeNet 在中间层加入辅助分类输出。

作用：

> 给浅层网络注入额外梯度，加快浅层参数更新，缓解深层网络训练困难。

训练时有辅助作用，测试时主要使用最终分类器输出。

---

# 8. ResNet 残差网络

ResNet 是本章重点。

## 8.1 ResNet 核心思想

普通网络希望直接学习：

$$
H(x)
$$

ResNet 改为学习残差：

$$
F(x)=H(x)-x
$$

因此输出为：

$$
H(x)=F(x)+x
$$

也就是残差块：

$$
x \rightarrow F(x) \rightarrow F(x)+x
$$

其中 $x$ 通过 shortcut connection 直接加到输出上。

---

## 8.2 残差连接的作用

残差连接可以：

1. 加快信号反向传播；
2. 缓解梯度消失；
3. 加快网络收敛；
4. 让深层网络更容易训练；
5. 解决网络加深后的退化问题。

如果某些层暂时没有学到有用特征，可以令：

$$
F(x)=0
$$

那么输出就是：

$$
F(x)+x=x
$$

相当于保留原输入，不会因为网络太深而破坏特征。

---

## 8.3 ResNet 结构特点

PPT 中总结：

- 堆叠残差模块；
- 每个模块有两个卷积层；
- 相邻 stage 通道数 ×2；
- 空间分辨率 /2；
- 使用全局平均池化；
- 最后只接一个 FC 输出 1000 类。

一句话记忆：

> ResNet 通过残差连接让网络可以训练得非常深，例如 152 层。

---

# 9. 不同网络比较

PPT 对 VGG、GoogLeNet、ResNet 的比较：

| 网络 | 特点 |
|---|---|
| VGG | 参数最多，计算量最大，结构简单 |
| GoogLeNet | 参数少，效率最高 |
| ResNet | 效率适中，准确率高，深层网络训练效果好 |

记忆方式：

- AlexNet：CNN 图像识别突破起点；
- VGG：小卷积核堆深网络；
- GoogLeNet：Inception 多尺度并行；
- ResNet：残差连接解决深层训练；
- MobileNet：轻量化网络，适合移动端。

---

# 10. MobileNet

MobileNet 面向移动端和嵌入式设备，核心是：

$$
\text{Depthwise Separable Convolution}
$$

即深度可分离卷积。

---

## 10.1 标准卷积

标准卷积同时做：

1. 空间特征提取；
2. 通道信息混合。

参数量：

$$
K^2 C_{in} C_{out}
$$

---

## 10.2 深度可分离卷积

深度可分离卷积分两步：

### 第一步：Depthwise Convolution

每个输入通道单独做卷积。

参数量：

$$
K^2 C_{in}
$$

### 第二步：Pointwise Convolution

用 $1 \times 1$ 卷积混合通道。

参数量：

$$
C_{in}C_{out}
$$

总参数量：

$$
K^2C_{in}+C_{in}C_{out}
$$

相比标准卷积：

$$
K^2C_{in}C_{out}
$$

参数和计算量明显减少。

### MobileNet 结论

> MobileNet 使用深度可分离卷积减少模型参数，适合移动端、嵌入式和实时视觉任务。

---

# 11. CNN 特征解释

PPT 后半部分讨论：

> CNN 中间层到底学到了什么？

---

## 11.1 第一层卷积核可视化

第一层 filters 通常学习到：

- 边缘；
- 颜色对比；
- 方向纹理；
- 简单局部模式。

这说明 CNN 的浅层特征类似传统图像处理中人工设计的边缘检测器，但 CNN 的滤波器是通过训练自动学习出来的。

---

## 11.2 中间层与深层特征

PPT 总结：

> 浅层学习通用低级模式，例如边缘；深层学习与物体相关的高级模式。

可以这样理解：

| 网络层级 | 学到的内容 |
|---|---|
| 浅层 | 边缘、颜色、纹理 |
| 中层 | 局部形状、部件 |
| 深层 | 物体语义、高级类别特征 |

---

## 11.3 特征空间最近邻

PPT 比较了：

- 像素空间最近邻；
- CNN 特征空间最近邻。

像素空间直接比较 RGB 值，容易受光照、背景、位置影响。

CNN 特征空间比较的是深层语义特征，因此更能反映图像内容是否相似。

---

## 11.4 可视化激活图

如果某一层 feature map 是：

$$
128 \times 13 \times 13
$$

可以看作 128 张 $13 \times 13$ 的灰度图。

每个通道代表一种特征响应：

- 亮的地方表示该特征被强烈激活；
- 暗的地方表示该特征响应弱。

---

## 11.5 最大激活图像块

方法：

1. 选定某一层某一个通道；
2. 输入大量图像；
3. 记录该通道响应最大的区域；
4. 可视化这些图像 patch。

作用：

> 判断某个通道到底在检测什么模式。

例如某通道总是被“狗脸”区域激活，就说明该通道可能对狗脸结构敏感。

---

# 12. Saliency 显著性分析

显著性分析研究：

> 图像中哪些像素或区域对分类结果最重要？

---

## 12.1 Occlusion 遮挡法

方法：

1. 输入原图，得到类别概率；
2. 用小方块遮挡图像某一区域；
3. 再输入网络；
4. 观察目标类别概率下降多少。

如果遮挡某一区域后类别概率明显下降，说明该区域对分类很重要。

优点：

- 直观；
- 容易理解。

缺点：

- 计算慢，需要多次前向传播。

---

## 12.2 Backprop 反向传播法

方法：

1. 前向传播得到类别分数；
2. 选择某个类别；
3. 计算该类别分数对输入像素的梯度：

$$
\frac{\partial s_c}{\partial I}
$$

4. 梯度绝对值越大，说明该像素对分类影响越大。

得到的图称为 saliency map。

---

## 12.3 Saliency Map 用于弱监督分割

PPT 中提到，可以用 saliency map 再结合 GrabCut 进行无监督或弱监督分割。

核心思路：

> 分类网络虽然只学习图像级类别，但它内部仍然可能学到物体大致位置。

---

# 13. 本章公式汇总

## 13.1 卷积输出尺寸

$$
W'=\frac{W-F+2P}{S}+1
$$

## 13.2 卷积参数量

$$
F \times F \times C_{in} \times C_{out}
$$

## 13.3 Softmax

$$
P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_j e^{s_j}}
$$

## 13.4 交叉熵损失

$$
L_i=-\sum_j y_j\log p_j
$$

one-hot 情况下等价于：

$$
L_i=-\log(\text{真实类别概率})
$$

## 13.5 ResNet 残差连接

$$
H(x)=F(x)+x
$$

## 13.6 标准卷积参数量

$$
K^2C_{in}C_{out}
$$

## 13.7 深度可分离卷积参数量

$$
K^2C_{in}+C_{in}C_{out}
$$

---

# 14. 重点网络对比表

| 网络 | 核心思想 | 主要优点 | 主要问题/特点 |
|---|---|---|---|
| AlexNet | 大卷积核 + 多层 CNN + FC 分类 | 首个 CNN ImageNet 突破 | 全连接层参数多 |
| VGG | 全部用 $3 \times 3$ 小卷积 | 结构简单，表达能力强 | 参数和计算量大 |
| GoogLeNet | Inception 多分支结构 | 多尺度特征，参数少 | 结构复杂 |
| ResNet | 残差连接 | 可训练超深网络，精度高 | 结构更深 |
| MobileNet | 深度可分离卷积 | 参数少，计算量低 | 适合移动端，精度可能略低 |

---

# 15. 考试/复习重点

这章最可能考这些：

1. **Softmax 的作用**  
   把 logits 转换成概率，适合类别互斥分类。

2. **交叉熵损失的意义**  
   惩罚真实类别概率低的情况。

3. **卷积输出尺寸计算**  
   必背：

   $$
   W'=\frac{W-F+2P}{S}+1
   $$

4. **卷积参数量计算**  
   必背：

   $$
   F^2C_{in}C_{out}
   $$

5. **AlexNet 第一层计算**  
   输入 $227 \times 227 \times 3$，卷积核 $11 \times 11$，stride = 4，输出：

   $$
   55 \times 55 \times 96
   $$

   参数量：

   $$
   11 \times 11 \times 3 \times 96 \approx 35K
   $$

6. **VGG 为什么用小卷积核**  
   三个 $3 \times 3$ 等效一个 $7 \times 7$，参数更少、非线性更多。

7. **GoogLeNet 的 Inception 模块**  
   并行使用 $1 \times 1$、$3 \times 3$、$5 \times 5$、pooling 提取多尺度特征。

8. **$1 \times 1$ 卷积作用**  
   降维、减少参数、混合通道、增加非线性。

9. **ResNet 残差连接**  
   $$
   F(x)+x
   $$

   解决深层网络训练困难。

10. **MobileNet 深度可分离卷积**  
    depthwise 负责空间卷积，pointwise 负责通道混合，大幅减少参数。

11. **CNN 特征解释**  
    浅层学边缘纹理，深层学物体语义。

12. **Saliency 方法**  
    Occlusion 看遮挡后概率变化；Backprop 看类别分数对输入像素的梯度。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)