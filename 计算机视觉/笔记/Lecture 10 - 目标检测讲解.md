# 目标检测讲解

**User:** Anonymous  
**Created:** 5/23/2026 23:59:11  
**Updated:** 5/24/2026 18:05:36  
**Exported:** 5/24/2026 18:11:26  
**Link:** [https://chatgpt.com/g/g-p-69ca8f8ab6e881918bdad9b692c413de-ji-suan-ji-shi-jue-xue-xi/c/6a11ceb6-1394-83ec-b909-5898d671f0e2](https://chatgpt.com/g/g-p-69ca8f8ab6e881918bdad9b692c413de-ji-suan-ji-shi-jue-xue-xi/c/6a11ceb6-1394-83ec-b909-5898d671f0e2)  

## Response:
2026/5/24 18:03:46

> **检查PDF文件存在与否**
>
>
> **提取 PDF 文本并检查**
>
>
> **提取PDF文件文本内容**
>
>
> **提取PDF文本指定页码**
>
>
> **提取并检查PDF文本**
>
>
> **整理讲义并考虑引用**
>
> 我需要制作一些讲义笔记。可能要考虑引用，但不一定需要。看起来一些上传的旧文件已过期，不过没影响当前的Lecture10。我可以告诉用户：“这份Lecture10 还能读取，之前有些旧文件过期，但不影响本章内容。”不必特别引用，除非用户需要。
>
> **整理讲义笔记并考虑结构**
>
> 我需要制作简洁而详细的笔记，结构上按PPT顺序，包含AP/mAP修正。或许可以加一些“可直接背”的内容。文件不必制作除非用户要求。可能会加上页码，文本里有幻灯片页数。至于“旧文件过期”，可以提一下，但最好放在脚注里，不要干扰主要内容。
>
> **总结目标检测相关公式**
>
> 我需要覆盖一些关于目标检测的基本内容：分类与检测的区别、BoF（可能缺少空间布局）。关于模板匹配，要包括位置、尺度、方向对齐的差异。还要讲解模型方法、假设生成、评分、以及NMS和IoU等内容，解释TP/FP/FN标准。
>
> 已思考 46s

我会把这章整理成考试复习笔记，不再按“讲课口吻”展开太多，而是按 PPT 顺序保留定义、流程、公式、易考点和容易混的 AP/mAP。

下面给你整理成**考试复习笔记版**，按 PPT 顺序来。重点我会把**定义、流程、公式、易考点**放进去，AP/mAP 按刚才纠正后的说法写。

---

# 第十讲 目标检测 Object Detection 笔记

## 一、目标检测基础

### 1. 什么是目标检测

**目标检测 Object Detection** 的任务是：

> 在图像中找到目标物体实例，并给出它们的边界框。

输出通常包括：

$$
\text{类别} + \text{边界框} + \text{置信度}
$$

边界框一般表示为：

$$
(x,y,w,h)
$$

其中：

- $(x,y)$：边界框位置；
- $w$：宽度；
- $h$：高度。

---

### 2. 图像分类 vs 目标检测

#### 图像分类

图像分类只判断整张图是什么：

$$
\text{image} \rightarrow \text{class}
$$

例如：

> 这张图是 dog。

分类任务中，词袋模型 Bag of Features 可以使用，因为它主要关心图像中有哪些局部特征。

---

#### 目标检测

目标检测不仅要判断是什么，还要判断在哪里：

$$
\text{image} \rightarrow \text{class + bounding box}
$$

例如：

> 图中左下角有一只 dog。

目标检测中，Bag of Features 不太适合，因为它会丢失空间布局信息。

重点记：

> 图像分类关注“是什么”，目标检测关注“是什么 + 在哪里”。

---

## 二、模板匹配 Template Matching

PPT 先从最基础的目标检测方法讲起：**模板匹配**。

基本思想：

> 给定一个目标模板，在图像中不断移动模板，计算模板和图像局部区域的相似度。

流程：

1. **Align 空间对齐**  
   选择位置、尺度、方向。

2. **Compare 计算相似性**  
   比较候选区域和模板是否相似。

问题：

- 物体可能出现在不同位置；
- 物体可能有不同尺度；
- 物体可能旋转；
- 物体外观可能变化。

所以模板匹配是目标检测的最简单思想，但实际应用中要进一步改进。

---

# 三、目标检测的一般流程

PPT 给出的目标检测基本流程是四步：

$$
\text{Specify Object Model}
\rightarrow
\text{Generate Hypotheses}
\rightarrow
\text{Score Hypotheses}
\rightarrow
\text{Resolve Detections}
$$

中文：

1. **指定目标模型**
2. **产生假设**
3. **假设打分**
4. **检测解析**

可以理解成：

> 先定义目标长什么样，再提出一堆候选框，然后判断每个候选框像不像目标，最后去掉重复框。

---

## 1. 指定目标模型 Specify Object Model

目标模型就是：

> 用什么方式表示目标物体。

PPT 讲了四种。

---

### 1.1 统计模板模型 Statistical Template

目标用一个边界框表示：

$$
(x,y,w,h)
$$

在边界框内部提取特征，比如 SIFT、HOG 等。

特点：

- 适合形状比较固定的目标；
- 例如人脸、汽车、行人；
- 特征是相对于边界框坐标定义的。

---

### 1.2 关节部件模型 Articulated Parts Model

目标由多个部件组成：

$$
\text{object} = \text{parts} + \text{spatial configuration}
$$

例如人体可以分成：

- 头；
- 躯干；
- 手臂；
- 腿。

适合有变形的目标，比如人。

---

### 1.3 混合模板/部件模型 Hybrid Template/Parts Model

结合整体模板和局部部件：

- 粗尺度上看整体；
- 细尺度上看局部部件。

优点：

> 既保留整体结构，又允许局部变化。

---

### 1.4 可形变三维模型 Deformable 3D Model

用三维形状、姿态和变形参数来表示目标。

PPT 例子是 SMPL 人体模型。

特点：

- 更复杂；
- 能描述 3D 姿态和人体形变；
- 不只是二维框检测。

---

## 2. 产生假设 Generate Hypotheses

产生假设就是：

> 提出图像中可能包含目标的候选区域。

PPT 讲了两种方法。

---

### 2.1 滑动窗口 Sliding Window

做法：

> 用一个固定大小的窗口，在图像上从左到右、从上到下滑动，每个位置都判断一次。

如果目标大小不同，就用**图像金字塔**。

图像金字塔：

$$
\text{原图} \rightarrow \text{多尺度图像}
$$

然后在不同尺度图像上滑动窗口。

优点：

- 思路简单；
- 不容易漏掉位置。

缺点：

- 计算量大；
- 候选窗口非常多。

---

### 2.2 区域提议 Region Proposal

区域提议不是暴力遍历所有窗口，而是根据图像内容提出候选区域。

PPT 讲的是：

> Selective Search by Hierarchical Grouping

基本思想：

1. 先把图像分割成很多小区域；
2. 根据颜色、纹理、大小等相似性进行合并；
3. 产生不同尺度的候选框。

优点：

> 比滑动窗口更有针对性，候选区域更少。

---

## 3. 假设打分 Score Hypotheses

对每个候选区域判断它是不是目标。

一般流程：

1. 对候选区域提取特征；
2. 输入分类器；
3. 得到一个分数。

例如：

$$
score = w^T x + b
$$

其中：

- $x$：候选区域特征；
- $w,b$：分类器参数；
- score 越大，越可能是目标。

---

## 4. 检测解析 Resolve Detections

检测解析主要解决一个问题：

> 同一个目标可能被多个候选框同时检测出来。

所以需要 **NMS，非最大值抑制**。

---

# 四、IoU 与 NMS

## 1. IoU 交并比

IoU 全称：

$$
Intersection\ over\ Union
$$

公式：

$$
IoU = \frac{\text{预测框与真实框的交集面积}}{\text{预测框与真实框的并集面积}}
$$

即：

$$
IoU = \frac{Area(B_p \cap B_{gt})}{Area(B_p \cup B_{gt})}
$$

其中：

- $B_p$：预测框；
- $B_{gt}$：真实框。

IoU 越大，说明预测框和真实框重合越好。

---

## 2. NMS 非最大值抑制

NMS 用来删除重复检测框。

步骤：

1. 按置信度从高到低排序；
2. 选择分数最高的框；
3. 删除与它 IoU 大于阈值的其他框；
4. 对剩下的框重复这个过程。

作用：

> 保留最可靠的检测框，删除重复框。

---

# 五、检测结果评价

目标检测评价不只看类别对不对，还要看框准不准。

## 1. TP、FP、FN

### True Positive，真阳性

预测有目标，而且预测正确。

条件通常是：

$$
IoU \geq t
$$

例如 $t=0.5$。

---

### False Positive，假阳性

预测有目标，但预测错误。

常见情况：

- 框的位置不准，IoU 太低；
- 把背景误检为目标；
- 同一个目标重复检测出多个框。

---

### False Negative，假阴性

真实有目标，但模型没有检测出来。

也就是漏检。

---

## 2. Precision 精确率

Precision 回答的问题：

> 模型预测出来的目标中，有多少是真的？

公式：

$$
Precision = \frac{TP}{TP+FP}
$$

Precision 高，说明误检少。

---

## 3. Recall 召回率

Recall 回答的问题：

> 真实存在的目标中，有多少被检测出来了？

公式：

$$
Recall = \frac{TP}{TP+FN}
$$

Recall 高，说明漏检少。

---

# 六、AP 和 mAP

这一块很容易混，按这个记。

## 1. AP 是什么

AP 全称：

$$
Average\ Precision
$$

它不是简单等同于 AUC。

更严谨地说：

> AP 是在固定类别、固定 IoU 阈值下，改变分类置信度阈值，得到一系列 Precision-Recall 点后，对不同 Recall 水平下 Precision 的加权平均。

记成：

$$
AP_{c,\tau}
$$

表示：

> 类别 $c$，IoU 阈值为 $\tau$ 时的 AP。

公式可以写成：

$$
AP = \sum_n (R_n - R_{n-1})P_n
$$

其中：

- $R_n$：第 $n$ 个点的 Recall；
- $P_n$：对应的 Precision；
- $R_n - R_{n-1}$：Recall 增量。

---

## 2. AP 和 AUC 的区别

| 指标 | 含义 |
|---|---|
| AUC | 曲线下面积 |
| AP | 不同 Recall 水平下 Precision 的加权平均 |
| PR-AUC | PR 曲线下面积 |
| ROC-AUC | ROC 曲线下面积 |

重点记：

> AUC 强调“面积”，AP 强调“平均 Precision”。

AP 和 PR 曲线有关，但考试不要简单写成“AP 就是曲线下面积”。

---

## 3. mAP 是什么

mAP 全称：

$$
mean\ Average\ Precision
$$

如果只看一个类别、一个 IoU 阈值，就是 AP。

如果考虑多个 IoU 阈值，比如：

$$
0.5,\ 0.55,\ 0.6,\dots,\ 0.95
$$

那么对这些 IoU 阈值下的 AP 取平均：

$$
mAP_c = \frac{1}{T}\sum_{\tau} AP_{c,\tau}
$$

如果是多类别检测，还要对类别再平均：

$$
mAP = \frac{1}{C}\sum_{c=1}^{C}mAP_c
$$

展开就是：

$$
mAP =
\frac{1}{C \times T}
\sum_{c=1}^{C}
\sum_{\tau} AP_{c,\tau}
$$

一句话：

> AP 是固定类别、固定 IoU 下算出来的平均精度；mAP 是多个类别、多个 IoU 阈值下 AP 的平均。

---

# 七、行人检测：Dalal-Triggs 方法

PPT 第二部分讲经典行人检测方法：

$$
HOG + Linear\ SVM
$$

论文：

> Dalal and Triggs, Histograms of Oriented Gradients for Human Detection, CVPR 2005

---

## 1. Dalal-Triggs 行人检测流程

PPT 给了四步：

1. 在每个位置和尺度提取固定大小窗口；
2. 在窗口内计算 HOG 特征；
3. 用线性 SVM 给窗口打分；
4. 用 NMS 删除重复检测框。

经典检测窗口大小：

$$
64 \times 128
$$

因为行人通常是竖直的，宽高比接近 $1:2$。

---

## 2. 预处理 Preprocess

PPT 中提到：

输入可以是：

- RGB；
- LAB；
- Grayscale。

颜色空间对结果影响不大。

还可以做 Gamma Normalization，比如：

- square root；
- log。

作用：

> 减弱光照变化的影响。

---

# 八、HOG 特征

HOG 全称：

$$
Histogram\ of\ Oriented\ Gradients
$$

中文：

> 方向梯度直方图。

核心思想：

> 用局部区域中的梯度方向分布描述物体形状。

行人的头、肩、腿等轮廓会产生稳定的边缘方向，因此 HOG 适合行人检测。

---

## 1. 第一步：计算梯度

对图像求水平和竖直方向梯度：

$$
G_x,\ G_y
$$

梯度幅值：

$$
m = \sqrt{G_x^2 + G_y^2}
$$

梯度方向：

$$
\theta = \arctan \frac{G_y}{G_x}
$$

其中：

- $m$：边缘强度；
- $\theta$：边缘方向。

---

## 2. 第二步：划分 cell

经典设置：

- 检测窗口：

$$
64 \times 128
$$

- cell 大小：

$$
8 \times 8
$$

所以 cell 数量：

水平方向：

$$
64 / 8 = 8
$$

竖直方向：

$$
128 / 8 = 16
$$

总 cell 数：

$$
8 \times 16
$$

每个 cell 统计一个方向直方图。

如果方向 bin 数为 9，则每个 cell 是 9 维特征。

---

## 3. 第三步：方向直方图

每个像素根据自己的梯度方向投票到对应方向 bin。

投票权重通常是梯度幅值。

例如 9 个方向 bin 可以表示：

$$
0^\circ,\ 20^\circ,\ 40^\circ,\dots,\ 160^\circ
$$

---

## 4. 第四步：block 归一化

为了减小光照影响，需要对局部 block 进行归一化。

PPT 中：

- block 大小：

$$
16 \times 16
$$

- cell 大小：

$$
8 \times 8
$$

所以一个 block 包含：

$$
2 \times 2 = 4
$$

个 cell。

每个 cell 是 9 维，所以一个 block 是：

$$
4 \times 9 = 36
$$

维。

归一化作用：

> 让 HOG 特征对亮度和对比度变化不敏感。

---

## 5. HOG 特征维度计算

这是本章最容易考计算的地方。

已知：

- 检测窗口：

$$
64 \times 128
$$

- cell：

$$
8 \times 8
$$

- cell 数：

$$
8 \times 16
$$

- block：

$$
2 \times 2\ \text{cells}
$$

block 采用重叠滑动，步长是一个 cell。

水平方向 block 数：

$$
8 - 2 + 1 = 7
$$

竖直方向 block 数：

$$
16 - 2 + 1 = 15
$$

总 block 数：

$$
7 \times 15
$$

每个 block 维度：

$$
2 \times 2 \times 9 = 36
$$

所以整个窗口 HOG 特征维度：

$$
7 \times 15 \times 36 = 3780
$$

也就是：

$$
7 \times 15 \times 9 \times 4 = 3780
$$

这个一定要会。

---

# 九、HOG + SVM 分类

HOG 提取完后，一个 $64 \times 128$ 的窗口会变成一个 3780 维特征向量。

然后输入线性 SVM：

$$
f(x) = w^T x + b
$$

如果：

$$
f(x) > 0
$$

判断为行人。

如果：

$$
f(x) < 0
$$

判断为非行人。

线性 SVM 的作用：

> 学习一个超平面，把行人窗口和非行人窗口分开。

---

## 统计模板方法优缺点

### 优点

- 对非变形物体效果好；
- 适合固定姿态目标；
- 比如人脸、汽车、行人；
- 检测速度较快。

### 缺点

- 不适合高度可变形物体；
- 对遮挡不鲁棒；
- 需要大量训练数据。

---

# 十、人脸检测 Face Detection

PPT 第三部分讲 Viola-Jones 人脸检测。

基本任务：

> 把滑动窗口分类为 face 或 not face。

多尺度检测效果更好，因为人脸可能大小不同。

---

## 1. 人脸检测的挑战

滑动窗口会产生大量候选窗口。

一张百万像素图像，可能有接近：

$$
10^6
$$

个候选位置。

但真实人脸很少，通常：

$$
0 \sim 10
$$

个。

所以问题是：

1. 非人脸窗口极多；
2. 必须尽快排除非人脸窗口；
3. 假阳性率必须非常低。

PPT 中说，如果一张图有 $10^6$ 个候选窗口，为了避免每张图都有误检，假阳性率需要小于：

$$
10^{-6}
$$

---

# 十一、Viola-Jones 人脸检测器

Viola-Jones 是经典实时人脸检测方法。

特点：

> 训练慢，但检测非常快。

核心思想有三个：

1. **Integral Images 积分图**  
   快速计算矩形特征。

2. **Boosting 特征选择**  
   从大量特征中选择少量有效特征。

3. **Attentional Cascade 级联分类器**  
   快速拒绝大量非人脸窗口。

---

# 十二、积分图 Integral Image

## 1. 积分图定义

积分图中每个位置 $(x,y)$ 的值表示：

> 原图中该点左上方所有像素的灰度和。

记作：

$$
ii(x,y)
$$

---

## 2. 积分图计算

PPT 给出两个公式。

先计算行累计和：

$$
s(x,y) = s(x-1,y) + i(x,y)
$$

再计算积分图：

$$
ii(x,y) = ii(x,y-1) + s(x,y)
$$

其中：

- $i(x,y)$：原图像素值；
- $s(x,y)$：当前行的累计和；
- $ii(x,y)$：积分图值。

可以理解成：

> 积分图就是二维前缀和。

---

## 3. 用积分图计算矩形区域和

设矩形四个角的积分图值为：

- A：右下角；
- B：右上角；
- C：左下角；
- D：左上角。

则矩形区域像素和为：

$$
sum = A - B - C + D
$$

重点：

> 任意大小的矩形区域，只需要三次加减运算。

这就是积分图快的原因。

---

# 十三、Haar-like 特征

Viola-Jones 使用 Haar-like features。

基本形式：

$$
feature = \sum \text{white area} - \sum \text{black area}
$$

也就是：

> 比较相邻矩形区域的灰度差异。

它可以捕捉人脸中的明暗结构，例如：

- 眼睛通常比脸颊暗；
- 鼻梁和两侧有亮暗差异；
- 嘴部区域有明显边缘。

由于 Haar 特征都是矩形区域灰度和的差，所以可以用积分图快速计算。

---

## Haar 特征数量问题

PPT 中说，对于一个：

$$
24 \times 24
$$

的检测窗口，可能的矩形特征数量大约有：

$$
160000
$$

个。

测试时不可能全部计算。

所以需要：

> 从大量特征中选出少量最有用的特征。

这就引出 Boosting。

---

# 十四、Boosting 特征选择

## 1. AdaBoost 基本思想

AdaBoost 用多个弱分类器组合成强分类器。

弱分类器：

> 只根据一个 Haar 特征和一个阈值判断是不是人脸。

最终强分类器：

$$
f(x)=\sum_{m=1}^{M}\alpha_m G_m(x)
$$

其中：

- $G_m(x)$：第 $m$ 个弱分类器；
- $\alpha_m$：弱分类器权重。

注意：PPT 上写“系数 $\alpha_m$ 与精度成反比”这里容易有问题。  
标准 AdaBoost 中，弱分类器精度越高，错误率越低，$\alpha_m$ 越大。

所以考试如果不展开标准公式，可以写：

> 最终分类器由多个弱分类器加权组合而成，分类能力更强的弱分类器权重更大。

---

## 2. 样本权重初始化

初始时，所有样本权重相同：

$$
w_{1i} = \frac{1}{N}
$$

其中：

- $N$：训练样本总数；
- $w_{ki}$：第 $k$ 轮中第 $i$ 个样本的权重。

---

## 3. AdaBoost 每轮过程

每一轮：

1. 根据当前样本权重训练一个弱分类器；
2. 选择加权错误率最小的特征和阈值；
3. 提高被错分样本的权重；
4. 下一轮更关注难分类样本。

重点理解：

> AdaBoost 会把注意力逐渐集中到之前分错的样本上。

---

## 4. Boosting 为什么能选特征

在 Viola-Jones 中，每个弱分类器对应：

$$
\text{一个 Haar 特征} + \text{一个阈值}
$$

AdaBoost 每一轮都从大量 Haar 特征中选择一个当前最有用的特征。

所以它同时完成：

1. 分类器训练；
2. 特征选择。

---

# 十五、级联分类器 Cascade

## 1. 为什么需要级联

大多数滑动窗口都是非人脸。

如果每个窗口都用复杂分类器，会非常慢。

所以 Viola-Jones 使用级联结构：

$$
Classifier_1
\rightarrow
Classifier_2
\rightarrow
Classifier_3
\rightarrow
\dots
$$

一个窗口必须通过所有分类器，才被认为是人脸。

只要在某一级被判断为非人脸，就立即丢弃。

---

## 2. 级联分类器特点

前面的分类器：

- 简单；
- 计算快；
- 用来快速排除大量非人脸。

后面的分类器：

- 更复杂；
- 判断更精细；
- 只有少量候选窗口会进入后面。

一句话：

> 级联分类器用简单分类器先筛掉大量负样本，把计算量留给少数可疑窗口。

---

## 3. Viola-Jones 细节

PPT 中给了一些数字：

- 38 个级联阶段；
- 总共使用 6061 个特征；
- 候选特征约 180K；
- 测试时每个窗口平均只评估约 10 个特征；
- 训练样本包括 4916 个正样本；
- 每阶段收集 10000 个负样本；
- 尺度步长为 1.25；
- 对重叠框做 NMS。

这些数字不一定都要背，但可以记一个核心：

> 候选特征很多，但实际检测时平均只用很少特征，所以检测很快。

---

# 十六、本章核心流程总结

## 1. 目标检测通用流程

$$
\text{目标模型}
\rightarrow
\text{产生候选区域}
\rightarrow
\text{候选区域打分}
\rightarrow
\text{NMS解析检测结果}
$$

---

## 2. Dalal-Triggs 行人检测流程

$$
64 \times 128\ \text{窗口}
\rightarrow
HOG
\rightarrow
Linear\ SVM
\rightarrow
NMS
$$

重点：

$$
7 \times 15 \times 9 \times 4 = 3780
$$

---

## 3. Viola-Jones 人脸检测流程

$$
\text{滑动窗口}
\rightarrow
\text{Haar特征}
\rightarrow
\text{积分图快速计算}
\rightarrow
\text{AdaBoost选特征}
\rightarrow
\text{级联分类器快速拒绝非人脸}
$$

---

# 十七、考试重点整理

## 必背概念

1. 目标检测和图像分类区别；
2. 目标检测四步流程；
3. IoU 定义；
4. NMS 作用和步骤；
5. Precision 和 Recall；
6. AP 和 mAP 的区别；
7. HOG 特征思想；
8. 积分图思想；
9. Haar-like 特征；
10. AdaBoost 思想；
11. 级联分类器为什么快。

---

## 必会公式

### IoU

$$
IoU = \frac{Area(B_p \cap B_{gt})}{Area(B_p \cup B_{gt})}
$$

### Precision

$$
Precision = \frac{TP}{TP+FP}
$$

### Recall

$$
Recall = \frac{TP}{TP+FN}
$$

### AP

$$
AP = \sum_n (R_n - R_{n-1})P_n
$$

### mAP

$$
mAP =
\frac{1}{C \times T}
\sum_{c=1}^{C}
\sum_{\tau} AP_{c,\tau}
$$

### HOG 维度

$$
7 \times 15 \times 9 \times 4 = 3780
$$

### 积分图行累计和

$$
s(x,y) = s(x-1,y) + i(x,y)
$$

### 积分图

$$
ii(x,y) = ii(x,y-1) + s(x,y)
$$

### 矩形区域和

$$
sum = A - B - C + D
$$

---

# 十八、易混点

## 1. AP 不是简单等于 AUC

正确理解：

> AP 是 Precision 在不同 Recall 水平下的加权平均。

AUC 才是严格的曲线下面积。

---

## 2. AP 和 mAP 的区别

$$
AP_{c,\tau}
$$

表示某一类别、某一 IoU 阈值下的 AP。

mAP 是多个 AP 的平均：

- 可以对多个 IoU 阈值平均；
- 可以对多个类别平均。

---

## 3. HOG 中 cell 和 block 不一样

cell 是统计梯度方向直方图的基本单位。

block 是多个 cell 组成的归一化区域。

本章经典设置：

$$
cell = 8 \times 8
$$

$$
block = 2 \times 2\ cells
$$

---

## 4. Boosting 和 Cascade 作用不一样

Boosting：

> 从大量 Haar 特征中选出少量有用特征，并组合弱分类器。

Cascade：

> 快速排除大量非人脸窗口，提高检测速度。

---

# 十九、一句话总复习

这章主要讲传统目标检测方法：

> 目标检测先产生候选框，再对候选框分类打分，最后用 NMS 去重；行人检测的经典方法是 HOG + SVM，人脸检测的经典方法是 Viola-Jones，其中 Viola-Jones 依靠积分图快速计算 Haar 特征，用 AdaBoost 选择特征，并通过级联分类器快速排除非人脸窗口。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)