# 目标检测概述

**User:** Anonymous  
**Created:** 5/24/2026 21:20:46  
**Updated:** 5/24/2026 22:29:31  
**Exported:** 5/25/2026 9:30:44  
**Link:** [https://chatgpt.com/g/g-p-69ca8f8ab6e881918bdad9b692c413de-ji-suan-ji-shi-jue-xue-xi/c/6a12fb14-ade4-83ec-a2ca-e2c609f86a32](https://chatgpt.com/g/g-p-69ca8f8ab6e881918bdad9b692c413de-ji-suan-ji-shi-jue-xue-xi/c/6a12fb14-ade4-83ec-a2ca-e2c609f86a32)  

## Response:
2026/5/24 22:28:23

> **整理并总结笔记要点**
>
> 用户要求“总结这一章的笔记”，这意味着需要精简而完整的信息，可能会用于考试。要包括章节标题、关键公式、算法、比较表等。确保引用资料正确，避免过度引用。应重点提及“YOLO输出维度”及相关公式，按需求简洁总结，不掺杂过多细节。
>
> 已思考 21s

# 第十三讲：基于 CNN 的目标检测--复习笔记

## 0. 本章主线

这一章核心就是一句话：

> **目标检测 = 分类 Classification + 定位 Localization**

分类只回答：图里是什么？  
目标检测要回答：**图里有哪些物体？每个物体是什么？在哪里？**

本章方法发展路线：

**滑动窗口 → Region Proposal → R-CNN → Fast R-CNN → Faster R-CNN → YOLO / SSD 单阶段检测器**。`Lecture13_CNN_image_detection.pdf`

---

## 1. 图像识别任务的层次

| 任务 | 输出 | 特点 |
|---|---|---|
| 分类 Classification | 整张图一个类别 | 没有空间位置 |
| 语义分割 Semantic Segmentation | 每个像素的类别 | 不区分同类不同个体 |
| 目标检测 Object Detection | 类别 + 边界框 | 能检测多个物体 |
| 实例分割 Instance Segmentation | 类别 + 每个实例的像素区域 | 检测 + 精细分割 |

目标检测输出一般是：

$$
类别 + (x, y, w, h)
$$

其中 $(x,y)$ 表示框的位置，$(w,h)$ 表示框的宽和高。

---

## 2. 单目标检测：分类 + 边框回归

如果图像中只有一个目标，可以让 CNN 输出两部分：

一部分输出类别分数：

$$
Cat:0.9,\ Dog:0.05,\ Car:0.01
$$

另一部分输出边界框坐标：

$$
(x,y,w,h)
$$

训练时总损失为：

$$
L = L_{cls} + L_{bbox}
$$

其中：

| 损失 | 作用 |
|---|---|
| 分类损失 | 判断类别对不对 |
| 边框回归损失 | 判断预测框位置准不准 |

所以 PPT 说：**将定位问题视作回归问题**。

---

## 3. 多目标检测的难点

单目标检测比较简单，但多目标检测难在：

> 每张图中的物体数量不固定。

比如一张图有 3 个目标，另一张图有 7 个目标，如果直接用固定数量的全连接节点输出，就很难处理。

所以目标检测不能简单地设计成固定输出，而要想办法产生一批候选框，再判断这些框里有没有物体。

---

## 4. 滑动窗口 Sliding Window

最朴素的方法是滑动窗口：

1. 在图像上裁剪很多窗口；
2. 每个窗口送入 CNN；
3. 判断这个窗口是狗、猫，还是背景。

优点：思路简单。

缺点：计算量太大。因为要考虑：

- 不同位置；
- 不同尺度；
- 不同长宽比。

也就是说，窗口数量会非常多，每个窗口都跑一次 CNN，速度很慢。PPT 明确指出该方法需要在大量位置、尺度和长宽比上应用 CNN，计算代价很高。`Lecture13_CNN_image_detection.pdf`

---

## 5. Region Proposal 区域提议

为了解决滑动窗口太慢的问题，引入 **Region Proposal**。

核心思想：

> 不再枚举所有窗口，而是先找出一批“可能包含物体”的候选区域。

常见方法：

| 方法 | 含义 |
|---|---|
| SLIC Super-pixel | 先把图像分成超像素 |
| Selective Search | 通过分层聚类合并区域，产生不同尺度的候选框 |

Selective Search 可以在 CPU 上较快地产生约 2000 个候选区域。`Lecture13_CNN_image_detection.pdf`

---

## 6. R-CNN

R-CNN 是第一个重要方法。

流程：

1. 输入图像；
2. 用 Selective Search 得到约 2000 个候选区域；
3. 每个候选区域 resize 到固定大小，比如 $224 \times 224$；
4. 每个区域单独送入 CNN 提取特征；
5. 用 SVM 分类；
6. 用 Bounding Box Regression 修正边界框。

优点：  
CNN 特征比传统人工特征强，检测精度提高。

缺点：  
非常慢。因为每张图大约有 2000 个 RoI，每个 RoI 都要单独跑一次 CNN，很多重叠区域会被重复计算。`Lecture13_CNN_image_detection.pdf`

记法：

> **R-CNN = Region Proposal + CNN 特征提取 + SVM 分类 + BBox 回归**

---

## 7. Fast R-CNN

Fast R-CNN 解决 R-CNN 的重复计算问题。

核心改进：

> 先对整张图跑一次 CNN，再在特征图上裁剪 RoI。

流程：

1. 输入整张图；
2. backbone CNN 提取整图特征图；
3. 将 proposal 映射到 feature map 上；
4. 用 ROI Pool 得到固定大小的区域特征；
5. 后面接分类分支和边框回归分支。

和 R-CNN 相比：

| 方法 | CNN 计算方式 |
|---|---|
| R-CNN | 每个候选框单独跑 CNN |
| Fast R-CNN | 整张图只跑一次 CNN |

所以 Fast R-CNN 快很多。

---

## 8. ROI Pool 和 ROI Align

### 8.1 ROI Pool

ROI Pool 的作用：

> 把任意大小的 RoI 区域变成固定大小的特征表示。

步骤：

1. 把原图 proposal 映射到 feature map；
2. 坐标取整；
3. 分成固定网格，比如 $7 \times 7$；
4. 每个小格做 max pooling；
5. 得到固定大小的特征。

问题：

> 坐标取整会导致空间错位。

### 8.2 ROI Align

ROI Align 的改进：

- 不进行坐标取整；
- 使用双线性插值；
- 保持更准确的空间对齐关系。

所以：

| 方法 | 坐标处理 | 问题/优点 |
|---|---|---|
| ROI Pool | 坐标取整 | 可能错位 |
| ROI Align | 不取整，双线性插值 | 空间对齐更好 |

PPT 中也强调 ROI Align 可以保持空间对齐关系。`Lecture13_CNN_image_detection.pdf`

---

## 9. Fast R-CNN 的瓶颈

Fast R-CNN 虽然减少了 CNN 重复计算，但还有一个问题：

> Region Proposal 仍然由 Selective Search 生成，而这个过程主要在 CPU 上完成，耗时较大。

所以接下来 Faster R-CNN 的核心就是：

> 让 CNN 自己生成候选框。

---

## 10. Faster R-CNN

Faster R-CNN 的核心模块是 **RPN：Region Proposal Network**。

它的思想是：

> 在 CNN 的 feature map 上直接预测候选框。

整体流程：

1. 输入图像；
2. CNN backbone 提取 feature map；
3. RPN 在 feature map 上生成 proposals；
4. 对 proposals 做 ROI Pool / ROI Align；
5. 最后分类 + 边框回归。

Faster R-CNN 仍然是 **two-stage detector，两阶段检测器**：

| 阶段 | 作用 |
|---|---|
| 第一阶段 RPN | 生成候选框 |
| 第二阶段检测头 | 对候选框分类，并精修边界框 |

PPT 明确总结 Faster R-CNN 是两阶段检测器：先获得 region proposal，再预测每个区域。`Lecture13_CNN_image_detection.pdf`

---

## 11. Anchor 机制

RPN 里面的重要概念是 **anchor box**。

在 feature map 的每个位置，都预设若干个 anchor。每个 anchor 有不同的：

- 尺度；
- 长宽比。

对于每个 anchor，RPN 要预测两件事：

1. 这个 anchor 里面有没有物体；
2. 这个 anchor 应该如何调整，才能更接近真实框。

如果每个位置有 $K$ 个 anchor，feature map 大小是 $20 \times 15$，那么：

| 输出 | 大小 |
|---|---|
| objectness 分数 | $K \times 20 \times 15$ |
| 边框修正量 | $4K \times 20 \times 15$ |

然后按照 objectness 分数排序，选取前约 300 个作为 proposals。`Lecture13_CNN_image_detection.pdf`

---

## 12. Faster R-CNN 的 4 个损失

Faster R-CNN 需要联合训练 4 个损失：

| 损失 | 含义 |
|---|---|
| RPN 分类损失 | anchor 是物体还是背景，二分类 |
| RPN 边框回归损失 | anchor 到真实框的偏移 |
| 最终分类损失 | 判断 proposal 属于哪个类别 |
| 最终边框回归损失 | 进一步修正检测框 |

所以 Faster R-CNN 的训练目标是：

$$
L = L_{rpn\_cls} + L_{rpn\_bbox} + L_{final\_cls} + L_{final\_bbox}
$$

---

## 13. 两阶段检测 vs 单阶段检测

| 类型 | 代表方法 | 思路 | 特点 |
|---|---|---|---|
| 两阶段检测 | R-CNN、Fast R-CNN、Faster R-CNN | 先生成候选框，再分类回归 | 精度高，但速度慢 |
| 单阶段检测 | YOLO、SSD、RetinaNet | 直接预测类别和边界框 | 速度快，适合实时检测 |

PPT 中也说明：两阶段检测精度高但速度慢，单阶段检测速度快，适合实时检测。`Lecture13_CNN_image_detection.pdf`

---

## 14. YOLO

YOLO 是典型的单阶段检测器。

核心思想：

> 把图像划分成网格，每个网格直接预测边界框和类别。

例如把图像划分为 $7 \times 7$ 个 grid cell。

每个 grid cell 预测：

1. $B$ 个边界框；
2. 每个框 5 个数：
   $$
   (dx, dy, dh, dw, confidence)
   $$
3. $C$ 个类别分数。

所以每个 grid cell 的输出维度是：

$$
5B + C
$$

整张图的输出大小是：

$$
7 \times 7 \times (5B + C)
$$

PPT 举例：

$$
B = 2,\quad C = 20
$$

所以每个格子的输出维度为：

$$
5 \times 2 + 20 = 30
$$

整张图完整输出应理解为：

$$
7 \times 7 \times 30
$$

PPT 第31页写“最终输出维度为30”，更准确地说是：**每个网格单元的输出维度为 30**。`Lecture13_CNN_image_detection.pdf`

---

## 15. SSD

SSD 全称是 **Single-Shot MultiBox Detector**。

核心特点：

1. **单阶段检测**：不单独生成 proposals；
2. **多尺度预测**：在不同层 feature map 上检测不同大小的目标；
3. **默认框 / 预设框**：在固定长宽比的预设框基础上预测偏移量。

SSD 的优势是速度较快，同时精度也不错。PPT 中提到 SSD 在 Pascal VOC2007 test 上 mAP 可达到 80% 以上，并且可以实时检测。`Lecture13_CNN_image_detection.pdf`

---

## 16. YOLO 版本发展

### YOLOv3

YOLOv3 使用 Darknet-53，主要改进：

- 更深的网络；
- 多尺度预测；
- 对不同尺度目标更友好。`Lecture13_CNN_image_detection.pdf`

### YOLOv8

PPT 中总结 YOLOv8 的特点：

- 新骨干网络，使用 C2f 结构；
- 解耦头，把分类和定位分开；
- 使用 TaskAlignedAssigner 做正样本分配；
- 引入 Distribution Focal Loss，提高检测框定位质量。`Lecture13_CNN_image_detection.pdf`

这部分不需要死记版本细节，知道 YOLO 的发展方向就行：

> 更快、更准、更适合多尺度目标检测。

---

# 17. 本章方法对比总表

| 方法 | 核心思想 | 优点 | 缺点 |
|---|---|---|---|
| 滑动窗口 | 枚举窗口并分类 | 简单直接 | 计算量巨大 |
| Region Proposal | 先找可能有物体的区域 | 减少无效窗口 | 仍依赖传统算法 |
| R-CNN | proposal 单独跑 CNN | 精度提升 | 非常慢 |
| Fast R-CNN | 整图跑一次 CNN，共享特征 | 减少重复计算 | Selective Search 仍慢 |
| Faster R-CNN | 用 RPN 生成 proposals | 精度高，端到端程度更强 | 两阶段，速度相对慢 |
| YOLO | 网格直接预测框和类别 | 速度快 | 小目标和密集目标较难 |
| SSD | 多尺度特征图预测默认框偏移 | 速度快，兼顾多尺度 | 依赖默认框设计 |

---

# 18. 考点版总结

这一章最可能考这些：

## 1. 目标检测和分类的区别

分类只输出类别；目标检测输出类别和边界框。

## 2. 目标检测损失函数

$$
L = L_{cls} + L_{bbox}
$$

分类损失判断类别，边框回归损失判断位置。

## 3. 滑动窗口为什么慢

因为要遍历大量位置、尺度、长宽比，每个窗口都要跑 CNN。

## 4. R-CNN、Fast R-CNN、Faster R-CNN 的区别

| 方法 | 关键区别 |
|---|---|
| R-CNN | 每个 proposal 单独跑 CNN |
| Fast R-CNN | 整图只跑一次 CNN，在 feature map 上裁 RoI |
| Faster R-CNN | 用 RPN 替代 Selective Search 生成 proposals |

## 5. ROI Pool 和 ROI Align 区别

ROI Pool 会坐标取整，可能错位；ROI Align 不取整，用双线性插值，空间对齐更好。

## 6. Faster R-CNN 为什么是两阶段

第一阶段 RPN 生成候选框；第二阶段对候选框分类和回归。

## 7. Anchor 是什么

Anchor 是 feature map 每个位置上的预设框，用来作为边界框回归的初始参考框。

## 8. YOLO 输出维度怎么算

如果划分为 $S \times S$ 网格，每个格子预测 $B$ 个框、$C$ 个类别，则输出为：

$$
S \times S \times (5B + C)
$$

PPT 例子：

$$
S=7,\ B=2,\ C=20
$$

所以：

$$
7 \times 7 \times 30
$$

## 9. 两阶段和单阶段区别

两阶段：精度高但慢。  
单阶段：速度快，适合实时检测。

---

# 19. 一句话背诵版

**目标检测是在分类基础上增加边界框定位；R-CNN 用候选区域做 CNN 特征提取但很慢，Fast R-CNN 共享整图特征，Faster R-CNN 用 RPN 自动生成候选框；YOLO 和 SSD 则跳过候选框阶段，直接预测类别和边界框，因此速度更快。**



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)