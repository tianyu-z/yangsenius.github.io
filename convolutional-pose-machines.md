# 论文阅读 Convolutional Pose Machines 

## 概述

Pose Machines provide a sequential prediction framework for learning rich implicit spatial models.

论文旨在提供一个一般性的序列预测框架，这个框架处理的学习任务是丰富的隐含关系模型。

人体姿态模型属于一个有复杂隐含关系的模型，如果形象地描述?

> 人体姿态模型是个铰链式的骨架模型，人体骨架由关节和肢干（从机械的角度：齿轮和杠杆）构成。

传统的解决方式，就是用表征关节点之间直接依赖关系的图模型对人体姿态进行建模。（如果检测到肘关节，那么肩部很大概率会在附近出现）

这种建模方式属于 参数模型，用图模型来描述节点间复杂的相互依赖关系，并推断是一个极其复杂的过程，在2014年ECCV的一篇文章Pose Machines: Articulated Pose Estimation via Inference Machine，提出了pose machine的概念，这也是本篇文章的思路来源。pose machine中提到：

> State-of-the-art approaches for articulated human pose estimation are rooted in parts-based graphical models. These models are often restricted to tree-structured representations and simple parametric potentials in order to enable tractable inference. However, these simple dependencies fail to capture all the interactions between body parts.
> It incorporates rich spatial interactions among multiple parts and information across parts of diﬀerent scales

## CPM 

**1.核心**：Combine the **pose machine architecture** and **convolutional architecture** 

**2.构成**：
A sequence of convolutional networks that **repeatedly produce 2D belief maps** for the location of each part

**3.方法**：

上一阶段产生的图像特征`image features`和信念图`belief maps`作为下一个阶段的输入。

> The belief maps provide the subsequent stage an expressive non-parametric encoding of the spatial uncertainty of location for each part, allowing the CPM to learn rich image-dependent spatial models of the relationships between parts

**信念图** 为下一阶段 提供一个可以表示**关节部分不确定关系**的**非参表示编码**

不去精确地解析信念图，而是通过对中间的信念图的操作，来学习隐含的依赖图像的空间部分关系模型

前面的部分关节信念为下阶段消除歧义`disambiguating`，每个阶段预测越来越精确的位置

> In order to capture long range interactions between parts, the design of the network in each stage of our sequential prediction framework is motivated by the goal of achieving a large receptive ﬁeld on both the image and the belief maps

 **intermediate supervision:**

 >A systematic framework that replenishes gradients and guides the network to produce increasingly accurate belief maps by enforcing intermediate supervision periodically through the network. 

##网络设计和公式描述

<center> <img src="https://raw.githubusercontent.com/yangsenius/images/master/CPM.png" width="800"> </center>


##接受域的概念 -- Receptive field of CNN

$$ l_k = l_{k-1} + ((f_k - 1) * \prod_{i=1}^{k-1}s_i)$$
