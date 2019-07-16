# Pose machine：Articulated Pose Estimation via Inference Machines

由显式的图模型推断人体铰链模型到隐式地表达丰富的多部件关系的转折点！

## 介绍

估计铰链式人体姿态的复杂度来源有两个：

- 潜在骨架模型的自由度（20大约）导致的高位配置空间的搜索范围
- 图像中人体姿态的多变性

传统的图模型，基于树结构或者星状结构的简单图模型难以捕捉多部件之间的关系和依赖性，而且容易导致双重计数错误产生。
非树模型需要近似推断来解决，参数学习非常复杂！

图模型的第二个困难的地方是：
> A second limitation of graphical models is that deﬁning the potential functions requires careful consideration when specifying the types of interactions. This choice is usually dominated by parametric forms such as simple quadratic models in order to enable tractable inference [1]. Finally, to further enable eﬃcient inference in practice, many approaches are also restricted to use simple classiﬁers such as mixtures of linear models for part detection [5]. These are choices guided by tractabilty of inference rather than the complexity of the data. Such tradeoﬀs result in a restrictive model that do not address the inherent complexity of the problem. 

论文建立了**一个类似于场景解析的层级推断机制来估计人体姿态**

## Pose machine

它是一个序列预测算法，它**效仿了信息传递的机制**来预测每一个人体部件的置信度，在每个阶段迭代提升它的预测能力

- 1.它联合了多变量之间的交互关系

- 2.它从数据中学习空间关系，而不需要精确的参数模型

- 3.它模块化的架构能够允许使用高承载的预测器来处理不同形状的姿态

解决在传统模型人体姿态估计的两大难点。

**可以这么说，pose machine最重要的想法就是：
运用级联，将对每个单独部件的估计信息结合起来，从中提取语义特征再进一步地精确估计下个阶段的每个部件位置。这个思想是开创性的，它不仅符合人类的直觉，并且实验证明是有提升效果的，后来的CPM，用卷积网络的架构准确实现了这一方法，取得了当时的最好效果，对各种形状的人体姿态都有帮组，在2016年，CPM在MPII,LSP,FLIC数据库上都到达了最好表现，而近年来成功的关于姿态估计的所有论文都几乎蕴含了这一思想：试图归纳出图中全局的信息作为下个级联阶段的预测，比如stacker hourglass、cpn**

## 方法

话不多说，上图上公式

**1.整体框架图与“信息”整合公式**


<center> <img src="https://raw.githubusercontent.com/yangsenius/yangsenius.github.io/master/pose-machine-images/1.png" width="800"> </center>

$$ {^{l}}{g_{t}}\left ( x_{z}^{l},\bigoplus _{l\in 1...L}\psi (z,{b_{t-1}}^{l}) \right )\rightarrow \left \{{^{l}}{b_{t}^{p}}\left ( Y_{p}=z \right )\right \}_{p\in 0...P_{l}}$$

$g_{t}\left (\cdot \right)$是预测器（多部件类别分类器），它的输入为将**原始的图像特征信息**和上个阶段计算得到的部件响应值经过$\psi(\cdot )$语义特征映射函数后的**语义信息**相结合的特征。
```
它符合我们认知的一个信息加工过程，利用已经判断好的信息作为下一步判断的依据！
```
注意：上个阶段的部件响应值包括所有层级$(l\in1,2,...L)$下、所有部件$(p\in1,2,...P)$的输出响应值。这里的层级表示一个由粗略（整个人体响应）到精细（小部件关节响应）的等级梯度

**2.图像语义特征映射$\psi_{1},\psi_{2}$**

<center> <img src="https://raw.githubusercontent.com/yangsenius/yangsenius.github.io/master/pose-machine-images/2.png" width="800"> </center>

- 2.1语义片区（Patch）特征$\psi_{1}$，结合邻域信息

``` 
突然发觉把“patch”译为“片区”非常合适，一个是发音比较像，另外，片区反映了patch是对一小部分区域进行处理的含义。
```


$$\psi_{1}\left ( z, ^{l}b_{t-1}\right )= \bigoplus_{p\in 0...P_{l}}c_{1}\left ( z,^{l}b_{t-1}^{p} \right ).$$

$c_{1}\left ( z,^{l}b_{t-1}^{p} \right )$表示的是在某个层级$l$下，在某个位置$z$的邻域patch范围内，编码了其他所有部件的响应值（confidence map）图产生的信息对该特定部件的影响，见上图a区域

- 2.2语义偏移特征$\psi_{2}$，考虑远距离（long range interactions among the parts at non-uniform, relative oﬀsets）信息

$$c_{2}\left ( z,^{l}b_{t-1}^{p} \right )=\left [^{l}o_{1}^{p};...;^{l}o_{k}^{p}  \right ]$$

$$\psi_{2}\left ( z, ^{l}b_{t-1}\right )= \bigoplus_{p\in 0...P_{l}}c_{2}\left ( z,^{l}b_{t-1}^{p} \right ).$$

$c_{2}\left ( z,^{l}b_{t-1}^{p} \right )=\left [^{l}o_{1}^{p};...;^{l}o_{K}^{p}  \right ]$表示的是，对于位置$z$,到在一个特定部件的confidence map图，经过极大值抑制会产生$K$个尖峰位置的向量集合，见上图b区域

总结：语义片区特征映射$\psi_{1}$根据附近其他部件的响应值$confidence$来捕捉粗略的信息，偏移特征映射$\psi_{2}$捕捉精确的相对位置信息。最终的的语义特征通过连接两个特征映射得到$\psi \left (\cdot\right )=\left [\psi_{1};\psi_{2}\right ]$

``` 
这个想法是非常具有启发式的，这是在日后的深度卷积网络设计中常见的一种技巧：#用抽象的信息融合来替代精准的模型表达#，然后交给分类器或者神经网络去共同学习融合后的信息带来的约束，在卷积网络中，这种操作更加精简，比如在之后的2016年CVPR卷积姿态机CPM的论文中，这种操作可以用卷积特征图的叠加来实现，还有有些其他论文，类似FCN，它的想法是利用上采样，把高分辨率低语义信息与低分辨率高语义信息相融合，等等，类似的设计还有很多
```

## 训练过程(Trainings)

<center> <img src="https://raw.githubusercontent.com/yangsenius/yangsenius.github.io/master/blog/pose-machine/training.png" width="800"> </center>

训练推断程序，需要训练每个层级$(l\in1,2,...L)$下、每个部件$(p\in1,2,...P)$的预测器${^{l}g_{t}}$.上图中反映了训练过程，第一个阶段$t=1$的预测器${^{l}g_{1}}$，使用的数据集为$D_{0}$它使用了从标注好的图像中patch中提取的特征。接下来的阶段，数据集$D_{t}$通过连接图像patch中提取的特征和从$confidence$ $ map$ 即 $\left [ ^{l}b_{t-1}\right ]_{l=1}^{L}$中提取的语义特征


## 堆积(Stack)

为了防止模型过拟合，阻止在下个阶段的预测器继续对上阶段相同的训练数据进行拟合，采用了**stack training**方法（类似于交叉验证）来为下一个阶段的预测器产生训练数据集。将数据集分成$M$份，拿一份用来训练，剩余的作为held-out数据。在第一阶段时，我们把预测器复制$M$份，分别训练这$M$份数据集。**然后在下一个阶段时，对于每个训练样本，我们选择（上阶段）没有碰到过该样本的预测器作为这个阶段该样本的预测器，进行训练(很巧妙的思想！)**，然后我们迭代这种构造数据集的方法！

## 推断（Inference）

$$\forall l,\forall p,^{l}y_{p}^{*}= \underset{z}{\arg max}\,  \: ^{l}b_{T}^{p}\left ( z \right )$$

找到confidence map中响应值最大的位置，即为人体部件位置

## Implementation

- Choice of Predictor：常见的传统方式有：boosting，random forest，CNN是在2016年的论文CPM中得以使用。

- Training：构造正负样本集，从图像的patch中获取

- Image Features：HOG等

- Context Features：对于context patch features，patch的尺寸为21x21，然后使用max-pooling以2x2的区域获取一个121维的特征，对于语义偏移特征，K的取值为3

以上的介绍就是对2014年ECCV上的Pose Machine论文的介绍。它的贡献是，提出了一个推断机制框架，适合于学习具有丰富的空间关系的模型，也就是它不单单局限于人体姿态问题。而它结合高性能的预测器能够对人体姿态估计问题有很好的表现性能。在当时遇到的难题就是拥挤姿态问题和罕见姿态问题。

## 参考文献：

[1.Ramakrishna, Varun et al. “Pose Machines: Articulated Pose Estimation via Inference Machines.” ECCV (2014)][1]

[2.Wei, Shih-En et al. “Convolutional Pose Machines.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 4724-4732.][2]

------
[1]:https://pdfs.semanticscholar.org/67dc/de46f8188f3f0676b7529a2e5828ab611e4d.pdf
[2]:
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780880&tag=1
