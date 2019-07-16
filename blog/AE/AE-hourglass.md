# Associative embedding: End-to-End Learning for Joint Detection and Grouping


2018.04.07：

-------------------

## 论文阅读

这篇文章的核心思想是比较精炼概括的，它的亮点是用一个框架解决了在计算机视觉中常见的任务中经常遇到的两个通用环节：Detection and Grouping，用中文来讲就是，检测（小的视觉单元作为候选）和（根据得分）重组（一个合理的结构）。

从以下的视觉任务中可以体现：

 - **人体姿态估计问题**：一般按照bottom-to-up的方式，先检测出body key points然后按照约束来组合完整的人体，但多人姿态估计的问题又衍生出另一种up-down的方式，就是先检测出单个人体再识别其姿态，比如Mask R-CNN, RMPE等方法。 
 
 - **目标检测**：往往先寻找不同位置和尺度下的bounding boxes，然后打分筛选
 - **实例分割**：寻找相关联的像素，然后将像素合理重组成物体实例（mask）。
 - **多目标追踪**：检测物体实例，重组其运动轨迹

这些方式，本质上都符合人类自身视觉从部分认识整体，以整体推理部分的直觉。以往的工作都认识到这一点，只是这篇论文做了一次提炼概括了，并指出了一个问题：

 

以往的两步策略（detection ﬁrst and grouping second）忽略了两个环节之间内在的紧密耦合。

> （在之前看的CMU的Realtime Multi-Person2D Pose Estimation using Part Affinity
> Fields, 他们的论文当中，除了人体关键点作为监督信息外，还引入了Part Affinity
> Fields，也就是和肢体方向保持一致的单位向量作为监督信号，我的感觉是，这实际上就是没有充分利用两个环节上的耦合性，或者说是人体关键点与肢体连接的耦合性信息，毕竟人体的关节与整体的关系是统一的，
> 而OpenPose用的是寻找最佳的图匹配的方式，但同时将关键点位置，和肢体向量同时作为监督信息，会导致信息冗余，增加复杂度吧？所以我觉得作者这种融合两步的思想就很实际，很前卫）

所以作者针对多人体姿态估计，将两步工作融入到一个框架里，即在一般的输出Heatmap层，附加了一层作为“tag“（也就是论文提到的embedding的含义），并设计了一个grouping loss作为监督关键点是否分配给了正确的人体的函数。论文巧妙的地方就是没有给“tag”赋予”ground-truth”来作为强监督，而是用“tag“值的相似与差异来表示多个人体。用于预测Heatmap的网络架构基于作者之前的工作“Stacked Hourglass”.

论文中Related work中的Perceptual Organization的叙述部分，给我了比较多的启示：


> Perceptual Organization是感知组织的意思，我理解成人类在认识事物或概念所遵循的层级组织关系。所谓的强人工智能，就需要解决这一棘手问题吧。作者提到了这一工作涉及到的许多任务，有Figure–ground segmentation   
(perception)，hierarchical image parsing， spectral clustering，conditional random ﬁelds，generative probabilistic models等等一系列问题，这些方法都遵循，从pre-detect visual units到measure affinity，再到grouping，但是目前没有统一到一个统一的架构上来，作者就是从这角度出发，不加一些额外的设计来完成一个端到端的网络架构。作者提到了图像层级解析，特别符合人类认知图像，所以，作者的Hourglass模块设计成沙漏状，先是压缩图像，获得全局信息，然后利用全局信息与低层特征融合，输出一个与同样大小的heatmap，其实就是想将这样的层级解析的思想间接地蕴含在内，只不过网络的训练将这些信息都隐含在了参数里，无法与人的解析思路类比


## 改进想法
2018.7.5

论文重要的一点引入了‘embedding’，或者叫‘tag’。它是沟通detection和grouping中间的媒介，作者用1D的标量值来表示‘tag’，论文的原话是：
> "In practice we have found that 1D embedding is sufﬁcient for multiperson pose estimation,and higher dimensions do not lead to signiﬁcant improvement."

作者没有做出详细的论证，只是从实践的角度说明了1D表示的合理性，这个标量的数值的具体含义是什么呢，不清楚，但它和‘mask’概念有着很大的关联。如果说，用向量或者矩阵来表示这个‘tag’是否可行呢？比如可以表示人的尺度、姿态、位置、等等属性，（参考capsules）。

**回头做个实验，用单纯的背景，复制相同的人体，或者对比变换，喂给test做实验，来判断输出的tag的变化**

作者别出心裁的设计是，他没有给‘tag’引入绝对的数值标签作为监督，而是把“相同的实例，tag有着相似的取值，不同的实例的tag的取值存在差异”的原则作为一个监督，来设计loss函数--（tagloss，group loss）。

$$L_{group}\left ( h,T \right )=\frac{1}{N}\sum_{n}\sum_{k}\left ( \bar{h}_{n}-h_{k}\left ( x_{nk} \right ) \right )^{2}+\frac{1}{N^{2}}\sum_{n}\sum_{{n^{`}}}exp\left \{ -\frac{1}{2\sigma ^{2}} \left (  \bar{h}_{n}-\bar{h}_{{n^{`}}}\right )^{2}\right \}$$

其中：
$h_{k}\in R^{W\times H},h(x)$代表像素位置$x$的tag取值,$T$代表的groudtruth构成的坐标集$T=\left \{ \left ( x_{nk} \right ) \right \},n=1,...,N,k=1,...,K$  
$\bar{h}_{n}=\frac{1}{K}\sum_{k}h_{k}\left ( x_{nk} \right )$

这里对于单个人求所有关键点的tag均值$\bar{h}_{n}$作为一个参考嵌入（reference embedding）,然后希望隶属用一人体的关键点位置的tag值尽可能接近$\bar{h}_{n}$。不同的人体，$\bar{h}_{n}$差距尽可能大。我觉得这一点可以先做几个思考：

1. 求$\bar{h}_{n}$值时用的参考groudtruth的位置求取所有关键点位置tag值的均值，是否合理，初始值如何设定？为什么不求一个区域mask的tag均值？

2. 两人各出现重叠时，tag的值是否就意味着接近呢？比如某个人的手臂与另外一个人连在了一起？

3. 这样的设计对哪些情况效果不好，loss发散导致关键点检测率的下降？因为论文说了做实验说明在用关键点的groudtruth下， 多人检测的准确率从59.2%多提高到94%,进而说明，多人关键点的定位精确度是影响最终准确率的主要因素。一副图像有3个 人的关节，比如右手肘关节，那么标签是在这3个位置产生高斯分布，训练是用单个heatmap图去逼近。CMU-POSE采用了part affine field似乎可以 提高精准率吗？那么我能用什么方法，将多个人的关节响应图更精准一些？！是一个点
4. 在Groudtruth中，如果某个关键点未标记的话，论文直接将该关键点Heatmap全部赋值零，输出的网络就很难去拟合这个关键点，我的想法是用一个先验的模型去推测大概率这个未标记点的位置，然后生成高斯的heatmap

我的想法是：
1.  1D的tag值变成矩阵或者向量，参考capsules，引入坐标添加技术，人体大概坐标的位置会影响tag的取值，[CoordConv][1]，capsules也有！
          
 1.  把多个关键点当作数据（观察，果），把实例当作隐变量（因），寻找最佳匹配？
 2.  连续的mask区域，代替稀疏的点构成的骨架？

## 工程实践

2018.4.27

``` python
'''coco_pose/dp.py'''
    class GenerateHeatmap():#从标签点生成关节点的Heatmap热值图响应，
        '''resolution：128x128  parts：17'''
        def __init__(self, output_res, num_parts):
            self.output_res = output_res
            self.num_parts = num_parts
            sigma = self.output_res/64  #sigma=2
            self.sigma = sigma
            size = 6*sigma + 3  #size=15
            x = np.arange(0, size, 1, float)
            #返回一个array([0.,1.,...,15.])的数据
            y = x[:, np.newaxis]
            #np.newaxis 为 numpy.ndarray（多维数组）增加一个轴，
            #np.newaxis 在使用和功能上等价于 None，其实就是 None 的一个别名
            #x[:, np.newaxis]=array([[0],[1],[2]]) 行向量变列向量
            x0, y0 = 3*sigma + 1, 3*sigma + 1    #x0=y0=7
            self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            

            '''根据keypoints数量生成heatmaps张量'''
        def __call__(self, keypoints):
            hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
            #hms是 17x128x128的张量
            sigma = self.sigma
            for p in keypoints:
                for idx, pt in enumerate(p):
                #eumerate用法：遍历list[],输出序号和序号对应的数值
                    if pt[2]>0: #pt[2]代表是否可见？
                        x, y = int(pt[0]), int(pt[1])
                        if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                            #print('not in', x, y)
                            continue
                        ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                        br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                        c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                        a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                        cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                        aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                        hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
            return hms
2018.6.27
'''utils/misc.py'''
    '''作者调用网路写的代码'''
    def importNet(net):
        t = net.split('.')  #比如net='models.posenet.PoseNet'  t=['models', 'posenet', 'PoseNet']
        path, name = '.'.join(t[:-1]), t[-1]     # path='models.posenet' name=PoseNet，
        module = importlib.import_module(path)    # 这些net的名称全部储存在一个configJSON文件里
        return eval('module.{}'.format(name))    #这里的eval 什么意思？？是怎么用在make_network里面的。字符串变成了函数。

        """def eval(source, globals, locals) 这里就是eval的定义，似乎是把 字符串变成了 函数或者python对象 的入口？
    这是注释：Evaluate the given source in the context of globals and locals.
    The source may be a string representing a Python expression or a code object as returned by compile(). The globals must be a dictionary and locals can be any mapping, defaulting to the current globals and locals. If only globals is given, locals defaults to it."""

'''train.py  主函数'''
    def main():
        func, config = init()#初始化训练参数，其中func是make_network()读取batch数据的函数
        data_func = config['data_provider'].init(config)#产生train和valid两种phase的数据迭代器
        train(func, data_func, config)#训练函数，batch生成+batch送入训练+配置
       #在训练函数的里面要考虑每个迭代周期epoch里面，保存checkpoint，每次实验海可以选择从上一次进行的checkpoint进行。
    if __name__ == '__main__':
        main()
'''task/pose.py   make_network函数处理batch数据，在train（）函数里phase的选择是train和valid两种数据迭代器。然后把迭代器里的每个batch-size数据送到make_network()'''
    def make_network(configs):    
        PoseNet = importNet(configs['network'])    #现在的PoseNet变成了什么
        train_cfg = configs['train']
        config = configs['inference']

        poseNet = PoseNet(**config) # **config直接表示configs['inference']里的一组 字典

        forward_net = DataParallel(poseNet.cuda()) #平行的前向计算
        def calc_loss(*args, **kwargs):
            return poseNet.calc_loss(*args, **kwargs)

        config['net'] = Trainer(forward_net, configs['inference']['keys'], calc_loss) #trianer类，config[]能指代类？，
        train_cfg['optimizer'] = torch.optim.Adam(config['net'].parameters(), train_cfg['learning_rate'])

        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        logger = open(os.path.join(exp_path, 'log'), 'a+')

        def make_train(batch_id, config, phase, **inputs):
            for i in inputs:
                inputs[i] = make_input(inputs[i])

            net = config['inference']['net']
            config['batch_id'] = batch_id

            if phase != 'inference':
                result = net(inputs['imgs'], **{i:inputs[i] for i in inputs if i!='imgs'})

                num_loss = len(config['train']['loss'])  # =3?
                '''config['train']['loss']
                'loss': [
                ['push_loss', 1e-3],
                ['pull_loss', 1e-3],
                ['detection_loss', 1],'''
                ## I use the last outputs as the loss
                ## the weights of the loss are controlled by config['train']['loss'] 
                losses = {i[0]: result[-num_loss + idx]*i[1] for idx, i in enumerate(config['train']['loss'])}

                loss = 0
                toprint = '\n{}: '.format(batch_id)
                for i in losses:
                    loss = loss + torch.mean(losses[i])

                    my_loss = make_output( losses[i] )
                    my_loss = my_loss.mean(axis = 0)

                    if my_loss.size == 1:
                        toprint += ' {}: {}'.format(i, format(my_loss.mean(), '.8f'))
                    else:
                        toprint += '\n{}'.format(i)
                        for j in my_loss:
                            toprint += ' {}'.format(format(j.mean(), '.8f'))

                logger.write(toprint)
                logger.flush()

                if batch_id == 200000:
                    ## decrease the learning rate after 200000 iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 1e-5

                optimizer = train_cfg['optimizer']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                return None
            else:
                out = {}
                net = net.eval()
                result = net(**inputs)
                if type(result)!=list and type(result)!=tuple:
                    result = [result]
                out['preds'] = [make_output(i) for i in result]
                return out
        return make_train

'''data/coco_pose/dp.py  init(config)这个函数负责从phase是train和valid从数据库里生成迭代器'''
    def init(config):
        batchsize = config['train']['batchsize']
        current_path = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_path)
        import ref as ds
        ds.init()

        train, valid = ds.setup_val_split()
        dataset = { key: Dataset(config, ds, data) for key, data in zip( ['train', 'valid'], [train, valid] ) }

        use_data_loader = config['train']['use_data_loader']

        loaders = {}
        for key in dataset:
            loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True, num_workers=config['train']['num_workers'], pin_memory=False)

        def gen(phase):
            batchsize = config['train']['batchsize']
            batchnum = config['train']['{}_iters'.format(phase)]
            loader = loaders[phase].__iter__()
            for i in range(batchnum):
                imgs, masks, keypoints, heatmaps = next(loader)
                yield {
                    'imgs': imgs,
                    'masks': masks,
                    'heatmaps': heatmaps,
                    'keypoints': keypoints
                }


        return lambda key: gen(key)

'''下面的这个操作是打开一个保存序号的文件：'525232\n', '132019\n', '349193\n', '263094\n',，转换成[525232,132019,349193,263094],使用了map(function,*iterables),其中
lambda x:func（x）是一种简化的表达式匿名函数，int(x.strip())为表达式，strip（）表示去掉换行符\n, f.readlines()表示map()函数的迭代器参数'''
        with open(ref_dir + '/valid_id', 'r') as f:
            valid_id = list(map(lambda x:int(x.strip()), f.readlines()))
        valid_id_set = set(valid_id)


'''task/loss.py  loss函数的误差是如何定义的，pytorch最方便的地方在于用loss.backward()一个函数直接封装了梯度
反向传播'''
    def singleTagLoss(pred_tag, keypoints):
        """
        associative embedding loss for one image
        """
        eps = 1e-6
        tags = []
        pull = 0
        for i in keypoints:
            tmp = []
            for j in i:
                if j[1]>0:
                    tmp.append(pred_tag[j[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull +  torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        if len(tags) == 0:
            return make_input(torch.zeros([1]).float()), make_input(torch.zeros([1]).float())

        tags = torch.stack(tags)[:,0]

        num = tags.size()[0]
        size = (num, num, tags.size()[1])
        A = tags.unsqueeze(dim=1).expand(*size)
        B = A.permute(1, 0, 2)

        diff = A - B
        diff = torch.pow(diff, 2).sum(dim=2)[:,:,0]
        push = torch.exp(-diff)
        push = (torch.sum(push) - num)
        return push/((num - 1) * num + eps) * 0.5, pull/(num + eps)

    def tagLoss(tags, keypoints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        keypoints = keypoints.cpu().data.numpy()
        for i in range(tags.size()[0]):
            push, pull = singleTagLoss(tags[i], keypoints[i%len(keypoints)])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)

    def test_tag_loss():
        t = make_input( torch.Tensor((1, 2)), requires_grad=True )
        t.register_hook(lambda x: print('t', x))
        loss = singleTagLoss((t, [[[0,1]], [[1,1]]]))[0]
        loss.backward()
        print(loss)

    if __name__ == '__main__':
        test_tag_loss()

'''test.py'''
    def multiperson(img, func, mode):
    """
    1. Resize the image to different scales and pass each scale through the network
    2. Merge the outputs across scales and find people by HeatmapParser
    3. Find the missing joints of the people with a second pass of the heatmaps
    """
    if mode == 'multi':
        scales = [2, 1., 0.5]
    else:
        scales = [1]

    height, width = img.shape[0:2]      # 假设 h=1000，w=500
    center = (width/2, height/2)  #（250，500）
    dets, tags = None, []
    for idx, i in enumerate(scales):
        scale = max(height, width)/200    # 5
        input_res = max(height, width)
        inp_res = int((i * 512 + 63)//64 * 64)  #取整 对于i=2，1，0.5， inp_res=1024,512,256
        res = (inp_res, inp_res)

        mat_ = get_transform(center, scale, res)[:2]       #task/misc.py
        inp = cv2.warpAffine(img, mat_, res)/255      #opencv的变换矩阵

        def array2dict(tmp):
            return {
                'det': tmp[0][:,:,:17],
                'tag': tmp[0][:,-1, 17:34]
            }

        tmp1 = array2dict(func([inp]))
        tmp2 = array2dict(func([inp[:,::-1]]))

        tmp = {}
        for ii in tmp1:
            tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]),axis=0)

        det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][flipRef]
        if det.max() > 10:
            continue
        if dets is None:
            dets = det
            mat = np.linalg.pinv(np.array(mat_).tolist() + [[0,0,1]])[:2]
        else:
            dets = dets + resize(det, dets.shape[1:3]) 

        if abs(i-1)<0.5:
            res = dets.shape[1:3]
            tags += [resize(tmp['tag'][0], res), resize(tmp['tag'][1,:, :, ::-1][flipRef], res)]

    if dets is None or len(tags) == 0:
        return [], []

    tags = np.concatenate([i[:,:,:,None] for i in tags], axis=3)  #shape[17,256,256,2]这里都是2，2代表什么？
    dets = dets/len(scales)/2
    
    dets = np.minimum(dets, 1)
    grouped = parser.parse(np.float32([dets]), np.float32([tags]))[0]   #shape=(x,17,5)  x代表检测的人数？5代表什么呢？


    scores = [i[:, 2].mean() for  i in grouped]                         #shape=(x)

    for i in range(len(grouped)):
        grouped[i] = refine(dets, tags, grouped[i])

    if len(grouped) > 0:
        grouped[:,:,:2] = kpt_affine(grouped[:,:,:2] * 4, mat)
    return grouped, scores
```
## checkpoint是什么
http://www.atyun.com/12192.html
防止在训练模型时信息丢失的检查点 checkpoint 机制。
如果你玩过电子游戏，你就会明白为什么检查点（chekpoint）是有用的了。举个例子，有时候你会在一个大Boss的城堡前把你的游戏的当前进度保存起来——以防进入城堡里面就Game Over了。

机器学习和深度学习实验中的检查点本质上是一样的，它们都是一种保存你实验状态的方法，这样你就可以从你离开的地方开始继续学习。

Keras文档为检查点提供了一个很好的解释:
- 模型的体系结构，允许你重新创建模型
- 模型的权重
- 训练配置(损失、优化器、epochs和其他元信息
- 优化器的状态，允许在你离开的地方恢复训练





- **Markdown和扩展Markdown简洁的语法**
- **代码块高亮**
- **图片链接和图片上传**
- ***LaTex*数学公式**
- **UML序列图和流程图**
- **离线写博客**
- **导入导出Markdown文件**
- **丰富的快捷键**
使用简单的符号标识不同的标题，将某些文字标记为**粗体**或者*斜体*，创建一个[链接](http://www.csdn.net)等，详细语法参考帮助？。

本编辑器支持 **Markdown Extra** , 　扩展了很多好用的功能。具体请参考[Github][2].  

### 表格

**Markdown　Extra**　表格语法：

| 项目     | 价格  |
| -------- | ----- |
| Computer | $1600 |
| Phone    | $12   |
| Pipe     | $1    |

可以使用冒号来定义对齐方式：

| 项目     | 价格    | 数量  |
| :------- | ------: | :---: |
| Computer | 1600 元 | 5     |
| Phone    | 12 元   | 12    |
| Pipe     | 1 元    | 234   |

###定义列表

**Markdown　Extra**　定义列表语法：
项目１
项目２
:   定义 A
:   定义 B

项目３
:   定义 C

:   定义 D

	> 定义D内容

### 代码块
代码块语法遵循标准markdown代码，例如：
``` python
@requires_authorization
def somefunc(param1='', param2=0):
    '''A docstring'''
    if param1 > param2: # interesting
        print 'Greater'
    return (param2 - param1 + 1) or None
class SomeClass:
    pass
>>> message = '''interpreter
... prompt'''
```

###脚注
生成一个脚注[^footnote].
  [^footnote]: 这里是 **脚注** 的 *内容*.
  
### 目录
用 `[TOC]`来生成目录：

[TOC]

### 数学公式
使用MathJax渲染*LaTex* 数学公式，详见[math.stackexchange.com][1].

 - 行内公式，数学公式为：$\Gamma(n) = (n-1)!\quad\forall n\in\mathbb N$。
 - 块级公式：

$$	x = \dfrac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$

更多LaTex语法请参考 [这儿][3].

### UML 图:

可以渲染序列图：

```sequence
张三->李四: 嘿，小四儿, 写博客了没?
Note right of 李四: 李四愣了一下，说：
李四-->张三: 忙得吐血，哪有时间写。
```

或者流程图：

```flow
st=>start: 开始
e=>end: 结束
op=>operation: 我的操作
cond=>condition: 确认？

st->op->cond
cond(yes)->e
cond(no)->op
```

- 关于 **序列图** 语法，参考 [这儿][4],
- 关于 **流程图** 语法，参考 [这儿][5].

## 离线写博客

即使用户在没有网络的情况下，也可以通过本编辑器离线写博客（直接在曾经使用过的浏览器中输入[write.blog.csdn.net/mdeditor](http://write.blog.csdn.net/mdeditor)即可。**Markdown编辑器**使用浏览器离线存储将内容保存在本地。

用户写博客的过程中，内容实时保存在浏览器缓存中，在用户关闭浏览器或者其它异常情况下，内容不会丢失。用户再次打开浏览器时，会显示上次用户正在编辑的没有发表的内容。

博客发表后，本地缓存将被删除。　

用户可以选择 <i class="icon-disk"></i> 把正在写的博客保存到服务器草稿箱，即使换浏览器或者清除缓存，内容也不会丢失。

> **注意：**虽然浏览器存储大部分时候都比较可靠，但为了您的数据安全，在联网后，**请务必及时发表或者保存到服务器草稿箱**。

##浏览器兼容

 1. 目前，本编辑器对Chrome浏览器支持最为完整。建议大家使用较新版本的Chrome。
 2. IE９以下不支持
 3. IE９，１０，１１存在以下问题
    1. s
    1. IE9不支持文件导入导出
    1. IE10不支持拖拽文件导入

---------

[1]: http://www.sohu.com/a/240960815_129720
[2]: https://github.com/jmcmanus/pagedown-extra "Pagedown Extra"
[3]: http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
[4]: http://bramp.github.io/js-sequence-diagrams/
[5]: http://adrai.github.io/flowchart.js/
[6]: https://github.com/benweet/stackedit
