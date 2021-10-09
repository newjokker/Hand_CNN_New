# Loss Function


### define

损失函数 （Loss Function） 也可称为代价函数 （Cost Function）或误差函数（Error Function），用于衡量预测值与实际值的偏离程度。
一般来说，我们在进行机器学习任务时，使用的每一个算法都有一个目标函数，算法便是对这个目标函数进行优化，特别是在分类或者回归任务中，
便是使用损失函数（Loss Function）作为其目标函数。机器学习的目标就是希望预测值与实际值偏离较小，也就是希望损失函数较小，
也就是所谓的最小化损失函数。


* 分类常用的 loss

* 定位常用的 loss

* 分割常用的 loss 

* 不同 loss 之间的差异有多大

* 

### Zero-one Loss

* 0-1 loss 是最原始的loss，它是一种较为简单的损失函数，如果预测值与目标值不相等，那么为1，否则为0, https://zhuanlan.zhihu.com/p/72589970

* 0-1损失可用于分类问题，但是由于该函数是非凸的，在最优化过程中求解不方便，有阶跃，不连续。0-1 loss无法对x进行求导，
在依赖于反向传播的深度学习任务中，无法被使用，所以使用不多。


# ---


### MSE 

* mean squared error, 均方误差损失函数

* loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

* 

### MAE

* mean absoult error, 平均绝对值误差 

* K.mean(K.abs(y_pred - y_true), axis=-1)

* 平均绝对误差（MAE）是另一种用于回归模型的损失函数。MAE是目标值和预测值之差的绝对值之和。其只衡量了预测值误差的平均模长，而不考虑方向，
取值范围也是从0到正无穷（如果考虑方向，则是残差/误差的总和——平均偏差（MBE））

### MAPE

* mean_absolute_percentage_error, 

* 代码实现
    * diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    * return 100. * K.mean(diff, axis=-1)

### MAPE

* mean_squared_logarithmic_error

* refer : https://www.cnblogs.com/guoyaohua/p/9217206.html

### MSLE 

* mean_squared_logarithmic_error

* refer : https://zhuanlan.zhihu.com/p/34667893 

### squared_hinge

* K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)

### hinge

* K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)

### categorical_hinge

* code 
    * pos = K.sum(y_true * y_pred, axis=-1)
    * neg = K.max((1. - y_true) * y_pred, axis=-1)
    * return K.maximum(0., neg - pos + 1.)

### logcosh

* code
    * def _logcosh(x):
          return x + K.softplus(-2. * x) - K.log(2.)
    * return K.mean(_logcosh(y_pred - y_true), axis=-1)


### soft-max

* ![equation](https://pic2.zhimg.com/80/v2-c3db01467ac0e926f64ba819be71d079_720w.jpg)
    
    * softmax loss 被广泛用于分类问题中，而且发展出了很多的变种，有针对不平衡样本问题的weighted softmax loss、focal loss，
    针对蒸馏学习的soft softmax loss，促进类内更加紧凑的L-softmax Loss等一系列改进。
   
    * softmax函数与softmax-loss函数是不一样的，千万，千万别记混了

### Logistic-loss

* ![equation](https://pic3.zhimg.com/80/v2-d88c6c12d49c72bf99b2b5a23f98d012_720w.jpg)

    * refer : https://zhuanlan.zhihu.com/p/71819342
    
    * 

### cross entropy 

* 用于度量两个概率分布之间的相似性

* ![equation](https://pic3.zhimg.com/80/v2-9a41671bae385d60b4aec73450712f36_720w.jpg)
    
    * yi 为样本的真实标签，取值只能为0或1；
    
    * pi 为预测样本属于类别 yi 的概率；
    
    * m 为类别的数量。

---

## pytorch 自带的 loss 

* refer : https://blog.csdn.net/weixin_36670529/article/details/105670337
* refer : https://pytorch.org/docs/stable/nn.html

### L1Loss

* refer : https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss

### SmoothL1Loss

* refer : https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

### MSELoss

* mean squared error

* refer : https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss

### CrossEntropyLoss

* refer : https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

* 用于多分类

### BCELoss

* refer : https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss

* 用于二分类，

### NLLLoss

* refer : https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss

* refer : https://blog.csdn.net/qq_22210253/article/details/85229988

* CrossEntropyLoss = softmax + log + NLLLoss

### NLLLoss2d

* 



 

















