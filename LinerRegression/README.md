#
Logistic回归
    
    假设输入的数据的特征向量Z∈R,一般都是通过感知机的过程通过决策函数的符号来判断其属于哪一类，而logistic回归则是更近一步
    ，通过比较概率值来判断类别。
    
![pic](https://github.com/ReOneK/pytorch/blob/master/LinerRegression/ref/1.png)

  
    其中ω 是权重， b 是偏置。现在介绍Logistic 模型的特点，先引人一个概念:一个事件发生的几率( odds ) 是指该事件发生的概率与不发生的概率的比值，比如一个事件发生的概率是p. 那么该事件发生的几率是log(p/1-p)，
    该事件的对数几率或logit 函数是:
![pic](https://github.com/ReOneK/pytorch/blob/master/LinerRegression/ref/2.png)

    对于Logistic 回归而言:
![s](https://github.com/ReOneK/pytorch/blob/master/LinerRegression/ref/3.png)

    这也就是说在Logistic 回归模型中，输出Y=l 的对数几率是输入Z 的线性函数，这也就是Logistic 回归名称的原因.




    
