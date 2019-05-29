# 关于参数初始化可参考[深度学习的weight initialization](https://zhuanlan.zhihu.com/p/25110150)
# 
# 可以从torch的[文档](http://pytorch.org/docs/master/nn.html?highlight=init%20xavier_normal#torch.nn.init.xavier_normal)中得到
# 
# - `init.xvaier_uniform()`一般用于tanh的初始化，结果采样于均匀分布 $$U(-a, a) \sim [-\frac {\sqrt{6}} {\sqrt{fan\_in + fan\_out}}, \frac {\sqrt{6}} {\sqrt{fan\_in + fan\_out}}]$$
# - `init.xvarier_normal()`，结果采样于正态分布 $$N(0, \sqrt{\frac 2 {fan\_in + fan\_out}})$$
# - `init.kaiming_uniform()` 结果采样于均匀分布 $$U(-a, a) \sim [-\frac {\sqrt{6}} {\sqrt{(1+a^2) \times fan\_out}}, \frac {\sqrt{6}} {\sqrt{(1+a^2) \times fan\_out}}]$$
# - `init.kaiming_normal()`一般用于ReLU的初始化，初始化方法为正态分布 $$N(0, \sqrt{\frac 2 {(1 + a^2) \times fan\_in}})$$


##在使用`add_module()`时有时会出现bug,推荐使用`load_state_dict()`