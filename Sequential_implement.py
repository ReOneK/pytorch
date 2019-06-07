#Sequential的三种实现


import torch.nn as nn
import torch

net1=nn.Sequential()
net1.add_module('conv',nn.Conv2d(3,3,3))
net1.add_module('batchnorm',nn.BatchNorm2d(3))
net1.add_module('activate_layer',nn.ReLU())


net2=nn.Sequential(
    nn.Conv2d(3,3,3),
    nn.BatchNorm2d(3),
    nn.ReLU(),
)


from collections import OrderedDict
net3=nn.Sequential(
    OrderedDict(
        [
          ('conv1', nn.Conv2d(3, 3, 3)),
          ('bn1', nn.BatchNorm2d(3)),
          ('relu1', nn.ReLU())
        ]
    )
)


print('net1:', net1)
print('net2:', net2)
print('net3:', net3)