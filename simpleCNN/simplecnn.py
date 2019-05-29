
#%%
import torch
from torch.autograd import Variable
from torch import nn, optim


#%%
class SimpleCNN(nn.Module) :
    def __init__(self) :
        # b, 3, 32, 32
        super().__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv_1', nn.Conv2d(3, 32, 3, 1, padding=1))
        #b, 32, 32, 32
        layer1.add_module('relu_1', nn.ReLU(True))
        layer1.add_module('pool_1', nn.MaxPool2d(2, 2)) # b, 32, 16, 16
        self.layer1 = layer1
        
        layer2 = nn.Sequential()
        layer2.add_module('conv_2', nn.Conv2d(32, 64, 3, 1, padding=1))
        # b, 64, 16, 16
        layer2.add_module('relu_2', nn.ReLU(True))
        layer2.add_module('pool_2', nn.MaxPool2d(2, 2)) # b, 64, 8, 8
        self.layer2 = layer2
        
        layer3 = nn.Sequential()
        layer3.add_module('conv_3', nn.Conv2d(64, 128, 3, 1, padding=1))
        # b, 128, 8, 8
        layer3.add_module('relu_3', nn.ReLU(True))
        layer3.add_module('pool_3', nn.MaxPool2d(2, 2)) # b, 128, 4, 4
        self.layer3 = layer3
        
        layer4 = nn.Sequential()
        layer4.add_module('fc_1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc_2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc_3', nn.Linear(64, 10))
        self.layer4 = layer4
    
    def forward(self, x) :
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        fc_input = conv3.view(conv3.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out


#%%
# 建立模型

model = SimpleCNN()
print(model)


#%%
# 提取前两层

new_model = nn.Sequential(*list(model.children())[:2])
print(new_model)


#%%
# 提取所有的卷积层

conv_model = nn.Sequential()
for name, module in model.named_modules() :
    if isinstance(module, nn.Conv2d) :
        conv_model.add_module(name, module)

print(conv_model)


#%%
# 提取模型中的参数

for name, param in model.named_parameters() :
    print('{} : {}'.format(name, param.shape))


#%%
# 权重初始化
from torch.nn import init

for m in model.modules() :
    if isinstance(m, nn.Conv2d) :
        init.normal_(m.weight.data)
        init.xavier_normal_(m.weight.data)
        init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear) :
        m.weight.data.normal_()

