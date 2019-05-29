import torch
from torchvision import datasets, models, transforms


class vgg19(torch.nn.Module):
    def __init__(self,requires_grad=False):
        super(vgg19,self).__init__()
        vgg_pretrained_features=models.vgg19(pretrained=True).features
        self.slice1=torch.nn.Sequential()
        self.slice2=torch.nn.Sequential()
        self.slice3=torch.nn.Sequential()

        for x in range(7):
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(7,21):
            self.slice2.add_module(str(x),vgg_pretrained_features[x])
        for x in range(21,30):
            self.slice3.add_module(str(x),vgg_pretrained_features[x])
        
        if not requires_grad:
            for parm in self.parametere():
                parm.requires_grad=False


    def forward(self,x):
        h_relu1=self.slice1(x)
        h_relu2=self.slice2(h_relu1)
        h_relu3=self.slice3(h_relu2)
        out=[h_relu1,h_relu2,h_relu3]
        return out

vgg_pretrained_features=models.vgg19(pretrained=True).features
print(vgg_pretrained_features)