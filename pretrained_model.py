import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.net = getattr(models, args.model)(pretrained=True)

        for params in self.net.parameters():
            params.requires_grad = False

    def forward(self, img):
        features = self.net(img)

        return features
  
            
