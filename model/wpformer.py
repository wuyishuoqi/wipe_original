import numpy as np

import torchvision
import torch.nn as nn
import torch
from torchvision.transforms import Resize

from model.channel_trans import ChannelTransformer


class Wpformer(nn.Module):
  """Our RoadSeg takes rgb and another (depth or normal) as input,
  and outputs freespace predictions.
  """

  def __init__(self, nInChannel: int, nOutChannel: int):
    super().__init__()

    self.upsample136x32 = nn.Upsample((136, 32))

    resnet_raw_model1 = torchvision.models.resnet34(pretrained=True)
    resnet_raw_model2 = torchvision.models.resnet34(pretrained=True)
    resnet_raw_model3 = torchvision.models.resnet34(pretrained=True)
    filters = [64, 64, 128, 256, 512]
    self.encoder_conv1_p1 = nn.Conv2d(
      in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
    )
    # self.encoder_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
    self.encoder_bn1_p1 = resnet_raw_model1.bn1
    self.encoder_relu_p1 = resnet_raw_model1.relu
    self.encoder_maxpool_p1 = resnet_raw_model1.maxpool
    self.encoder_layer1_p1 = resnet_raw_model1.layer1
    self.encoder_layer2_p1 = resnet_raw_model1.layer2
    self.encoder_layer3_p1 = resnet_raw_model1.layer3
    self.encoder_layer4_p1 = resnet_raw_model1.layer4

    self.tf = ChannelTransformer(
      vis=False, img_size=[17, 12], channel_num=512, num_layers=1, num_heads=3
    )

    self.decode = nn.Sequential(
      nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(2),
      nn.ReLU(inplace=True),
    )

    self.bn1 = nn.BatchNorm1d(2)
    self.bn2 = nn.BatchNorm2d(512)
    self.rl = nn.ReLU(inplace=True)

  def forward(self, x: torch.Tensor):  # 16,2,3,114,32
    x = x[:, :, :, 0, :]  # Nx9x114x3
    x = x.permute(0, 3, 2, 1)  # Nx3x114x9
    x = x[:, np.newaxis, :, :, :]  # Nx1x3x114x9

    x1 = x[:, :, 0, :, :]  # Nx1x114x9
    x2 = x[:, :, 1, :, :]
    x3 = x[:, :, 2, :, :]

    x1 = self.upsample136x32(x1)  # Nx1x136x32
    x2 = self.upsample136x32(x2)
    x3 = self.upsample136x32(x3)

    x1 = self.encoder_conv1_p1(x1)  ##16,2,136,136
    x1 = self.encoder_bn1_p1(x1)  ##size16,64,136,136
    x1 = self.encoder_relu_p1(x1)  ##size(1,64,192,624)

    x2 = self.encoder_conv1_p1(x2)  ##16,2,136,136
    x2 = self.encoder_bn1_p1(x2)  ##size16,64,136,136
    x2 = self.encoder_relu_p1(x2)  ##size(1,64,192,624)

    x3 = self.encoder_conv1_p1(x3)  ##16,2,136,136
    x3 = self.encoder_bn1_p1(x3)  ##size16,64,136,136
    x3 = self.encoder_relu_p1(x3)  ##size(1,64,192,624)

    # x = self.encoder_maxpool(x)

    x1 = self.encoder_layer1_p1(x1)
    x2 = self.encoder_layer1_p1(x2)
    x3 = self.encoder_layer1_p1(x3)

    x1 = self.encoder_layer2_p1(x1)
    x2 = self.encoder_layer2_p1(x2)
    x3 = self.encoder_layer2_p1(x3)

    x1 = self.encoder_layer3_p1(x1)
    x2 = self.encoder_layer3_p1(x2)
    x3 = self.encoder_layer3_p1(x3)  # 256*34*8

    x1 = self.encoder_layer4_p1(x1)
    x2 = self.encoder_layer4_p1(x2)
    x3 = self.encoder_layer4_p1(x3)  # 256*34*8

    x = torch.cat([x1, x2, x3], dim=3)

    x = self.bn2(x)

    x, weight = self.tf(x)

    x = self.decode(x)

    m = torch.nn.AvgPool2d((1, 12), stride=(1, 1))
    x = m(x).squeeze(dim=3)
    x = self.bn1(x)
    # x = self.rl(x)

    x = torch.transpose(x, 1, 2)  # Nx17x2

    return x


def weights_init(m):
  # classname = m.__class__.__name__
  # if classname.find('Conv') != -1:
  #     nn.init.xavier_normal_(m.weight.data)
  #     nn.init.xavier_normal_(m.bias.data)
  # elif classname.find('BatchNorm2d') != -1:
  #     nn.init.normal_(m.weight.data, 1.0)
  #     nn.init.constant_(m.bias.data, 0.0)
  if isinstance(m, nn.Conv2d):
    nn.init.xavier_normal_(m.weight.data)
    # nn.init.xavier_normal_(m.bias.data)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.BatchNorm1d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
