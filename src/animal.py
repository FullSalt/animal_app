from torchvision import transforms

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

# from torchvision.models import resnet18 

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

#　ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(37632, 2)

    def forward(self, x):
        h = self.conv(x)
        h = F.relu(h)
        h = self.bn(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(-1, 37632)
        h = self.fc(h)
        return h

# ↓ 下記講義資料と同じ内容

# # 学習済みモデルに合わせた前処理を追加
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# #　ネットワークの定義
# class Net(pl.LightningModule):

#     def __init__(self):
#         super().__init__()

#         #学習時に使ったのと同じ学習済みモデルを定義
#         self.feature = resnet18(pretrained=True) 
#         self.fc = nn.Linear(1000, 2)

#     def forward(self, x):
#         #学習時に使ったのと同じ順伝播
#         h = self.feature(x)
#         h = self.fc(h)
#         return h