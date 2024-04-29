"""
    modules library
"""
import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models.resnet import resnet50, ResNet50_Weights

class ConvBlock(nn.Module):
    """
        residual conv block
    """
    def __init__(self, inchs, outchs):
        super().__init__()

        self.conv1 = nn.Conv2d(inchs, outchs, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(outchs)
        self.conv2 = nn.Conv2d(outchs, outchs, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(outchs)

        self.pool = nn.AvgPool2d(5, 4, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """
            torch module forward
        """
        x = self.pool(x)

        x0 = self.conv1(x)
        x = self.bn1(x0)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x0 + self.relu(x)
        return x

class ColModule(nn.Module):
    """
        parameter estimation module
    """
    def __init__(self, backbone=None, in_chs=3, out_chs=12+25):
        super().__init__()

        if backbone is None:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_chs, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                ConvBlock(256, 512)
            )
        else:
            self.backbone = backbone

        self.gpool = nn.AdaptiveAvgPool2d((1,1))
        self.out = nn.Linear(512, out_chs)

    def forward(self, x):
        """
            torch module forward
        """
        x = self.backbone(x)

        bs, ch, _, _ = x.shape
        x = self.gpool(x).reshape(bs,ch)
        x = self.out(x)
        return x

class ColWarp(nn.Module):
    """
        differential warping module
    """
    def forward(self, im, flat_col):
        """
            torch module forward
        """
        bs, _, _, _ = im.shape

        colwarp = flat_col[:, :9].reshape(bs,3,3)
        colshift = flat_col[:, 9:12].unsqueeze(-1).unsqueeze(-1)
        spatial = flat_col[:, 12:].reshape(bs, 1, 1, 5, 5).repeat(1, 3, 1, 1, 1)

        im = im + colshift
        im = torch.einsum('bdhw,bdc->bchw', im, colwarp) # batched matmul

        out = torch.zeros_like(im)
        for b in range(bs):
            out[b] = F.conv2d(im[b], spatial[b], padding=2, groups=3) 

        return out

class Clamp(nn.Module):
    """
        channel-wise max clamping
    """
    def __init__(self,
                 rate=.1,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()

        self.rate = rate

        self.mean = torch.tensor(mean, requires_grad=False)
        self.std = torch.tensor(std, requires_grad=False)

        vmin = -self.mean/self.std
        vmax = (1-self.mean)/self.std

        self.register_buffer("min", vmin)
        self.register_buffer("max", vmax)

    def forward(self, x):
        """
            torch module forward
        """
        if self.training:
            return x

        for c in range(x.shape[1]):
            xc = x[:,c]

            vmin, vmax = self.min[c], self.max[c]

            xmin = xc < vmin
            xmax = xc > vmax

            xc[xmin] = vmin
            xc[xmax] = vmax
        return x

class CorrectCol(nn.Module):
    """
        color estimation + warping
    """
    def __init__(self, classifier=None, backbone_weights=None):
        super().__init__()

        if classifier is None:
            self.classifier = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.classifier = classifier

        try:
            self.clas_state_dict = self.classifier.state_dict()
            self.classifier.eval()
            for p in self.classifier.parameters():
                p.requires_grad = False
        except AttributeError:
            self.clas_state_dict = None

        self.colm = ColModule()
        if backbone_weights is not None:
            self.colm.backbone.load_state_dict(backbone_weights)
        self.colw = ColWarp()
        self.clamp = Clamp()

    def forward(self, x):
        """
            torch module forward
            single classifier
        """
        col = self.colm(x)
        x = self.colw(x, col)
        x = self.clamp(x)
        return self.classifier(x), col, x

class CorrectColMulti(nn.Module):
    """
        color estimation + warping
        multiple classifiers
    """
    def __init__(self, classifiers, backbone_weights=None):
        super().__init__()

        self.classifiers = nn.ModuleList(classifiers)
        for clas in classifiers:
            clas.eval()
            for p in clas.parameters():
                p.requires_grad = False

        self.colm = ColModule()
        if backbone_weights is not None:
            self.colm.backbone.load_state_dict(backbone_weights)
        self.colw = ColWarp()
        self.clamp = Clamp()

    def forward(self, x, clasid):
        """
            torch module forward
        """
        col = self.colm(x)
        x = self.colw(x, col)
        x = self.clamp(x)
        return self.classifiers[clasid](x), col, x
