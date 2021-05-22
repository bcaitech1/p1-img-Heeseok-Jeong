import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        x = self.backbone(x)
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.resnet152(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

class VGGNet11_BN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.vgg11_bn(pretrained=True)
        self.back_in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.back_in_features, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

class My_VGGNet11_BN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.vgg11_bn(pretrained=True)
        self.back_in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Linear(self.back_in_features, num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        return x

class VGGNet19_BN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.vgg19_bn(pretrained=True)
        self.back_in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.back_in_features, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

class My_VGGNet19_BN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.vgg19_bn(pretrained=True)
        self.back_in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Linear(self.back_in_features, num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        return x

class EfficientNet_B0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = timm.models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

class EfficientNet_B3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = timm.models.efficientnet_b3(pretrained=True)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

class EfficientNet_B4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x
