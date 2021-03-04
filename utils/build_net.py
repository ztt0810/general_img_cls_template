from torch import nn
from config import Config
import timm
import torch


class BuildNet:
    def __init__(self, backbone, num_classes):
        self.backbone = backbone
        self.num_classes = num_classes
        self.device = torch.device(Config.device)

    def __call__(self, *args, **kwargs):
        return self.make_model()

    def make_model(self):
        model = timm.create_model(
            model_name=self.backbone,
            num_classes=self.num_classes,
            pretrained=True
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, Config.num_classes)
        )
        model.to(self.device)

        return model