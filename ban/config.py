# -*- coding: utf-8 -*-
from ban.models.mlp import MLP
from ban.models.lenet import LeNet
from ban.models.resnet import ResNet18
from ban.models.resnet import ResNet34
from ban.models.resnet import ResNet50
from ban.models.resnet import ResNet101
from ban.models.resnet import ResNet152

from ban.models.resnet import resnet20

"""
add your model.
from your_model_file import Model
model = Model()
"""

# model = ResNet50()
def get_model():
    # model = ResNet18()
    model = resnet20()
    return model
