# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from ban import config


class BANUpdater(object):
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.distloss = kwargs.pop("distloss")
        self.last_model = None
        self.gen = 0
        self.T = kwargs.pop("temperature")
        self.alpha = kwargs.pop("alpha")

    def update(self, inputs, targets, criterion):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if self.gen > 0 and self.distloss == "default":
            teacher_outputs = self.last_model(inputs).detach()
            loss = self.kd_loss(outputs, targets, teacher_outputs)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()
        return loss

    def register_last_model(self, weight):
        self.last_model = config.get_model().to("cuda")
        self.last_model.load_state_dict(torch.load(weight))

    def kd_loss(self, outputs, labels, teacher_outputs):
        KD_loss = nn.KLDivLoss()(
            F.log_softmax(outputs / self.T, dim=1),
            F.softmax(teacher_outputs / self.T, dim=1),
        ) * self.alpha + F.cross_entropy(outputs, labels) * (1.0 - self.alpha)

        return KD_loss

    def __model(self):
        return self.model

    def __last_model(self):
        return self.last_model

    def __gen(self):
        return self.gen
