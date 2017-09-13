#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('..')

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


class BoxNet(chainer.Chain):

    """
    BoxNet
    - It takes vgg feature maps of size (224, 224, 3) sized image as imput
    """

    def __init__(self):
        super(BoxNet, self).__init__(
            conv=L.Convolution2D(512, 6, 3, stride=1, pad=1)
        )
        self.train = False

    def __call__(self, f):
        h = F.relu(self.conv1_1(x))
        return h