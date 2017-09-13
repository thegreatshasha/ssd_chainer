import sys

""" Have to find a better solution to this instead of absolute imports """
sys.path.append('..')
sys.path.append('../models')

import numpy as np
import unittest

from vgg import VGGNet
from boxnet import BoxNet

class TestFunctions(unittest.TestCase):
	""" Tests functions in functionlist.py """

	def test_demo(self):
		np.testing.assert_equal(1,1)

	def test_vggmini(self):
		""" Unit tests which test your own understanding """
		""" Tests that vggmini transform convert batch of images into correct feature map dimensions """
		img = np.random.rand(2, 3, 256, 128).astype(np.float32)
		print(img.shape)

		""" Tests that vggmini reduces the image resolution by 1/2^4 which it should because of 4 max pooling layers """
		vgg = VGGNet()
		fm = vgg(img)
		np.testing.assert_equal(fm.shape, (2, 512, 256/2**4, 128/2**4))

	def test_vggmini_visualize(self):
		""" Tests vgg by preloading the set of weights and visualizing the features. They should be sensible looking """
		pass

	def test_boxnet(self):
		""" Tests generation of normalized shifts and class scores (scores they are, softmaxes not) """
		img = np.random.rand(2, 3, 256, 128)
		
		vgg = VGGNet()
		boxnet = BoxNet()

		fm = vgg(img)
		boxes = boxnet(fm)

		np.testing.assert_equal(boxes.shape, (2,6,256/2**4,128/2**4))

		""" Dimension check with random shifts """

		""" Visualize boxes with random shifts """

	def test_training(self):
		""" Test that the training is working for all the model classes """
		pass

if __name__ == "__main__":
	unittest.main()