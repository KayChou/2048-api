import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import time

batch_size = 64
conv1_out = 16
conv2_out = 32
conv3_out = 64
conv4_out = 128
conv5_out = 64
conv6_out = 32

fc1_num = 16
fc2_num = 16

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_out, kernel_size=[1, 2])
		self.conv2 = nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=[2, 1])
		self.conv3 = nn.Conv2d(in_channels=conv2_out, out_channels=conv3_out, kernel_size=[1, 2])
		self.conv4 = nn.Conv2d(in_channels=conv3_out, out_channels=conv4_out, kernel_size=[2, 1])
		self.conv5 = nn.Conv2d(in_channels=conv4_out, out_channels=conv5_out, kernel_size=[1, 2])
		self.conv6 = nn.Conv2d(in_channels=conv5_out, out_channels=conv6_out, kernel_size=[2, 1])

		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(conv6_out, fc1_num)
		self.fc2 = nn.Linear(fc1_num, 4)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = x.view(-1, conv6_out)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x)

