import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


batch_size = 64
conv1_out = 30
conv2_out = 30
conv3_out = 30
fc1_num = 40

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1a = nn.Conv2d(in_channels=1, out_channels=conv1_out, kernel_size=[1, 2])
		self.conv1b = nn.Conv2d(in_channels=1, out_channels=conv1_out, kernel_size=[2, 1])
		self.conv2a = nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=[2, 1])
		self.conv2b = nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=[1, 2])
		self.conv3 = nn.Conv2d(in_channels=conv2_out, out_channels=conv3_out, kernel_size=1)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(conv2_out*9, fc1_num)
		self.fc2 = nn.Linear(fc1_num, 4)

	def forward(self, x):
	    conv1a = F.relu(self.conv1a(x))
	    conv1b = F.relu(self.conv1b(x))

	    conv2a = F.relu(self.conv2a(conv1a))
	    conv2b = F.relu(self.conv2b(conv1b))

	    x = conv2a + conv2b

	    x = F.relu(x)
	    x = x.view(-1, conv2_out*9)
	    x = F.relu(self.fc1(x))
	    x = F.dropout(x, training=self.training)
	    x = self.fc2(x)
	    return F.log_softmax(x, dim=0)

