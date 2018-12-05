import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataLoader import MyDataset
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


def load_data():
	train_data = MyDataset(
		root = './Datasets/Train.csv',
		transform=transforms.Compose(
			[transforms.ToTensor()]))

	train = torch.utils.data.DataLoader(
		train_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0)

	test_data = MyDataset(
		root = './Datasets/Test.csv',
		transform=transforms.Compose(
			[transforms.ToTensor()]))

	test = torch.utils.data.DataLoader(
		test_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0)
	return train, test


# Train the net
def train(model, epoch, train_loader, optimizer):
	model.train()

	for idx, (data, target) in enumerate(train_loader):
		
		data = data.type(torch.float)

		if torch.cuda.is_available():
			data = Variable(data).cuda()
			target = Variable(target).cuda()
			model.cuda()

		output = model(data)

		optimizer.zero_grad()
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		
		if idx % 50 == 0:
			print('Train epoch: %d   Loss: %.3f    ' % (epoch+1, loss))
			predict = output.data.max(1)[1]
			num = predict.eq(target.data).sum()
			correct = 100*num/batch_size

			print("\t\t\t\t\t", predict[0:20])
			print("\t\t\t\t\t", target[0:20])
			print('Accuracy: %0.2f' % correct, '%')


def main():
	start_time = time.time()
	model = Net()

	train_loader, test_loader = load_data()

	optimizer = optim.Adam(model.parameters(), lr=0.001)
	epochs = 20

	for epoch in range(epochs):
		train(model, epoch, train_loader, optimizer)

	torch.save(model, 'model.pth')
	model = torch.load('model.pth')
	print("Time used:", time.time()-start_time)


if __name__ == '__main__':
	main()
