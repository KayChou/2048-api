import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from dataLoader import MyDataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from  net import simpleNet, Batch_Net

grid = 4
batch_size = 6400

model = simpleNet(16, 32, 16, 4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)

epochs = 20

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

def load_data():
	train_data = MyDataset(
		root = './Datasets/Train.csv',
		transform = data_tf)

	train = torch.utils.data.DataLoader(
		train_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0)

	test_data = MyDataset(
		root = './Datasets/Test.csv',
		transform = data_tf)

	test = torch.utils.data.DataLoader(
		test_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0)
	return train, test


def train(model, epoch, train_loader):
	model.train()
	for idx, (data, target) in enumerate(train_loader):

		data = data.view(data.size(0), -1)
		data = data.type(torch.float)
		data = Variable(data)
		target = Variable(target)

		output = model(data)
		loss = criterion(output, target)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if idx % 10 == 0:
			print('Train epoch: %d   Loss: %.3f    ' % (epoch+1, loss.data.item()))
			predict = output.data.max(1)[1]
			num = predict.eq(target.data).sum()
			correct = 100*num/batch_size

			print("\t", predict[0:20])
			print("\t", target[0:20])
			print('Accuracy: %0.2f' % correct, '%')


def main():
	global model
	train_loader, test_loader = load_data()
	for epoch in range(epochs):
		train(model, epoch, train_loader)

	torch.save(model, 'model.pth')
	model = torch.load('model.pth')


if __name__ == '__main__':
	main()