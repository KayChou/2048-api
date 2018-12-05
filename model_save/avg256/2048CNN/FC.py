import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataLoader import MyDataset
from torchvision import transforms

grid = 4
batch_size = 1024
fc1_num = 32
fc2_num = 16


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(16, fc1_num)
		self.fc2 = nn.Linear(fc1_num, fc2_num)
		self.fc3 = nn.Linear(fc2_num, 4)

	def forward(self, x):
	    x = x.view(-1, grid**2)

	    x = F.relu(self.fc1(x))
	    x = F.relu(self.fc2(x))
	    x = F.relu(self.fc3(x))
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

			print("\t", predict[0:20])
			print("\t", target[0:20])
			print('Accuracy: %0.2f' % correct, '%')


def main():
	train_loader, test_loader = load_data()

	model = Net()
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	epochs = 20

	for epoch in range(epochs):
		train(model, epoch, train_loader, optimizer)

	torch.save(model, 'model.pth')
	model = torch.load('model.pth')


if __name__ == '__main__':
	main()
