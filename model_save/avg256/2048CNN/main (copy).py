import os
import torchvision.datasets as datasets
import torch.utils.data
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
	    x = self.conv1(x)
	    x = F.max_pool2d(x, kernel_size=2)
	    x = F.relu(x)

	    x = self.conv2(x)
	    x = F.max_pool2d(x, kernel_size=2)
	    x = F.relu(x)

	    x = x.view(-1, 320)
	    x = F.relu(self.fc1(x))
	    x = F.dropout(x, training=self.training)
	    x = self.fc2(x)
	    return F.log_softmax(x, dim=0)

# Load training and test data
def load_data(path):
	train_data = datasets.MNIST(root=path,
		train=True,transform=transforms.Compose(
			[transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))]),
		target_transform=None,
		download=True)
	train = torch.utils.data.DataLoader(
		train_data,
		batch_size=64,
		shuffle=True,
		num_workers=0)
	test_data = datasets.MNIST(
		root=path,
		train=False,
		transform=transforms.Compose(
			[transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))]),
		target_transform=None,
		download=True)
	test = torch.utils.data.DataLoader(
		test_data,
		batch_size=64,
		shuffle=True,
		num_workers=0)
	return train, test
# Train the net
def train(model, epoch, train_loader, optimizer):
	model.train()
	for idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if idx % 50 == 49:
			print('Train epoch: %d   Loss: %.3f    ' % (epoch+1, loss))

# Test the net
def test(model, test_loader):
	model.eval()
	correct = 0
	for data, target in test_loader:
		output = model(data)
		predict = output.data.max(1)[1]
		correct = correct + predict.eq(target.data).sum()
	print('Accuracy: %2d' % (100*correct/10000), '%')


def main():
	data_base = './Datasets'
	mnist_path = os.path.join(data_base, 'MNIST')
	train_loader, test_loader = load_data(mnist_path)

	model = Net()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
	epochs = 10

	for epoch in range(epochs):
		train(model, epoch, train_loader, optimizer)
		test(model, test_loader)

if __name__ == '__main__':
	main()