import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataLoader import MyDataset as MyDataset
from torchvision import transforms
from torch.autograd import Variable
import time

batch_size = 64
start_time = time.time()

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.RNN = nn.LSTM(
			input_size = 4,
			hidden_size = 300,
			num_layers = 4,
			batch_first=True)
		self.fc1 = nn.Linear(300, 64)
		self.fc2 = nn.Linear(64, 4)

	def forward(self, x):
		x, (h_n, h_c) = self.RNN(x, None)
		x = x[:, -1 ,:]
		x = self.fc1(x)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)
	

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
		data = Variable(data.view(-1,4,4))

		if torch.cuda.is_available():
			data = Variable(data).cuda()
			target = Variable(target).cuda()
			model.cuda()

		output = model(data)

		optimizer.zero_grad()
		# target = target.repeat(12)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		
		if idx % 10 == 0:
			predict = output.data.max(1)[1]
			num = predict.eq(target.data).sum()
			correct = 100.0*num/batch_size
			t = time.time()-start_time
			# print("\t\t\t\t\t", predict[0:20])
			# print("\t\t\t\t\t", target[0:20])
			print('Train epoch: %d   Loss: %.3f    ' % (epoch+1, loss), \
				'Accuracy: %0.2f' % correct, '%', '\tTotal Time: %0.2f' % t)


def main():
	model = Net()

	train_loader, test_loader = load_data()

	# optimizer = optim.RMSprop(model.parameters(), lr=0.001)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	epochs = 20

	for epoch in range(epochs):
		if(epoch>10):
			optimizer = optim.Adam(model.parameters(), lr=0.0001)
		train(model, epoch, train_loader, optimizer)
		torch.save(model, 'model_'+str(epoch) + '.pth')

	torch.save(model, 'model.pth')
	print("Time used:", time.time()-start_time)


if __name__ == '__main__':
	main()
