import pandas as pd
import numpy as np
from torchvision import transforms
import torch

class MyDataset(torch.utils.data.Dataset):

	def __init__(self, root, transform=None, target_transform=None):
		dataframe = pd.read_csv(root)
		data_array = dataframe.values

		self.data = data_array[:, 0:16]
		self.label = data_array[:, 16]
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		board = self.data[index].reshape((4, 4))
		board = board[:, :, np.newaxis]
		board = board/11.0
		# board = torch.from_numpy(board)
		
		label = self.label[index]

		if self.transform is not None:
			board = self.transform(board)
		return board, label

	def __len__(self):
		return len(self.label)


class generator():

	def __init__(self, filename):
		self.file = filename
		self.index = 0

		dataframe = pd.read_csv(self.file)
		data_array = dataframe.values

		self.x = data_array[:, 0:16]
		self.y = data_array[:, 16]
		self.item_num = len(self.x)
		self.idx = 0

	def getitem(self, idx):
		board = self.x[idx].reshape((4,4))
		# board = torch.from_numpy(board)
		return board

	def get_next_batch(self, batch_size):
		return

	def data_loader(self, batch_size=64, shuffle=False):
		batch = self.item_num/batch_size

		data = np.array([batch_size, 1, 4, 4])
		target = np.array()

		for i in range(batch_size):
			board = self.x.reshape((4, 4))
