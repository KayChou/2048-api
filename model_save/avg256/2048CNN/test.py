from dataLoader import generator, MyDataset
from torchvision import transforms
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

a = np.array([1,2,3])
a = torch.from_numpy(a)
print(type(a))
a = a.type(torch.cuda.FloatTensor)
print(type(a))

'''
#train_loader = MyDataset(root='./Datasets/Train.csv')
test = MyDataset(root='./Datasets/Train.csv',
						transform = transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test, batch_size = 64)

for idx, (data, target) in enumerate(test_loader):
	print(data.shape, type(target), len(target))
nums = np.array([1,2,3,3,4])


counts = np.bincount(nums)
#返回众数
a = np.argmax(counts)
print(a)

print(test_loader.__getitem__(0))
'''