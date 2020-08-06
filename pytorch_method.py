import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 6, 3, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 12, 3, 2),
            nn.ReLU(True),
            nn.Conv2d(12, 24, 3, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(True),
            nn.Linear(24, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class trainCustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, img_path):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """
        # 图像路径
        self.img_path = img_path

        # Transforms
        self.to_tensor = transforms.ToTensor()

        # 读取 csv 文件
        self.data_info = pd.read_csv(csv_path, header=None)

        # 文件第一列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[:14000, 0])

        # 第二列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[:14000, 1])

        # 第三列是决定是否进行额外操作
        #self.operation_arr = np.asarray(self.data_info.iloc[:, 2])

        # 计算 length
        self.data_len = len(self.image_arr)
        
        
    def __getitem__(self, index):
        # 从 pandas df 中得到文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.img_path + single_image_name)
 
        # 检查需不需要额外操作
        #some_operation = self.operation_arr[index]
        # 如果需要额外操作
#         if some_operation:
#             # ...
#             # ...
#             pass

        # 把图像转换成 tensor
        img_as_tensor = self.to_tensor(img_as_img)
 
        # 得到图像的 label
        single_image_label = self.label_arr[index]
 
        return img_as_tensor, single_image_label
 
    def __len__(self):
        return self.data_len
        
transformations = transforms.Compose([transforms.ToTensor()])

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, img_path):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """
        self.img_path = img_path
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # 读取 csv 文件
        self.data_info = pd.read_csv(csv_path, header=None)
        # 文件第一列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[:15750, 0])
        # 第二列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[:15750, 1])
        # 第三列是决定是否进行额外操作
        #self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # 计算 length
        self.data_len = len(self.image_arr)
        #self.transformations = transformations 
        
    def __getitem__(self, index):
        # 从 pandas df 中得到文件名
        single_image_name = self.image_arr[index]
        # 读取图像文件
        img_as_img = Image.open(self.img_path + single_image_name)
 
        # 检查需不需要额外操作
        #some_operation = self.operation_arr[index]
        # 如果需要额外操作
#         if some_operation:
#             # ...
#             # ...
#             pass
        # 把图像转换成 tensor
        img_as_tensor = self.to_tensor(img_as_img)
 
        # 得到图像的 label
        single_image_label = self.label_arr[index]
 
        return img_as_tensor, single_image_label
 
    def __len__(self):
        return self.data_len


class TestCustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, img_path):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """
        self.img_path = img_path
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # 读取 csv 文件
        self.data_info = pd.read_csv(csv_path, header=None)
        # 文件第一列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[15750:, 0])
        # 第二列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[15750:, 1])
        # 第三列是决定是否进行额外操作
        #self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # 计算 length
        self.data_len = len(self.image_arr)
        #self.transformations = transformations 
        
    def __getitem__(self, index):
        # 从 pandas df 中得到文件名
        single_image_name = self.image_arr[index]
        # 读取图像文件
        img_as_img = Image.open(self.img_path + single_image_name)
 
        # 检查需不需要额外操作
        #some_operation = self.operation_arr[index]
        # 如果需要额外操作
#         if some_operation:
#             # ...
#             # ...
#             pass
        # 把图像转换成 tensor
        img_as_tensor = self.to_tensor(img_as_img)
 
        # 得到图像的 label
        single_image_label = self.label_arr[index]
 
        return img_as_tensor, single_image_label
 
    def __len__(self):
        return self.data_len


def main():
    net = Net()
    customdataset = CustomDatasetFromImages('train.csv', 'train/')
    trainloader = DataLoader(customdataset, batch_size=4, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cactus_net.pth'
    torch.save(net.state_dict(), PATH)

def test(PATH):
    testcustomdataset = TestCustomDatasetFromImages('train.csv', 'train/')
    testloader = DataLoader(testcustomdataset, batch_size=1750, shuffle=False, num_workers=0)
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    print(labels)
    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    predicted = torch.max(outputs, 1)
    pred_y = predicted[1].numpy()
    y_label = labels.numpy()
    accuracy = (pred_y == y_label).sum() / len(y_label)
    print(accuracy)

test(PATH='./cactus_net.pth' )