"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
numpy
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 3
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    # Converts a PIL.Image or numpy.ndarray to
    transform=torchvision.transforms.ToTensor(),
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    # download it if you don't have it
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)

# 显示第三张图片
plt.imshow(train_data.train_data[3].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[3])
plt.pause(1)
plt.close()
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            # compress to 3 features which can be visualized in plt
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# 初始化一个2行5列的画布
# plt.subplots() 返回一个 Figure实例fig 和一个 AxesSubplot实例ax 。这个很好理解，fig代表整个图像，ax代表坐标轴和画的图。
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # 开启动态作图

# original data (first row) for viewing
# view() 对tensor进行形变，如（8,8) --> (-1,4) == (2,4)
view_data = train_data.train_data[:N_TEST_IMG].view(
    -1, 28*28).type(torch.FloatTensor)/255.

'''
由其文档可知，在 colormap 类别上，有如下分类：
perceptual uniform sequential colormaps：感知均匀的序列化 colormap
sequential colormaps：序列化（连续化）色图 colormap；
      gray：0-255 级灰度，0：黑色，1：白色，黑底白字；
      gray_r：翻转 gray 的显示，如果 gray 将图像显示为黑底白字，gray_r 会将其显示为白底黑字；
      binary
diverging colormaps：两端发散的色图 colormaps；
      seismic
qualitative colormaps：量化（离散化）色图；
miscellaneous colormaps：其他色图；
      rainbow
'''
# 展示5张图片
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()
                              [i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())  # 清除x坐标的刻度标签
    a[0][i].set_yticks(())  # 清除y坐标的刻度标签

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):

        # print(x.shape) ([64, 1, 28, 28])
        # print(b_label.shape)([64])
        b_x = x.view(-1, 28*28)   # batch x, shape (64, 784)
        b_y = x.view(-1, 28*28)   # batch y, shape (64, 784)
        # print(b_x.shape)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # plotting decoded image (second row)
            # _.shape = (5,3) , decode_date.shape = (5,784)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[
                               i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.01)

plt.ioff()
plt.close()

# 3D展示数据
view_data = train_data.train_data[:200].view(
    -1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(
), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
plt.pause(1)
plt.close()
