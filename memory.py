#  #################################################################
#  This file contains the main DROO operations, including building DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- February 2020. Written based on Tensorflow 2 by Weijian Pan and 
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  ###################################################################

from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


# DNN network for memory
class MemoryDNN:
    def __init__(
            self,
            net,
            learning_rate=0.01,
            training_interval=10,
            batch_size=100,
            memory_size=1000,
            output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval  # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
            nn.Linear(self.net[0], self.net[1]),
            nn.ReLU(),
            nn.Linear(self.net[1], self.net[2]),
            nn.ReLU(),
            nn.Linear(self.net[2], self.net[3]),
            nn.Sigmoid()
        )

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        """
        encode 表示Critic编码过程，将观测值(h,Q,Y)和动作m分别存储到内存中。具体操作是调用self.remember(h, m)函数，将h和m存储到内存中。
        并且在每隔一定的时间间隔内，调用self.learn()函数，对模型进行训练。 self.learn()
        :param h: 观测值，包括信道增益、数据队列和能量队列；h是一个N=3*k维向量，目前的k=10因此h是一个30维向量
        :param m: 动作，包括二进制卸载决策；m是一个N=k维向量，目前的k=10因此m是一个10维向量
        :return: None
        """
        # encoding the entry
        self.remember(h, m)
        # train the DNN every multiple steps

        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])

        # train the DNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.09, 0.999), weight_decay=0.0001)
        criterion = nn.BCELoss()
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        assert (self.cost > 0)
        self.cost_his.append(self.cost)

    # 解码过程，根据不同的mode选择不同的解码方式
    def decode(self, h, k=1, mode='OP'):
        """
        decode 表示Actor解码过程，根据不同的mode选择不同的解码方式
        将处理过的h输入到模型中，获取模型的action输出m_pred。具体操作是通过self.model(h)获得模型的输出。
        并将得到的m_pred根据不同的mode选择不同的解码方式，最终返回解码后的结果。

        :param h: 观测值，包括信道增益、数据队列和能量队列；h是一个N=3*k维向量，目前的k=10因此h是一个30维向量
        :param k: 选择的动作个数，和用户数目一致
        :param mode: 解码器模式
        """

        # to have batch dimension when feed into Tensor
        # 将传入的观测值h转换为PyTorch张量，并在其外部添加一个维度，以便符合模型输入的要求。具体操作是将h数组或张量通过torch.Tensor(h[np.newaxis, :])转换为一个1*30的张量。
        h = torch.Tensor(h[np.newaxis, :])

        # 将模型设置为评估模式，通过self.model.eval()实现。
        self.model.eval()
        # 将处理过的h输入到模型中，获取模型的action输出m_pred。具体操作是通过self.model(h)获得模型的输出。
        m_pred = self.model(h)
        # 将模型输出的预测结果m_pred转换为NumPy数组，以便后续处理。通过m_pred.detach().numpy()可以实现这一步操作。
        m_pred = m_pred.detach().numpy()

        # 根据不同的mode选择不同的解码方式
        # OP表示order-preserving解码器，KNN表示k-nearest neighbor解码器，OPN表示order-preserving neighbor解码器
        # 使用m_pred[0]获取二维矩阵m_pred的第一行，即模型输出的预测结果。
        if mode == 'OP':
            return self.knm(m_pred[0], k)
        elif mode == 'KNN':
            return self.knn(m_pred[0], k)
        elif mode == 'OPN':
            return self.opn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN' or 'OPN'")

    def knm(self, m, k=1, N_channel=5):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_temp_0 = m
        # m_temp_0中最大的N_channel个值且大于0.5的位置为1，其余位置为0
        sorted_m_temp_0 = np.sort(m_temp_0)
        threshold = sorted_m_temp_0[-N_channel]
        result = np.zeros_like(m_temp_0)
        threshold = max(threshold, 0.5)
        result[m_temp_0 > threshold] = 1
        m_list.append(result)

        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            # 计算 m 与 0.5 的绝对值差值 m_abs。
            m_abs = abs(m - 0.5)
            # 使用 np.argsort 函数对 m_abs 进行排序，返回排序后的索引列表 idx_list。
            idx_list = np.argsort(m_abs)[:k - 1]
            # NOP量化方法防止过快收敛到次优解，采用了噪声保序(NOP)量化方法[31]，该方法可以生成Mt≤2N个候选动作
            # 使用循环从 idx_list 中取出每个索引，进行下面的操作：
            #   如果 m 中的值大于 0.5，则将 m 减去当前索引对应的值，判断结果是否大于 0，如果大于 0，则将1添加到 m_list，否则将0添加到 m_list。
            #   如果 m 中的值小于或等于 0.5，则将 m 减去当前索引对应的值，判断结果是否大于等于 0，如果大于等于 0，则将1添加到 m_list，否则将0添加到 m_list。
            for i in range(k - 1):
                if m[idx_list[i]] > 0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_temp = m - m[idx_list[i]]
                    # m_temp中最大的N_channel个值且大于0的位置为1，其余位置为0
                    sorted_m_temp = np.sort(m_temp)
                    threshold = sorted_m_temp[-N_channel]
                    result = np.zeros_like(m_temp)
                    threshold = max(threshold, 0)
                    result[m_temp > threshold] = 1
                    m_list.append(result)
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_temp = m - m[idx_list[i]]
                    # m_temp中最大的N_channel个值且大于等于0的位置为0，其余位置为1
                    sorted_m_temp = np.sort(m_temp)
                    threshold = sorted_m_temp[-N_channel]
                    result = np.zeros_like(m_temp)
                    threshold = max(threshold, 0)
                    result[m_temp >= threshold] = 1
                    m_list.append(1 * result)
        return m_list

    def opn(self, m, k=1):
        """
        self.knm(m + np.random.normal(0, 1, len(m)), k)：同样调用 knm 函数，但这次的输入是 m 加上一个随机生成的服从正态分布（均值为0，标准差为1）的数组。
        这样做的目的是在原始的 m 向量上添加一些随机噪声，以产生略微不同的二进制决策列表。
        这样能够防止模型过度拟合，提高模型的泛化能力。
        """
        return self.knm(m, k) + self.knm(m + np.random.normal(0, 1, len(m)), k)

    def knn(self, m, k=1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m) ** 2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()
