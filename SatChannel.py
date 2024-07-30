import numpy as np
import scipy.io


def sat_channel(h):
    """
    通过读取mat文件中存储的信道增益矩阵，将其转换为每个用户的信道增益h，返回h
    :param h: 信道增益矩阵
    :return: 每个用户的信道增益h
    """
    n = len(h)
    # 读取.mat文件
    mat = scipy.io.loadmat('UAV100sat231015_LEOs_30sats.mat')
    # print(mat)
    H_ASL = mat['H_ASL']
    # print(H_ASL)
    # 对每一行取最大值，得到一个向量
    h_users = np.amax(H_ASL, axis=1)
    # 取前n个用户
    h_users = h_users[:n]
    return h_users
if __name__ == "__main__":
    N = 10
    h0 = np.ones((N))
    h = sat_channel(h0)
    print(h)
