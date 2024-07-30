# Distributed Online Scheduling for Mobile Edge Computing
import scipy.io as sio  # import scipy.io for .mat file I/
import numpy as np  # import numpy

from SatChannel import sat_channel
# Implementated based on the PyTorch
from memory import MemoryDNN
# import the resource allocation function
# replace it with your algorithm when applying LyDROO in other problems
from ResourceAllocation import Algo1_NUM
import MECResourceAllocation as MEC

import math

from processBar import progress_bar

CHANNELMODE = "1111"

def ResourceAllocation(mecRA, MEC_user_Map, user_MEC_Map, m, h, w, Q, Y, V, t, flag = "EVALUATE"):
    """
    分布式在线优化算法的性能评价流程
    :param mecRA:
    :return: f_val,rate,energy
    f_val: 目标值
    rate: 计算速率，表示这个时隙内每个用户的数据Q减少的数量
    energy: 能耗， 表示这个时隙内每个用户能量Y消耗的数量
    """
    f_val = 0
    rate = np.zeros(Nuser)
    energy = np.zeros(Nuser)
    yitaa = np.zeros(Nuser)
    for i in range(Nuser):
        # TODO: 修改yitaa[i]的计算方式，计算每个用户的能耗，调研传输1bit需要的能耗
        yitaa[i] = 1/h[i] * 0.0001

    Ati = np.zeros(Nuser)
    for i in range(Nuser):
        if m[i] == 1:
            Ati[i] = Q[i]
    # 利用MEC_user_Map组装一个1*Nuser的向量，作为参数传入Getaits函数
    aits, Phi_t, transCost, computationCost, TaskBenefits, BenefitD = mecRA.QueueUpdate(t, Ati, Ati, yitaa,user_MEC_Map)

    # f_val = 传输开销-atis接入收益
    UtransCost = sum(yitaa * aits)
    f_val = TaskBenefits - UtransCost
    rate = aits
    energy = yitaa
    # print("f_val: ", f_val, "rate: ", rate, "energy: ", energy)
    return f_val, rate, energy

def SingleMECRAEvaluate(mecRA,MECNum, MEC_user_Map, m, h, w, Q, Y, V, t, user_MEC_Map):
    """
    对单个mec服务器的动作进行评估
    :param mecRA: MECRA类
    :param MECNum: 对应的MEC服务器编号
    :param MEC_user_Map: 用户到MEC服务器的映射
    :return:
    f_val: 目标值
    rate: 计算速率，表示这个时隙内每个用户的数据Q减少的数量
    energy: 能耗， 表示这个时隙内每个用户能量Y消耗的数量
    """
    NuserNum = len(MEC_user_Map[MECNum])
    f_val = 0
    rate = np.zeros(Nuser)
    energy = np.zeros(Nuser)
    yitaa = np.zeros(Nuser)
    # 组装yitaa，Ati，作为MECEvaluate的输入
    for i in range(NuserNum):
        yitaa[MEC_user_Map[MECNum][i]] = 1/h[MEC_user_Map[MECNum][i]] * 0.0001
    Ati = np.zeros(NuserNum)
    for i in range(NuserNum):
        if m[MEC_user_Map[MECNum][i]] == 1:
            Ati[MEC_user_Map[MECNum][i]] = Q[MEC_user_Map[MECNum][i]]
    aits, Phi_t, transCost, computationCost, TaskBenefits, BenefitD = mecRA.MECEvaluate(t, MECNum, Ati, yitaa, user_MEC_Map)
    # 给f_val赋值，作为目标函数
    UtransCost = sum(yitaa * aits)
    f_val = TaskBenefits - UtransCost
    rate = aits
    energy = yitaa * aits
    return f_val, rate, energy
def plot_rate(rate_his, rolling_intv=50, ylabel='Normalized Computation Rate'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))

    plt.plot(np.arange(len(rate_array)) + 1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    # plt.fill_between(np.arange(len(rate_array)) + 1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values),
    #                  np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color='b', alpha=0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.show()

def plot_optFrame(FrameX, FrameY):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))

    plt.plot(FrameX, FrameY, 'b')
    plt.ylabel('Objective Value')
    plt.xlabel('Time Frames')
    plt.show()
def plot_double_rate(rate_his1, rate_his2, rolling_intv=50, legend1='1', legend2='2'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl


    mpl.style.use('seaborn')

    plt.plot(np.arange(len(rate_his1)) + 1, np.hstack(rate_his1), 'b')
    plt.plot(np.arange(len(rate_his2)) + 1, np.hstack(rate_his2), 'r')
    plt.ylabel('Rate')
    plt.xlabel('Time Frames')
    plt.legend([legend1, legend2])
    plt.title('Comparison of ' + legend1 + ' and ' + legend2)
    plt.show()
#利用功率h和视线比因子生成racian衰落信道
#必要时用您自己的频道代替换它
def racian_mec(h, factor):
    n = len(h)
    beta = np.sqrt(h * factor)  # LOS channel amplitude
    sigma = np.sqrt(h * (1 - factor) / 2)  # scattering sdv
    x = np.multiply(sigma * np.ones((n)), np.random.randn(n)) + beta * np.ones((n))
    y = np.multiply(sigma * np.ones((n)), np.random.randn(n))
    g = np.power(x, 2) + np.power(y, 2)
    return g

if __name__ == "__main__":
    '''
        LyDROO algorithm composed of four steps:
            1) 'Actor module'
            2) 'Critic module'
            3) 'Policy update module'
            4) ‘Queueing module’ of
    '''

    Nuser = 10  # number of users
    n = 500  # number of time frames
    K = Nuser  # initialize K = Nuser
    decoder_mode = 'OPN'  # the quantization mode could be 'OP' (Order-preserving) or 'KNN' or 'OPN' (Order-Preserving with noise)
    Memory = 1024  # capacity of memory structure
    Delta = 32  # Update interval for adaptive K
    CHFACT = 10 ** 10  # The factor for scaling channel value
    energy_thresh = np.ones((Nuser)) * 0.08  # energy comsumption threshold in J per time slot
    nu = 1000  # energy queue factor;
    w = [1.5 if i % 2 == 0 else 1 for i in range(Nuser)]  # weights for each user
    V = 20
    userComputingRate = 0.07

    arrival_lambda = 0.1 * np.ones((Nuser))  # average data arrival, 3 Mbps per user

    print(
        '#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d' % (
        Nuser, n, K, decoder_mode, Memory, Delta))

    # initialize data
    channel = np.zeros((n, Nuser))  # chanel gains
    dataA = np.zeros((n, Nuser))  # arrival data size

    # generate channel
    dist_v = np.linspace(start=120, stop=255, num=Nuser)
    Ad = 3
    fc = 915 * 10 ** 6
    loss_exponent = 3  # path loss exponent
    light = 3 * 10 ** 8
    h0 = np.ones((Nuser))
    for j in range(0, Nuser):
        h0[j] = Ad * (light / 4 / math.pi / fc / dist_v[j]) ** (loss_exponent)

    # TODO: 给mem net加1个维度，输入的时候加1个维度mec.Q，表示根据mec队列长度进行决策
    mem = MemoryDNN(net=[Nuser * 3, 256, 128, Nuser],
                    learning_rate=0.01,
                    training_interval=20,
                    batch_size=128,
                    memory_size=Memory
                    )
    mode_his = []  # store the offloading mode
    k_idx_his = []  # store the index of optimal offloading actor
    FrameX = []
    FrameY = []
    m_temp = np.zeros((Nuser))
    Q = np.zeros((n, Nuser))  # data queue in MbitsW
    Y = np.zeros((n, Nuser))  # virtual energy queue in mJ
    Obj = np.zeros(n)  # objective values after solving problem (26)
    energy = np.zeros((n, Nuser))  # energy consumption
    rate = np.zeros((n, Nuser))  # achieved computation rate

    # MECRA类的初始化
    # 参数设置
    NMEC = 10  # MEC服务器数量
    user_MEC_Map = {}  # user to MEC server mapping
    MEC_user_Map = {}  # MEC server to user mapping
    for i in range(Nuser):
        user_MEC_Map[i] = i // NMEC
        if i // NMEC not in MEC_user_Map:
            MEC_user_Map[i // NMEC] = []
        MEC_user_Map[i // NMEC].append(i)
    n = n  # 时间步数
    epsilon = 1 / 50  # SGD步长
    rhos = np.ones(NMEC) * 200  # 计算密度
    xi_s = np.ones(NMEC) * 1  # 处理后数据比例
    Qtis = np.zeros((n, NMEC, NMEC))  # data queue in MbitsW
    Dtis = np.zeros((n, NMEC, NMEC))  # answer data queue in MbitsW
    Ati = np.ones((n, NMEC))  # access data queue in MbitsW
    Fti = np.random.rand(n, NMEC) * 2000  # computing resource queue in CPU cycles
    Ctij = np.ones((n, NMEC, NMEC))  # channel capacity queue
    alphai = np.ones(NMEC) * 3  # 边缘服务器i接入单位大小任务所得收益
    zetai = np.ones(NMEC) / 100000  # 边缘服务器i的计算开销系数
    zetaij = np.ones((NMEC, NMEC)) * 0.01  # 边缘服务器i到j的传输开销系数
    # 初始化
    mecRA = MEC.MECResourceAllocation(NMEC, Nuser, n, rhos, xi_s, Qtis, Dtis, Fti, Ctij, alphai, epsilon, zetai, zetaij)

    for i in range(n):
        progress_bar(i, n - 1)

        if i > 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(np.array(k_idx_his[-Delta:-1]) % K) + 1
            else:
                max_k = k_idx_his[-1] + 1
            K = min(max_k + 1, Nuser)

        i_idx = i

        # real-time channel generation
        h_tmp = racian_mec(h0, 0.3)
        if CHANNELMODE == "SAT":
            h_tmp = sat_channel(np.ones((Nuser)))
        # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
        h = h_tmp * CHFACT
        channel[i, :] = h
        # real-time arrival generation
        # 从指数分布中采样的实时到达时间，arrival_lambda是每个用户的平均到达速率,为一个N-dimensional array(ndarray)
        dataA[i, :] = np.random.exponential(arrival_lambda)

        # 4) ‘Queueing module’ of LyDROO
        if i_idx > 0:
            # update queues
            # Q: 数据队列，Y：能量队列
            # dataA[i_idx - 1, :]：上一时刻的数据到达量，
            # rate[i_idx - 1, :]：上一时刻的计算速率
            ResourceAllocation(mecRA, MEC_user_Map, user_MEC_Map, m_temp, h, w, Q[i_idx - 1, :], Y[i_idx - 1, :], V,
                               i_idx - 1)
            Q[i_idx, :] = Q[i_idx - 1, :] + dataA[i_idx - 1, :] - rate[i_idx - 1, :]  # current data queue
            # assert Q is positive due to float error
            Q[i_idx, Q[i_idx, :] < 0] = 0
            # energy_thresh: 能量消耗阈值
            # nu: 能量队列因子
            Y[i_idx, :] = np.maximum(Y[i_idx - 1, :] + (energy[i_idx - 1, :] - energy_thresh) * nu,
                                     0)  # current energy queue
            # assert Y is positive due to float error
            Y[i_idx, Y[i_idx, :] < 0] = 0

        # scale Q and Y to close to 1; a deep learning trick
        # TODO:分NMEC次分别调用，得到每个MEC服务器的动作
        m_temp = np.zeros((Nuser))
        for i_MEC in range(NMEC):
            # 输入每个用户信道增益h、数据队列Q和能量队列Y
            # TODO: 基于i_MEC和MEC_user_Map[i_MEC]得到每个MEC服务器的所有用户的h和数据队列和能量队列，组装nn_input
            # nn_input = np.concatenate((h, Q[i_idx, :] / 10000, Y[i_idx, :] / 10000))
            if i_MEC not in MEC_user_Map:
                continue
            MECuserList_temp = MEC_user_Map[i_MEC]
            if len(MECuserList_temp) == 0:
                continue
            nn_input = np.concatenate(
                (h[MECuserList_temp], Q[i_idx, MECuserList_temp] / 10000, Y[i_idx, MECuserList_temp] / 10000))

            # 1) 'Actor module' of LyDROO
            # generate a batch of actions
            # nn_input表示观测值，K表示离散化的动作空间大小（用户数目），decoder_mode表示解码器模式
            m_list = mem.decode(nn_input, K, decoder_mode)

            r_list = []  # all results of candidate offloading modes
            v_list = []  # the objective values of candidate offloading modes
            # 2) Critic：对m_list中的每个动作m，调用Algo1_NUM函数计算其目标值、计算速率和能耗
            for m in m_list:
                # 2) 'Critic module' of LyDROO
                # allocate resource for all generated offloading modes saved in m_list
                r_list.append(SingleMECRAEvaluate(mecRA, 0, MEC_user_Map, m, h, w, Q[i_idx, :], Y[i_idx, :], V, i_idx,
                                                  user_MEC_Map))
                # v_list用于记录并比较每个MEC服务器的目标值，找到最大的目标值对应的m作为决策动作
                v_list.append(r_list[-1][0])

            # print('Frame %d, Max Reward = %f' % (i, max(v_list)))
            # FrameX.append(i)
            # FrameY.append(max(v_list))

            # 取f_val最大值的作为动作记录
            k_idx_his.append(np.argmax(v_list))

            # 3) 'Policy update module' of LyDROO
            # 调用mem的encode函数，将最大的目标值对应的m作为动作记录，用于训练
            mem.encode(nn_input, m_list[k_idx_his[-1]])
            mode_his.append(m_list[k_idx_his[-1]])
            # TODO: 根据选择的动作，组装MEC所对应的users的m_temp和rate和energy,Obj
            # Obj: 目标值，取所有的MEC服务器的目标值的和
            # rate: 计算速率，对所有MEC服务器管辖的用户，使用MEC_user_Map组装成1*Nuser的向量，计算每个用户的计算速率，用于更新Q
            # energy: 能耗，对所有MEC服务器管辖的用户，使用MEC_user_Map组装成1*Nuser的向量，计算每个用户的能耗，用于更新Y
            MECuserList = MEC_user_Map[i_MEC]
            MECuserNumber = len(MECuserList)
            MEC_m_temp = m_list[k_idx_his[-1]]
            Objtemp, ratetemp, energytemp = r_list[k_idx_his[-1]]
            for i_user in range(MECuserNumber):
                m_temp[MECuserList[i_user]] = MEC_m_temp[i_user]
                rate[i_idx, MECuserList[i_user]] = ratetemp[MECuserList[i_user]] + userComputingRate
                energy[i_idx, MECuserList[i_user]] = energytemp[MECuserList[i_user]]
            Obj[i_idx] += Objtemp

    mem.plot_cost()
    mecRA.QuePlot()

    plot_rate(Q.sum(axis=1) / Nuser, 100, 'Average Data Queue')
    plot_rate(energy.sum(axis=1) / Nuser, 100, 'Average Energy Consumption')

    plot_double_rate(Q.sum(axis=1) / Nuser, rate.sum(axis=1) / Nuser, 100, 'Average Data Queue',
                     'Average Computation Rate')

    plot_optFrame(FrameX, FrameY)

    # save all data
    # sio.savemat('./result_%d.mat' % Nuser,
    #             {'input_h': channel / CHFACT, 'data_arrival': dataA, 'data_queue': Q, 'energy_queue': Y,
    #              'off_mode': mode_his, 'rate': rate, 'energy_consumption': energy, 'data_rate': rate, 'objective': Obj})