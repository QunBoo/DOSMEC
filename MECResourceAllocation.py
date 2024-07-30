from processBar import progress_bar

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['simhei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class MECResourceAllocation:
    def __init__(self, N, Nuser, n, rhos, xi_s, Qtis, Dtis, Fti, Ctij, alphai, epsilon, zetai, zetaij):
        self.N = N
        self.Nuser = Nuser
        self.n = n
        self.rhos = rhos
        self.xi_s = xi_s
        self.Qtis = Qtis
        self.Dtis = Dtis
        self.Fti = Fti
        self.Ctij = Ctij
        self.alphai = alphai
        self.epsilon = epsilon
        self.zetai = zetai
        self.zetaij = zetaij
        self.fitSum = []
        self.bitsSum = []
        self.dtijSum = []

    def QueueUpdate(self, t, X, A, yitaa,user_MEC_Map):
        """
        队列更新函数
        :param t: 时隙t
        :param X: 接入决策
        :param A: 数据准备到达量
        :return: aits, Phi_t, transCost, computationCost, TaskBenefits, BenefitD
        TaskBenefits: 任务收益, 表示任务接入带来的收益
        """
        # 计算aits
        aits = self.Getaits(t, A, yitaa, user_MEC_Map)
        aits_sum = np.zeros(self.N)
        for i_user in range(self.Nuser):
            aits_sum[user_MEC_Map[i_user]] += aits[i_user]
        # 计算fits
        fits = self.Getfits(t, A)
        # 计算btijs
        btijs, dtijs = self.Getbtijsdtijs(t, A)

        # 计算Qtis
        Qtis = np.zeros((self.N, self.N))
        for i in range(self.N):
            for s in range(self.N):
                total1 = 0
                total2 = 0
                for j in range(self.N):
                    total1 += btijs[i][j][s]
                    total2 += btijs[j][i][s]
                Qtis[i][s] = (max(self.Qtis[t][i][s] - fits[i][s] / self.rhos[s] - total1, 0) + total2 + aits_sum[s])
        # 计算Dtis
        Dtis = np.zeros((self.N, self.N))
        BenefitD = 0
        for i in range(self.N):
            for s in range(self.N):

                total1 = 0
                total2 = 0
                for j in range(self.N):
                    total1 += dtijs[i][j][s]
                    total2 += dtijs[j][i][s]
                if i == s:
                    # 如果是自己，说明任务已经完成，计算收益
                    BenefitD += self.alphai[i] * total2
                    continue
                Dtis[i][s] = (max(self.Dtis[t][i][s] - total1, 0) + total2 + self.xi_s[s] * fits[i][s] / self.rhos[s])

        self.Qtis[t + 1] = Qtis
        self.Dtis[t + 1] = Dtis

        Phi_t, transCost, computationCost, TaskBenefits = self.GetPhi_t(t, btijs, dtijs, fits, self.zetai, self.zetaij, self.alphai, aits)
        return aits, Phi_t, transCost, computationCost, TaskBenefits, BenefitD
    def MECEvaluate(self, t, MECNum, A, yitaa, user_MEC_Map):
        """
        评估函数, 输入给一个MEC服务器的任务量， 计算aits以及收益
        :param t: 时隙t
        :param MECNum: MEC服务器编号
        :param A: 数据准备到达量，1*Nuser的矩阵
        :param yitaa: 通信开销，传输1bit需要的能耗, 1*Nuser
        :param user_MEC_Map: 用户到MEC服务器的映射, 1*Nuser的矩阵
        :return: aits: 1*Nuser的矩阵
        """
        # 计算aits
        # 使用输入A、yitaa, user_MEC_Map计算aits，用于评估能被接入的数据量
        aits = self.Getaits(t, A, yitaa, user_MEC_Map)
        Phi_t, transCost, computationCost, BenefitD = {}, {}, {}, {}
        TaskBenefits = 0
        for i in range(self.Nuser):
            TaskBenefits += aits[i]
        return aits, Phi_t, transCost, computationCost, TaskBenefits, BenefitD
    def Getaits(self, t, A, yitaa, user_MEC_Map):
        """
        计算aits
        :param t: 时隙t
        :param A: 数据准备到达量, 1*Nuser的矩阵
        :param yitaa: 通信开销，传输1bit需要的能耗
        :return: aits 1*Nuser的矩阵
        """
        aits = np.zeros(self.Nuser)
        if len(A)!=self.Nuser:
            print("aits输入参数错误")
            exit(0)
        # 遍历A的每个用户，通过Map找到对应服务器的Qtis，计算aits
        for i_user in range(self.Nuser):
            if A[i_user] > 0:
                # 组装后待接入数据量>0
                i_MECNum = user_MEC_Map[i_user]
                alphai = self.alphai[i_MECNum]
                epsilon = self.epsilon
                QtisTemp = self.Qtis[t][i_MECNum][i_MECNum]
                if QtisTemp < (alphai - yitaa[i_user]) / epsilon:
                    aits[i_user] = A[i_user]
        return aits

    def Getfits(self, t, A):
        """
        计算fits
        :param t: 时隙t
        :param A: 数据准备到达量
        :return: fits
        """
        fits = np.zeros((self.N, self.N))
        kappa_tis = self.Getkappa_tis(t)
        for i in range(self.N):
            row = kappa_tis[i]
            max_index = np.argmax(row)
            max_value = row[max_index]
            if max_value > 0:
                fits[i][max_index] = self.Fti[t][i]
        self.fitSum.append(np.sum(fits))
        return fits

    def Getbtijsdtijs(self, t, A):
        """
        计算btijs
        :param t: 时隙t
        :return: btijs
        """
        btijs = np.zeros((self.N, self.N, self.N))
        dtijs = np.zeros((self.N, self.N, self.N))
        beta_tijs = self.Getbeta_tijs(t)
        gamma_tijs = self.Getgamma_tijs(t)
        for i in range(self.N):
            for j in range(self.N):
                maxbtijs_index = np.argmax(beta_tijs[i][j])
                maxbtijs = beta_tijs[i][j][maxbtijs_index]
                maxgammatijs_index = np.argmax(gamma_tijs[i][j])
                maxgammatijs = gamma_tijs[i][j][maxgammatijs_index]
                maxbtjis_index = np.argmax(beta_tijs[j][i])
                maxbtjis = beta_tijs[j][i][maxbtjis_index]
                maxgammatjis_index = np.argmax(gamma_tijs[j][i])
                maxgammatjis = gamma_tijs[j][i][maxgammatjis_index]
                if max(maxbtijs, maxgammatijs) < 0 or max(maxbtijs, maxgammatijs)<max(maxbtjis, maxgammatjis):
                    btijs[i][j][maxbtijs_index] = 0
                elif maxbtijs > maxgammatijs:
                    btijs[i][j][maxbtijs_index] = self.Ctij[t][i][j]
                else:
                    dtijs[i][j][maxgammatijs_index] = self.Ctij[t][i][j]
        self.bitsSum.append(np.sum(btijs))
        self.dtijSum.append(np.sum(dtijs))
        return btijs, dtijs


    def Getkappa_tis(self, t):
        """
        计算kappa_tis, (4-20a)
        :return: kappa_tis N*N的矩阵
        """
        kappa_tis = np.zeros((self.N, self.N))
        epsilon = self.epsilon
        for i in range(self.N):
            for s in range(self.N):
                kappa_tis[i][s] = epsilon * (self.Qtis[t][i][s] - self.xi_s[s]*self.Dtis[t][i][s]) / self.rhos[s] - self.zetai[i]
        return kappa_tis

    def Getbeta_tijs(self, t):
        """
        计算beta_tis, (4-20b)
        :param t:
        :return: beta_tijs N*N*N的矩阵
        """
        beta_tijs = np.zeros((self.N, self.N, self.N))
        epsilon = self.epsilon
        for i in range(self.N):
            for j in range(self.N):
                for s in range(self.N):
                    beta_tijs[i][j][s] = epsilon * (self.Qtis[t][i][s] - self.Qtis[t][j][s]) - self.zetaij[i][j]
        return beta_tijs

    def Getgamma_tijs(self, t):
        """
        计算gamma_tis, (4-20c)
        :param t:
        :return: gamma_tijs N*N*N的矩阵
        """
        gamma_tijs = np.zeros((self.N, self.N, self.N))
        epsilon = self.epsilon
        for i in range(self.N):
            for j in range(self.N):
                for s in range(self.N):
                    gamma_tijs[i][j][s] = epsilon * (self.Dtis[t][i][s] - self.Dtis[t][j][s]) - self.zetaij[i][j]
        return gamma_tijs

    def GetPhi_t(self, t, btijs, dtijs, ftis, zetai, zetaij, alphai, atis):
        """
        计算Phi_t, (4-6)
        :param t:
        :return: Phi_t N*N的矩阵
        """
        Phi_t = 0
        transCost = 0
        for i in range(self.N):
            for j in range(self.N):
                total1 = 0
                for s in range(self.N):
                    total1 += (btijs[i][j][s] + dtijs[i][j][s])
                transCost += zetaij[i][j] * total1

        computationCost = 0
        for i in range(self.N):
            total1 = 0
            for s in range(self.N):
                total1 += ftis[i][s]
            computationCost += zetai[i] * total1

        TaskBenefits = 0
        for i_user in range(self.Nuser):
            TaskBenefits += alphai[i_user] * atis[i_user]
        Phi_t = transCost + computationCost - TaskBenefits
        return Phi_t, transCost, computationCost, TaskBenefits

    def QuePlot(self):
        """
        对Qtis, Dtis, Atis, Ftis进行画图
        :return:
        """
        # 画Qtis
        import pandas as pd
        rolling_intv = 100
        sumsQtis = np.sum(self.Qtis, axis=2)
        plt.figure()
        for i in range(self.N):
            # 对当前线进行滚动窗口求平均值
            rolled_mean = pd.Series(sumsQtis[:, i]).rolling(window=rolling_intv, min_periods=1).mean()
            plt.plot(range(self.n), rolled_mean, label=f'Line {i + 1}')
        plt.xlabel('time slot')
        plt.ylabel('每个 MEC 服务器的 Qtis 长度')
        plt.title('Qtis')
        plt.legend()
        plt.show()
        # 画Dtis
        sumsDtis = np.sum(self.Dtis, axis=2)
        plt.figure()
        for i in range(self.N):
            # 对当前线进行滚动窗口求平均值
            rolled_mean = pd.Series(sumsDtis[:, i]).rolling(window=rolling_intv, min_periods=1).mean()
            plt.plot(range(self.n), rolled_mean, label=f'Line {i + 1}')
        plt.xlabel('time slot')
        plt.ylabel('每个MEC服务器的Dtis长度')
        plt.title('Dtis')
        plt.legend()
        plt.show()
        # 画Ftis
        plt.figure()
        for i in range(self.N):
            rolled_mean = pd.Series(self.Fti[:, i]).rolling(window=rolling_intv, min_periods=1).mean()
            plt.plot(range(self.n), rolled_mean, label=f'Line {i + 1}')
        plt.xlabel('time slot')
        plt.ylabel('每个MEC服务器的计算量Ftis')
        plt.title('Fti')
        plt.legend()
        plt.show()

        return

if __name__ == "__main__":
    # 参数设置
    N = 10  # MEC服务器数量
    n = 3000  # 时间步数
    epsilon = 1/50  # SGD步长
    rhos = np.ones(N) * 2000   # 计算密度
    xi_s = np.ones(N) * 1      # 处理后数据比例
    Qtis = np.zeros((n, N, N))  # data queue in MbitsW
    Dtis = np.zeros((n, N, N))  # answer data queue in MbitsW
    Ati = np.ones((n, N))  # access data queue in MbitsW
    Fti = np.random.rand(n, N) * 2000  # computing resource queue in CPU cycles
    Ctij = np.ones((n, N, N))  # channel capacity queue
    alphai = np.ones(N)*3   # 边缘服务器i接入单位大小任务所得收益
    zetai = np.ones(N)/100000   # 边缘服务器i的计算开销系数
    zetaij = np.ones((N, N))*0.01  # 边缘服务器i到j的传输开销系数

    # 随机生成A
    Ati = np.random.rand(n, N)
    for i in range(n):
        for j in range(N):
            if Ati[i][j] < 0.8:
                Ati[i][j] = 0
            else:
                Ati[i][j] = 1
    A_temp = Ati[0]

    # 实例化MECResourceAllocation类
    mecRA = MECResourceAllocation(N, n, rhos, xi_s, Qtis, Dtis, Fti, Ctij, alphai, epsilon, zetai, zetaij)
    ati_plt = []
    A_temp_plt = []
    Phi_plt = []
    BenefitD_plt = []
    transCost_plt = []
    A_temp_plt.append(sum(A_temp))
    computationCost_plt = []
    TaskBenefits_plt = []
    # 测试QueueUpdate函数
    for i in range(n-1):
        progress_bar(i,n-1)
        aits, Phi_t, transCost, computationCost, TaskBenefits, BenefitD = mecRA.QueueUpdate(i, A_temp, A_temp)
        A_temp += (Ati[i+1]-aits)
        A_temp_plt.append(sum(A_temp))
        ati_plt.append(sum(aits))
        Phi_plt.append(Phi_t)
        BenefitD_plt.append(BenefitD)
        transCost_plt.append(transCost)
        computationCost_plt.append(computationCost)
        TaskBenefits_plt.append(TaskBenefits)

    # 画图
    print("对于每个时隙，计算Phi_t，准备画图")
    plt.plot(Phi_plt, label='Phi')
    # plt.plot(BenefitD_plt, label='BenefitD')
    plt.plot(transCost_plt, label='transCost')
    plt.plot(computationCost_plt, label='computationCost')
    # plt.plot(TaskBenefits_plt, label='TaskBenefits')
    plt.title("Phi_t, 任务开销随时间变化")
    plt.legend()
    plt.show()
#     求mecRA.Qtis的和
    print("对于每个时隙，计算atis的和，准备画图")
    plt.plot(ati_plt, label='ati_plt')
    plt.plot(A_temp_plt, label='A_temp_plt')
    plt.title("ati sum, 队列和随时间变化")
    plt.legend()
    plt.show()

    print("对于每个时隙，计算Qtis的和，准备画图")
    sum = []
    for i in range(n):
        sum.append(np.sum(mecRA.Qtis[i]))
    plt.plot(sum)
    plt.title("Qtis sum, 队列和随时间变化")
    plt.show()

    print("对于每个时隙，计算Dtis的和，准备画图")
    sum = []
    for i in range(n):
        sum.append(np.sum(mecRA.Dtis[i]))
    plt.plot(sum)
    plt.title("Dtis sum, 队列和随时间变化")
    plt.show()

    print("对于每个时隙，计算fits的和，准备画图")
    plt.plot(mecRA.fitSum)
    plt.title("fits sum, 队列和随时间变化")
    plt.show()

    print("对于每个时隙，计算btijs的和，准备画图")
    plt.plot(mecRA.bitsSum)
    plt.title("btijs sum, 队列和随时间变化")
    plt.show()

    print("对于每个时隙，计算dtijs的和，准备画图")
    plt.plot(mecRA.dtijSum)
    plt.title("dtijs sum, 队列和随时间变化")
    plt.show()
