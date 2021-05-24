import numpy as np
import matplotlib.pyplot as plt
from random import random


class Percolation(object):
    def __init__(self, n=2, p=0.5):
        self.N = n
        self.p = p
        self.state = 0
        self.cfg = np.zeros((self.N, self.N))

    def change_cfg(self):
        for i in range(self.N):
            for j in range(self.N):

                if random() <= self.p:
                    self.cfg[i, j] = 1
                else:
                    self.cfg[i, j] = 0

    def plot_cfg(self):
        plt.imshow(self.cfg, cmap=plt.cm.cool, vmin=0, vmax=1)
        plt.colorbar()
        plt.show()

    def judge_perco(self):
        self.change_cfg()
        y_verticle = self.cfg.sum(axis=0)
        y_horizontal = self.cfg.sum(axis=1)
        if y_verticle.all() >= 1 or y_horizontal.all() >= 1:
            self.state = 1

        else:
            self.state = 0

    def find_prob(self, times=100000):
        count = 0
        for i in range(times):
            self.judge_perco()
            if self.state == 1:
                count += 1

        return count / times

    def threshold_find(self, tol):
        """
        Find the one that make the probability =0.5?
        Since the percolation probability changes with p, use bisection to find the first value that its neighborhood is close to 0 and 1
        """
        self.p = 0.5
        per_pro = self.find_prob(10000)
        if per_pro > 0.5:
            a, b = 0, 0.5
        elif per_pro == 0.5:
            return self.p
        else:
            a, b = 0.5, 1

        while abs(per_pro - 0.5) > tol:
            self.p = (a + b) / 2
            per_pro = self.find_prob(10000)
            if per_pro > 0.5:
                a, b = a, self.p
            elif per_pro == 0.5:
                return self.p
            else:
                a, b = self.p, b

        self.p = (a + b) / 2
        return (a + b) / 2

    def cluster_find(self):
        """
        find the size of cluster, return an array
         1.stack(list). Use it to track the element that has been checked
         2.ndarray flag:check whether an position has been checked or not
         :return nadarray: each element represent size of a cluaster
        """
        flag = np.zeros_like(self.cfg)
        checked = []
        result = []
        count = 0

        def check_ele(a):
            i, j = a[0], a[1]
            if flag[i, j] == 1:
                return 0
            else:
                flag[i, j] = 1
                if self.cfg[i, j] == 0:
                    return 0
                else:
                    if j + 1 < self.N:
                        checked.append((i, j + 1))
                    if j - 1 >= 0:
                        checked.append((i, j - 1))
                    if i + 1 < self.N:
                        checked.append((i + 1, j))
                    if i - 1 >= 0:
                        checked.append((i - 1, j))
                    return 1

        for i in range(self.N):
            for j in range(self.N):
                if check_ele((i, j)) == 0:
                    continue
                else:
                    count += 1
                    while len(checked) > 0:
                        count += check_ele(checked.pop())

                    result.append(count)
                    count = 0
        return result

    def clusters_hist(self, times = 10000):
        result = []
        for i in range(times):
            self.change_cfg()
            result += self.cluster_find()

        fig, ax = plt.subplots(1, 1, figsize=plt.figaspect(1 / 2), tight_layout=True)
        titles = 'Percolation cluster size, with N ={N},p={p}, repeat {times} configurations'.format(N=self.N,p=self.p,times= times)
        ax.set_title(titles)
        ax.set_xlabel("size")
        ax.set_ylabel("count")

        #print(result)
        ax.hist(result, bins=self.N * self.N, range=(0, self.N * self.N - 1))
        plt.show()


def anal(p):
    return p ** 4 + 4 * (p ** 3) * (1 - p) + 6 * (p ** 2) * ((1 - p) ** 2)


def Ex2():
    two_dim = Percolation(n=2, p=0.1)
    step_p = np.linspace(0, 1, 100)
    pro_p = np.linspace(0, 1, 100)
    for i in range(100):
        two_dim.p = step_p[i]
        pro_p[i] = two_dim.find_prob(10000)
    fig, ax = plt.subplots(1, 1, figsize=plt.figaspect(1 / 2), tight_layout=True)
    titles = 'Percolation with n = 2'
    ax.set_title(titles)
    ax.set_xlabel("p")
    ax.set_ylabel("probability")
    ax.plot(step_p, pro_p, label='MC')
    ax.plot(np.linspace(0, 1, 100), anal(np.linspace(0, 1, 100)), label='Analytical solution')
    plt.legend()
    plt.show()


def Ex3():
    dimension_step = np.array(range(2, 10))
    threshold = np.array(range(2, 10),dtype = float)

    for i in range(2, 10):
        two_dim = Percolation(n=dimension_step[i-2], p=0.4)
        a = two_dim.threshold_find(1e-3)
        print(i-2)
        #print(a)
        threshold[i-2] = a
        #print(threshold[i-2])

    print(threshold)

    fig, ax = plt.subplots(1, 1, figsize=plt.figaspect(1 / 2), tight_layout=True)
    titles = 'Percolation with respect to N'
    ax.set_title(titles)
    ax.set_xlabel("dimension")
    ax.set_ylabel("threshold")
    ax.scatter(dimension_step, threshold)
    for a, b in zip(dimension_step, threshold):
        ax.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)
    plt.show()


if __name__ == '__main__':
    two_dim = Percolation(n=7 , p = 0.4)
    two_dim.clusters_hist()
