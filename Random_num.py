import numpy as np
import matplotlib.pyplot as plt
import random
import time


class Plot_random(object):
    def __init__(self):
        self.ran = random.random


    def plot_distribution(self, N):
        """
        plot the distribution of the random using method f with a histogram
        """
        x = np.zeros(N)
        for i in range(N):
            x[i] = self.ran()

        titles = 'distribution of the generator with N = {N}'.format(N = N)
        fig, ax = plt.subplots(1, 1, figsize=plt.figaspect(1 / 2), tight_layout=True)
        n, bins, patches = ax.hist(x, bins=10, range=(0, 1))
        ax.set_title(titles)
        plt.show()
        print('bins:', bins)

    def plot_relation(self, N):
        """
        Plot the relation of the element
        """
        x = np.zeros(N)
        for i in range(N):
            x[i] = self.ran()

        y = np.delete(x, 0, 0)
        x = np.delete(x, N - 1, 0)
        titles = 'distribution of the relation with N = {N}'.format(N = N)
        fig, ax = plt.subplots(1, 1, figsize=plt.figaspect(1 / 2), tight_layout=True)
        ax.set_title(titles)
        ax.scatter(x, y)
        ax.set_xlabel("U_i")
        ax.set_ylabel("U_{i+1}")
        plt.show()


class Ran2(Plot_random):
    a = 1229
    m = 2048
    b = 1
    x_n = 1

    def ran2(self):
        self.x_n = (self.x_n * self.a) % self.m - self.b
        return self.x_n / self.m

    def __init__(self):
        self.ran = self.ran2


class Ran4(Plot_random):
    def ran2(self):
        #time.sleep(0.0001)
        return time.time()%1

    def __init__(self):
        self.ran = self.ran2



if __name__ == '__main__':
    ran_t = Ran4()
    ran_t.plot_distribution(5000)
