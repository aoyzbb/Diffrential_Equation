import numpy as np
import matplotlib.pyplot as plt
import math

def boundary_check(i,j):
    r = ((i/5 - 4.5)**2 + (j/5 -4.5)**2)**0.5
    if r > 0.1:
        return False
    return True

def init_guess(i,j):
    if (i,j)==(25,25):
        return 0
    r = ((i/5 - 5)**2 + (j/5 -5)**2)**0.5
    return 10 * math.e **(-r) / r


if __name__ == '__main__':
    xx = np.linspace(0,10,11)
    yy = np.linspace(0,10,11)
    X, Y = np.meshgrid(xx,yy)
    phi = np.zeros_like(X)
    phi_subdiv = np.zeros((51,51))
    tol = 1e-3
    diff = 10
    for i in range(51):
        for j in range(51):
            phi_subdiv[i,j] = init_guess(i,j)

    while diff < tol:
        diff = 0
        i = 1
        while i < 50:
            j=1
            while j <50:
                if boundary_check(i,j) == 0:
                    buffer = (phi_subdiv[i,j+1]+phi_subdiv[i,j-1]+phi_subdiv[i-1,j]+ phi_subdiv[i+1,j] - 0.04 * math.sinh(phi_subdiv[i,j]))/4
                    buffer =buffer * 1.3 - 0.3 * phi_subdiv[i,j]
                    diff += abs(buffer-phi_subdiv[i,j])
                    phi_subdiv[i,j] =buffer
                j+=1
            i+=1



    for i in range(11):
        for j in range(11):
            phi[i,j] = phi_subdiv[i * 5,j *5]

    fig = plt.figure()  #定义新的三维坐标轴
    titles = 'Ex4, Poisson-Boltzmann equation in 3d '

    ax3 = plt.axes(projection='3d')
    ax3.set_title(titles)
    ax3.plot_surface(X,Y,phi,cmap='rainbow')
    plt.show()
