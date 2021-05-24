import numpy as np
import matplotlib.pyplot as plt
import math


def rho(x):
    return 2 * math.sin(x) + x * math.cos(x)


def test_phi(x):
    return x * np.cos(x)




def epsilon(x):
    return 1


def get_k(x, dx, phi1, phi2):
    return ((epsilon(x) + epsilon(x + dx)) * phi1 + (epsilon(x) + epsilon(x - dx)) * phi2 + 2 * (dx ** 2) * rho(x)) / (
                4 * epsilon(x))

def relaxation_kernel(phi,rho,epsilon,N,a,b,tol):
    diff = 10
    h = (b-a)/N
    phi_dis = np.zeros(N)
    rho_dis = np.zeros(N)
    epsilon_dis = np.zeros(N)
    for i in range(N):
        phi_dis[i] = phi(a + i * h)
        rho_dis[i] = rho(a + i*h)
        epsilon_dis[i] = epsilon(a+i*h)

    while diff > tol:
        diff = 0
        i = 1
        while i < N-1:
            buffer = ((epsilon_dis[i+1] + epsilon_dis[i]) * phi_dis[i+1] + (epsilon_dis[i] + epsilon_dis[i-1]) * phi_dis[i-1] + 2 * (h ** 2) * rho_dis[i]) / (
                4 * epsilon_dis[i])
            buffer =buffer * 1.3 - 0.3 * phi_dis[i]
            diff+= (abs(phi_dis[i]-buffer)**2)
            phi_dis[i] = buffer
            i+=1

    return phi_dis


if __name__ == '__main__':
    N = 64
    a = 0
    b= 2 * math.pi
    tol = 1e-6
    def linear_test(x):
        return x
    x_step = np.linspace(0,2 * math.pi, N)
    phi_step_8 = relaxation_kernel(linear_test,rho,epsilon,8,0,2* math.pi, tol)
    phi_step_16 = relaxation_kernel(linear_test,rho,epsilon,16,0,2* math.pi, tol)
    phi_step_32 = relaxation_kernel(linear_test,rho,epsilon,32,0,2* math.pi, tol)
    phi_step_64 = relaxation_kernel(linear_test,rho,epsilon,64,0,2* math.pi, tol)
    fig,ax = plt.subplots(1, 1, figsize=plt.figaspect(1 / 2), tight_layout=True)
    titles = 'poisson equation for one dimention,with linear guess, tolerance = {tol}, produced by Yang Ao'.format(tol=tol)
    ax.set_title(titles)
    ax.set_xlabel("x")
    ax.set_ylabel("phi")
    ax.plot(np.linspace(0,2 * math.pi, 8),phi_step_8,label ='approximation N = 8')
    ax.plot(np.linspace(0,2 * math.pi, 16),phi_step_16,label ='approximation N = 16')
    ax.plot(np.linspace(0,2 * math.pi, 32),phi_step_32,label ='approximation N = 32')
    ax.plot(np.linspace(0,2 * math.pi, 64),phi_step_64,label ='approximation N = 64')
    ax.plot(x_step,test_phi(x_step) , label = 'Analytical solution')
    plt.legend()
    plt.show()
