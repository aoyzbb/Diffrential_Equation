import numpy as np
from computational_physics.roots.secant import secant
import matplotlib.pyplot as plt
import math

def forward_get_next(t_k,y_k,func,dt,*args):
    t_new = t_k + dt
    y_new = func(t_k,y_k,*args) * dt + y_k

def backeuler_get_next(t_k,y_k,func,dt, *args):
    """
    Calculate the next point of the system.

    :param callable func: function ,note that y_{k+1} = y_k + dt * func(y_{k+1})
    """
    t_new = t_k + dt
    def help_func(y):
        return func(y,*args)*dt + y_k - y

    y_new = secant(help_func,y_k, y_k+func(y_k,*args)*dt,1e-4)

    return t_new , y_new

def diff_x(x_k1,v_k1):
    return v_k1

def diff_genrator_v(dt, omega, ksi):
    def diff_v(v_k1, x_k):
        return -2 * v_k1 * ksi * omega - x_k * (omega**2) -dt  *v_k1 * (omega ** 2)

    return diff_v

if __name__ == '__main__':
    fig,ax = plt.subplots(1, 1, figsize=plt.figaspect(1 / 2), tight_layout=True)
    dt = 0.5
    t_point = np.arange(0,100,dt)

    x_point=np.zeros(t_point.shape)
    v_point= np.zeros(t_point.shape)
    x_point[0],v_point[0]=2,0
    omega = 2 * math.pi
    ksi = 0.25
    diff_v = diff_genrator_v(dt,omega,ksi)
    i = 0
    N = t_point.size
    while i<N-1:
        t1 ,v_point[i+1] = backeuler_get_next(t_point[i],v_point[i],diff_v,0.01,x_point[i])
        t2 ,x_point[i+1] = backeuler_get_next(t_point[i],x_point[i],diff_x,0.01,v_point[i+1])
        assert abs(t1-t2)<1e-4
        i+=1

    titles = 'Back Euler to solve Spring Mass Damp, with omega = {omega} ,ksi ={ksi},dt = {dt}, produced by Yang Ao'.format(omega = omega,ksi =ksi,dt =dt)
    ax.set_title(titles)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.plot(x_point,v_point)
    plt.show()
