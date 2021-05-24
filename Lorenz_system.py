import numpy as np
import matplotlib.pyplot as plt


def get_new_point(y, dt, k1, k2, k3, k4):
    '''
    Use Runge-Kutta to 4th to calculate the new y_k+1 , k1,k2,k3,k4 should be determined previously.
    '''
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def lorenz_sys(sigma=10, rho=28, beta=8 / 3):
    '''
    (x,y,z)->(x,y,z)
    '''

    def diff_x(dt, dx, x, y, z):
        return sigma * (y + dt * (x * (rho - z) - y) - x - dx)

    def diff_y(dt, dy, x, y, z):
        return (x + dt * sigma * (y - z)) * (rho - (z + dt * (x * y - beta * z))) - y - dy

    def diff_z(dt, dz, x, y, z):
        return (x + dt * sigma * (y - z)) * (y + dt * (x * (rho - z))) - beta * (z + dz)

    return diff_x, diff_y, diff_z


if __name__ == '__main__':
    rho = 26
    diff_x, diff_y, diff_z = lorenz_sys(rho = rho)

    t_point = np.arange(0, 100, 0.01)
    x_point = np.zeros(t_point.shape)
    y_point = np.zeros(t_point.shape)
    z_point = np.zeros(t_point.shape)
    dt = 0.01
    x_point[0] , y_point[0],z_point[0]  = 0,1,0
    i = 0
    while i<9999:
        kx1 = diff_x(0,0,x_point[i],y_point[i],z_point[i])
        kx2 = diff_x(dt/2,dt * kx1 /2,x_point[i],y_point[i],z_point[i])
        kx3 = diff_x(dt/2, dt * kx1 /2,x_point[i],y_point[i],z_point[i])
        kx4 = diff_x(dt, dt*kx3,x_point[i],y_point[i],z_point[i])

        ky1 = diff_y(0,0,x_point[i],y_point[i],z_point[i])
        ky2 = diff_y(dt/2,dt * ky1 /2,x_point[i],y_point[i],z_point[i])
        ky3 = diff_y(dt/2,dt * ky1 /2,x_point[i],y_point[i],z_point[i])
        ky4 = diff_y(dt, dt*ky3,x_point[i],y_point[i],z_point[i])

        kz1 = diff_z(0,0,x_point[i],y_point[i],z_point[i])
        kz2 = diff_z(dt/2,dt * kz1 /2,x_point[i],y_point[i],z_point[i])
        kz3 = diff_z(dt/2,dt * kz1 /2,x_point[i],y_point[i],z_point[i])
        kz4 = diff_z(dt, dt*kz3,x_point[i],y_point[i],z_point[i])

        x_point[i+1] = get_new_point(x_point[i],dt,kx1,kx2,kx3,kx4)
        y_point[i+1] = get_new_point(y_point[i],dt,ky1,ky2,ky3,ky4)
        z_point[i+1] = get_new_point(z_point[i],dt,kz1,kz2,kz3,kz4)
        i+=1
        #print(i,x_point[i],y_point[i],z_point[i])
    titles = 'Lorenz_system, with rho = {rho},dt={dt}'.format(rho = rho, dt=dt)
# set figure information
#    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(titles)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    figure = ax.plot(x_point,y_point,z_point)
    plt.show()
    '''
    fig,ax = plt.subplots(1, 1, figsize=plt.figaspect(1 / 2), tight_layout=True)
    titles = 'Lorenz system for dt = {dt} '.format(dt =dt)
    ax.set_title(titles)
    ax.set_xlabel("y_point")
    ax.set_ylabel("z_point")
    ax.plot(y_point,z_point)
    plt.show()
    '''
