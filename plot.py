import numpy as np
import matplotlib.pyplot as plt
import paths
import parameters as para

states_path = paths.states
energies_path = paths.energies
x = np.linspace(-para.x_width, para.x_width, para.m ,endpoint=True)
y = np.linspace(-para.y_width, para.y_width, para.n ,endpoint=True)
com = np.linspace(-para.com_width, para.com_width, para.o ,endpoint=True)
X, Y, COM = np.meshgrid(x, y, com, indexing='ij')


f_x = 0.4 

name = 'V_0={}.npy'.format(f_x)


states = np.load( states_path + name )
energies = np.load( energies_path + name ) 

for i,energy in enumerate(energies):
    plt.plot( states[int(para.m/2) , int(para.n/2) , :, i ] , com , label = energy) 

plt.legend()
plt.show()

