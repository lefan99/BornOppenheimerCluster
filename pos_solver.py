import numpy as np 
import sys
from scipy.sparse import diags, coo_array
from scipy.sparse.linalg import eigsh
from time import time
from scipy import constants as const
from scipy.special import struve, yv, erf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import parameters as para
from solver import grid, laplacian 
import os


#Import the data from the COM calculations, convention follows the local version written by timo 
BO_energy = np.array([])
BO_states = np.zeros([para.m , para.n , para.o])
BO_array = np.linspace(-para.com_width, para.com_width, para.o, endpoint=True)

pot_index = sys.argv[1]
for i in range(150):
   
   if os.path.exists('data/states/pot{}/com{}.npy'.format(pot_index, i)):
        
        np.append( BO_energy , np.load('data/energy/pot{}/com{}.npy'.format(pot_index, i)) )
        state = np.load('data/states/pot{}/com{}.npy'.format(pot_index, i)) ) 

        BO_states[:,:,counter] = state * np.sign(state)

#set up grid and hamiltonian to solve the positional schrodinger equation

GRID = grid(1, 0 ,1 ,0 , para.o , para.com_width)
Ham = laplacian(GRID)
V_BO = diags(BO_energy)

H = Ham.Hkin + V_BO
energies, states = eigsh(H , k = 10, which='SA') 

print(energies/para.joul_to_eV)

#plot the BO energy surface
fig, ax = plt.subplots()
ax.plot(BO_array*1e9, BO_energy/para.joul_to_eV)
ax.axhline(energies[0]/para.joul_to_eV)



for i in energies:

    ax.axhline(i/para.joul_to_eV)
fig.savefig('COM_potential/test_fx_{}.pdf'.format(pot_index))


#calculate the final state as a product of the POS and REL state
shape = (para.m, para.n , para.o , 1) 
BO_states = np.reshape(BO_states , newshape = (shape) )
shape_states = ( 1, 1 , para.o , 10 ) 
states = BO_states * states

#Normalize the exciton wave function using trapezoidal integration.
states_squared = np.abs(states)**2
normalize = np.trapz(states_squared, 	 BO_array, axis=2)
normalize = np.trapz(normalize,			 GRID_1.y, axis=1)
normalize = np.trapz(normalize,			 GRID_1.x, axis=0)
states = states / np.sqrt(normalize)

#Save the normalized, sorted and processed states for further use. One can do any further postprocessing with them now.
np.save('statesData/V_0={}'.format(f_x), states)
np.save('energiesData/V_0={}'.format(f_x), energies)

#Compute Oscillator strength for every state.
k_l_square = np.trapz(states[int(para.m/2), int(para.n/2), :, :], BO_array, axis = 0)**2

#Transfer oscillator strength density from 1/m to 1/nm
k_l_square = k_l_square / 1e9
osci.append(k_l_square[0])

#Make a plot of the obtained lowest lying motional state at relative coordinate r=0.
fig, ax = plt.subplots()
ax.plot(BO_array / 1e-9, states[int(para.m/2), int(para.n/2), :, 0] * (1e-9)**(3/2), label='f_x={}'.format(f_x))
ax.legend()
ax.grid(True)
ax.set_ylabel(r'$|\phi(X,r=0)|$ [1/$nm^(3/2)$]')
ax.set_xlabel(r'X [nm]')
plt.tight_layout()
fig.savefig('COM_groundstate/mls_at_overlap_fx_{}.pdf'.format(f_x))

#Plot the field dependence of the oscillator strength density.
#fig, ax = plt.subplots()
#ax.plot(para.fields*1e3, osci)
#ax.grid(True)
#ax.legend()
#ax.set_xlabel(r'Electric field strength $[V/\mu m]$')
#ax.set_ylabel(r'Oscillator strength density [1/nm]')
#plt.tight_layout()
#fig.savefig('Oscillator_strength_ground_bound_state_fx_dependece.pdf')
#
##Save for further use.
#np.save('osci_BO.npy', osci)
#



