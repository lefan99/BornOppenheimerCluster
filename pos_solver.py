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

class Hamiltonian():
	'''Hamiltonian class that takes care of the construction of the kinetic Hamiltonian in the framework of scipy sparse matrices. 
	The only necessary input is the created grid class, the necessary parameters get directly imported from
	parameters.py'''

	def __init__(self, grid):
		'''Initialization of the class. Takes care of all the necessary method calls and actually builds the Hamiltonian.'''
		
		#Declaration for the sparse matrix constructors. Total nonzero contributions:
		#for every combination of x,y,com, the loop appends at most 9 elements.
		#It's actually a bit less than that, but who cares
		self._row = np.zeros(9*len(grid.x)*len(grid.y)*len(grid.com))
		self._col = np.zeros(9*len(grid.x)*len(grid.y)*len(grid.com))
		self._data = np.zeros(9*len(grid.x)*len(grid.y)*len(grid.com))
		self._counter = 0

		#Call build method
		self._build_Hkin(grid)

	def _append(self, value, row, coloumn):
		'''Helper function for appending values to sparse matrix constructors. Given the matrix indices
		row and coloumn and the entry of the sparse matrix value.'''
		self._row[self._counter] = row
		self._col[self._counter] = coloumn
		self._data[self._counter] = value
		self._counter += 1

	def _build_Hkin(self, grid):
		'''Method that fills constructor arrays with the kinetic energy information as inferred from the grid.
		This corresponds to building the discretized sparse laplacian modulo a prefactor. For every possible dicretized 
		point in the three dimensional space, a finite difference stencil of the second derrivative is applied. We apply
		the second-order accuracy finite difference coefficients corresponding to a stencil of the form [-1, 2, -1]. The linear
		superposition of this stencil with different prefactors for every direction that depend on discretization resolution and
		the effective masses, gives us the discretized kinetic energy operator.

		The np.ravel_multi_index function is particularly helpful, as it gives us a way to easily switch between the one dimensional
		indentation of the matrix and the multidimensional indentation corresponding to coordinates in 3D space. For more info on this
		see the documentation of the function.

		In contrast to the SLEPc version of this method, this method explicitly allows one of the dimensions to be trivial.
		This is of course helpful for the Born-Oppenheimer approximation, where we decouple the relative and COM motion.'''

		#Iterate over all of the linear dimensions
		dimensions = (len(grid.x), len(grid.y), len(grid.com))
		for x_index in range(dimensions[0]):
			for y_index in range(dimensions[1]):
				for com_index in range(dimensions[2]):
					#First get the central point of the stencil.
					origin = np.ravel_multi_index((x_index, y_index, com_index), dimensions, mode='raise')

					#Central contribution of the finite difference stencil with prefactor 2.
					#For every direction a different pre-factor that depens on the discretization
					#and on the mass along this direction will have to be applied. The mass dependent
					#term is just the prefactor of the laplacian in the definition of the kinetic energy,
					#the discretization dependent term comes from the finite difference scheme.
					#The if statement makes sure there is no data appended for a trivial dimension.
					for dxi, mass in zip([grid.dx, grid.dy, grid.dcom], [para.mu, para.mu, para.M]):
						if dxi != 0:
							self._append((-2 / dxi**2 * (-para.h_bar**2/(2*mass))), origin, origin)

					#Now we deal with the neighbour contributions of the stencil with prefactor -1. Before
					#assigning the values in the matrix there is an if condition that makes sure the
					#neighbouring point does not cross the finite boundary of the grid. By ignoring
					#these points we are effectivly imposing a boundary condition on the differential equation:
					#namely that the wave function should vanish at the boundary. Care must therefore be taken
					#when chosing the size of the box: It ought to be large enough that boundary effects make no
					#difference for the bound wave functions.
					#This also automatically takes care of a trivial dimension. If the trivial dimension index
					#is always zero, the corresponding if statements will never be True.

					if x_index+1 != dimensions[0]: #Checking for upper bound
						origin_right_x_neighbour = np.ravel_multi_index((x_index+1, y_index, com_index), dimensions, mode='raise')
						self._append((1 / grid.dx**2 *(-para.h_bar**2/(2*para.mu))), origin, origin_right_x_neighbour)

					if x_index-1 != -1: #Checking for lower bound
						origin_left_x_neighbour = np.ravel_multi_index((x_index-1, y_index, com_index), dimensions, mode='raise')
						self._append((1 / grid.dx**2 *(-para.h_bar**2/(2*para.mu))), origin, origin_left_x_neighbour)

					if y_index+1 != dimensions[1]: #Checking for upper bound
						origin_right_y_neighbour  = np.ravel_multi_index((x_index, y_index+1, com_index), dimensions, mode='raise')
						self._append((1 / grid.dy**2 *(-para.h_bar**2/(2*para.mu))), origin, origin_right_y_neighbour)

					if y_index-1 != -1: #Checking for lower bound
						origin_left_y_neighbour  = np.ravel_multi_index((x_index, y_index-1, com_index), dimensions, mode='raise')
						self._append((1 / grid.dy**2 *(-para.h_bar**2/(2*para.mu))), origin, origin_left_y_neighbour)

					if com_index+1 != dimensions[2]: #Checking for upper bound
						origin_right_com_neighbour  = np.ravel_multi_index((x_index, y_index, com_index+1), dimensions, mode='raise')
						self._append((1 / grid.dcom**2 *(-para.h_bar**2/(2*para.M))), origin, origin_right_com_neighbour)

					if com_index-1 != -1: #Checking for lower bound
						origin_left_com_neighbour  = np.ravel_multi_index((x_index, y_index, com_index-1), dimensions, mode='raise')
						self._append((1 / grid.dcom**2 *(-para.h_bar**2/(2*para.M))), origin, origin_left_com_neighbour)

		#Get rid of the excess zeros in the constructor arrays before assembling the sparse matrix.
		self._row = self._row[self._data != 0]
		self._col = self._col[self._data != 0]
		self._data = self._data[self._data != 0]
		#Assemble the sparse coo.
		self.Hkin = coo_array((self._data, (self._row, self._col)), shape=(dimensions[0]*dimensions[1]*dimensions[2], dimensions[0]*dimensions[1]*dimensions[2]))

GRID_1 = grid(para.m , para.x_width , para.n , para.y_width , 1 , 0)

#Import the data from the COM calculations, convention follows the local version written by timo 
BO_energy = []
BO_states = np.zeros([para.m , para.n , para.o])
BO_array = np.linspace(-para.com_width, para.com_width, para.o, endpoint=True)
f_x = 0.4
pot_index = sys.argv[1]
for j in range(len(BO_array)):

   i = BO_array[j]

   if os.path.exists('../hamiltonian/rel_data/states/pot{}/com{}.npy'.format(pot_index, i)):
       
        BO_energy.append( np.load('../hamiltonian/rel_data/energies/pot{}/com{}.npy'.format(pot_index, i)) )
        state = np.load('../hamiltonian/rel_data/states/pot{}/com{}.npy'.format(pot_index, i))  

        BO_states[:,:,j] = state * np.sign(state)


#print(len(BO_energy) , len(BO_states))
BO_energy = np.asarray( BO_energy)
#print('energy ' ,BO_energy ,'states', BO_states) 

#set up grid and hamiltonian to solve the positional schrodinger equation

GRID = grid(1, 0 ,1 ,0 , para.o , para.com_width)
Ham = Hamiltonian(GRID)
V_BO = diags(BO_energy)


#print( 'HKIN' , Ham.Hkin, 'potential'  , V_BO ,'bo energy surface',  BO_energy)
H = Ham.Hkin + V_BO
energies, states = eigsh(H , k = 10, which='SA') 

print(energies/para.joul_to_eV)

#plot the BO energy surface
fig, ax = plt.subplots()
ax.plot(BO_array*1e9, BO_energy/para.joul_to_eV)
ax.axhline(energies[0]/para.joul_to_eV)



for i in energies:

    ax.axhline(i/para.joul_to_eV)
fig.savefig('../plots/COM_potential/test_fx_{}.pdf'.format(pot_index))


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
np.save('../hamiltonian/statesData/V_0={}'.format(f_x), states)
np.save('../hamiltonian/energiesData/V_0={}'.format(f_x), energies)

#Compute Oscillator strength for every state.
k_l_square = np.trapz(states[int(para.m/2), int(para.n/2), :, :], BO_array, axis = 0)**2

#Transfer oscillator strength density from 1/m to 1/nm
k_l_square = k_l_square / 1e9
#osci.append(k_l_square[0])

#Make a plot of the obtained lowest lying motional state at relative coordinate r=0.
fig, ax = plt.subplots()
ax.plot(BO_array / 1e-9, states[int(para.m/2), int(para.n/2), :, 0] * (1e-9)**(3/2), label='f_x={}'.format(f_x))
ax.legend()
ax.grid(True)
ax.set_ylabel(r'$|\phi(X,r=0)|$ [1/$nm^(3/2)$]')
ax.set_xlabel(r'X [nm]')
plt.tight_layout()
fig.savefig('../plots/COM_groundstate/mls_at_overlap_fx_{}.pdf'.format(f_x))

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



