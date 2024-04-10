# -*- coding: utf-8 -*-
# @Author: timoreinartz
# @Date:   2023-11-02 11:12:41
# @Last Modified by:   timoreinartz
# @Last Modified time: 2023-11-17 16:51:06

import numpy as np 
from scipy.sparse import diags, csr_matrix, coo_array
from scipy.sparse.linalg import eigsh
from time import time
from scipy.special import struve, yv, erf
import sys
import os
import parameters as para
import confining_potential as conf

class grid():
	'''Simple Class to hold the necessary data for the construction of the grid. Given the necessary geometric input parameters,
	which are defined in parameters.py, this just sets up the necessary meshgrids for the 1D confinement case.'''
	def __init__(self, x_points, x_width, y_points, y_width, com_points, com_width):
		self.dx = 2*x_width / x_points
		self.dy = 2*y_width / y_points 
		self.dcom = 2*com_width / com_points

		self.x = np.linspace(-x_width, x_width, x_points ,endpoint=True)
		self.y = np.linspace(-y_width, y_width, y_points ,endpoint=True)
		self.com = np.linspace(-com_width, com_width, com_points ,endpoint=True)

		self.X, self.Y, self.COM = np.meshgrid(self.x, self.y, self.com, indexing='ij')

class laplacian():
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

	def _append(self, value, origin, neighbour):
		'''Helper function for appending values to sparse matrix constructors. Given the matrix indices
		row and coloumn and the entry of the sparse matrix value.'''
		self._row[self._counter] = origin
		self._col[self._counter] = neighbour
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

def Keyldish(r):
	'''Just the definition of the Keldysh potential in real space in terms of Bessel functions.'''
	V = -para.c * np.pi / (2*para.epsilon_r*para.r_0) * (struve(0, r/para.r_0) - yv(0, r/para.r_0))
	return V

def main():
	#Init Grid with COM set to zero
	GRID = grid(para.m, para.x_width, para.n, para.y_width, 1, 0)

	#Build (relative) Laplacian and Keyldish potential
	LAP = laplacian(GRID)
	V_keyldish = diags(Keyldish(np.sqrt(GRID.X.reshape(-1)**2 + GRID.Y.reshape(-1)**2)))

	#Specify adiabatic dependence and parameter dependence of the potential
	BO_array = np.linspace(-para.com_width, para.com_width, para.o, endpoint=True)

	#The idea here is that sys.argv[1] will specify the current adiabatic center of mass
	#position, while sys.argv[2] will specify the current potential specification.
	#That way we can easily parallelize the task of solving the relative Hamiltonian
	#by assigning a job for each combination of COM position/potential
	print(sys.argv[1])
	print(sys.argv[2])
	current_com = BO_array[int(sys.argv[1])]
	current_pot = int(sys.argv[2])

	#Build Hamiltonian for erf case
	if para.potential_mode == 'erf':
		V_el_conf  = diags(-para.e * conf.V_x( para.m_valence/para.M*GRID.X.reshape(-1)+    current_com, para.fields[current_pot], para.sigma[current_pot]))
		V_hl_conf  = diags( para.e * conf.V_x(-para.m_conduction/para.M*GRID.X.reshape(-1)+    current_com, para.fields[current_pot], para.sigma[current_pot]))
		Ham = LAP.Hkin + V_keyldish + V_el_conf + V_hl_conf

	#Build Hamiltonian for interpolation case
	if para.potential_mode == 'interp':
		V_el_conf  = diags(-para.e * conf.interpolation( para.m_valence/para.M*GRID.X.reshape(-1)+    current_com, para.potential_index[current_pot]))
		V_hl_conf  = diags( para.e * conf.interpolation(-para.m_conduction/para.M*GRID.X.reshape(-1)+    current_com, para.potential_index[current_pot]))
		Ham = LAP.Hkin + V_keyldish + V_el_conf + V_hl_conf

	#Solve relative Hamiltonian
	t1 = time()
	energies, states = eigsh(Ham, k=para.eigenstates_relative, which='SA')
	t2 = time()
	t = (t2-t1)/60

	#Sort the energies and states. Depending on which solver was used, they might be unsorted.
	#If they aren't this doesn't do anything.
	order = np.argsort(energies)
	energies = energies[order]
	states = states[:, order]

	#Bring the states from the flattened format back into a coordinate representation. Their format is now
	#[x,y,1,n] where the first two indices correspond to relative coordinates, com direction is trivial
	#and n labels the eigenstates. This reshaping is consistent with the flattened encoding,
	#since we used the numpy unravel routine to define the Hamiltonian.
	new_shape = GRID.X.shape + (para.eigenstates_relative,)
	states = np.reshape(states, newshape=(new_shape))

	#Normalize the exciton wave function using trapezoidal integration.
	states_squared = np.abs(states)**2
	normalize = states_squared[:,:,0,:]
	normalize = np.trapz(normalize,			 GRID.y, axis=1)
	normalize = np.trapz(normalize,			 GRID.x, axis=0)

	states = states / np.sqrt(normalize)

	#Find the most localized state at the origin. This will correspond to the lowest exciton bound state.
	k_l_square = np.abs(states[int(para.m/2),int(para.n/2),0,:])**2
	optical_order = np.argsort(k_l_square)
	mls = np.argmax(k_l_square)

	#Save the selected MLS state of this COM/potential combination and its energy for further use in construction
	#of the Born-Oppenheimer potential energy surface and complete wave function
	os.makedirs('data/states/pot{}'.format(current_pot), exist_ok=True)
	os.makedirs('data/energies/pot{}'.format(current_pot), exist_ok=True)
	np.save('data/states/pot{}/com{}.npy'.format(current_pot, current_com), states[:,:,0,mls])
	np.save('data/energies/pot{}/com{}.npy'.format(current_pot, current_com), energies[mls])

if __name__ == '__main__':
	main()
