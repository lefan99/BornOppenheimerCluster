# -*- coding: utf-8 -*-
# @Author: timoreinartz
# @Date:   2023-11-02 11:26:18
# @Last Modified by:   timoreinartz
# @Last Modified time: 2023-11-17 16:54:31

import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d

def V_x(x, f_x, sigma):
	'''One Dimensional electric potential. Mode const corresponds to constant electric field everywhere in space.
	Mode box has linear growth only in a box sized sigma and constant potential at the edges'''
	return f_x*erf(x/sigma)

def interpolation(x, n):
	'''Return the interpolated potential from the COMSOL Data for a flat positional array x. Parameter n determines the top-Gate voltage.'''

	#Read in all computed potentials from file
	data_pot = np.genfromtxt('potential.txt', skip_header=9, delimiter=',')
	x_data = data_pot[:,0]
	potentials_data = data_pot[:, 2:]

	#Center the nth potential around x=0 and y=0
	offset = (potentials_data[-1,n] - potentials_data[0,n])/2 + potentials_data[0,n]
	y = potentials_data[:, n] - offset
	shift_index = np.where(y<0)[0]
	x_shift = x_data - x_data[shift_index[-1]]

	#Compute cubic interpolation
	interpolation = interp1d(x_shift, y, kind='cubic')

	#Return interpolated potential which takes input as nanometers and outputs in Volts
	return interpolation(x*1e9)

