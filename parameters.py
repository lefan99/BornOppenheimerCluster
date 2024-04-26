# -*- coding: utf-8 -*-
# @Author: timoreinartz
# @Date:   2023-09-25 10:17:13
# @Last Modified by:   timoreinartz
# @Last Modified time: 2023-11-08 16:20:25

from scipy import constants as const
import numpy as np

#Define physical constants in standard units
epsilon = const.epsilon_0
e = const.e
c = e**2 / (4*np.pi*epsilon) #Coloumb constant
h_bar = const.hbar
joul_to_eV = e

#effective masses
m_conduction = 0.7*const.m_e
m_valence = 0.6*const.m_e

#reduced and COM mass
mu = m_conduction * m_valence / (m_conduction+m_valence)
M = m_conduction + m_valence

#relative dielectric and screening length of 2D Coloumb
epsilon_r = 4.4
r_0 = 3.9/epsilon_r * 1e-9

#Simulation Parameters
#eigenstates=400 #Amount of eigenstates to solve for
#Geometry
m = 150             # Number of xpoints
x_width = 20e-9
n = 150             # Number of ypoints
y_width = 20e-9
o = 100             # Number of compoints
com_width = 100e-9

eigenstates_relative = 200

potential_mode = 'erf'
if potential_mode == 'erf':
    #fields = np.arange(0e-3, 1025e-3, 25e-3)
    #fields = np.round(fields, decimals=3)
    fields = [400e-3]
    sigma = [20e-9 * np.sqrt(2)]*len(fields)

if potential_mode == 'interp':
    #potential_index = range(10)
    potential_index = [0]

if potential_mode == 'dot':

    fields = [400e-3]
    sigma = [20e-9 * np.sqrt(2)]*len(fields)
    

#saved_states = 70


