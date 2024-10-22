from scipy import constants as const
import numpy as np
import pandas as pd 


path_1D = '/work/kk472919/hamiltonian1D_2/' #absolute path for storage
path_0D = '/work/kk472919/hamiltonian0D/'

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
m = 400             # Number of xpoints
x_width = 35e-9
n = 400             # Number of ypoints
y_width = 35e-9
o = 200             # Number of compoints
com_width = 150e-9

eigenstates_relative = 200

potential_mode = 'interp'

if potential_mode == 'dot_interp':
    fields = ['dot']
    sigma = [0.0]


if potential_mode == 'erf':
    #fields = np.arange(0e-3, 1025e-3, 25e-3)
    #fields = np.round(fields, decimals=3)
    fields = [1.5 , 2 , 3.5 , 4]
    sigma = [20e-9 * np.sqrt(2)]

if potential_mode == 'interp':
    #potential_index = range(10)
    potential_index = [0]
    fields_1 = list(pd.read_csv('../COMSOL/fine_sweep.csv').columns)
    del fields_1[-1]#columns have the fields strength, delete the last column bc it does not belong with the other column, rest of code sorts the column by field strength. and puts them into a str for in order to be readable by the solver code. (pandas df needs columns str name to find the right column)
    
    fields = []

    for field in fields_1:

        if 'Unnamed' not in field:
            fields.append(eval(field))

    fields = sorted(fields)
    fields = [str(i) for i in fields]

    sigma = [20e-9 * np.sqrt(2)]

if potential_mode == 'dot':

    fields = [425e-3, 0.725]
    sigma = 20e-9 * np.sqrt(2)
    

#saved_states = 70


