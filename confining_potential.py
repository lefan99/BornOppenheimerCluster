import paths
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d, RegularGridInterpolator,  LinearNDInterpolator
import pandas as pd


def V_x(x, f_x, sigma):
    '''One Dimensional electric potential. Mode const corresponds to constant electric field everywhere in space.
    Mode box has linear growth only in a box sized sigma and constant potential at the edges'''
    return f_x*erf(x/sigma)

def V_dot( f_x , x , y , sigma_x , sigma_y ):
    ''' Dot confinement Potential for 0D confinement case. Preliminary potetnial model of the one dimensional potential 
    enveloped by a gaussian profile'''

    return V_x( x, f_x , sigma_x) * np.exp( -0.5 * y**2 / (sigma_y**2))  





def interpolation(x, n , interp_type = 'cubic'):
    '''Return the interpolated potential from the COMSOL Data for a flat positional array x. Parameter n determines the top-Gate voltage.'''

    #Read in all computed potentials from file
    data_pot = pd.read_csv('../COMSOL/fine_sweep.csv')
    x_data = np.asarray(data_pot['x'])
    potentials_data =np.asarray( data_pot[n])

    #Center the nth potential around x=0 and y=0
    offset = (potentials_data[-1] - potentials_data[0])/2 + potentials_data[0]
    y = potentials_data - offset
    shift_index = np.where(y<0)[0]
    x_shift = x_data - x_data[shift_index[-1]]

    #Compute cubic interpolation
    interpolation = interp1d(x_shift, y, kind= interp_type)

    #Return interpolated potential which takes input as nanometers and outputs in Volts
    return interpolation(x*1e9)


def dot_comsol(x, y):
    '''Returns potential from COMSOL data using interpolation, if necessary. 2D confining potential'''

    #Read in all computed potentials from file
    data_pot = np.genfromtxt(paths.dot_COMSOL, skip_header=9, delimiter=';')
    sort = np.argsort(data_pot[:,0])
    data_pot = data_pot[sort]
    x_data = data_pot[:,0]
    print(data_pot[:,0])
    y_data = data_pot[:,1]
    potentials_data = data_pot[:, 2]

    #Center the nth potential around x=0 and y=0
    #offset = (potentials_data[-1,n] - potentials_data[0,n])/2 + potentials_data[0,n]
    #y = potentials_data[:, n] - offset
    #shift_index = np.where(y<0)[0]
    #x_shift = x_data - x_data[shift_index[-1]]

    #Compute cubic interpolation
    points = np.transpose(np.asarray([x_data, y_data]))
    interpolation =  LinearNDInterpolator(points , potentials_data)

    #Return interpolated potential which takes input as nanometers and outputs in Volts
    return interpolation(x*1e9 , y*1e9)

def interpolation_high(x ,n):
    '''Returns the interpolated data for high top gate voltages gained from OCMSOL simulations'''
    data_pot = np.genfromtxt('/home/kk472919/PhD/BO_parallel/COMSOL/high_potential.csv', skip_header=1)
    x_data = data_pot[:,1]
    potentials_data = data_pot[:, 2:]

    offset = (potentials_data[-1,n] - potentials_data[0,n])/2 + potentials_data[0,n]
    y = potentials_data[:, n] - offset
    shift_index = np.where(y<0)[0]
    x_shift = x_data - x_data[shift_index[-1]]

    #Compute cubic interpolation
    interpolation = interp1d(x_shift, y, kind='cubic')

    #Return interpolated potential which takes input as nanometers and outputs in Volts
    return interpolation(x*1e9)






