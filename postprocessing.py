import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import os
import parameters as para

box_array =  np.linspace(-para.x_width, para.x_width, para.m ,endpoint=True)
BO_array = np.linspace(-para.com_width , para.com_width , para.o, endpoint =True)

# package providing plotting functions and postprocessing scripts! men va fan


def delta(h_bar_omega, energies, broadening=10e-6):
    '''Helper function for plotting purposes: This function approximates the delta shaped resonances
     in the modified Elliot formula (see notes). For the energy of every exciton state we need an
     omega dependent function. The format of the output of this therefore hat to be [w,n] where w
     is the discretized frequency/energy axis and n has the dimensions of the converged states.
     The delta function itself is approximated by a narrow cosh resonance. The broadening factor is
     chosen arbitrarily to make nice plots.'''
    omega_shape = h_bar_omega.shape
    energies_shape = energies.shape
    omega_shape = omega_shape + (1,)
    energies_shape = (1,) + energies_shape
    h_bar_omega = np.reshape(h_bar_omega, omega_shape)
    energies = np.reshape(energies, energies_shape)
    return 1/(np.pi*broadening) * 1/np.cosh((energies-h_bar_omega)/broadening)


def load_data(path):
    ''' Load the states and eneergy array from the calculations, energy in eV. Path as list like with absolute paths to storage'''
    energies = np.load(path[0])/para.joul_to_eV
    states = np.load(path[1])

    return energies, states


def osci_strength(state, energies, h_bar_omega , dx = para.com_width*2/para.o  , n = 10000):
    ''' calculate the oscillator strength of a SINGLE state using trapezoidal integration methods, n equal resolution of the optical spectrum has to be equal to n in h_bar_omega function'''
    
    shape = list(state.shape)
    dimensions = len(shape) 
    r_0 = np.zeros(dimensions-2)
    r_0[0] = int(shape[0]/2)
    r_0[1] = int(shape[1]/2)
    print(type(r_0[0]))
    for i in range( dimensions - 2):
        k_l = np.trapz(state[int(r_0[0]) , int(r_0[1]),:] , dx=dx , axis = 0)**2
    
    func = delta( h_bar_omega , energies) * k_l
    osci_den = np.sum( func , axis=1)

    return osci_den

def mev_to_field(x, sigma = para.sigma/np.sqrt(2)):
    '''Helper function for plotting purposes: This simply relates the height of the junction to the 
    amplitude of the electric field gaussian, see the corresponding chapter in the notes.'''
    
    return x /1e3 / (sigma*np.sqrt(np.pi/2)) /1e6


def h_bar_omega( emin = -0.232 , emax = -0.218 ,n= 10000):
    '''Return the energy range of the exciton states, n is number of array entries. Is used as the input of the delta function in the calcualation of the osci strength'''
    return np.linspace( emin, emax, n)


def plot_osci_strength( x , optical, h_bar_omega, scale = 0.5 , save = False , name = 'oscillator_strength_density.pdf'):
    '''plot the osci strength for multiple potential configuration. , k_l is the osci strength array for the states'''

    FIELDS, H_BAR_OMEGA = np.meshgrid(x, h_bar_omega, indexing='ij')
    fig = plt.figure()
    ax = Axes3D(fig)
    fig, ax = plt.subplots()
    plt.grid(False)
    plt.pcolormesh(FIELDS, H_BAR_OMEGA * 1000, optical/np.max(optical), cmap = 'binary', vmax=scale)
    plt.colorbar()
    plt.xlabel(r'$V_{TG}$ in $[V]$')
    plt.ylabel(r'Energy [meV]')
#sec = ax.secondary_xaxis('top', functions=(mev_to_field, field_to_mev))
#sec.set_xlabel(r'Electric field $[V\mu m]$')
    plt.tight_layout()
    if save:
        plt.savefig(name)

    else:
        plt.show()

def r_square2D( state_n = 15 , dim = para.m , arr = box_array):
    '''gives r^2 in karthesian coordinates array for the the 2D case used for plotting purposes'''
    r_sq = np.zeros([ dim, dim , state_n])
    for i in range(dim):
        for j in range(dim):
            r_sq[i,j,:] = arr[i]**2 + arr[j]**2
    return r_sq

def r_2D(state_n , dim = para.m , arr = box_array):
    '''helper function outputs x**2+y**2 array'''
    r_sq=np.zeros([dim , dim , state_n])
    for i in range(dim):
        for j in range(dim):
            r_sq[i,j,:] = np.sqrt(arr[i]**2 + arr[j]**2)
    return r_sq 


def r_square1D( state_n = 15 , dim = para.o , arr = BO_array):
    '''see above but 1D case'''
    r_sq = np.zeros([ dim, state_n])
    for j in range(dim):
        r_sq[i,j,:] = arr[i]**2 + arr[j]**2
    return r_sq

def eh_dist(state , r , X =  BO_array, x = box_array):
    '''returns the average distance between electron and hole squareroot(<r^2>),'''

    dim = len(state.shape)-3 #dimensinon determination whtere 0D confinement or 1D confinement
    for i in range(dim):
        integral = np.trapz(state**2 , X , axis=2)  
        integral = np.trapz(integral , X, axis =2)
    integral = integral*r
    integral = np.trapz(integral, x , axis=0)
    integral = np.trapz(integral, x , axis=0)
    return np.sqrt(integral)


def COM_pos(state , X =  BO_array, x = box_array , dim= 15):
    '''returns <X> COM for the 1D confinement case'''

    dim = len(state.shape)-3 #dimensinon determination whtere 0D confinement or 1D confinement
    X1 = np.reshape(X, (len(X) , dim))
    for i in range(dim):
        integral = np.trapz(state**2 ,  x , axis=0)
        integral = np.trapz(integral, x , axis=0)
    integral = integral * X1
    integral = np.trapz(integral , X, axis =0)
    return integral


def COM_pos_square(state , X =  BO_array, x = box_array):
    '''returns <X^2> COM for the 1D confinement case'''

    dim = len(state.shape)-3 #dimensinon determination whtere 0D confinement or 1D confinement
    X1 = np.reshape(X, (len(X) , dim))
    for i in range(dim):
        integral = np.trapz(state**2 ,  x , axis=0)
        integral = np.trapz(integral, x , axis=0)
    integral = integral * X1**2
    integral = np.trapz(integral , X, axis =0)
    return integral

