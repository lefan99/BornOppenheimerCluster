import paths
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import os
import parameters as para

def delta(h_bar_omega, energies, broadening=0.00001*para.joul_to_eV):
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

def mev_to_field(x, sigma = para.sigma/np.sqrt(2)):
    '''Helper function for plotting purposes: This simply relates the height of the junction to the 
    amplitude of the electric field gaussian, see the corresponding chapter in the notes.'''
    return x /1e3 / (sigma*np.sqrt(np.pi/2)) /1e6

def field_to_mev(x, sigma = para.sigma/np.sqrt(2)):
    '''Helper function for plotting purposes: This simply relates the height of the junction to the 
    amplitude of the electric field gaussian, see the corresponding chapter in the notes.'''
    return x *1e3 * sigma * np.sqrt(np.pi/2) *1e6

#Pick Resolution and boundaries for the spectrum plot.
optial_resolution = 1000
h_bar_omega = np.linspace(-228e-3*para.joul_to_eV, -219e-3*para.joul_to_eV, optial_resolution)

#It is convenient to have the spatial meshgrids on hand for plotting purposes
#so we quickly redefine them.
x = np.linspace(-para.x_width, para.x_width, para.m ,endpoint=True)
y = np.linspace(-para.y_width, para.y_width, para.n ,endpoint=True)
com = np.linspace(-para.com_width, para.com_width, para.o ,endpoint=True)
X, Y, COM = np.meshgrid(x, y, com, indexing='ij')

#Container arrays for Data that will be needed for making plots.
k_l_square0 = []
mls_distr = np.zeros((len(para.fields), para.o))
loc1_distr = np.zeros((len(para.fields), para.o))
loc2_distr = np.zeros((len(para.fields), para.o))
optical   = np.zeros((len(para.fields), optial_resolution))
mls_wave_function = np.zeros((len(para.fields), para.m, para.n, para.o))
mls_energy = np.zeros(len(para.fields))

#Iterating over all the independent simulations.
for f_index, f_x in enumerate(para.fields):
    print(f_x)

    #Loading the postprocessed data arrays in
    states   = np.load('statesData/V_0={}.npy'.format(f_x))
    energies = np.load('energiesData/V_0={}.npy'.format(f_x))

    #Compute Oscillator strength for every state.
    k_l_square = np.trapz(states[int(para.m/2), int(para.n/2), :, :], com, axis = 0)**2

    #Find the most localized state at the origin, that corresponds to the lowest exciton bound state.
    psi_0 = np.abs(states[int(para.m/2), int(para.n/2), int(para.o/2), :])**2

    '''The following lines of code were my attempt at writing a filtering mechanism in order to select out the motionally excited
    exciton bound states. My reasoning was the following: the bound states are distinguished by having a large amount of their
    probability density at the radial origin coordinate. The second thing that distinguishes them is, that looking at it from the COM
    coordinate, the probability density of this overlap should be concentrated mostly around the origin of the COM coordinate, because
    this is precisely what makes them confined states. The resulting filtering mechanism is that I therefore looked the amount of
    probability density that is given in an area around the COM origin at relative coordinate =0.


    This ad-hoc procedure can however be prone to mistakes. One should always verify that it in fact picked the correct state!'''
    radial_zero_of_wave_function = np.abs(states[int(para.m/2), int(para.n/2), :, :])**2 #Probability density when setting the relative coordinate to zero.
    localisation = np.trapz(radial_zero_of_wave_function[int(para.o/2)-10:int(para.o/2)+10], com[int(para.o/2)-10:int(para.o/2)+10], axis=0) #Integral of that quantity around the COM origin.
    
    #The filtering now correspond to just sorting according to that quantity.
    localisation = np.argsort(localisation, axis=None)[::-1]

    mls = np.argmax(psi_0) #Most localized state. This will always correspond to the lowest lying exciton bound state.
    #Most optical state. For low fields, this will be equal to mls, for high fields this becomes the free exciton resonance.
    #Plotting its oscillator strength however, is not really useful for the presentation, when one wants to visualize the confined states,
    #because it is much larger. It is therefore filtered for plotting purposes.
    mos = np.argmax(k_l_square)

    #One can compare which of the states has which quality.
    print('Most localized state:', mls, energies[mls]/para.joul_to_eV)
    print('Other method:', localisation[0], energies[localisation[0]]/para.joul_to_eV)
    print('Most optical state:', mos, energies[mos]/para.joul_to_eV)

    k_l_square0.append(k_l_square[mls]) #We save the oscillator strength of the lowest lying exciton bound state for later.

    #Filling in the placeholder array for these two quantities for later plotting purposes.
    mls_wave_function[f_index] = states[:,:,:,mls]
    mls_energy[f_index] = energies[mls] / para.joul_to_eV

    #Plotting the Oscillator strengths with the selected states from the localisation procedure in a distinguished color.
    #Using this, once can gauge wether the correct states were selected.
    fig, ax = plt.subplots()
    ax.plot(np.delete(range(0, len(energies)), mos), np.delete(k_l_square, mos), linestyle='none', marker='.')
    plt.vlines(np.delete(range(0, len(energies)), mos), 0, np.delete(k_l_square, mos))
    ax.plot(mls, k_l_square[mls], linestyle='none', marker='*', color='red')
    plt.axvline(mos, 0, 1, color='black')
    ax.plot(localisation[0], k_l_square[localisation[0]], linestyle='none', marker='.', color='orange')
    ax.plot(localisation[1], k_l_square[localisation[1]], linestyle='none', marker='.', color='violet')
    ax.plot(localisation[2], k_l_square[localisation[2]], linestyle='none', marker='.', color='green')
    plt.vlines(localisation[0], 0, k_l_square[localisation[0]], color='orange')
    plt.vlines(localisation[1], 0, k_l_square[localisation[1]], color='violet')
    plt.vlines(localisation[2], 0, k_l_square[localisation[2]], color='green')
    ax.grid(True)
    fig.savefig('Filtered_optical_plot/fx={}.pdf'.format(f_x))

    #Compute the optical response of this particular electrostatic potential. Compare with the modified Elliot formula.
    #func corresponds to a linecut in the field/energy dependent optical spectrum colormap.
    series = delta(h_bar_omega, energies) * k_l_square
    func = np.sum(series, axis=1)

    optical[f_index, :] = func #We save the linecut for doing the colormap later.

    #The following lines save the COM wave function at radial coordinate zero of the different
    #motional confined states for later plotting.
    #Also fix the phase of the wavefunction by multiplying with a sign for prettier plots.

    mls_wave_at_overlap = states[int(para.m/2), int(para.n/2), :, mls]
    mls_distr[f_index, :] = mls_wave_at_overlap * np.sign(mls_wave_at_overlap[int(para.o/2)-1])

    loc1_wave_at_overlap = states[int(para.m/2), int(para.n/2), :, localisation[1]]
    loc1_distr[f_index, :] = loc1_wave_at_overlap * np.sign(loc1_wave_at_overlap[int(para.o/2)-1])

    loc2_wave_at_overlap = states[int(para.m/2), int(para.n/2), :, localisation[2]]
    loc2_distr[f_index, :] = loc2_wave_at_overlap * np.sign(loc2_wave_at_overlap[int(para.o/2)-1])

#Plot the evolution of the oscillator strength for the lowest lying exciton bound state.
fig, ax = plt.subplots()
ax.plot(para.fields*1e-6, k_l_square0/k_l_square0[0])
ax.grid(True)
ax.set_xlabel(r'Electric field strength $[V/\mu m]$')
ax.set_ylabel(r'Oscillator strength [arbitrary units]')
plt.tight_layout()
fig.savefig('Oscillator_strength_ground_bound_state_fx_dependece.pdf')

#Plot the different motional states we extracted earlier for a selected choice of fields.
fig, ax = plt.subplots()
for f_index, f_x in zip([0, 4, 8, 12, 17], para.fields[[0, 4, 8, 12, 17]]):
    ax.plot(com*1e9, mls_distr[f_index, :]/(1e9)**(3/2) , label=r'$V_0={} [meV]$'.format(np.round(f_x*1e3, 0)))
ax.legend()
ax.grid(True)
ax.set_ylabel(r'$|\phi(X,r=0)|$ [1/$nm^(3/2)$]')
ax.set_xlabel(r'X [nm]')
plt.tight_layout()
fig.savefig('mls_at_overlap.pdf')

fig, ax = plt.subplots()
for f_index, f_x in zip([8, 11, 14, 17], para.fields[[8, 11, 14, 17]]):
    ax.plot(com*1e9, loc1_distr[f_index, :]/(1e9)**(3/2) , label=r'$V_0={} [meV]$'.format(np.round(f_x*1e3, 0)))
ax.legend()
ax.grid(True)
ax.set_ylabel(r'$|\phi(X,r=0)|$ [1/$nm^(3/2)$]')
ax.set_xlabel(r'X [nm]')
plt.tight_layout()
fig.savefig('loc[1].pdf')

fig, ax = plt.subplots()
for f_index, f_x in zip([14, 15, 17, 21, 23], para.fields[[14, 15, 17, 21, 23]]):
    ax.plot(com*1e9, loc2_distr[f_index, :]/(1e9)**(3/2) , label=r'$V_0={} [meV]$'.format(np.round(f_x*1e3, 0)))
ax.legend()
ax.grid(True)
ax.set_ylabel(r'$|\phi(X,r=0)|$ [1/$nm^(3/2)$]')
ax.set_xlabel(r'X [nm]')
plt.tight_layout()
fig.savefig('loc[2].pdf')

#Plot full optical spectrum colormap
FIELDS, H_BAR_OMEGA = np.meshgrid(para.fields, h_bar_omega, indexing='ij')
fig = plt.figure()
ax = Axes3D(fig)
fig, ax = plt.subplots()
plt.grid(False)
plt.pcolormesh(FIELDS*1e3, H_BAR_OMEGA/para.joul_to_eV*1000, optical/np.max(optical), cmap = 'binary')
plt.colorbar()
plt.xlabel(r'Junction height $[meV]$')
plt.ylabel(r'Energy [meV]')
sec = ax.secondary_xaxis('top', functions=(mev_to_field, field_to_mev))
sec.set_xlabel(r'Electric field $[V\mu m]$')
plt.tight_layout()
fig.savefig('spectrum_colormap.pdf')

#Plot the energy field dependence of the lowest lying exciton bound state
fig, ax = plt.subplots()
ax.plot(para.fields*1e3, mls_energy*1e3, marker='.')
ax.set_xlabel(r'Junction height $[meV]$')
ax.set_ylabel(r'mls state energy $[meV]$')
ax.grid(True)
fig.savefig('mls_energy.pdf')

#Save for later use
np.save('mls_energy.npy', mls_energy)
np.save('mls_wave_function.npy', mls_wave_function)
