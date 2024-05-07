import parameters as para
import numpy as np
import os



BOx_array = np.linspace(-para.com_width, para.com_width, para.o, endpoint=True)
x_list = []
y_list = []


for i,x in enumerate(BOx_array):

    for j,y in enumerate(BOx_array):

       if not os.path.exists('../hamiltonian/rel_data/states/pot{}/com_x{}_y{}.npy'.format(0.425, x, y)):
            
            x_list.append( i)
            y_list.append(j)


np.save('../missing_x.npy', np.asarray(x_list))
np.save('../missing_y.npy', np.asarray(y_list))
print(len(x_list))
