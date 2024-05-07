import solver
import sys
import numpy as np


missing_x = np.load('../missing_x.npy') 
missing_y = np.load('../missing_y.npy') 


def main():
    index = int(sys.argv[1]) * 100
    for i in range(100): 
        if index+i in range(len(missing_x)):
            solver.solver( missing_x[index+i] , missing_y[index+i]  , 0)
            print( ' calc done for xcom ='  , missing_x[index+i] , 'and ycom' , missing_y[index+i] , 'given as index of BO_array') 
        
if __name__ =='__main__':
    main()
    
