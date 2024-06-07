import solver
import sys


def main():
    
    for y_com_index in range(200, int(sys.argv[2])): 
        solver.solver( int(sys.argv[1]) , y_com_index  , int(sys.argv[3]))
        print( ' calc done for xcom ='  , sys.argv[1] , 'and ycom' , y_com_index , 'given as index of BO_array') 

if __name__ =='__main__':
    main()
    
