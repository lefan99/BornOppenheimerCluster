import solver
import sys


def main(): 
    for n in range( 100): #n equal to potential index , sys arg gives index on BO array
        solver.solver1D( n, 1)

        print( ' calc done for xcom ='  , n, 'given as index of BO_array' , "with potential index" , 1) 

if __name__ =='__main__':
    main()
    
