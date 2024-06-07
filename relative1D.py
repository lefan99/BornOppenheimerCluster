import solver
import sys


def main(): 
    for n in range( 10): #n equal to potential index , sys arg gives index on BO array
        solver.solver1D( int(sys.argv[1]), n)

        print( ' calc done for xcom ='  , sys.argv[1] , 'given as index of BO_array' , "with potential index" , n) 

if __name__ =='__main__':
    main()
    
