import solver
import sys
import parameters as para
import pos_solver as pos
import os

def main(): 

    
    
    pot = para.fields[int(sys.argv[1])]
#   if 'Unnamed' in pot:
#       print('error unnamed')
#       exit()
    for i in range( para.o): #n equal to potential index , sys arg gives index on BO array
        solver.solver1D( i, int(sys.argv[1]))

        print( ' calc done for xcom ='  , i, 'given as index of BO_array' , "with potential index" , sys.argv[1]) 

    en, st = pos.retrieve_array1D(pot)
    pos.solve_ham1D(en,st, pot)
    print('finished')


if __name__ =='__main__':
    main()
    
