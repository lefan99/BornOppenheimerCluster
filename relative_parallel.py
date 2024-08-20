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
    solver.solver1D( int(sys.argv[2]), int(sys.argv[1]))

    print( ' -----------calc done for xcom ='   , sys.argv[1], 'given as index of BO_array' , "with potential index" , sys.argv[2] , '-----------------------') 

    #en, st = pos.retrieve_array1D(pot)
    #pos.solve_ham1D(en,st, pot)
    #print('finished')


if __name__ =='__main__':
    main()
    
