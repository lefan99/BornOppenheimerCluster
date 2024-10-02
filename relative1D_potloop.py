import solver
import sys
import parameters as para
import pos_solver as pos
import os

def main(): 

    fields = para.fields
    pot = []
    f_name = []


    for index, field in enumerate(fields):

        if 'Unnamed' not in field:
            f_name.append(field)
        
            if eval(field) > -10:
                pot.append(index)

    
#   if 'Unnamed' in pot:
#       print('error unnamed')
#       exit()
    for i in pot: #n equal to potential index , sys arg gives index on BO array
        solver.solver1D(  int(sys.argv[1]) , i)

        print( ' -----------calc done for xcom ='   , sys.argv[1] , 'given as index of BO_array' , "with potential index" , i, '-----------------------') 

    #en, st = pos.retrieve_array1D(pot)
    #pos.solve_ham1D(en,st, pot)
    print('finished')


if __name__ =='__main__':
    main()
    
