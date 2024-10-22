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
    print(pot[2])
    
#   if 'Unnamed' in pot:
#       print('error unnamed')
#       exit()
    #for i in range(int(len(pot)/2) , len(pot) ): #n equal to potential index , sys arg gives index on BO array
        #solver.solver1D(  int(sys.argv[1]) , pot[i])
    #solver.solver1D(int(sys.argv[1]) , pot[50])
    #   print( ' -----------calc done for xcom ='   , sys.argv[1] , 'given as index of BO_array' , "with potential index" , pot[i], '-----------------------') 
    #print(pot[50])

    if int(sys.argv[1]) in pot:

        en, st = pos.retrieve_array1D(para.fields[int(sys.argv[1])])
        print(en/para.joul_to_eV)
        pos.solve_ham1D(en,st, para.fields[int(sys.argv[1])])
        print('finished', para.fields[int(sys.argv[1])])
    else:

        print('MEN VA FAN!' , para.fields[int(sys.argv[1])])

if __name__ =='__main__':
    main()
    
