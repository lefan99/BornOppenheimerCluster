import pos_solver as ps


for n in [10 , 15 ,20 , 25 ,30 ,35]:

    energy,states = ps.retrieve_array1D(n)
    print(energy , states)
    print(ps.solve_ham1D( energy , states, n))
    
