import pos_solver as ps


for n in [186]:

    energy,states = ps.retrieve_array1D(n)
    print(energy , states)
    print(ps.solve_ham1D( energy , states, n))
    
