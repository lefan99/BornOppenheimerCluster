import pos_solver as ps


for n in range(3,13):

    energy,states = ps.retrieve_array1D(n)
    print(energy , states)
    print(ps.solve_ham1D( energy , states, n))
    
