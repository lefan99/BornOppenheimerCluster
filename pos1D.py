import pos_solver as ps


for n in range(10):

    energy,states = ps.retrieve_array1D(n)
    ps.solve_ham( energy , states, n) 
