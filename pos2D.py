import pos_solver as ps

n = 0 #index of parafields amplitude list

energy,states = ps.retrieve_array2D(n)
print(energy , states)
ps.solve_ham2D( energy , states, n)
