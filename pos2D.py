import pos_solver as ps

n = 0 #index of parafields amplitude list

energy,states = ps.retrieve_array2D(n)

e, s = ps.solve_2D_full(energy, states,n)

print(e , s)
