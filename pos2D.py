import pos_solver as ps

n = 0 #index of parafields amplitude list

energy,states = ps.retrieve_array2D(n)

print(energy , states)
for i in range(15):
    ps.combine2D( energy , states, n, i)
