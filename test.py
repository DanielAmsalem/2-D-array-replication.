def return_neighbours(n, J, I):  # Return positions of neighbours of (j,i) in nxn matrix
    I, J = int(I), int(J)
    neighbours = []
    if J + 1 < n:  # right neighbour
        neighbours += [(I, J + 1)]
    if J > 0:  # left neighbour
        neighbours += [(I, J - 1)]
    if I + 1 < n:  # down neighbour
        neighbours += [(I + 1, J)]
    if I > 0:  # up neighbour
        neighbours += [(I - 1, J)]

    return neighbours

a=[]
for i in range(4):
    for j in range(4):
        a += [return_neighbours(4,i,j)]
print(a)
