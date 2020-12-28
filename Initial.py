import numpy as np
import matplotlib.pyplot as plt

#define grid
dim = 2
nb_node_per_edge = 2
nb_node_per_ele = nb_node_per_edge**2
nb_node_per_ele_dim = nb_node_per_ele*dim
pixel_size = 128
nb_pixel_y = 256
nb_pixel_x = nb_pixel_y
nb_cell_x = int(nb_pixel_x / pixel_size)
nb_cell_y = int(nb_pixel_y / pixel_size)
nb_node_x = nb_cell_x*(nb_node_per_edge-1) + 1
nb_node_y = nb_cell_y*(nb_node_per_edge-1) + 1

#define material
volume_fraction = 0.5
E = 1.0
nu = 0.3
ke_indices = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2],
                    [2, 7, 0, 5, 6, 3, 4, 1], [3, 6, 5, 0, 7, 2, 1, 4],
                    [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6],
                    [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0],], 
                    dtype=np.int8)

ke_factor = E/(1.0 - nu**2)
ke_entries = np.array([(1.0 / 2.0 - nu / 6.0)*ke_factor,
                    (1 / 8.0 + nu / 8.0)*ke_factor,
                    (-1 / 4.0 - nu / 12.0)*ke_factor,
                    (-1 / 8.0 + 3 * nu / 8.0)*ke_factor,
                    (-1 / 4.0 + nu / 12.0)*ke_factor,
                    (-1 / 8.0 - nu / 8.0)*ke_factor,
                    (nu / 6.0)*ke_factor,
                    (1 / 8.0 - 3.0 * nu / 8.0)*ke_factor],
                    dtype=np.float)

Ke = np.arange((nb_node_per_ele_dim)**2, dtype=float).reshape(nb_node_per_ele_dim, nb_node_per_ele_dim)

for i in range(nb_node_per_ele_dim):
    for j in range(nb_node_per_ele_dim):
        Ke[i,j] = ke_entries[ke_indices[i,j]]
        #Ke[i,j] = 1.0

###################################
cells = np.arange(nb_cell_x*nb_cell_y, dtype=float).reshape(nb_cell_x, nb_cell_y)
density = np.arange(nb_cell_x*nb_cell_y, dtype=float).reshape(nb_cell_x, nb_cell_y)
cell = np.arange(pixel_size*pixel_size, dtype=float).reshape(pixel_size, pixel_size)
pixels = np.arange(nb_pixel_x*nb_pixel_y, dtype=float).reshape(nb_pixel_y, nb_pixel_x)
u = np.arange(nb_node_x*nb_node_y*2, dtype=float).reshape(nb_node_x, nb_node_y, 2)
last_u = np.arange(nb_node_x*nb_node_y*2, dtype=float).reshape(nb_node_x, nb_node_y, 2)
f = np.arange(nb_node_x*nb_node_y*2, dtype=float).reshape(nb_node_x, nb_node_y, 2)
x = np.arange(nb_node_x*nb_node_y*2, dtype=float).reshape(nb_node_x, nb_node_y, 2)
Kx = np.arange(nb_node_x*nb_node_y*2, dtype=float).reshape(nb_node_x, nb_node_y, 2)

K_index_scatter = np.arange(nb_cell_x*nb_cell_y*nb_node_per_ele_dim).reshape(nb_cell_x, nb_cell_y, nb_node_per_ele_dim)
A = np.mat(np.zeros((nb_node_x*nb_node_y*dim, nb_node_x*nb_node_y*dim)))
b = np.mat(np.zeros(nb_node_x*nb_node_y*dim)).T

def get_index(i0, i1, d) -> int:
    ret = 0
    ret += i1
    ret += 2*i0
    return ret*dim + d

def set_K_index_scatter():
    for ele_x in range(nb_cell_x):
        for ele_y in range(nb_cell_y):
            for ind in range(nb_node_per_ele_dim):
                i = ind % (nb_node_per_edge*dim)
                j = ind // (nb_node_per_edge*dim)
                K_index_scatter[ele_x,ele_y,ind] = (ele_y*(nb_node_x*dim) + ele_x*dim) + i + j*(nb_node_x*dim)

def assembe_A():
    global A 
    A = np.mat(np.zeros((nb_node_x*nb_node_y*dim, nb_node_x*nb_node_y*dim)))
    for ele_x in range(nb_cell_x):
        for ele_y in range(nb_cell_y):
            for ind_x in range(nb_node_per_ele_dim):
                for ind_y in range(nb_node_per_ele_dim):
                    global_x = K_index_scatter[ele_x,ele_y,ind_x]
                    global_y = K_index_scatter[ele_x,ele_y,ind_y]
                    A[global_x, global_y] += Ke[ind_x,ind_y]

def get_b():
    global b
    for ele_x in range(nb_node_x):
        for ele_y in range(nb_node_y):
            b[ele_x*nb_node_y + ele_y] = 1.0

def update_boundary_condition():
    f[0,nb_node_y-1,1] = -1.0
    for j in range(nb_node_y):
        u[0,j,0] = 0.0
    u[nb_node_x-1,0,1] = 0.0

def initial():
    for i in range(nb_cell_x):
        for j in range(nb_cell_y):
            if i < j:
                cells[i, j] = 1.0
            else:
                cells[i, j] = 0.0
    
    for i in range(nb_cell_x):
        for j in range(nb_cell_y):
            density[i, j] = volume_fraction

    for i in range(nb_cell_x):
        for j in range(nb_cell_y):
            for p in range(pixel_size):
                for q in range(pixel_size):
                    index_x = i*pixel_size + p
                    index_y = j*pixel_size + q
                    pixels[index_y, index_x] = cells[i, nb_cell_y-j-1]

    for i in range(nb_node_x):
        for j in range(nb_node_y):
            u[i,j,0] = 0.0
            u[i,j,1] = 0.0
            last_u[i,j,0] = 0.0
            last_u[i,j,1] = 0.0
            f[i,j,0] = 0.0
            f[i,j,1] = 0.0

    update_boundary_condition()

    


def main(output_img=True):
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    set_K_index_scatter()
    assembe_A()
    #print(K_index_scatter.T)
    print(A.shape)
    print(A.T)
    #res = np.linalg.lstsq(A,b)
    #res = np.linalg.solve(A,b)
    #print(res)

    initial()
    plt.imshow(pixels, cmap = 'gray')


    #plt.show()


if __name__ == '__main__':
    main()