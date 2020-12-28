import numpy as np
import matplotlib.pyplot as plt
import time
import math

#define grid
dim = 2
nb_node_per_edge = 2
nb_node_per_ele = nb_node_per_edge**2
nb_node_per_ele_dim = nb_node_per_ele*dim
pixel_size = 16
nb_pixel_y = 256
nb_pixel_x = nb_pixel_y * 2
nb_cell_x = int(nb_pixel_x / pixel_size)
nb_cell_y = int(nb_pixel_y / pixel_size)
nb_node_x = nb_cell_x*(nb_node_per_edge-1) + 1
nb_node_y = nb_cell_y*(nb_node_per_edge-1) + 1
penalty = 3


#define material
volume_fraction = 0.5
change_limit = 0.2
minimum_density = 1e-2
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

map = np.array([0, 3, 1, 2])

def get_M(i, p):
    return map[i]*dim + p

for i in range(dim**2):
    for p in range(dim):        
        for j in range(dim**2):
            for q in range(dim):
                id_x = i*dim + p
                id_y = j*dim + q
                Ke[id_x,id_y] = ke_entries[ke_indices[get_M(i,p),get_M(j,q)]]
                #Ke[i,j] = 1.0


###################################
cell = np.arange(pixel_size*pixel_size, dtype=float).reshape(pixel_size, pixel_size)
pixels = np.arange(nb_pixel_x*nb_pixel_y, dtype=float).reshape(nb_pixel_y, nb_pixel_x)
u = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)
last_u = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)
f = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)
x = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)
Kx = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)
res = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)

K_index_scatter = np.arange(nb_cell_x*nb_cell_y*nb_node_per_ele_dim).reshape(nb_cell_x, nb_cell_y, nb_node_per_ele_dim)
A = np.mat(np.zeros((nb_node_x*nb_node_y*dim, nb_node_x*nb_node_y*dim)))
b = np.mat(np.zeros(nb_node_x*nb_node_y*dim)).T

class CG:
    def __init__(self):
        # grid parameters
        self.show_matrix = False
        self.iter = 0
        self.max_iteration = 5000

        # setup sparse simulation data arrays
        self.r = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)  # residual
        self.x = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)  # solution
        #self.x_ref = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)  # solution
        self.b = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)  # b
        self.p = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)  # conjugate gradient
        self.Ap = np.arange(nb_node_x*nb_node_y*dim, dtype=float).reshape(nb_node_x, nb_node_y, dim)  # matrix-vector product
        self.alpha = 0.0  # step size
        self.beta = 0.0  # step size
        self.sum = 0.0  # storage for reductions

    
    def update_boundary_condition(self):
        self.b[0,nb_node_y-1,1] = -1.0
        for j in range(nb_node_y):
            self.x[0,j,0] = 0.0
        self.x[nb_node_x-1,0,1] = 0.0

    def init(self):
        self.x = np.zeros((nb_node_x, nb_node_y, dim), dtype=float)
        self.Ap = np.zeros((nb_node_x, nb_node_y, dim), dtype=float)
        self.p = np.zeros((nb_node_x, nb_node_y, dim), dtype=float)
        self.b = np.zeros((nb_node_x, nb_node_y, dim), dtype=float)
        self.update_boundary_condition()

    def compute_Ap(self, density, x):
        self.Ap = np.zeros((nb_node_x, nb_node_y, dim), dtype=float)
        for ele_x in range(nb_cell_x):
            for ele_y in range(nb_cell_y):
                d = density[ele_x, ele_y]
                scale = 1.0
                for _ in range(penalty):
                    scale *= d
                for ind_Kx in range(nb_node_per_ele_dim):
                    for ind_x in range(nb_node_per_ele_dim):
                        ind_node = ind_Kx // dim
                        ind_xy = ind_Kx % dim
                        id_node_x = ele_x + ind_node // nb_node_per_edge
                        id_node_y = ele_y + ind_node % nb_node_per_edge
                    
                        x_ind_node = ind_x // dim
                        x_ind_xy = ind_x % dim
                        x_id_node_x = ele_x + x_ind_node // nb_node_per_edge
                        x_id_node_y = ele_y + x_ind_node % nb_node_per_edge
                        coeff = Ke[ind_Kx,ind_x]
                        self.Ap[id_node_x, id_node_y, ind_xy] += coeff * x[x_id_node_x, x_id_node_y, x_ind_xy] * scale
    
    def compute_r0(self, density):
        self.compute_Ap(density, self.x)
        for i in range(nb_node_x):
            for j in range(nb_node_y):
                for xy in range(dim):
                    self.r[i,j,xy] = self.b[i,j,xy] - self.Ap[i,j,xy]
    
    def update_p(self):
        for i in range(nb_node_x):
            for j in range(nb_node_y):
                for xy in range(dim):
                    self.p[i,j,xy] = self.r[i,j,xy] + self.beta * self.p[i,j,xy]
                    
    def reduce(self, p, q):
        self.sum = 0
        for i in range(nb_node_x):
            for j in range(nb_node_y):
                for xy in range(dim):
                    self.sum += p[i,j,xy] * q[i,j,xy]
    
    def update_x(self):
        for i in range(nb_node_x):
            for j in range(nb_node_y):
                for xy in range(dim):
                    self.x[i,j,xy] += self.alpha * self.p[i,j,xy]

    def update_r(self):
        for i in range(nb_node_x):
            for j in range(nb_node_y):
                for xy in range(dim):
                    self.r[i,j,xy] -= self.alpha * self.Ap[i,j,xy]
    
    def apply_dirichlet(self):
        for j in range(nb_node_y):
            self.r[0,j,0] = 0.0
        self.r[nb_node_x-1,0,1] = 0.0

    def run(self, density):
        self.init()
        self.compute_r0(density)
        #print("r0:",self.r)
        self.update_p()
        #print("p:",self.p)
        self.reduce(self.p, self.r)
        old_rTr = self.sum
        # CG
        while self.iter < self.max_iteration:
            # self.alpha = rTr / pTAp
            self.compute_Ap(density, self.p)
            #print("Ap:",self.Ap)
            self.reduce(self.p, self.Ap)
            pAp = self.sum
            self.alpha = old_rTr / pAp
            #print("alpha:",self.alpha[None])

            # self.x = self.x + self.alpha self.p
            self.update_x()
            
            # self.r = self.r - self.alpha self.Ap
            self.update_r()
            self.apply_dirichlet()

            # check for convergence
            self.reduce(self.r, self.r)
            rTr = self.sum[None]
            if rTr < 1.0e-9:
                break

            # self.beta = new_rTr / old_rTr
            self.reduce(self.r, self.r)
            new_rTr = self.sum
            self.beta = new_rTr / old_rTr

            # self.p = self.e + self.beta self.p
            self.update_p()
            
            old_rTr = new_rTr

            print(f'iter {self.iter}, residual={rTr}')
            self.iter += 1

        # ti.kernel_profiler_print()
        if(self.show_matrix):
            print("Solution:",self.x)


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

def update_pixles(new_density):
    for i in range(nb_cell_x):
        for j in range(nb_cell_y):
            for p in range(pixel_size):
                for q in range(pixel_size):
                    index_x = i*pixel_size + p
                    index_y = j*pixel_size + q
                    val = 1.0 - new_density[i, nb_cell_y-j-1]
                    pixels[index_y, index_x] = val

def initial(density):
    for i in range(nb_cell_x):
        for j in range(nb_cell_y):
            density[i, j] = volume_fraction
    #density[0,0] = 0.0       
    update_pixles(density)

    for i in range(nb_node_x):
        for j in range(nb_node_y):
            for xy in range(dim):
                u[i,j,xy] = 0.0
                last_u[i,j,xy] = 0.0
                f[i,j,xy] = 0.0

    update_boundary_condition()

def test_Kx():
    set_K_index_scatter()
    assembe_A()
    print(A.shape)
    print(A.T)

    #tmp = np.ones((nb_node_x, nb_node_y, dim))
    tmp = np.arange(0, nb_node_x*nb_node_y*2, 1).reshape(nb_node_x*nb_node_y*dim,1)
    tmp2 = tmp.reshape(nb_node_x, nb_node_y, dim)
    print(tmp.T)
    print(tmp2.T)
    #apply_K(tmp2)
    print(Kx.shape)
    print(Kx.T)
    print((A*tmp).T)

def clamp(a, min, max):
    if (a<min):
        return min
    elif (a>max):
        return max
    else:
        return a


def compute_sensitivity(u, density, sensitivity):
    for ele_x in range(nb_cell_x):
        for ele_y in range(nb_cell_y):
            dc_tmp = 0.0
            for ind_Kx in range(nb_node_per_ele_dim):
                for ind_x in range(nb_node_per_ele_dim):
                    ind_node = ind_Kx // dim
                    ind_xy = ind_Kx % dim
                    id_node_x = ele_x + ind_node // nb_node_per_edge
                    id_node_y = ele_y + ind_node % nb_node_per_edge
                    
                    x_ind_node = ind_x // dim
                    x_ind_xy = ind_x % dim
                    x_id_node_x = ele_x + x_ind_node // nb_node_per_edge
                    x_id_node_y = ele_y + x_ind_node % nb_node_per_edge

                    dc_tmp += u[id_node_x, id_node_y, ind_xy] * Ke[ind_Kx,ind_x] * u[x_id_node_x, x_id_node_y, x_ind_xy]
            sensitivity[ele_x, ele_y] = max(0.0, density[ele_x, ele_y]**(penalty - 1)*penalty*dc_tmp)

def optimality_criteria(sensitivity, density, new_density):
    lower = 0.0
    upper = 1e15
    while(lower*(1+1e-15) < upper):
        mid = 0.5 * (lower + upper)
        for ele_x in range(nb_cell_x):
            for ele_y in range(nb_cell_y):
                old_d = density[ele_x, ele_y]
                new_density[ele_x, ele_y] = clamp(old_d * math.sqrt(sensitivity[ele_x, ele_y]/mid), old_d-change_limit, old_d+change_limit)
                new_density[ele_x, ele_y] = clamp(new_density[ele_x, ele_y], minimum_density, 1.0)
        if( np.sum(new_density) - volume_fraction*nb_cell_x*nb_cell_y > 0):
            lower = mid
        else:
            upper = mid

def sensitivity_filtering(sensitivity, density, new_sensitivity):
    filter_radius = 1.5
    radius_int = math.ceil(filter_radius)
    for ele_x in range(nb_cell_x):
        for ele_y in range(nb_cell_y):
            total_s = 0.0
            total_w = 0.0
            for dx in range(-radius_int, radius_int, 1):
                for dy in range(-radius_int, radius_int, 1):
                    ni = ele_x + dx
                    nj = ele_y + dy
                    if(ni < 0 or nj < 0 or ni >= nb_cell_x or nj >= nb_cell_y):
                        continue
                    nu = density[ni, nj]
                    w = max(0.0, filter_radius-math.hypot(dx, dy))
                    total_s += w * nu * sensitivity[ni][nj]
                    total_w += w * nu
            new_sensitivity[ele_x,ele_y] = total_s/total_w

def main(output_img=True):
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    density = np.zeros((nb_cell_x, nb_cell_y), dtype=float)
    initial(density)
    #print("Ke:",Ke)
    change = 1.0
    
    while (change > 0.1):
        # Solve FEM
        solver = CG()
        t = time.time()
        solver.run(density)
        print(f'Solver time: {time.time() - t:.3f} s')

        # compute sensitivity
        sensitivity = np.zeros((nb_cell_x, nb_cell_y), dtype=float)
        compute_sensitivity(solver.x, density, sensitivity)
        #print("sensitivity:",sensitivity)
        new_sensitivity = np.zeros((nb_cell_x, nb_cell_y), dtype=float)
        sensitivity_filtering(sensitivity, density, new_sensitivity)
        #print("new_sensitivity:",new_sensitivity)

        # optimality criteria
        new_density = np.zeros((nb_cell_x, nb_cell_y), dtype=float)
        optimality_criteria(new_sensitivity, density, new_density)
        #print("new_density:",new_density)

        update_pixles(new_density)   
        change = np.max(np.abs(new_density - density))
        density = np.copy(new_density)
        print("change:",change)
        plt.imshow(pixels, cmap = 'gray')
        plt.show()

    plt.imshow(pixels, cmap = 'gray')
    plt.show()


if __name__ == '__main__':
    main()