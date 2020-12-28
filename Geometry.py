import numpy as np
import time
import taichi as ti

lp = ti.f32
ti.init(default_fp=lp, arch=ti.x64, kernel_profiler=False)
def vec(*xs):
  return ti.Vector(list(xs))

#define grid
pixel_size = 8
nb_pixel_y = 128
nb_pixel_x = 2 * nb_pixel_y
nb_cell_y = int(nb_pixel_y / pixel_size)
nb_cell_x = int(nb_pixel_x / pixel_size)
pixels = ti.var(dt=lp)
cell = ti.var(dt=lp)
cells = ti.var(dt=lp)

indicies = ti.ij
ti.root.dense(indicies, (nb_cell_x, nb_cell_y)).place(cells)
ti.root.dense(indicies, (pixel_size, pixel_size)).place(cell)
ti.root.dense(indicies, (nb_pixel_x, nb_pixel_y)).place(pixels)

def initial():
    for i in range(nb_cell_x):
        for j in range(nb_cell_y):
            if i < j:
                cells[i, j] = 1.0
            else:
                cells[i, j] = 0.0

    for i in range(nb_cell_x):
        for j in range(nb_cell_y):
            for p in range(pixel_size):
                for q in range(pixel_size):
                    index_x = i*pixel_size + p
                    index_y = j*pixel_size + q
                    val = cells[i, j]
                    pixels[index_x, index_y] = val


def main(output_img=True):

    gui = ti.GUI("Topo_Optimize", res=(nb_pixel_x, nb_pixel_y))
    initial()
    while True:
        gui.set_image(pixels.to_numpy())
        gui.show()
        if gui.get_event(ti.GUI.ESCAPE):
            exit()

if __name__ == '__main__':
    main()