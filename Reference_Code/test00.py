import numpy as np

import taichi as ti

ti.init(default_fp=ti.float32, arch=ti.x64, kernel_profiler=True)


x = ti.var(ti.f32)
ti.root.place(x)
# is equivalent to:
x = ti.var(ti.f32, shape=())

x = ti.var(ti.f32)
ti.root.dense(ti.i, 3).place(x)
# is equivalent to:
x = ti.var(ti.f32, shape=3)

x = ti.var(ti.f32)
ti.root.dense(ti.ij, (3, 4)).place(x)
# is equivalent to:
x = ti.var(ti.f32, shape=(3, 4))


x = ti.var(ti.f32)
p = ti.var(ti.f32)

# ti.root.dense(ti.ij, 4).dense(ti.ij, 2).place(x)

# for i in range(4):
#     for j in range(4):
#         x[i,j] = 1.0

# print(x.to_numpy())

grid = ti.root.pointer(ti.ij, [4]).dense(ti.ij, (2,1)).place(x,p)

for i in range(4):
    for j in range(4):
        x[i,j] = 1.0
        p[i,j] = 1.0

print(x.to_numpy())
print(p.to_numpy())

for l in reversed(range(2)):
    print(l)