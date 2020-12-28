import taichi as ti
import numpy as np
import math
import time

ti.init(arch=ti.gpu)

def vec(*xs):
  return ti.Vector(list(xs))

base_size = 256
pixel = ti.var(ti.f32, (2*base_size, 2*base_size))
shadow = ti.var(ti.f32, (base_size, base_size, base_size))
fram_iter = ti.var(ti.i32, ())

@ti.func
def rlight():
  return vec(-1.0, 0.0, 0.0).normalized()

@ti.func
def rdstep():
  return 0.006

@ti.func
def rrange():
  return 750

# Computes intersection with an AABB cube.
@ti.func
def intersect_cube(ray_o, ray_dir):
    xmin, xmax = -1.0, 1.0
    ymin, ymax = -1.0, 1.0
    zmin, zmax = -1.0, 1.0

    # Solves: ro + t*rd == [x|y|z][min|max]
    t_side_a = (vec(xmin, ymin, zmin) - ray_o) / ray_dir
    t_side_b = (vec(xmax, ymax, zmax) - ray_o) / ray_dir
    txmin = ti.min(t_side_a[0], t_side_b[0])
    tymin = ti.min(t_side_a[1], t_side_b[1])
    tzmin = ti.min(t_side_a[2], t_side_b[2])
    txmax = ti.max(t_side_a[0], t_side_b[0])
    tymax = ti.max(t_side_a[1], t_side_b[1])
    tzmax = ti.max(t_side_a[2], t_side_b[2])

    tmin_argmax = 1.0
    if txmin > tzmin:
        tmin_argmax = 0
    if tymin > txmin and tymin > tzmin:
        tmin_argmax = 1

    t = math.inf

    t_earliest_exit = ti.min(txmax, ti.min(tymax, tzmax))
    t_latest_enter = ti.max(txmin, ti.max(tymin, tzmin))

    if t_earliest_exit < t_latest_enter:
        pass
    elif tmin_argmax == 0:
        t = txmin
    elif tmin_argmax == 1:
        t = tymin
    else:
        t = tzmin
        
    return t

# Rayleigh Scattering
@ti.func
def rayleigh_phase(light_dir, ray_dir):
  cos = light_dir.dot(ray_dir)
  return (0.1875 / math.pi) * (1 + cos ** 2)

@ti.func
def phase_at_surface(light_dir, ray_dir, cube_pos):
    realy_phase = 1.0
    if(cube_pos[0] > -1.0):
        realy_phase = rayleigh_phase(light_dir, ray_dir)
    return realy_phase

@ti.func
def extinAt(p):
  return 0.9

@ti.func
def lightAt(p, d, length):
  trans = 1.0
  trans *= ti.exp(- rdstep() * length * extinAt(p))
  return 40 * trans

@ti.func
def radiance(ray_o, ray_dir):
    trans = 1.0
    total = 0.0
    ray_t = math.inf
    cube_t = intersect_cube(ray_o, ray_dir)
    insca = 0.9
    if(cube_t < ray_t):
        tmp_len = int(cube_t/rdstep())   
        for i in range(tmp_len):
            total += trans * lightAt(ray_o, ray_dir, tmp_len) * insca * rayleigh_phase(rlight(), ray_dir) * rdstep()   
    else:
        for i in range(rrange()):
            total += trans * lightAt(ray_o, ray_dir, rrange()) * insca * rayleigh_phase(rlight(), ray_dir) * rdstep()
    return total

@ti.func
def compute_barycentric_coord(a: ti.Vector, b: ti.Vector, c: ti.Vector, p: ti.Vector) -> ti.Vector:
    # S_abp = ab x ap / 2
    # S_abc = ab x ac / 2
    (pa, pb, pc) = a-p, b-p, c-p
    (ca, cb) = a-c, b-c
    S_abp = pa[0] * pb[1] - pa[1] * pb[0]
    S_bcp = pb[0] * pc[1] - pb[1] * pc[0]
    S_cap = pc[0] * pa[1] - pc[1] * pa[0]
    S_abc = ca[0] * cb[1] - ca[1] * cb[0]
    return ti.Vector([S_bcp / S_abc, S_cap / S_abc, S_abp / S_abc])

# Defines the General Linear Camera
@ti.func
def defineGLC(frame_count: ti.f32):
    distance = frame_count / 400 + 1
    camera_c = ti.Vector([0.0, 0.0, distance])

    # Defines the GLC.
    GLC_rotate_angle = 0.0  # Change angle to 0 for a standard pinhole camera.
    cos_angle, sin_angle = ti.cos(GLC_rotate_angle), ti.sin(GLC_rotate_angle)
    GLC_rotater = ti.Matrix([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # Computes the transform to a rotating camera.
    # --------------------------------------------
    camera_rotation_radius = 6.0
    # Camera rotates around the scene about 6 seconds a period.
    camera_rotation_phi = frame_count / 200
    # Camera looks slightly down upon the y=0 plane. 
    camera_rotation_theta = math.pi / 3

    camera_c = camera_rotation_radius * \
            ti.Vector([ti.sin(camera_rotation_theta) * ti.cos(camera_rotation_phi),
                       ti.cos(camera_rotation_theta),
                       ti.sin(camera_rotation_theta) * ti.sin(camera_rotation_phi)])
    forward = ti.Vector([0.0, 1.0, 0.0], dt=ti.f32) - camera_c
    up = ti.Vector([0.0, 1.0, 0.0], dt=ti.f32)
    # right x up = -forward -> up x -forward = right
    right = (forward.cross(up)).normalized()
    up = (right.cross(forward)).normalized()
    forward = forward.normalized()

    camera_orientation = ti.Matrix(cols = [right, up, -forward])
    return GLC_rotater, camera_c, camera_orientation

@ti.kernel
def render(frame_count: ti.f32):
    
    GLC_rotater, camera_c, camera_orientation = defineGLC(frame_count)

    A1 = GLC_rotater @ ti.Vector([0.0, 1.0])
    B1 = GLC_rotater @ ti.Vector([-1.0, -1.0])
    C1 = GLC_rotater @ ti.Vector([1.0, -1.0])

    A0 = ti.Vector([0.0, 0.5])
    B0 = ti.Vector([-0.5, -0.5])
    C0 = ti.Vector([0.5, -0.5])

    resolution = 2*base_size

    for i, j in pixel:
        # Computes the ray origin and dir using GLC.
        near_point = ti.Vector([i *2.0 / resolution - 1, j * 2.0 / resolution - 1])
        bc_coord = compute_barycentric_coord(A0, B0, C0, near_point)
        far_point = A1*bc_coord[0] + B1*bc_coord[1] + C1*bc_coord[2]

        ray_o = ti.Vector([near_point[0], near_point[1], 0.0])
        ray_dir = ti.Vector([far_point[0] - near_point[0], far_point[1] - near_point[1], -1.0])
        ray_o = (camera_orientation @ ray_o) + camera_c
        ray_dir = (camera_orientation @ ray_dir).normalized()
        pixel[i, j] = radiance(ray_o, ray_dir)

def main():
    gui = ti.GUI('Volumetric Light + General Linear Camera')
    fram_iter[None] = 0
    while True:
        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                exit()
        render(fram_iter[None])
        gui.set_image(pixel.to_numpy())
        gui.show()
        fram_iter[None] += 1
            
if __name__ == '__main__':
    main()