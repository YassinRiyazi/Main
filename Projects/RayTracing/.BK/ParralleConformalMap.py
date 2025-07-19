import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

# Parameters
R = 1.0
n_air = 1.0
n_glass = 1.5
num_rays = 50
beam_width = 1.8

# Generate spherical cap
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi/2, 50)
theta_grid, phi_grid = np.meshgrid(theta, phi)
x = R * np.sin(phi_grid) * np.cos(theta_grid)
y = R * np.sin(phi_grid) * np.sin(theta_grid)
z = R * np.cos(phi_grid)

# Apply inverse Joukowsky transform to x, y
def inverse_joukowsky(x, y):
    zeta = x + 1j*y
    sqrt_term = np.sqrt(zeta**2 - 4 + 1e-5)  # Prevent negative sqrt
    inv = 0.5 * (zeta + sqrt_term)
    return np.real(inv), np.imag(inv)

x_j, y_j = inverse_joukowsky(x, y)

# Stack into points and faces for triangulated mesh
points = np.vstack([x_j.ravel(), y_j.ravel(), z.ravel()]).T
faces = []

rows, cols = x_j.shape
for i in range(rows - 1):
    for j in range(cols - 1):
        idx = lambda r, c: r * cols + c
        faces.append([idx(i, j), idx(i+1, j), idx(i, j+1)])
        faces.append([idx(i+1, j), idx(i+1, j+1), idx(i, j+1)])
faces = np.array(faces)

# Create mesh
mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

# Build KDTree for normal estimation
kdtree = cKDTree(mesh.vertices)

def surface_normal(point):
    # Use KDTree to get nearest vertex normal
    dist, idx = kdtree.query(point)
    tri_ids = mesh.vertex_faces[idx]
    tri_ids = tri_ids[tri_ids != -1]
    normals = mesh.face_normals[tri_ids]
    return np.mean(normals, axis=0)

def intersect_mesh(ray_origin, ray_dir):
    locs, index_ray, index_tri = mesh.ray.intersects_location([ray_origin], [ray_dir])
    if len(locs) == 0:
        return None
    return locs[0]

def reflect(d, n):
    return d - 2 * np.dot(d, n) * n

def refract(d, n, n1, n2):
    d = d / np.linalg.norm(d)
    n = n / np.linalg.norm(n)
    cos_theta1 = -np.dot(d, n)
    sin_theta1_sq = 1 - cos_theta1**2
    sin_theta2_sq = (n1 / n2)**2 * sin_theta1_sq

    if sin_theta2_sq > 1:
        return reflect(d, n)

    cos_theta2 = np.sqrt(1 - sin_theta2_sq)
    return (n1 / n2) * d + (n1 / n2 * cos_theta1 - cos_theta2) * n

# Plot setup
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
mesh.show(ax=ax)

# Ray tracing
ray_dir = np.array([0, 1, 0])

for j in range(0, 20):
    for i in range(num_rays):
        x_pos = beam_width * (i / (num_rays - 1) - 0.5)
        ray_origin = np.array([x_pos, -2, j / 10])
        segments = [ray_origin]

        hit = intersect_mesh(ray_origin, ray_dir)
        if hit is None:
            continue

        normal1 = surface_normal(hit)
        segments.append(hit)

        refracted_dir = refract(ray_dir, normal1, n_air, n_glass)

        # Trace inside the object to find the exit
        internal_origin = hit + 1e-4 * refracted_dir
        hit_exit = intersect_mesh(internal_origin, refracted_dir)
        if hit_exit is None:
            continue

        normal2 = surface_normal(hit_exit)
        segments.append(hit_exit)

        final_dir = refract(refracted_dir, -normal2, n_glass, n_air)
        final_point = hit_exit + 0.5 * final_dir
        segments.append(final_point)

        segments = np.array(segments)
        ax.plot(segments[:, 0], segments[:, 1], segments[:, 2], 'r-', alpha=0.6)

# Axes
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-2, 2)
ax.set_zlim(-0.5, 2.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ray Tracing Through Drop-shaped Lens (Inverse Joukowsky)')
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.show()
