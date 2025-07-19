import numpy as np
import trimesh

R = 1.0  # radius

# Hemisphere points as structured grid (phi, theta)
theta = np.linspace(0, 2*np.pi, 50)
phi = np.linspace(0, np.pi/2, 25)
theta_grid, phi_grid = np.meshgrid(theta, phi)

x = R * np.sin(phi_grid) * np.cos(theta_grid)
y = R * np.sin(phi_grid) * np.sin(theta_grid)
z = R * np.cos(phi_grid)

# Flatten for vertices
curved_vertices = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

# Triangulate hemisphere surface from structured grid (phi x theta)
rows, cols = phi_grid.shape
faces = []

for i in range(rows - 1):
    for j in range(cols - 1):
        # Indices of the quad corners
        p0 = i * cols + j
        p1 = p0 + 1
        p2 = p0 + cols
        p3 = p2 + 1
        # Two triangles per quad
        faces.append([p0, p1, p2])
        faces.append([p1, p3, p2])

faces = np.array(faces)

# Create flat base (disk at z=0)
res_flat = 50
x_flat = np.linspace(-R, R, res_flat)
y_flat = np.linspace(-R, R, res_flat)
x_flat_grid, y_flat_grid = np.meshgrid(x_flat, y_flat)
mask = x_flat_grid**2 + y_flat_grid**2 <= R**2

x_flat_masked = x_flat_grid[mask]
y_flat_masked = y_flat_grid[mask]
z_flat_masked = np.zeros_like(x_flat_masked)

flat_vertices = np.vstack((x_flat_masked, y_flat_masked, z_flat_masked)).T

# Triangulate flat disk with Delaunay on XY plane
from scipy.spatial import Delaunay
tri = Delaunay(flat_vertices[:, :2])
flat_faces = tri.simplices

# Combine vertices
vertices = np.vstack((curved_vertices, flat_vertices))

# Offset flat_faces indices by number of curved vertices
flat_faces_offset = flat_faces + len(curved_vertices)

# Combine faces
all_faces = np.vstack((faces, flat_faces_offset))

# Create trimesh mesh and export
mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces)
mesh.export("hemisphere_mesh.ply")

print("Mesh exported with:")
print(f" - {len(vertices)} vertices")
print(f" - {len(all_faces)} faces")
