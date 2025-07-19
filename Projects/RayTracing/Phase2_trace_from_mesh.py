# import  trimesh
# import  utils 
# import  numpy       as  np

# # Refractive indices
# num_x = 30
# num_z = 30
# # Sky blue color for mesh faces
# sky_blue = [135/255, 206/255, 235/255, 1.0]  # RGBA
# # When adding spheres and ray paths for beam, color them red
# red = [255, 0, 0, 255]


# # Load mesh
# # mesh = trimesh.load("hemisphere_mesh.ply")
# mesh = trimesh.load(r"Projects/RayTracing/Poly/hemisphere_mesh.ply")
# if not isinstance(mesh, trimesh.Trimesh):
#     raise ValueError("The loaded file is not a triangular mesh.")

# mesh.fix_normals()


# # Example usage
# min_x, min_z, max_x, max_z = utils.get_min_xz(mesh)

# x = np.linspace(min_x, max_x, num_x)
# z = np.linspace(0, max_z, num_z)
# origins = np.array([[xi, -2.0, zi] for zi in z for xi in x])

# # directions = np.tile([0, 1, 0], (len(origins), 1))  # pointing +Y
# # directions /= np.linalg.norm(directions, axis=1)[:, None]
# directions = np.tile([0, 1, 0], (len(origins), 1)).astype(np.float64)
# directions /= np.linalg.norm(directions, axis=1)[:, None]

# contact_points, new_dirs = utils.trace_rays(origins, directions, mesh)

# print(f"Traced {len(contact_points)} contact points")



# # Paint entire mesh faces sky blue
# mesh.visual.face_colors = (np.array(sky_blue)*255).astype(np.uint8)

# scene = trimesh.Scene(mesh)


# for i, p in enumerate(contact_points):
#     sphere = trimesh.creation.uv_sphere(radius=0.02)
#     sphere.apply_translation(p)
#     sphere.visual.face_colors = red
#     scene.add_geometry(sphere)

#     start = p
#     end = p + new_dirs[i] * 0.3
#     path = trimesh.load_path(np.vstack([start, end]))

#     # Set color for each entity (line) in the path
#     n_entities = len(path.entities)
#     colors = np.tile(red, (n_entities, 1))
#     path.colors = colors

#     scene.add_geometry(path)


# # Create a black rectangular ground plane under the hemisphere
# ground_size = 3.0  # size of the ground plane
# ground_z = -0.01   # slightly below the hemisphere base (which is at z=0)

# # Create vertices of the square ground plane
# vertices = np.array([
#     [-ground_size/2, -ground_size/2, ground_z],
#     [ ground_size/2, -ground_size/2, ground_z],
#     [ ground_size/2,  ground_size/2, ground_z],
#     [-ground_size/2,  ground_size/2, ground_z]
# ])

# # Create two triangles to make a square plane
# faces = np.array([
#     [0, 1, 2],
#     [0, 2, 3]
# ])

# # Create mesh for ground
# ground = trimesh.Trimesh(vertices=vertices, faces=faces)
# ground.visual.face_colors = [0, 0, 0, 255]  # black with full opacity

# # Add ground to the scene
# scene.add_geometry(ground)


# scene.show()

import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import numpy as np
import trimesh
import pyvista as pv
import utils

# Parameters
num_x = 30
num_z = 30
sky_blue = [135 / 255, 206 / 255, 235 / 255]
red = [1.0, 0.0, 0.0]

# Load mesh
mesh_path = "Projects/RayTracing/Poly/hemisphere_mesh.ply"
mesh = trimesh.load(mesh_path)
if not isinstance(mesh, trimesh.Trimesh):
    raise ValueError("Loaded file is not a triangular mesh.")
mesh.fix_normals()

# Ray source grid
min_x, min_z, max_x, max_z = utils.get_min_xz(mesh)
x = np.linspace(min_x, max_x, num_x)
z = np.linspace(0, max_z, num_z)
origins = np.array([[xi, -2.0, zi] for zi in z for xi in x])
directions = np.tile([0, 1, 0], (len(origins), 1)).astype(np.float64)
directions /= np.linalg.norm(directions, axis=1)[:, None]

# Trace rays
contact_points, new_dirs = utils.trace_rays(origins, directions, mesh)
print(f"Traced {len(contact_points)} contact points")

# PyVista plotter
pl = pv.Plotter(off_screen=True)
pl.set_background("white")

# Convert mesh to PyVista
vertices = mesh.vertices
faces = mesh.faces
faces_pv = np.hstack([[3] + list(face) for face in faces]).reshape(-1, 4)
pv_mesh = pv.PolyData(vertices, faces_pv)
pv_mesh["colors"] = np.tile(np.array(sky_blue), (len(faces), 1))
pl.add_mesh(pv_mesh, color=sky_blue, opacity=1.0, show_edges=False)

# Add ground plane
ground_size = 3.0
ground_z = -0.01
ground_vertices = np.array([
    [-ground_size/2, -ground_size/2, ground_z],
    [ ground_size/2, -ground_size/2, ground_z],
    [ ground_size/2,  ground_size/2, ground_z],
    [-ground_size/2,  ground_size/2, ground_z]
])
ground_faces = [[3, 0, 1, 2], [3, 0, 2, 3]]
ground = pv.PolyData(ground_vertices, np.hstack(ground_faces))
pl.add_mesh(ground, color="black", opacity=1.0)

# Add contact points and refracted paths
for i, p in enumerate(contact_points):
    # Sphere at contact
    sphere = pv.Sphere(radius=0.02, center=p)
    pl.add_mesh(sphere, color=red)

    # Ray continuation line
    end = p + new_dirs[i] * 0.3
    line = pv.Line(p, end)
    pl.add_mesh(line, color=red)

# Show or save
# Save to file instead of showing
pl.screenshot("render_output.png")
print("Saved to render_output.png")
# Or save to file:
# pl.screenshot("output.png")
