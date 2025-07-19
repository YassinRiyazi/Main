# generate_hemisphere_stl.py
import numpy as np
from stl import mesh

def create_undulating_hemisphere_stl(filename='hemisphere.stl'):
    # Parameters
    R = 1.0  # Base radius
    mesh_density = 50  # Angular resolution
    undulation_scale = 0.08  # Surface variation scale
    undulation_freq = 3  # Number of undulations
    
    # Create angular grid
    theta = np.linspace(0, 2*np.pi, mesh_density)
    phi = np.linspace(0, np.pi/2, mesh_density//2)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Add smooth undulations
    undulations = np.sin(undulation_freq * theta_grid) * np.cos(2 * phi_grid) * undulation_scale * R
    radius = R + undulations
    
    # Create 3D points
    x = radius * np.sin(phi_grid) * np.cos(theta_grid)
    y = radius * np.sin(phi_grid) * np.sin(theta_grid)
    z = radius * np.cos(phi_grid)
    
    # Create vertices array
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Create faces (triangles)
    faces = []
    for i in range(phi_grid.shape[0]-1):
        for j in range(theta_grid.shape[1]-1):
            v0 = i * mesh_density + j
            v1 = v0 + 1
            v2 = v0 + mesh_density
            v3 = v1 + mesh_density
            
            # Create two triangles per grid cell
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    # Create the mesh
    hemisphere = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            hemisphere.vectors[i][j] = vertices[f[j]]
    
    # Create flat base
    theta = np.linspace(0, 2*np.pi, mesh_density)
    r = np.linspace(0, R, mesh_density//2)
    theta_grid, r_grid = np.meshgrid(theta, r)
    x_flat = r_grid * np.cos(theta_grid)
    y_flat = r_grid * np.sin(theta_grid)
    z_flat = np.zeros_like(x_flat)
    
    flat_vertices = np.column_stack([x_flat.ravel(), y_flat.ravel(), z_flat.ravel()])
    
    # Create faces for flat base
    flat_faces = []
    for i in range(r_grid.shape[0]-1):
        for j in range(theta_grid.shape[1]-1):
            v0 = i * mesh_density + j
            v1 = v0 + 1
            v2 = v0 + mesh_density
            v3 = v1 + mesh_density
            
            # Create two triangles per grid cell
            flat_faces.append([v0, v2, v1])
            flat_faces.append([v1, v2, v3])
    
    # Create flat mesh
    flat_mesh = mesh.Mesh(np.zeros(len(flat_faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(flat_faces):
        for j in range(3):
            flat_mesh.vectors[i][j] = flat_vertices[f[j]]
    
    # Combine meshes
    combined = mesh.Mesh(np.concatenate([hemisphere.data, flat_mesh.data]))
    
    # Save to STL
    combined.save(filename)
    print(f"Saved hemisphere mesh to {filename}")

if __name__ == "__main__":
    create_undulating_hemisphere_stl()