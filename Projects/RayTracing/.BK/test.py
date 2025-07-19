import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from stl import mesh  # Requires numpy-stl (pip install numpy-stl)

# Parameters
R = 1.0  # Base radius of hemisphere
n_air = 1.0  # Refractive index of air
n_glass = 1.5  # Refractive index of glass
num_rays = 150  # Number of rays in parallel beam
beam_width = 1.5  # Width of parallel beam
mesh_density = 50  # Points along each angular dimension
undulation_scale = 0.08  # Scale of surface undulations

# Generate mesh for undulating hemisphere
def create_hemisphere_mesh():
    theta = np.linspace(0, 2*np.pi, mesh_density)
    phi = np.linspace(0, np.pi/2, mesh_density//2)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Add smooth undulations
    undulations = np.sin(3*theta_grid) * np.cos(2*phi_grid) * undulation_scale * R
    
    # Create 3D points
    radius = R #+ undulations
    x = radius * np.sin(phi_grid) * np.cos(theta_grid)
    y = radius * np.sin(phi_grid) * np.sin(theta_grid)
    z = radius * np.cos(phi_grid)
    
    # Create triangular mesh
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Create faces (triangles)
    faces = []
    for i in range(phi_grid.shape[0]-1):
        for j in range(theta_grid.shape[1]-1):
            v0 = i * mesh_density + j
            v1 = v0 + 1
            v2 = v0 + mesh_density
            v3 = v1 + mesh_density
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    # Create mesh object
    hemisphere_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            hemisphere_mesh.vectors[i][j] = vertices[f[j]]
    
    return vertices, np.array(faces), hemisphere_mesh

# Create mesh data
vertices, faces, hemisphere_mesh = create_hemisphere_mesh()
kdtree = KDTree(vertices)  # For fast nearest-neighbor searches

# Create flat base mesh
theta = np.linspace(0, 2*np.pi, mesh_density)
r = np.linspace(0, R, mesh_density//2)
theta_grid, r_grid = np.meshgrid(theta, r)
x_flat = r_grid * np.cos(theta_grid)
y_flat = r_grid * np.sin(theta_grid)
z_flat = np.zeros_like(x_flat)
flat_vertices = np.column_stack([x_flat.ravel(), y_flat.ravel(), z_flat.ravel()])

# Ray-mesh intersection functions
def ray_triangle_intersection(ray_origin, ray_dir, triangle):
    """Möller–Trumbore intersection algorithm"""
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < 1e-6:
        return None  # Ray parallel to triangle
    
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return None
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)
    
    if v < 0.0 or u + v > 1.0:
        return None
    
    t = f * np.dot(edge2, q)
    if t > 1e-6:
        point = ray_origin + ray_dir * t
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        return point, normal
    return None

def intersect_mesh(ray_origin, ray_dir, mesh_vertices, mesh_faces):
    """Find closest intersection with mesh"""
    closest_hit = None
    min_dist = float('inf')
    
    # First find approximate nearest faces using KDTree
    _, approx_idx = kdtree.query(ray_origin)
    candidate_faces = [f for f in faces if approx_idx in f]
    
    for face in candidate_faces:
        triangle = [mesh_vertices[face[0]], 
                    mesh_vertices[face[1]], 
                    mesh_vertices[face[2]]]
        hit = ray_triangle_intersection(ray_origin, ray_dir, triangle)
        if hit:
            point, normal = hit
            dist = np.linalg.norm(point - ray_origin)
            if dist < min_dist:
                min_dist = dist
                closest_hit = (point, normal)
    
    return closest_hit

# Optical physics functions (same as before)
def refract(direction, normal, n1, n2):
    direction = direction / np.linalg.norm(direction)
    normal = normal / np.linalg.norm(normal)
    cos_theta1 = -np.dot(direction, normal)
    sin_theta1 = np.sqrt(1 - cos_theta1**2)
    sin_theta2 = (n1 / n2) * sin_theta1
    if sin_theta2 > 1:
        return reflect(direction, normal)
    cos_theta2 = np.sqrt(1 - sin_theta2**2)
    return n1/n2 * direction + (n1/n2 * cos_theta1 - cos_theta2) * normal

def reflect(direction, normal):
    direction = direction / np.linalg.norm(direction)
    normal = normal / np.linalg.norm(normal)
    return direction - 2 * np.dot(direction, normal) * normal

# Visualization setup
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh surface (simplified for visualization)
ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], 
               triangles=faces, color='cyan', alpha=0.3)
ax.scatter(x_flat.ravel(), y_flat.ravel(), z_flat.ravel(), 
           color='cyan', alpha=0.2)

# Ray tracing with parallel beam
beam_direction = np.array([0, 1, 0])  # Coming straight down

for i in range(num_rays):
    x_pos = beam_width * (i / (num_rays-1) - 0.5)
    ray_origin = np.array([x_pos, -1, 0.5])
    segments = [ray_origin]
    
    # First intersection with hemisphere
    hit = intersect_mesh(ray_origin, beam_direction, vertices, faces)
    if hit:
        point1, normal1 = hit
        segments.append(point1)
        
        # Refraction into glass
        refracted_dir = refract(beam_direction, normal1, n_air, n_glass)
        
        # Find exit point
        hit_inside = intersect_mesh(point1 + 1e-5*refracted_dir, refracted_dir, vertices, faces)
        hit_flat = intersect_mesh(point1 + 1e-5*refracted_dir, refracted_dir, flat_vertices, [])
        
        if hit_inside and hit_flat:
            dist_inside = np.linalg.norm(hit_inside[0] - point1)
            dist_flat = np.linalg.norm(hit_flat[0] - point1)
            exit_point, exit_normal = hit_inside if dist_inside < dist_flat else hit_flat
        elif hit_inside:
            exit_point, exit_normal = hit_inside
        elif hit_flat:
            exit_point, exit_normal = hit_flat
        else:
            reflected_dir = reflect(refracted_dir, normal1)
            exit_point = point1 + reflected_dir * 0.5
            segments.append(exit_point)
            segments = np.array(segments)
            ax.plot(segments[:,0], segments[:,1], segments[:,2], 'm-', alpha=0.8, linewidth=2)
            continue
            
        segments.append(exit_point)
        final_dir = refract(refracted_dir, -exit_normal, n_glass, n_air)
        final_point = exit_point + final_dir * 1.5
        segments.append(final_point)
        
        segments = np.array(segments)
        ax.plot(segments[:,0], segments[:,1], segments[:,2], 'r-', alpha=0.9, linewidth=2)
    else:
        miss_point = ray_origin + beam_direction * 2.5
        segments.append(miss_point)
        segments = np.array(segments)
        ax.plot(segments[:,0], segments[:,1], segments[:,2], 'k--', alpha=0.4, linewidth=1)

# Final plot adjustments
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Mesh-Based Ray Tracing with Smoothly Undulating Hemisphere', fontsize=12)
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-0.5, 2.5])
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.show()