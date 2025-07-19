# ray_trace_stl.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from scipy.spatial import KDTree

# Optical physics functions
def refract(direction, normal, n1, n2):
    """Compute refraction direction using Snell's law"""
    direction = direction / np.linalg.norm(direction)
    normal = normal / np.linalg.norm(normal)
    
    cos_theta1 = -np.dot(direction, normal)
    sin_theta1 = np.sqrt(1 - cos_theta1**2)
    sin_theta2 = (n1 / n2) * sin_theta1
    
    if sin_theta2 > 1:
        # Total internal reflection
        return reflect(direction, normal)
    
    cos_theta2 = np.sqrt(1 - sin_theta2**2)
    return n1/n2 * direction + (n1/n2 * cos_theta1 - cos_theta2) * normal

def reflect(direction, normal):
    """Compute reflection direction"""
    direction = direction / np.linalg.norm(direction)
    normal = normal / np.linalg.norm(normal)
    return direction - 2 * np.dot(direction, normal) * normal

def ray_triangle_intersection(ray_origin, ray_direction, triangle):
    """Möller-Trumbore ray-triangle intersection algorithm"""
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < 1e-6:
        return None  # Ray parallel to triangle
    
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return None
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    
    if v < 0.0 or u + v > 1.0:
        return None
    
    t = f * np.dot(edge2, q)
    if t > 1e-6:  # Intersection in front of ray origin
        point = ray_origin + ray_direction * t
        # Calculate normal
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        return point, normal
    
    return None

def find_closest_intersection(ray_origin, ray_direction, stl_mesh, kdtree):
    """Find closest intersection point in STL mesh"""
    closest_hit = None
    min_distance = float('inf')
    
    # Find nearest vertex using KDTree
    _, idx = kdtree.query(ray_origin)
    candidate_triangles = []
    
    # Find triangles containing this vertex
    for i, triangle in enumerate(stl_mesh.vectors):
        if any(np.array_equal(stl_mesh.v0[i], kdtree.data[idx]) or
               np.array_equal(stl_mesh.v1[i], kdtree.data[idx]) or
               np.array_equal(stl_mesh.v2[i], kdtree.data[idx])):
            candidate_triangles.append(triangle)
    
    # Check candidate triangles
    for triangle in candidate_triangles:
        hit = ray_triangle_intersection(ray_origin, ray_direction, triangle)
        if hit:
            point, normal = hit
            distance = np.linalg.norm(point - ray_origin)
            if distance < min_distance:
                min_distance = distance
                closest_hit = (point, normal)
    
    return closest_hit

def main():
    # Parameters
    n_air = 1.0
    n_glass = 1.5
    num_rays = 15
    beam_width = 1.5
    
    # Load STL mesh
    stl_mesh = mesh.Mesh.from_file('hemisphere.stl')
    
    # Create KDTree for vertices
    vertices = np.vstack([stl_mesh.v0, stl_mesh.v1, stl_mesh.v2])
    kdtree = KDTree(vertices)
    
    # Set up plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh (simplified)
    ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], 
                   triangles=np.arange(len(vertices)).reshape(-1, 3), 
                   alpha=0.2, color='cyan')
    
    # Create parallel beam
    beam_direction = np.array([0, 0, -1])  # Coming straight down
    
    # Trace rays
    for i in range(num_rays):
        # Ray starting position
        x_pos = beam_width * (i / (num_rays-1) - 0.5)
        ray_origin = np.array([x_pos, 0, 2])
        segments = [ray_origin]
        
        # First intersection with hemisphere
        hit = find_closest_intersection(ray_origin, beam_direction, stl_mesh, kdtree)
        
        if hit:
            point1, normal1 = hit
            segments.append(point1)
            
            # Refraction into glass
            refracted_dir = refract(beam_direction, normal1, n_air, n_glass)
            
            # Find exit point (from inside)
            hit_exit = find_closest_intersection(point1 + 1e-5*refracted_dir, refracted_dir, stl_mesh, kdtree)
            
            if hit_exit:
                exit_point, exit_normal = hit_exit
                segments.append(exit_point)
                
                # Refraction out of glass
                final_dir = refract(refracted_dir, -exit_normal, n_glass, n_air)
                
                # Extend ray
                final_point = exit_point + final_dir * 1.5
                segments.append(final_point)
                
                # Plot ray path
                segments = np.array(segments)
                ax.plot(segments[:,0], segments[:,1], segments[:,2], 'r-', linewidth=2, alpha=0.8)
            else:
                # Internal reflection if no exit found
                reflected_dir = reflect(refracted_dir, normal1)
                reflection_point = point1 + reflected_dir * 0.5
                segments.append(reflection_point)
                segments = np.array(segments)
                ax.plot(segments[:,0], segments[:,1], segments[:,2], 'm-', linewidth=2, alpha=0.8)
        else:
            # Ray misses hemisphere
            miss_point = ray_origin + beam_direction * 2.5
            segments.append(miss_point)
            segments = np.array(segments)
            ax.plot(segments[:,0], segments[:,1], segments[:,2], 'k--', alpha=0.4)
    
    # Final plot adjustments
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ray Tracing through Undulating Hemisphere (STL Mesh)', fontsize=12)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-0.5, 2.5])
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()