import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
R = 1.0  # Radius of the hemisphere
n_air = 1.0  # Refractive index of air
n_glass = 1.5  # Refractive index of glass
light_pos = np.array([0, 3, 0])  # Position of light source
num_rays = 20  # Number of rays to trace

# Create hemisphere surface points for visualization
theta = np.linspace(0, 2*np.pi, 50)
phi = np.linspace(0, np.pi/2, 25)
theta_grid, phi_grid = np.meshgrid(theta, phi)
x = R * np.sin(phi_grid) * np.cos(theta_grid)
y = R * np.sin(phi_grid) * np.sin(theta_grid)
z = R * np.cos(phi_grid)

# Flat surface of the hemisphere (z=0 plane)
x_flat = np.linspace(-R, R, 20)
y_flat = np.linspace(-R, R, 20)
x_flat, y_flat = np.meshgrid(x_flat, y_flat)
z_flat = np.zeros_like(x_flat)
mask = x_flat**2 + y_flat**2 <= R**2
x_flat, y_flat, z_flat = x_flat[mask], y_flat[mask], z_flat[mask]

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

def intersect_hemisphere(ray_origin, ray_direction):
    """Find intersection of ray with hemisphere"""
    a = np.dot(ray_direction, ray_direction)
    b = 2 * np.dot(ray_origin, ray_direction)
    c = np.dot(ray_origin, ray_origin) - R**2
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    
    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    # We want the smallest positive t
    t = min(t for t in [t1, t2] if t > 0)
    if t <= 0:
        return None
    
    point = ray_origin + t * ray_direction
    # Check if it's the hemisphere part (z >= 0)
    if point[2] < 0:
        return None
    
    normal = point / np.linalg.norm(point)
    return point, normal

def intersect_plane(ray_origin, ray_direction):
    """Find intersection with z=0 plane"""
    if np.abs(ray_direction[2]) < 1e-6:
        return None
    
    t = -ray_origin[2] / ray_direction[2]
    if t <= 0:
        return None
    
    point = ray_origin + t * ray_direction
    # Check if within hemisphere base
    if point[0]**2 + point[1]**2 > R**2:
        return None
    
    normal = np.array([0, 0, -1])
    return point, normal

# Set up plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot hemisphere
ax.plot_surface(x, y, z, color='cyan', alpha=0.3)
ax.scatter(x_flat, y_flat, z_flat, color='cyan', alpha=0.3)

# Plot light source
ax.scatter(*light_pos, color='yellow', s=100, label='Light Source')

# Generate and trace rays
for i in range(num_rays):
    # Create a ray from the light source to the hemisphere
    angle = 2 * np.pi * i / num_rays
    target = np.array([0.7 * R * np.cos(angle), 0.7 * R * np.sin(angle), 0])
    ray_dir = target - light_pos
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    
    # Trace the ray
    segments = [light_pos]
    
    # First intersection with hemisphere
    hit = intersect_hemisphere(light_pos, ray_dir)
    if hit:
        point1, normal1 = hit
        segments.append(point1)
        
        # Refraction into glass
        refracted_dir = refract(ray_dir, normal1, n_air, n_glass)
        
        # Find exit point (could be through curved surface or flat surface)
        hit_inside = intersect_hemisphere(point1 + 1e-5*refracted_dir, refracted_dir)
        hit_flat = intersect_plane(point1 + 1e-5*refracted_dir, refracted_dir)
        
        if hit_inside and hit_flat:
            # Choose the closer intersection
            dist_inside = np.linalg.norm(hit_inside[0] - point1)
            dist_flat = np.linalg.norm(hit_flat[0] - point1)
            exit_point, exit_normal = hit_inside if dist_inside < dist_flat else hit_flat
        elif hit_inside:
            exit_point, exit_normal = hit_inside
        elif hit_flat:
            exit_point, exit_normal = hit_flat
        else:
            continue
            
        segments.append(exit_point)
        
        # Refraction out of glass
        final_dir = refract(refracted_dir, -exit_normal, n_glass, n_air)
        
        # Extend the ray a bit
        final_point = exit_point + 2 * final_dir
        segments.append(final_point)
        
        # Plot the ray path
        segments = np.array(segments)
        ax.plot(segments[:, 0], segments[:, 1], segments[:, 2], 'r-', alpha=0.7)

# Add labels and adjust view
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Ray Tracing through a Transparent Hemisphere')
ax.legend()
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-0.5, 3.5])
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()