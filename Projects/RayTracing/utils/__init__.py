import  trimesh
import  numpy           as      np

from    .PLYModifier    import  *
from    .PLY            import  *

n_air = 1.0
n_glass = 1.5

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def refract(direction, normal, n1, n2):
    direction = normalize(direction)
    normal = normalize(normal)
    cos_theta1 = -np.einsum('ij,ij->i', direction, normal)
    sin_theta1 = np.sqrt(1 - cos_theta1**2)
    sin_theta2 = (n1 / n2) * sin_theta1

    # Handle total internal reflection
    tir_mask = sin_theta2 > 1
    refracted = np.zeros_like(direction)

    cos_theta2 = np.sqrt(1 - sin_theta2**2, where=~tir_mask, out=np.zeros_like(sin_theta2))

    refracted[~tir_mask] = (n1 / n2) * direction[~tir_mask] + \
        ((n1 / n2) * cos_theta1[~tir_mask] - cos_theta2[~tir_mask])[:, None] * normal[~tir_mask]

    # Reflection for TIR rays
    refracted[tir_mask] = reflect(direction[tir_mask], normal[tir_mask])
    refracted   = normalize(refracted)
    return refracted

def reflect(direction, normal):
    direction   = normalize(direction)
    normal      = normalize(normal)
    return direction - 2 * np.einsum('ij,ij->i', direction, normal)[:, None] * normal

# Ray tracing function for multiple rays
def trace_rays(ray_origins, ray_directions,mesh, max_bounces=2):
    points = []
    directions = []
    
    current_origins = ray_origins
    current_directions = ray_directions
    
    # Start in air
    current_n = np.full(len(ray_origins), n_air)
    inside_glass = np.zeros(len(ray_origins), dtype=bool)
    
    for bounce in range(max_bounces):
        # Intersect rays with mesh
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=current_origins, ray_directions=current_directions)
        
        if len(locations) == 0:
            # No more intersections
            break
        
        # We'll handle intersections per ray:
        # For rays with multiple intersections, choose closest point (min distance)
        
        # Calculate distances from current origins to intersection points
        distances = np.linalg.norm(locations - current_origins[index_ray], axis=1)
        
        # Find closest intersection per ray
        closest_indices = {}
        for i, ray_idx in enumerate(index_ray):
            if ray_idx not in closest_indices or distances[i] < closest_indices[ray_idx][1]:
                closest_indices[ray_idx] = (i, distances[i])
        
        # Prepare new arrays to store next origins and directions
        new_origins = []
        new_directions = []
        
        for ray_idx in range(len(current_origins)):
            if ray_idx not in closest_indices:
                # No intersection for this ray, it goes off to infinity
                continue
            
            i_closest = closest_indices[ray_idx][0]
            contact_point = locations[i_closest]
            triangle_index = index_tri[i_closest]
            
            # Surface normal at intersection (face normal)
            normal = mesh.face_normals[triangle_index]
            
            # Flip normal if necessary: should point against ray direction
            if np.dot(normal, current_directions[ray_idx]) > 0:
                normal = -normal
            
            # Determine indices for refraction based on current medium
            n1 = current_n[ray_idx]
            n2 = n_glass if not inside_glass[ray_idx] else n_air
            
            # Compute refracted (or reflected) direction
            new_dir = refract(current_directions[ray_idx][None, :], normal[None, :], n1, n2)[0]
            
            # Save results
            points.append(contact_point)
            directions.append(new_dir)
            
            # Update ray for next iteration
            new_origins.append(contact_point + 1e-6 * new_dir)  # offset slightly to avoid self intersection
            new_directions.append(new_dir)
            
            # Update inside/outside state
            inside_glass[ray_idx] = not inside_glass[ray_idx]
            current_n[ray_idx] = n2
        
        if len(new_origins) == 0:
            break
        
        current_origins = np.array(new_origins)
        current_directions = np.array(new_directions)
    
    return np.array(points), np.array(directions)
