import trimesh
import numpy as np

def meshLoader(meshPath:str)-> trimesh.Geometry:
    """
        Loading the poly of the drop
    """
    mesh = trimesh.load(r"Poly\hemisphere_mesh_exponential.ply")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The loaded file is not a triangular mesh.")
    return mesh

def get_min_xz(mesh)->tuple:
    """
        Simply finding the min and max for X andf Z
    """
    min_x = np.min(mesh.vertices[:, 0])
    max_x = np.max(mesh.vertices[:, 0])
    min_z = np.min(mesh.vertices[:, 2])
    max_z = np.max(mesh.vertices[:, 2])
    
    return min_x, min_z, max_x, max_z