import trimesh
import numpy as np


def scale_mesh_x(input_file: str, output_file: str = "scaled.ply", scale_factor: float = 2.0):
    """
    Loads a PLY file, scales it along the X-axis, and saves the result.
    
    Args:
        input_file (str): Path to the input PLY file.
        output_file (str): Path to save the scaled PLY file. Default is 'scaled.ply'.
        scale_factor (float): Scale factor for the X-axis. Default is 2.0.
    """
    # Load the mesh
    mesh = trimesh.load(input_file)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The loaded file is not a valid triangular mesh.")

    # Create scaling matrix: scale X axis by `scale_factor`, leave Y and Z unchanged
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = scale_factor  # scale X-axis

    # Apply transformation
    mesh.apply_transform(scale_matrix)

    # Export the modified mesh
    mesh.export(output_file)
    print(f"Mesh scaled along X-axis by {scale_factor} and saved to {output_file}")

def scale_mesh_x_exponentially(input_file: str, output_file: str = "scaled_exponential.ply", k: float = 0.5):
    """
    Loads a PLY file, scales the X-axis exponentially (x' = x * exp(k * x)), and saves the result.
    
    Args:
        input_file (str): Path to the input PLY file.
        output_file (str): Path to save the scaled mesh.
        k (float): Exponential scaling factor. Positive values stretch; negative values compress.
    """
    # Load the mesh
    mesh = trimesh.load(input_file)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The loaded file is not a valid triangular mesh.")

    # Copy the vertices and apply exponential scaling on X-axis
    vertices = mesh.vertices.copy()
    x = vertices[:, 0]
    vertices[:, 0] = x * np.exp(k * x)

    # Assign new vertices
    mesh.vertices = vertices

    # Export the modified mesh
    mesh.export(output_file)
    print(f"Mesh saved to {output_file} with exponential X scaling (k={k})")


if __name__ == "__main__":
    scale_mesh_x(input_file="hemisphere_mesh.ply", output_file= "hemisphere_mesh_scaled.ply", scale_factor = 2.0)
    scale_mesh_x_exponentially(input_file="hemisphere_mesh.ply", output_file= "scaled_exponential.ply", k = 0.5)