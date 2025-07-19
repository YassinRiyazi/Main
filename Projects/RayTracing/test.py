import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your mesh from disk
mesh = trimesh.load('Projects/RayTracing/Poly/hemisphere_mesh_scaled.ply')

# 2. Convert to a pyrender.Mesh
render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

# 3. Create a scene with a black background
scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[0.1, 0.1, 0.1])
scene.add(render_mesh)

# 4. Add a directional (parallel) light
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
# Identity pose: light shines along the scene’s –Z axis
scene.add(light, pose=np.eye(4))

# 5. Place a camera looking at the mesh
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
cam_pose = np.eye(4)
cam_pose[:3, 3] = [0, 0, 3]   # move camera 3 units along +Z
scene.add(camera, pose=cam_pose)

# 6. Render offscreen
r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
color, depth = r.render(scene)

# 7. Display the result
plt.figure(figsize=(8, 6))
plt.imshow(color)
plt.axis('off')
plt.show()
