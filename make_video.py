import open3d as o3d
import os
import numpy as np
import imageio

pc_dir = "logs/endonerf_run/reconstructed_pcds_40000"
files = sorted([f for f in os.listdir(pc_dir) if f.endswith(".ply")])

images = []

# Use OffscreenRenderer (better for HPC)
w, h = 800, 600
renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)

scene = renderer.scene
scene.set_background([0, 0, 0, 1])  # black background

material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultUnlit"

for f in files:
    pcd = o3d.io.read_point_cloud(os.path.join(pc_dir, f))

    scene.clear_geometry()
    scene.add_geometry("pcd", pcd, material)

    # setup camera
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent().max()

    cam = scene.camera
    cam.look_at(center, center + [0, 0, extent], [0, -1, 0])

    img = renderer.render_to_image()
    images.append(np.asarray(img))

renderer.release()

imageio.mimsave("output.mp4", images, fps=10)
print("Saved video: output.mp4")