import os
import open3d as o3d
import numpy as np
import imageio

pc_dir = "logs/endonerf_run/reconstructed_pcds_40000"
out_dir = "frames"
os.makedirs(out_dir, exist_ok=True)

files = sorted([f for f in os.listdir(pc_dir) if f.endswith(".ply")])

w, h = 800, 600
renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
scene = renderer.scene

material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultUnlit"

for i, f in enumerate(files):
    pcd = o3d.io.read_point_cloud(os.path.join(pc_dir, f))
    
    scene.clear_geometry()
    scene.add_geometry("pcd", pcd, material)

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = max(bbox.get_extent())

    cam = scene.camera
    cam.look_at(center, center + [0, 0, extent], [0, -1, 0])

    img = renderer.render_to_image()
    imageio.imwrite(f"{out_dir}/frame_{i:04d}.png", np.asarray(img))

renderer.release()
print("Frames saved.")
