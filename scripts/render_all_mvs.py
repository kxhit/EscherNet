import os
import shutil

DATA_ROOT = "/home/xin/data/EscherNet/Data/GSO30" # replace with path to GSO dataset
blender_path = "/home/xin/code/zero123/objaverse-rendering/blender-3.2.2-linux-x64/blender" # replace with path to blender

# get the list of subfolders, only folders
filenames = os.listdir(DATA_ROOT)

print(filenames)
print(len(filenames))


for filename in filenames:
    model_folder = os.path.join(DATA_ROOT, filename)
    obj_path = os.path.join(model_folder, "meshes/model.obj")
    out_dir = os.path.join(model_folder, "render_mvs_25")

    if not os.path.exists(obj_path):
        continue

    if os.path.exists(out_dir):
        # remove
        shutil.rmtree(out_dir)

    cmd = f"{blender_path} -b -P blender_script_mvs.py -- \
        --object_path {obj_path} \
        --output_dir {out_dir} \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 25 \
        --camera_dist 1.2"
    os.system(cmd)
