import os
import numpy as np
from tqdm import tqdm

# ours + NeuS
DATA_DIR = "/home/xin/data/EscherNet/Data/GSO30"   # GSO
exp_dir = "/home/xin/6DoF/GSO3D/"

config = "configs/neus_36.yaml"
exps = [1]
# exps = [1, 2, 3, 5, 10]

for exp in exps:
    OUTPUT_DIR = os.path.join(exp_dir, f"logs_GSO_T{exp}M36_99k")
    output_NeuS = f"ours_GSO_T{exp}"
    os.makedirs(output_NeuS, exist_ok=True)
    obj_names = os.listdir(DATA_DIR)
    for obj_name in tqdm(obj_names):
        if os.path.exists(os.path.join(output_NeuS, "NeuS", obj_name, "mesh.ply")):
            print("NeuS already trained for: ", obj_name)
            continue
        # remove the folder for new training
        os.system(f"rm -rf {output_NeuS}/NeuS/{obj_name}")
        print("Training NeuS for: ", obj_name)
        input_img = os.path.join(OUTPUT_DIR, obj_name, "0.png")
        # input_img = os.path.join(OUTPUT_DIR, obj_name, "gt.png")    #  ground truth image
        cmd = f"python train_renderer.py -i {input_img} \
              -d {DATA_DIR} \
              -n {obj_name} \
              -b {config} \
              -l {output_NeuS}/NeuS"
        os.system(cmd)