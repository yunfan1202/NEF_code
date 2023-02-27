import os
import time

dataset_dir = "ABC_NEF_examples"
valid_scene_names = os.listdir(dataset_dir)
# valid_scene_names = ["00000006"]
print(len(valid_scene_names))

for i in range(0, 10000):
    scene = str(i).zfill(8)
    if scene not in valid_scene_names:
        continue

    print("Processing ", scene)

    ckpt_dir = os.path.join("./ckpts_ABC", scene)
    ckpts = os.listdir(ckpt_dir)
    ckpts.sort()
    ckpt_name = ckpts[-1]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    command = "python eval.py --root_dir " + os.path.join(dataset_dir, scene) + \
              " --dataset_name blender --img_wh 400 400 --N_importance 64 --split video " \
              "--ckpt_path " + ckpt_path + " --scene_name " + scene     # + " --save_depth"

    print(command)
    os.system(command)
    time.sleep(5)



