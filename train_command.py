import os
import time

dataset_dir = "ABC_NEF_examples"
valid_scene_names = os.listdir(dataset_dir)
# valid_scene_names = ["00000006"]
print(len(valid_scene_names))
epoch = 6

for i in range(0, 10000):
    scene = str(i).zfill(8)
    if scene in valid_scene_names:
        print("Processing ", scene)
        command = "python train.py --dataset_name blender " \
                  "--root_dir " + os.path.join(dataset_dir, scene) +\
                  " --N_importance 64 --img_wh 400 400 --noise_std 0 --num_epochs "+str(epoch) + " --batch_size 1024 " \
                  "--optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 --exp_name " + scene
        print(command)
        os.system(command)
        time.sleep(5)

