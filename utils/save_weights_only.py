import torch
import argparse
import glob
import os

def collect_ckpt_path(file_dir):
    L=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == '.ckpt':
                L.append(os.path.join(dirpath, file))
    return L

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='checkpoint path')

    return parser.parse_args()

if __name__ == "__main__":
    # args = get_opts()

    ckpt_dir = "/home/yyf/Workspace/NeRF/codes/nerf_pl_edge/ckpts_ABC"
    ckpts_path = collect_ckpt_path(ckpt_dir)
    print(len(ckpts_path), ckpts_path)
    # checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    # torch.save(checkpoint['state_dict'], args.ckpt_path.split('/')[-2]+'.ckpt')
    # print('Done!')