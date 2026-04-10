from argparse import ArgumentParser
from pathlib import Path

import glob 
import os 
import h5py
import numpy as np 
import pandas as pd 
from typing import Optional
import torch

from marugoto_pancpraid import MILModel,loadPANCprAId
from pacpaint_uni import MLP, loadPACpAInt,deployPACpAInt,getTumoralTiles

from tqdm import tqdm
import warnings
warnings.filterwarnings(
    action='ignore', category=UserWarning, message=r"Boolean Series.*"
)
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_pacpaint_path",
        type=Path,
        default=Path(r"/mnt/d/pacpaint_uni/model.pth"),
        help="Path to the model pth",
        required=False,
    )
    parser.add_argument(
        "--model_pancpraid_gem",
        type=Path,
        default=Path(r"/mnt/c/Users/inserm/Documents/models/lr_1e-05_l1_0.0001_l2_0.001_best_model_gem.pth"),
        help="Path to the pancpraid gem model pth",
        required=False,
    )
    parser.add_argument(
        "--model_pancpraid_ffx",
        type=Path,
        default=Path(r"/mnt/c/Users/inserm/Documents/models/lr_1e-05_l1_0.0001_l2_0.001_best_model_ffx.pth"),
        help="Path to the pancpraid ffx model pth",
        required=False,
    )
    parser.add_argument(
        "--h5_path",
        type=Path,
        default=Path(r'/mnt/d/Tools/UNI_extractor_All_Tiles/data/'),
        help="Path of h5 directory",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(r'/mnt/d/Tools/UNI_extractor_All_Tiles/data/out/'),
        help="Path of output directiry",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default="cuda",
        help="Path of output directiry",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1024,
        help="Number of input features",
    )
    parser.add_argument(
        "--tumor_th",
        type=float,
        default=0.35,
        help="Number of input features",
    )
    parser.add_argument(
        "--tumor_cells_th",
        type=float,
        default=0.5,
        help="Number of input features",
    )
    parser.add_argument(
        "--save_pacpaint",
        type=bool,
        default=False,
        help="Number of input features",
    )


    return parser.parse_args()
    


    
def read_h5(path):
    # load features
    with h5py.File(h5_path, "r") as f:
        features = f["feats"][:]
        coords = f["coords"][:]
    return features, coords




if __name__ == "__main__":
    args = parse_args()

    device = args.device
    input_dim = 1024  
    os.makedirs(args.output_path, exist_ok=True)
    

    pacpaint_model = loadPACpAInt(args.model_pacpaint_path,input_dim,device) 
    hFFXmodel,hGEMmodel = loadPANCprAId(gem_pth=args.model_pancpraid_gem,ffx_pth=args.model_pancpraid_gem,input_dim=input_dim,device=device)
    

    

    h5_paths = glob.glob(f"{args.h5_path}/*.h5")
    pbar = tqdm(h5_paths, desc="Deploying PANCprAId", leave=False)
    pancpraid_scores = []

    for h5_path in h5_paths:
        slide_name = os.path.basename(h5_path).replace(".h5", "")
        features, coords =read_h5(h5_path)
        prediction_tumor,feats_tumor,coord_tumor = getTumoralTiles(features,coords,args.tumor_th, args.tumor_cells_th)
        ffx_sc,gem_sc = deployPANCprAId(hFFXmodel,hGEMmodel,feats_tumor)
        pancpraid_scores.append([slide_name,ffx_sc.cpu().item(),gem_sc.cpu().item()])
        pbar.update(1)
        if args.save_pacpaint : 
            prediction_tumor.to_csv(f"{args.output_path}/{slide_name}_pacpaint_results.csv")
        
    pancpraid_df = pd.DataFrame(pancpraid_scores,columns=["PATIENT","hFFXmodel","hGEMmodel"])
    scores_df.to_csv(f"{args.output_path}/pancpraid_results.csv")
    print(f"All results are saved in {args.output_path}")
