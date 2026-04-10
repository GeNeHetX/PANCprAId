from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
import glob 
import os 
import h5py
import numpy as np 
import pandas as pd 
from typing import Optional
import torch

from marugoto_pancpraid import MILModel,loadPANCprAId,deployPANCprAId
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
        default=Path(r"../models/PACpAInt_uni.pth"),
        help="Path to the model pth",
        required=False,
    )
    parser.add_argument(
        "--model_pancpraid_gem",
        type=Path,
        default=Path(r"../models/PANCprAId_hGEMmodel.pth"),
        help="Path to the pancpraid gem model pth",
        required=False,
    )
    parser.add_argument(
        "--model_pancpraid_ffx",
        type=Path,
        default=Path(r"../models/PANCprAId_hFFXmodel.pth"),
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
        help="Torch device cuda, mps, cpu ...",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1024,
        help="Number of input features if UNIv1 1024",
    )
    parser.add_argument(
        "--tumor_th",
        type=float,
        default=0.35,
        help="Threshold of neoplasic zone of pacpaint",
    )
    parser.add_argument(
        "--tumor_cells_th",
        type=float,
        default=0.5,
        help="Threshold of tumoral cells of pacpaint",
    )
    parser.add_argument(
        "--save_pacpaint",
        type=bool,
        default=False,
        help="Boolean to save pacpaint results",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default="/mnt/d/pacpaint_uni/csv_files.csv",
        help="Path to csv file with filename and patient column",
    )


    return parser.parse_args()
    


    
def read_h5(path):

    with h5py.File(h5_path, "r") as f:
        features = f["feats"][:]
        coords = f["coords"][:]
    return features, coords




if __name__ == "__main__":
    args = parse_args()

    device = args.device
    input_dim = 1024  
    os.makedirs(args.output_path, exist_ok=True)
    
    df = pd.read_csv(args.csv_path)  # filename, patient
    pacpaint_model = loadPACpAInt(args.model_pacpaint_path,input_dim,device) 
    hFFXmodel,hGEMmodel = loadPANCprAId(gem_pth=args.model_pancpraid_gem,ffx_pth=args.model_pancpraid_ffx,input_dim=input_dim,device=device)
    
    # Group by patients
    patient_files = defaultdict(list)
    for _, row in df.iterrows():
        full_path = os.path.join(args.h5_path, row["filename"])
        patient_files[row["patient"]].append(full_path)
    

 
    pbar = tqdm(patient_files.items(), desc="Deploying PANCprAId", leave=False)
    pancpraid_scores = []

    for patient, files in pbar:

        all_features = []
        all_coords = []

        for h5_path in files:
            features, coords = read_h5(h5_path)
            all_features.append(features)
            all_coords.append(coords)
        features = np.concatenate(all_features, axis=0)
        coords = np.concatenate(all_coords, axis=0)

        slide_name = patient
        features, coords =read_h5(h5_path)
        prediction_tumor,feats_tumor,coord_tumor = getTumoralTiles(pacpaint_model,features,coords,args.tumor_th, args.tumor_cells_th,device)
        ffx_sc,gem_sc = deployPANCprAId(hFFXmodel,hGEMmodel,feats_tumor,device)
        pancpraid_scores.append([slide_name,ffx_sc.cpu().item(),gem_sc.cpu().item()])
        #pbar.update(1)
        if args.save_pacpaint : 
            prediction_tumor.to_csv(f"{args.output_path}/{slide_name}_pacpaint_results.csv")
    
    pancpraid_df = pd.DataFrame(pancpraid_scores,columns=["PATIENT","hFFXmodel","hGEMmodel"])
    pancpraid_df.to_csv(f"{args.output_path}/pancpraid_results.csv")
    print(f"All results are saved in {args.output_path}")


