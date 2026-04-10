import torch
from torch import nn
import numpy as np 
import pandas as pd 
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
        


            nn.Linear(256, 2)  # 2 regression outputs
        )

    def forward(self, x):
        return self.net(x)


def loadPACpAInt(model_path,input_dim,device) : 
    model = MLP(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def deployPACpAInt(model,features,coords,device) : 
    preds = []

    with torch.no_grad():
        for i in range(0, len(features), 512):
            x = torch.from_numpy(features[i:i+512]).float().to(device)
            pred = model(x).cpu().numpy()
            preds.append(pred)
    prediction_np = np.vstack(preds)
    coord_pred = np.concatenate([coords,prediction_np],axis=1)

    df = pd.DataFrame(coord_pred, columns=["z","index","x","y","pred_tumor", "pred_tumor_cells"])


    return df

def getTumoralTiles(model,features,coords,tumor_th,tumor_cells_th,device) : 
    prediction_tumor = deployPACpAInt(model,features,coords,device)
    tumor_index = prediction_tumor[(prediction_tumor["pred_tumor"] > tumor_th) & (prediction_tumor["pred_tumor_cells"] > tumor_cells_th)].index

    feats_tumor = features[tumor_index]
    coord_tumor = coords[tumor_index]



    return prediction_tumor,feats_tumor,coord_tumor
