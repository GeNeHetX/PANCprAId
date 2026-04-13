import os
import numpy as np
import h5py

OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from display_results import display_wsi_results
# from extract_tiles import filter_tiles
from extract_tiles import filter_whites
from extract_features import extract_features


from pathlib import Path
from torch import device

from argparse import ArgumentParser

from torch import device


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--temp_dir",
        type=Path,
        default=Path(r"/mnt/d/UNI_multicentric_features2/"),
        help="Path to the temporary directory where the features will be saved",
        required=True,
    )
    parser.add_argument(
        "--wsi",
        type=Path,
        default=Path(r"//home/inserm/wbobhysto/datasets/PDAC_PancMulticentric/01_HESCohort_PitieSaintAntoine/"),
        help="Path to the WSI. Can be a .svs, .ndpi, .qptiff",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(r"/mnt/c/Users/inserm/Documents/pytorch_model.bin"),
        help="Path to the UNI model",
    )
    parser.add_argument(
        "--device", type=device, default="cuda:0", help="Device to use for the predictions"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the feature extraction")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the feature extraction. Set to 0 if using windows.",
    )
    parser.add_argument(
        "--amp",
        type=bool,
        default=True,
        help="Extract features with mixed precision",
    )
    
    parsed_args = parser.parse_args()
    return parser.parse_args()


def main(args):

    import glob
    from pathlib import Path
    
    files = glob.glob(f'{args.wsi}/*.svs')
    nb_files = len(files)
    print(str(nb_files) + " SVS were found")
    export_path = Path(args.temp_dir)
    

    export_path.mkdir(parents=True, exist_ok=True)
    i = 1
    for file in files :
        slidename = Path(os.path.basename(file)).stem
        print(str(i) + "/" + str(nb_files) + " : " +str(slidename))
        i = i + 1
        no = ["B00155099_AMBP01_HES"]
        if not os.path.exists(f'{args.temp_dir}/{slidename}.h5') and slidename not in no  :
            
            print("Filtering tiles...")
            print(file)
            try : 
                tiles_coord = filter_whites(file)
            except : 
                print("Couln't open slide")
            else : 

                print("Extracting all tiles features...")
                try : 
                    features = extract_features(
                        file,
                        args.device,
                        args.batch_size,
                        outdir = args.temp_dir,
                        tiles_coords=tiles_coord,
                        num_workers=args.num_workers,
                        checkpoint_path = args.model_path,
                        mixed_precision = args.amp
                    )
                except : 
                    print("couldn't extract features")
                else : 
                    with h5py.File(f'{args.temp_dir}/{slidename}.h5', 'w') as f:
                        f['coords'] = tiles_coord
                        f['feats'] = features

                        f['extractor'] = 'UNI'



            print("Done")



if __name__ == "__main__":
    args = parse_arg()
    main(args)
