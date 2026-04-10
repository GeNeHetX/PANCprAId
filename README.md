# PANCprAId

This repository contains the code for extract UNI features [1] and then deploy PANCpAId algorithm[2].
 
# Installation

To utilise the package, you need to download the code from this repository. You can do this by using the following command:

```bash
git clone https://github.com/GeNeHetX/PANCprAId.git
```

Then, you need to install the required packages. 
First, having OpenSlide is mandatory. Considerer visiting [their website](https://openslide.org/download/). In short for Linux users, choose the command corresponding to your distro. For windows users, download the Windows 64-bit Binaries and follow [these insctructions](https://openslide.org/api/python/). You will have to change the value of `OPENSLIDE_PATH` in python files. I suggest using Linux or WSL.
Then to install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
```
### Getting model access
Request access to the model weights from the Huggingface model page at: https://huggingface.co/mahmoodlab/UNI.

## Usage
### Extract neoplasic features
To use the model, you can use the following code:

```bash
usage: python process_wsi.py [-h] --temp_dir TEMP_DIR [--wsi WSI] [--model_path MODEL] --device {cuda:0,cpu,mps}]
                      [--batch_size BATCH_SIZE] [--num_workers NUM_WORKER] [--amp AMP```

Where:
- `--temp_dir` is the directory where the temporary files will be stored.
- `--wsi` is the path to the directory where the WSI are . It accepts ".svs", ".ndpi" and ".qptiff" files. More formats can be added in the `extract_features.py` and `extract_tiles.py` files.
- `--model_path` is the path to the pretrained uni extractor features model `pytorch_model.bin`.
- `--device` is the device to use for the prediction. It can be "cuda", "mps" if you work on macOS or "cpu" if you don't have a GPU. default is "cuda".
- `--batch_size` is the batch size to use for the prediction.
- `--num_workers` is the number of workers to use for the prediction. If on Windows, it should be set to 0.
- `--amp` is the use of mixed precision.



For example, you can use the following command to predict a WSI:

```bash
python src/UNI_extractor/process_wsi.py --temp_dir /home/results/ --wsi /home/data/01_HESCohort --model_path /home/model/pytorch_model_uni1.bin --device cuda:0 --batch_size 128 --num_workers 8
```


### Deploy pancpraid algorithm 
To use the PANCprAId pipeline, you can run the following command:

```bash
usage: python deployPANCprAId.py [-h] --model_pacpaint_path MODEL_PACPAINT_PATH 
                                --model_pancpraid_gem MODEL_PANCPRAID_GEM 
                                --model_pancpraid_ffx MODEL_PANCPRAID_FFX 
                                --h5_path H5_PATH 
                                --output_path OUTPUT_PATH 
                                [--device {cuda,cpu,mps}] 
                                [--input_dim INPUT_DIM] 
                                --csv_path CSV_PATH 
                                [--tumor_th TUMOR_TH] 
                                [--tumor_cells_th TUMOR_CELLS_TH] 
                                [--save_pacpaint SAVE_PACPAINT]
                                ```
    
 Where:
--model_pacpaint_path is the path to the pretrained PACpAInt model used for tumor detection. You could find it into models/PACpAInt_uni.pth.
--model_pancpraid_gem is the path to the pretrained PANCprAId GEM model. You could find it into models/PANCprAId_hGEMmodel.pth.
--model_pancpraid_ffx is the path to the pretrained PANCprAId FFX model. You could find it into models/PANCprAId_hGEMmodel.pth.
--h5_path is the directory containing the input .h5 feature files.
--output_path is the directory where predictions and outputs will be saved.
--device is the device used for inference. It can be "cuda", "mps" (for macOS), or "cpu". Default is "cuda".
--input_dim is the dimensionality of the input features (e.g., 1024 for UNI features).
--csv_path is the path to the CSV file listing the samples to process and the name of patient associated to it, it must contains at least columns filename and patient. Youcould find an exemple into pancpraid_mode/exemple_files.csv
--tumor_th is the threshold used for tumor region detection. Default is 0.35
--tumor_cells_th is the threshold used to classify tumor cells. Default is 0.5
--save_pacpaint enables saving PACpAInt predictions (True/False).                           
 
 For example, you can use the following command to predict a WSI:
 ```bash
python deployPANCprAId.py --model_pacpaint_path ../models/PACpAInt_uni.pth --model_pancpraid_gem ../models/PANCprAId_hGEMmodel.pth --model_pancpraid_ffx ../models/PANCprAId_hFFXmodel.pth --h5_path /home/results/ --output_path /home/results/pancpraid/ --device cuda --input_dim 1024 --csv_path /home/csv_files.csv --tumor_th 0.35 --tumor_cells_th 0.5 --save_pacpaint True
```


## References
[1] Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Song, Andrew H and Shaban, Muhammad and others « Towards a General-Purpose Foundation Model for Computational Pathology ». Nature Medicine (2024) https://doi.org/10.1038/s41591-024-02857-3.
[2] coming soon 

