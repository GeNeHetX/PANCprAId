# PACPaint Neo


python PACPaint_Neo/src/pacpaint_neo/process_wsi.py --temp_dir temp/ --wsi data/slide_HES.svs --neo PANCprAId/PACPaint_Neo/src/models/model_neo.pth --device mps



pytorch_model.bin refers to UNI model, you could download it at : https://huggingface.co/MahmoodLab/UNI

python3 UNI_extractor_Neoplasic/src/UNI_extractor/process_wsi.py --temp_dir temp/ --wsi data/slide_HES.svs --model_path pytorch_model.bin  --pred_tiles temp/slide_HES/pred_neo.csv --device mps --batch_size 16

