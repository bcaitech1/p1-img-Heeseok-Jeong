# pstage_02_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`
e.g. `python train.py --epochs 4 --model EfficientNet_B4 --name My_Eff_1`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`
e.g. `python inference.py --model EfficientNet_B4 --name My_Eff_1`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`

### SGD search
By using `SGD_search.py`, you can more train trained model with SGD optimizer.  
This may help models get to reach global minimum loss.
