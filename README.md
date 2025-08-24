# Thesis-RCNN-Size
- Uses Object Detection 
- the dataset directory is at the `TODO: ` which is located default dataset
```bash
cleaned_frames_by_20_og/cleaned_frames_by_20
```
## Requirements
- image dataset for training and testing it
- annotations in .xml format
## Training File
```bash
python mango-detect.py
```
#### Sample Output
```bash
C:\Users\Kenan\thesis-rcnn-size>python mango-detect.py
Labels found:
  0: bruised
  1: not_bruised
  2: yellow
  3: green_yellow
  4: green
  5: mango

Total images: 488
Valid images found: 488
Using Faster R-CNN MobileNetV3-Large-FPN 

Starting training with 488 images...
```
## Testing Files
- there are two options to test the `.pth` model file
### 1. Manually Check each Image
```bash
python test_mango detection.py
```
### 2. Comprehensive Output
```bash
python batch-test-mango.py
```