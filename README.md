# YOLOK_for_goals
---
A YOLO inspired key-points detection model to detect corners of goals.  

### Dependencies:
* torch==1.3.1
* torchvision==0.4.2
* opencv-python==4.1.2.30
* albumentations==0.4.3
* numba==0.46.0

### installation:
clone and install the requirements
```bash
git clone https://github.com/UnderExpectations/YOLOK_for_goals.git
cd YOLOK_for_goals
pip3 install -r requirements.txt
```
### Demo :
first download trained model from [here](https://drive.google.com/drive/folders/1wXXf4H7nXo_BPayyeIMPrWlUbYyIwk3J?usp=sharing) and use the config file with the model

**to run an image**
```bash
python3 predict.py path/to/image
```
**to run video**
```bash
python3 video_demo.py path/to/video
```
the output will be a video named output.avi in the same directory

### Traininig:
* download goals dataset from [here](https://drive.google.com/file/d/1FFWIDTW9MK9AhLKZH-vN0r0YR7UMIe9x/view?usp=sharing)
* edit the dataset path in **config.py** 
* run 
```bash
python3 train.py
```

