import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from utils import *
import numba
import cv2
import configs as cfg
import sys
from time import time
from predict import *
from tqdm import tqdm

def main():

    model = torch.load(cfg.MODEL_PATH)
    video_path =  sys.argv[1]
    video  = cv2.VideoCapture(video_path)

    ret, frame = video.read()
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))
    frames_count = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frames_count)
    #while ret:
    for i in tqdm(range(500)):
        s = time()
        result = predict(frame ,model)
        e = time()

        print(1/(e-s))
        result_dict = {}
        for center, class_name, prob in result:
            cv2.circle(frame, center, 2, (255, 0, 0), -1, )
            label = class_name + str(round(prob, 2))
            if class_name not in result_dict:
                result_dict[class_name]= (prob,center)
            elif result_dict[class_name][0]<prob:
                result_dict[class_name] = (prob, center)
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (center[0], center[1] - text_size[1])
            # cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), (124,32,225),-1)

            cv2.putText(frame, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, 8)

        if 'LT' in result_dict and 'LB' in result_dict:
            cv2.line(frame,result_dict['LT'][1],result_dict['LB'][1],(0,0,255),3)
        if 'LT' in result_dict and 'RT' in result_dict:
            cv2.line(frame, result_dict['LT'][1], result_dict['RT'][1], (0, 255, 0), 3)
        if 'RT' in result_dict and 'RB' in result_dict:
            cv2.line(frame, result_dict['RT'][1], result_dict['RB'][1], (255, 0, 0), 3)

        out.write(frame)
        ret, frame = video.read()
    out.release()


if __name__ == '__main__':

    main()
