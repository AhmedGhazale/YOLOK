import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from utils import *
import numba
import cv2
import configs as cfg
import sys
import configs as cfg

def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=min_visibility, label_fields=['category_id']))


def predict(img_org,model, det_threshold= .5):
    model.eval()
    result= []
    w = img_org.shape[1]
    h = img_org.shape[0]

    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    aug_test = get_aug([
        A.Resize(448, 448),
        A.Normalize(),
        ToTensorV2()
    ])
    sample = {'image': img, 'bboxes': [], 'category_id': []}
    sample = aug_test(**sample)
    img = sample['image']
    img = img.unsqueeze(0)
    img = img.cuda()
    with torch.no_grad():

        out = model(img)
        out = out.cpu()[0]

    boxes, cls, score = post_processing_torch(out,det_threshold=.5,s = cfg.GRID_SIZE, b = cfg.BOXES_PER_CELL)

    for i, box in enumerate(boxes):
        x = int(box[0] * w)
        y = int(box[1] * h)

        cls_index = cls[i]
        cls_index = int(cls_index)
        prob = score[i]
        prob = float(prob)
        result.append([(x, y), cfg.CLASSES[cls_index], prob])
    return result

def main():
  
  model = torch.load(cfg.MODEL_PATH)
  image_name =  sys.argv[1]
  #image_name = 'test_images/goal.jpg'

  image = cv2.imread(image_name)
  result = predict(image ,model)

  for center , class_name, prob in result:
      cv2.circle(image, center, 2, (255,0,0), -1, )
      label = class_name + str(round(prob, 2))
      text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
      p1 = (center[0], center[1] - text_size[1])
      #cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), (124,32,225),-1)

      cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, 8)

  cv2.imwrite('result.jpg', image)

if __name__ == '__main__':
	main()
    
    
