import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import cv2
from predict import predict
import configs as cfg
from tqdm import tqdm
import json
def eval_coco(json_path, images_path, model):

    coco_gt = COCO(json_path)
    cat_ids = coco_gt.getCatIds(catNms='goal')
    images_ids = coco_gt.getImgIds(catIds=cat_ids)
    images = coco_gt.loadImgs(images_ids)
    found_images = [x['id'] for x in images if os.path.exists(images_path + x['file_name'])]
    images = [x for x in images if x['id'] in found_images]
    result = []

    for image in tqdm(images):
        image_path = os.path.join(images_path,image['file_name'])
        img = cv2.imread(image_path)
        res = predict(img,model,det_threshold=0)
        result_dict = {}
        for center, class_name, prob in res:
            if class_name not in result_dict:
                result_dict[class_name] = (prob, center)
            elif result_dict[class_name][0] < prob:
                result_dict[class_name] = (prob, center)

        keypoints = []
        for label in cfg.CLASSES:
            if label in result_dict:
                keypoints.extend([result_dict[label][1][0], result_dict[label][1][1], 1] )
            else:
                keypoints.extend([0,0,0])

        result.append({'image_id':image['id'],
                       'category_id': cat_ids[0],
                       'keypoints':keypoints,
                       'score':1})
    with open('results.json','w') as file:
        json.dump(result,file)


    cocoGt = COCO('/home/ahmed/PycharmProjects/coco_eval/goals_pose_749_fixed.json')

    cocoDt = cocoGt.loadRes('results.json')
    # cocoDt=COCO('goals_pose_518.json')

    print(len(cocoDt.getImgIds()))
    print(len(cocoGt.getImgIds()))

    imgIds = sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt, cocoDt, iouType='keypoints')
    cocoEval.params.imgIds = found_images
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

model = torch.load(cfg.MODEL_PATH)
eval_coco(cfg.DATASET_PATH+'goals_pose_749.json',cfg.DATASET_PATH+"test/",model)