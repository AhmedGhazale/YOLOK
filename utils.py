import torch
import numpy as np
import numba
import configs as cfg
#@numba.njit()
def post_processing(out, det_threshold = .1, nms_threshold = .8 , s= 14, b=2, classes_num=20):
    boxes = []
    conf = []
    classes_index = []
    for i in range(s):
        for j in range(s):
            for k in range(b):
                boxes.append([(out[i][j][k*5+0]+j)/s,(out[i][j][k*5+1]+i)/s,out[i][j][k*5+2], out[i][j][k*5+3]])
                conf.append(out[i][j][k*5+4]*np.max(out[i][j][b*5+1:]))
                classes_index.append(np.argmax(out[i][j][b*5:]))


    boxes = np.array(boxes)

    box_xy = np.zeros_like(boxes)

    box_xy[...,:2] = boxes[...,:2] - 0.5 * boxes[...,2:]
    box_xy[...,2:] = boxes[...,:2] + 0.5 * boxes[...,2:]

    boxes = box_xy
    conf = np.array(conf)
    classes_index = np.array(classes_index)

    boxes2,classes, conf2  = post_processing_torch(torch.tensor(out))

    import ipdb
    ipdb.set_trace()

    chosen = np.where(conf>det_threshold)

    boxes = boxes[chosen]
    classes_index = classes_index[chosen]
    conf = conf[chosen]
  

    import ipdb
    ipdb.set_trace()

    keep = nms(boxes,conf,nms_threshold)
    #return boxes,classes_index,conf
    return boxes[keep],classes_index[keep],conf[keep]
    
  
#@numba.njit()

def post_processing_torch(out, det_threshold = .1, s= 14, b=2):

    boxes = out[..., :3 * b].view( s, s, b, 3).contiguous()
    conf = boxes[..., 2]
    classes = out[..., 3 * b:]
    boxes = boxes[..., 0:2]

    offset = np.transpose(np.reshape(np.array([np.arange(s)] * s * b),(b, s, s)), (1, 2, 0))
    offset_x = torch.tensor(offset)
    offset_y = offset_x.permute([1, 0, 2])

    boxes = torch.stack([(boxes[..., 0] + offset_x)/s,
                         (boxes[..., 1] + offset_y)/s], dim=-1).view(-1,2)

    classes_val,classses_idx = classes.max(-1)

    conf = (conf * classes_val.unsqueeze(-1)).view(-1)

    classes = classses_idx.unsqueeze(-1).repeat([1,1,b]).view(-1)

    indices = conf>det_threshold
    boxes = boxes[indices]
    conf = conf[indices]
    classes = classes[indices]

    return boxes,classes,conf

    #return boxes,classes,conf


def nms(dets, scores, thresh=.5):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep




@numba.njit()
def post_processing_test(out, det_threshold = .2, nms_threshold = .5 , s= 14, b=2, classes_num=20):
    bboxes = []
    conf = []
    classes_index = []
    for i in range(s):
        for j in range(s):
            for k in range(b):
                bboxes.append([(out[i][j][k*5+0]+j)/s,(out[i][j][k*5+1]+i)/s,out[i][j][k*5+2], out[i][j][k*5+3]])
                conf.append(out[i][j][k*5+4]*np.max(out[i][j][b*5+1:]))
                classes_index.append(np.argmax(out[i][j][b*5:]))


    bboxes = np.array(bboxes)
    return bboxes



if __name__ == '__main__':
    print(numba.__version__)
    out = np.ones( [16,16,30] )    
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        o = post_processing(out )




