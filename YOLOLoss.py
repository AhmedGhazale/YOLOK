import torch
import torch.nn.functional as F
import numpy as np

class YOLOLoss(torch.nn.Module):

    def __init__(self, s, b, l_coord = 5, l_noobj = .5):
        super(YOLOLoss, self).__init__()
        self.s = s
        self.b = b
        self.l_coord = l_coord
        self.l_noobj  = l_noobj
        self.offset = np.transpose(np.reshape(np.array([np.arange(s)] * s * b), (b, s, s)), (1, 2, 0))

    def compute_distance(self,boxes1, boxes2):

        offset_x = torch.tensor(self.offset).unsqueeze(0).repeat([boxes1.shape[0],1,1,1]).cuda()
        offset_y = offset_x.permute([0, 2, 1, 3])

        boxes1 = torch.stack([(boxes1[...,0]+offset_x)/self.s,
                              (boxes1[...,1]+offset_y)/self.s],dim=-1)

        boxes2 = torch.stack([(boxes2[..., 0] + offset_x) / self.s,
                              (boxes2[..., 1] + offset_y) / self.s], dim=-1)

        dist = torch.sqrt(torch.pow(boxes1[..., 0]-boxes2[..., 0], 2) + torch.pow(boxes1[..., 1]-boxes2[..., 1], 2))

        return dist

    def forward(self, pred, true):

        batch_size = pred.shape[0]
        pred_keypoints = pred[...,:3*self.b].view(-1,self.s,self.s,self.b,3).contiguous()
        pred_coord = pred_keypoints[...,0:2]
        pred_conf = pred_keypoints[...,2]
        pred_classes = pred[...,3*self.b:]

        true_keypoints = true[..., :3 * self.b].view(-1, self.s, self.s, self.b, 3).contiguous()
        true_coord = true_keypoints[..., 0:2]
        true_conf = true_keypoints[..., 2]
        true_classes = true[..., 3 * self.b:]

        # class loss
        obj_mask = true_conf[...,0]
        no_obj_mask = torch.ones_like(obj_mask) - obj_mask
        class_loss = torch.sum(obj_mask * torch.pow(pred_classes- true_classes,2).sum(-1))

        # coord_loss
        with torch.no_grad():
            dist = self.compute_distance( true_coord,pred_coord)
            ious = 1-dist

        val, idx = ious.max(-1,keepdim=True)

        iou_mask = torch.zeros_like(ious).scatter_(-1,idx,1)

        coord_obj_mask = iou_mask * true_conf
        coord_noobj_mask = (torch.ones_like(iou_mask) - iou_mask) * true_conf

        coord_loss =torch.sum(coord_obj_mask * (F.mse_loss(pred_coord[...,0:2], true_coord[...,0:2], reduction='none')).sum(-1))

        # conf loss
        no_obj_conf_loss =  torch.sum(no_obj_mask * (torch.pow(pred_conf,2).sum(-1)))
        obj_conf_loss =  torch.sum(coord_obj_mask * F.mse_loss(pred_conf, ious,reduction='none'))
        no_obj_conf_loss_2 = torch.sum(coord_noobj_mask * torch.pow(pred_conf,2))


        return (self.l_coord * coord_loss + 2 * obj_conf_loss + no_obj_conf_loss_2 + self.l_noobj * no_obj_conf_loss + class_loss )/ batch_size



    






