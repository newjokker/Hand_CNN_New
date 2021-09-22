"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union
#
import pdb


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")  # sum | mean

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        # [16,7,7,30]
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)


        # Calculate IoU for the two predicted bounding boxes with target bbox
        # [16,7,7,1]
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        # [2,16,7,7,1]
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)


        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        # [16,7,7,1], [16,7,7,1]
        # bestbox 矩阵为 1 代表最好的 box 在第二个预测框，否则在第一个预测框
        # iou_max 代表最好的预测结果的 p 值
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # [16,7,7,1]
        # exists_box 真值中心点所在的 cell
        exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        # fixme 找到两个预测框中预测结果比较好的预测框作为预测结果
        # [16,7,7,4] 预测出来的坐标
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )
        # [16,7,7,4] 真实框的坐标
        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        # fixme torch.sign 用于在执行 torch.sqrt 之后保留 box_predictions 的符号
        # fixme 将预测值的 2:4 进行开方运算，
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        #
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        #
        box_loss = self.mse(
            # [784,4] mse 比较容易处理 二维矩阵之间的距离 ？
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # pdb.set_trace()

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        # [16,7,7,1]
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            # exists_box * pred_box 将没有对应真值的预测 p 设置为 0
            # [784] 两个一维矩阵之间计算距离
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        #
        class_loss = self.mse(
            # exists_box * predictions[..., :20], [16,7,7,20]
            # flatten 之后 [784,20] 每一个 cell 对各个类型的预测值
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            #
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss                    # first two rows in paper
            + object_loss                                   # third row in paper
            + self.lambda_noobj * no_object_loss            # forth row
            + class_loss                                    # fifth row
        )

        # pdb.set_trace()

        return loss
