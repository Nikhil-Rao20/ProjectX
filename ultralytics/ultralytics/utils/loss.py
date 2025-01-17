# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.model = model
        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # Masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto,
                                                       pred_masks, imgsz, self.overlap)

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor,
                         area: torch.Tensor) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i],
                                              marea_i[fg_mask_i])

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes,
                                 pred_kpts):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
                                        device=keypoints.device)

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items

from collections import defaultdict

# class MultiTaskLoss(v8DetectionLoss):
#     scale_mode = 'min'
#     compute_stats  = False
#     class ProcrustesSolver:
#         @staticmethod
#         def apply(grads, scale_mode='min'):
#             assert (
#                 len(grads.shape) == 3
#             ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

#             with torch.no_grad():
#                 print(grads.shape, grads)
#                 cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
#                 cov_grad_matrix_e = cov_grad_matrix_e.mean(0)
#                 cov_grad_matrix_e = cov_grad_matrix_e.to(torch.float32)
#                 print("Shape of input :",cov_grad_matrix_e.shape)
#                 singulars, basis = torch.linalg.eigh(cov_grad_matrix_e, UPLO='U') #torch.symeig(cov_grad_matrix_e, eigenvectors=True)
#                 singulars = singulars.to(torch.float16)
#                 basis = basis.to(torch.float16)
#                 tol = (
#                     torch.max(singulars)
#                     * max(cov_grad_matrix_e.shape[-2:])
#                     * torch.finfo().eps
#                 )
#                 rank = sum(singulars > tol)
#                 print("TOL and Singular : ",tol, singulars)

#                 order = torch.argsort(singulars, dim=-1, descending=True)
#                 print("Shapes of outputs 1:",singulars.shape, basis.shape)
#                 print("rank, order: ", rank, order)
#                 singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]
#                 print("Shapes of outputs2 :",singulars.shape, basis.shape)

#                 if scale_mode == 'min':
#                     weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
#                 elif scale_mode == 'median':
#                     weights = basis * torch.sqrt(torch.median(singulars)).view(1, -1)
#                 elif scale_mode == 'rmse':
#                     weights = basis * torch.sqrt(singulars.mean())

#                 weights = weights / torch.sqrt(singulars).view(1, -1)
#                 weights = torch.matmul(weights, basis.T)
#                 grads = torch.matmul(grads, weights.unsqueeze(0))

#                 return grads, weights, singulars

#     def __init__(self, model):  # model must be de-paralleled
#         super().__init__(model)
#         self.pose_loss = v8PoseLoss(model)
#         self.seg_loss = v8SegmentationLoss(model)
#         self.loss_map = ['box', 'pose', 'kobj', 'seg', 'cls', 'dfl']
#         self.log_vars = torch.nn.Parameter(torch.zeros(len(self.loss_map), requires_grad=True))
#         self.losses = defaultdict(float)
#     def compute_alignment(self, grads):
#         """
#         Align task gradients using Procrustes analysis.
#         """
#         grads, weights, _ = MultiTaskLoss.ProcrustesSolver.apply(grads.T.unsqueeze(0), self.scale_mode)
#         return grads[0].sum(-1), weights.sum(-1)
    
#     def get_task_gradients(self, losses, shared_params):
#         """
#         Compute task gradients with respect to shared parameters.
#         """
#         grads = []
#         for task_loss in losses:
#             # print(f"task_loss type: {type(task_loss)}, shape: {task_loss.shape}, value: {task_loss}")
#             if not isinstance(task_loss, torch.Tensor):
#                 raise ValueError(f"Each task_loss must be a tensor. Got {type(task_loss)}")

#             task_grads = torch.autograd.grad(
#                 task_loss, shared_params, retain_graph=True, allow_unused=True
#             )

#             # Handle potential None values in gradients
#             grad_vector = torch.cat([
#                 g.flatten() if g is not None else torch.zeros_like(p).flatten()
#                 for g, p in zip(task_grads, shared_params)
#             ])

#             grads.append(grad_vector)

#         return torch.stack(grads, dim=0)

    
#     def set_shared_grad(self, shared_params, aligned_grad):
#         """
#         Update the gradients for shared parameters.
#         """
#         offset = 0
#         for param in shared_params:
#             if param.grad is not None:
#                 numel = param.numel()
#                 param.grad.data = aligned_grad[offset:offset + numel].view_as(param.grad)
#                 offset += numel
        
#     def update_loss_weights(self, val_losses):
#         """
#         Update dynamic loss weights based on the validation losses for each task.
#         """
#         total_inverse_loss = sum(1/(loss1+1e-6) for loss1 in val_losses)
#         dynamic_loss_weights = {task: (1 / (loss1 + 1e-6)) / total_inverse_loss for task, loss1 in zip(self.loss_map, val_losses)}
#         return dynamic_loss_weights

#     def __call__(self, preds, batch):
#         """Calculate the total loss and detach it."""
#         # box_loss, pose_loss, kobj_loss, seg_loss, cls_loss, dfl_loss
#         loss = torch.zeros(6, device=self.device)
#         feats, pred_kpts, pred_masks, proto = preds if len(preds) == 4 else preds[1]
#         batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
#         pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
#             (self.reg_max * 4, self.nc), 1)

#         # b, grids, ..
#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_distri = pred_distri.permute(0, 2, 1).contiguous()
#         pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
#         pred_masks = pred_masks.permute(0, 2, 1).contiguous()

#         dtype = pred_scores.dtype
#         imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
#         anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

#         # targets
#         batch_idx = batch['batch_idx'].view(-1, 1)
#         targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
#         targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#         gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
#         mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

#         # pboxes
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
#         pred_kpts = self.pose_loss.kpts_decode(anchor_points,
#                                                pred_kpts.view(batch_size, -1,
#                                                               *self.pose_loss.kpt_shape))  # (b, h*w, 17, 3)

#         _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
#             pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
#             anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

#         target_scores_sum = max(target_scores.sum(), 1)

#         # cls loss
#         loss[4] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

#         if fg_mask.any():
#             target_strided_bboxes = target_bboxes / stride_tensor

#             # bbox regression loss
#             loss[0], loss[5] = self.bbox_loss(
#                 pred_distri,
#                 pred_bboxes,
#                 anchor_points,
#                 target_strided_bboxes,
#                 target_scores,
#                 target_scores_sum,
#                 fg_mask,
#             )

#             # keypoints loss
#             keypoints = batch['keypoints'].to(self.device).float().clone()
#             keypoints[..., 0] *= imgsz[1]
#             keypoints[..., 1] *= imgsz[0]
#             loss[1], loss[2] = self.pose_loss.calculate_keypoints_loss(
#                 fg_mask,
#                 target_gt_idx,
#                 keypoints,
#                 batch_idx,
#                 stride_tensor,
#                 target_strided_bboxes,
#                 pred_kpts,
#             )

#             # segmentation loss
#             masks = batch['masks'].to(self.device).float()
#             if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
#                 masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

#             loss[3] = self.seg_loss.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx,
#                                                                 proto, pred_masks, imgsz, self.seg_loss.overlap)

#         # loss[0] *= self.hyp.box  # box gain
#         # loss[1] *= self.hyp.pose  # pose gain
#         # loss[2] *= self.hyp.kobj  # kobj gain
#         # loss[3] *= self.hyp.box  # seg gain
#         # loss[4] *= self.hyp.cls  # cls gain
#         # loss[5] *= self.hyp.dfl  # dfl gain
#         # print('Params shape :',len(list(self.params)))
#         grads = self.get_task_gradients(loss, torch.cat([p.view(-1) for p in self.params]))
#         aligned_grad, weights = self.compute_alignment(grads)
#         for i in range(len(loss)):
#             loss[i] *= weights[i]
#         self.set_shared_grad(list(self.hyp), aligned_grad)


#         # dynamic_loss_weights = self.update_loss_weights(loss)    
#         # loss[0] *= dynamic_loss_weights['box']  # box gain
#         # loss[1] *= dynamic_loss_weights['pose'] # pose gain
#         # loss[2] *= dynamic_loss_weights['kobj'] # kobj gain
#         # loss[3] *= dynamic_loss_weights['seg'] # seg gain
#         # loss[4] *= dynamic_loss_weights['cls']# cls gain
#         # loss[5] *= dynamic_loss_weights['dfl']  # dfl gain
        
#         return loss.sum() * batch_size, loss.detach()

import torch
import torch.nn.functional as F
from collections import defaultdict


class MultiTaskLoss(v8DetectionLoss):
    scale_mode = 'min'

    class ProcrustesSolver:
        @staticmethod
        def apply(grads, scale_mode='min'):
            assert (
                len(grads.shape) == 3
            ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

            with torch.no_grad():
                cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
                cov_grad_matrix_e = cov_grad_matrix_e.mean(0)
                cov_grad_matrix_e = cov_grad_matrix_e.to(torch.float32)
                if torch.any(torch.isnan(cov_grad_matrix_e)) :
                    raise ValueError("Matrix contains NaN  values")
                elif torch.any(torch.isinf(cov_grad_matrix_e)):
                    print("Matrix contains  Inf values")
                    cov_grad_matrix_e = torch.where(torch.isinf(cov_grad_matrix_e), torch.full_like(cov_grad_matrix_e, 1e6), cov_grad_matrix_e)
                if torch.norm(cov_grad_matrix_e) > 1e6:
                    print("Matrix is ill-conditioned with very large values")
                    cov_grad_matrix_e = cov_grad_matrix_e / torch.norm(cov_grad_matrix_e) * 1e6

                singulars, basis = torch.linalg.eigh(cov_grad_matrix_e, UPLO='U')
                singulars = singulars.to(torch.float32)
                basis = basis.to(torch.float32)
                tol = (
                    torch.max(singulars)
                    * max(cov_grad_matrix_e.shape[-2:])
                    * torch.finfo(singulars.dtype).eps
                )
                rank = sum(singulars > tol)
                
                if rank==0:
                     print("Warning: All singular values are zero. Skipping alignment.")
                     return grads, torch.eye(grads.shape[-1]), singulars, False

                order = torch.argsort(singulars, dim=-1, descending=True)
                singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]
                print('Singulars :',singulars.shape, basis.shape)
                if scale_mode == 'min':
                    weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
                elif scale_mode == 'median':
                    weights = basis * torch.sqrt(torch.median(singulars)).view(1, -1)
                elif scale_mode == 'rmse':
                    weights = basis * torch.sqrt(singulars.mean())

                weights = weights / torch.sqrt(singulars).view(1, -1)
                weights = torch.matmul(weights, basis.T)
                grads = torch.matmul(grads, weights.unsqueeze(0))

                return grads, weights, singulars, True

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.pose_loss = v8PoseLoss(model)
        self.seg_loss = v8SegmentationLoss(model)
        self.loss_map = ['box', 'pose', 'kobj', 'seg', 'cls', 'dfl']
        self.log_vars = torch.nn.Parameter(torch.zeros(len(self.loss_map), requires_grad=True))
        self.losses = defaultdict(float)
        self.useBalancer = True

    def compute_alignment(self, grads):
        grads, weights, _, useBalancer= MultiTaskLoss.ProcrustesSolver.apply(grads.T.unsqueeze(0), self.scale_mode)
        return grads[0].sum(-1), weights.sum(-1), useBalancer

    def get_task_gradients(self, losses, shared_params):
        grads = []
        
        for task_loss in losses:
            if not isinstance(task_loss, torch.Tensor):
                raise ValueError(f"Each task_loss must be a tensor. Got {type(task_loss)}")
            if task_loss.grad_fn is None:
                task_loss = task_loss.clone().detach().requires_grad_(True)
            task_grads = torch.autograd.grad(task_loss, shared_params, retain_graph=True, create_graph=True, allow_unused=True)

            grad_vector = torch.cat([
                g.flatten() if g is not None else torch.zeros_like(p).flatten()
                for g, p in zip(task_grads, shared_params)
            ])

            grads.append(grad_vector)

        return torch.stack(grads, dim=0)

    def set_shared_grad(self, shared_params, aligned_grad):
        offset = 0
        for param in shared_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data = aligned_grad[offset:offset + numel].view_as(param)
                offset += numel

    def update_loss_weights(self, val_losses):
        total_inverse_loss = sum(1 / (loss1 + 1e-6) for loss1 in val_losses)
        dynamic_loss_weights = {task: (1 / (loss1 + 1e-6)) / total_inverse_loss for task, loss1 in zip(self.loss_map, val_losses)}
        return dynamic_loss_weights

    def __call__(self, preds, batch):
        loss = torch.zeros(6, device=self.device)
        feats, pred_kpts, pred_masks, proto = preds if len(preds) == 4 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.pose_loss.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.pose_loss.kpt_shape))

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        loss[4] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.any():
            target_strided_bboxes = target_bboxes / stride_tensor
            loss[0], loss[5] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points,
                                              target_strided_bboxes, target_scores, target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[1], loss[2] = self.pose_loss.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                                       stride_tensor, target_strided_bboxes, pred_kpts)

            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]
            loss[3] = self.seg_loss.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx,
                                                                proto, pred_masks, imgsz, self.seg_loss.overlap)
        
        if self.useBalancer:
            shared_params = []
            for param in self.model.parameters():
                if not param.requires_grad:
                    param.requires_grad_(True)
                shared_params.append(param)
            grads = self.get_task_gradients(loss, shared_params)
            aligned_grad, weights,useBalancer = self.compute_alignment(grads)
            if useBalancer:
                for i in range(len(loss)):
                    loss[i] *= weights[i]
                self.set_shared_grad(shared_params, aligned_grad)
        else:
            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.pose # pose gain
            loss[2] *= self.hyp.kob # kobj gain
            loss[3] *= self.hyp.seg # seg gain
            loss[4] *= self.hyp.cls# cls gain
            loss[5] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()
