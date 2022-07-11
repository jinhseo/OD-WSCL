import torch
import torch.nn as nn
import numpy as np
import os
import time

from wetectron.config import cfg
from wetectron.structures.bounding_box import BoxList, BatchBoxList
from wetectron.structures.boxlist_ops import boxlist_iou, batch_boxlist_iou
from wetectron.modeling.box_coder import BoxCoder
from wetectron.utils.utils import clustering, to_boxlist, cal_iou, one_line_nms, cos_sim, get_share_class, easy_nms, easy_cluster
from wetectron.data.datasets.evaluation.voc.voc_eval import calc_detection_voc_prec_rec

class mist_layer(object):
    def __init__(self, p, iou=0.2):
        self.portion = p
        self.iou_th = iou
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)

    @torch.no_grad()
    def __call__(self, proposals, source_score, labels, device, target, iter_dict, output_dir, return_targets=False):
        num_rois = len(proposals)
        k = int(num_rois * self.portion)
        num_gt_cls = labels[1:].sum()
        if num_gt_cls != 0 and num_rois != 0:
            cls_prob = source_score[:, 1:]
            gt_cls_inds = labels[1:].nonzero(as_tuple=False)[:, 0]
            sorted_scores, max_inds = cls_prob[:, gt_cls_inds].sort(dim=0, descending=True)
            sorted_scores = sorted_scores[:k]
            max_inds = max_inds[:k]

            _boxes = proposals.bbox[max_inds.t().contiguous().view(-1)].view(num_gt_cls.int(), -1, 4)
            _boxes = BatchBoxList(_boxes, proposals.size, mode=proposals.mode)
            ious = batch_boxlist_iou(_boxes, _boxes)
            k_ind = torch.zeros(num_gt_cls.int(), k, dtype=torch.bool, device=device)
            k_ind[:, 0] = 1 # always take the one with max score
            for ii in range(1, k):
                max_iou, _ = torch.max(ious[:,ii:ii+1, :ii], dim=2)
                k_ind[:, ii] = (max_iou < self.iou_th).byte().squeeze(-1)

            gt_boxes = _boxes.bbox[k_ind]
            gt_cls_id = gt_cls_inds + 1
            temp_cls = torch.ones((_boxes.bbox.shape[:2]), device=device) * gt_cls_id.view(-1, 1).float()
            gt_classes = temp_cls[k_ind].view(-1, 1).long()
            gt_scores = sorted_scores.t().contiguous()[k_ind].view(-1, 1)

            ### cal precision/recall ###
            mist_pgt = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
            mist_pgt.add_field('labels', gt_classes.view(-1).detach().cpu())
            mist_pgt.add_field('scores', gt_scores.view(-1).detach().cpu())
            for iou_threshold in np.arange(0.25, 1.0, 0.25):
                prec_mist, rec_mist = calc_detection_voc_prec_rec([target], [mist_pgt], iou_threshold)
                for pos_c in gt_cls_inds:
                    iter_dict['prec_ours_%.2f'%iou_threshold][pos_c] += prec_mist[pos_c+1][-1]
                    iter_dict['rec_ours_%.2f'%iou_threshold][pos_c] += rec_mist[pos_c+1][-1]
            iter_dict['no_of_index'][pos_c] += 1

            if iter_dict['iter'] % 100 == 0:
                f = open(os.path.join(output_dir, "stats.txt"), 'a')
                f.write(str(iter_dict) + "\n")
            ### cal precision/recall ###
            if gt_boxes.shape[0] != 0:
                gt_boxes = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
                overlaps = boxlist_iou(proposals, gt_boxes)

                # TODO: pytorch and numpy argmax perform differently
                # max_overlaps, gt_assignment = overlaps.max(dim=1)
                max_overlaps  = torch.tensor(overlaps.cpu().numpy().max(axis=1), device=device)
                gt_assignment = torch.tensor(overlaps.cpu().numpy().argmax(axis=1), device=device)

                pseudo_labels = gt_classes[gt_assignment, 0]
                loss_weights = gt_scores[gt_assignment, 0]

                # fg_inds = max_overlaps.ge(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0]
                # Select background RoIs as those with <= FG_IOU_THRESHOLD
                bg_inds = max_overlaps.lt(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0]
                pseudo_labels[bg_inds] = 0

                # compute regression targets
                if return_targets:
                    matched_targets = gt_boxes[gt_assignment]
                    regression_targets = self.box_coder.encode(
                        matched_targets.bbox, proposals.bbox
                    )
                    return pseudo_labels, loss_weights, regression_targets

                return pseudo_labels, loss_weights

        # corner case
        pseudo_labels = torch.zeros(num_rois, dtype=torch.long, device=device)
        loss_weights = torch.zeros(num_rois, dtype=torch.float, device=device)
        if return_targets:
            regression_targets = torch.zeros(num_rois, 4, dtype=torch.float, device=device)
            return pseudo_labels, loss_weights, regression_targets
        return pseudo_labels, loss_weights
class oicr_layer(object):
    """ OICR. Tang et al. 2017 (https://arxiv.org/abs/1704.00138) """
    def __init__(self):
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)
    @torch.no_grad()
    def __call__(self, proposals, source_score, labels, device, return_targets=False):
        gt_boxes = torch.zeros((0, 4), dtype=torch.float, device=device)
        gt_classes = torch.zeros((0, 1), dtype=torch.long, device=device)
        gt_scores = torch.zeros((0, 1), dtype=torch.float, device=device)

        # not using the background class
        _prob = source_score[:, 1:].clone()
        _labels = labels[1:]
        positive_classes = _labels.eq(1).nonzero(as_tuple=False)[:, 0]
        for c in positive_classes:
            cls_prob = _prob[:, c]
            max_index = torch.argmax(cls_prob)
            gt_boxes = torch.cat((gt_boxes, proposals.bbox[max_index].view(1, -1)), dim=0)
            gt_classes = torch.cat((gt_classes, c.add(1).view(1, 1)), dim=0)
            gt_scores = torch.cat((gt_scores, cls_prob[max_index].view(1, 1)), dim=0)
            _prob[max_index].fill_(0)

        #if return_targets == True:
        #    gt_boxes = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
        #    gt_boxes.add_field('labels',  gt_classes[:, 0].float())
        #    # gt_boxes.add_field('difficult', bb)
        #    return gt_boxes

        if gt_boxes.shape[0]  == 0:
            num_rois = len(source_score)
            pseudo_labels = torch.zeros(num_rois, dtype=torch.long, device=device)
            loss_weights = torch.zeros(num_rois, dtype=torch.float, device=device)
        else:
            gt_boxes = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
            overlaps = boxlist_iou(proposals, gt_boxes)
            max_overlaps, gt_assignment = overlaps.max(dim=1)
            pseudo_labels = gt_classes[gt_assignment, 0]
            loss_weights = gt_scores[gt_assignment, 0]

            # Select background RoIs as those with <= FG_IOU_THRESHOLD
            bg_inds = max_overlaps.le(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0]
            pseudo_labels[bg_inds] = 0

            if return_targets:
                matched_targets = gt_boxes[gt_assignment]
                regression_targets = self.box_coder.encode(
                     matched_targets.bbox, proposals.bbox
                )
                return pseudo_labels, loss_weights, regression_targets

            # PCL_TRICK:
            # ignore_thres = 0.1
            # ignore_inds = max_overlaps.le(ignore_thres).nonzero(as_tuple=False)[:,0]
            # loss_weights[ignore_inds] = 0

        return pseudo_labels, loss_weights

class sim_layer(object):
    """ OICR. Tang et al. 2017 (https://arxiv.org/abs/1704.00138) """
    def __init__(self):
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)
    @torch.no_grad()
    def __call__(self, proposals, source_score, labels, device, pgt_instance, return_targets=False):
        gt_boxes = torch.zeros((0, 4), dtype=torch.float, device=device)
        gt_classes = torch.zeros((0, 1), dtype=torch.long, device=device)
        gt_scores = torch.zeros((0, 1), dtype=torch.float, device=device)

        # not using the background class
        _prob = source_score[:, 1:].clone()
        _labels = labels[1:]
        positive_classes = _labels.eq(1).nonzero(as_tuple=False)[:, 0]
        for c in positive_classes:
            cls_prob = _prob[:, c]
            max_index = torch.argmax(cls_prob)

            sim_box = pgt_instance[c]

            if sim_box.nelement() == 0:
                gt_boxes = torch.cat((gt_boxes, proposals.bbox[max_index].view(1, -1)), dim=0)
                gt_classes = torch.cat((gt_classes, c.add(1).view(1, 1)), dim=0)
                gt_scores = torch.cat((gt_scores, cls_prob[max_index].view(1, 1)), dim=0)
                _prob[max_index].fill_(0)
                #_prob[sim_box].fill_(0)
            else:
                gt_boxes = torch.cat((gt_boxes, proposals.bbox[sim_box]), dim=0)
                tmp_cls = torch.ones((sim_box.shape), dtype=torch.long, device=device) * c.add(1).item()
                gt_classes = torch.cat((gt_classes, tmp_cls.unsqueeze(1)), dim=0)
                gt_scores = torch.cat((gt_scores, cls_prob[sim_box].unsqueeze(1)), dim=0)
                _prob[max_index].fill_(0)
                #_prob[sim_box].fill_(0)

        if gt_boxes.shape[0]  == 0:
            num_rois = len(source_score)
            pseudo_labels = torch.zeros(num_rois, dtype=torch.long, device=device)
            loss_weights = torch.zeros(num_rois, dtype=torch.float, device=device)
        else:
            gt_boxes = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
            overlaps = boxlist_iou(proposals, gt_boxes)
            #max_overlaps, gt_assignment = overlaps.max(dim=1)

            max_overlaps  = torch.tensor(overlaps.cpu().numpy().max(axis=1), device=device)
            gt_assignment = torch.tensor(overlaps.cpu().numpy().argmax(axis=1), device=device)

            pseudo_labels = gt_classes[gt_assignment, 0]
            loss_weights = gt_scores[gt_assignment, 0]

            # Select background RoIs as those with <= FG_IOU_THRESHOLD
            bg_inds = max_overlaps.le(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0] ### IoU for instance supervision
            pseudo_labels[bg_inds] = 0

            if return_targets:
                matched_targets = gt_boxes[gt_assignment]
                regression_targets = self.box_coder.encode(
                     matched_targets.bbox, proposals.bbox
                )
                return pseudo_labels, loss_weights, regression_targets

            # PCL_TRICK:
            # ignore_thres = 0.1
            # ignore_inds = max_overlaps.le(ignore_thres).nonzero(as_tuple=False)[:,0]
            # loss_weights[ignore_inds] = 0

        return pseudo_labels, loss_weights

class cbs_layer(object):
    """ OICR. Tang et al. 2017 (https://arxiv.org/abs/1704.00138) """
    def __init__(self):
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)
    @torch.no_grad()
    def __call__(self, proposals, source_score, labels, device, close_box, n_cls, n_pgt, duplicate, return_targets=False):
    #def __call__(self, proposals, source_score, labels, device, close_box, duplicate, return_targets=False):
        gt_boxes = torch.zeros((0, 4), dtype=torch.float, device=device)
        gt_classes = torch.zeros((0, 1), dtype=torch.long, device=device)
        gt_scores = torch.zeros((0, 1), dtype=torch.float, device=device)

        # not using the background class
        _prob = source_score[:, 1:].clone()
        _labels = labels[1:]
        positive_classes = _labels.eq(1).nonzero(as_tuple=False)[:, 0]
        max_ind = torch.zeros((0, 1), dtype=torch.long, device=device)
        neg_pgt = []

        for c in positive_classes:
            cls_prob = _prob[:, c]
            max_index = torch.argmax(cls_prob)
            if duplicate == c.add(1).item():
                gt_boxes = torch.cat((gt_boxes, proposals.bbox[close_box]), dim=0)
                tmp_cls = torch.ones((close_box.shape), dtype=torch.long, device=device) * duplicate
                gt_classes = torch.cat((gt_classes, tmp_cls.unsqueeze(1)), dim=0)
                gt_scores = torch.cat((gt_scores, cls_prob[close_box].unsqueeze(1)), dim=0)
                _prob[max_index].fill_(0)
                ### TODO close_box should be fill(0)?
            else:
                neg_c_n = n_pgt[torch.nonzero(n_cls == c.add(1))[0]]
                gt_boxes = torch.cat((gt_boxes, proposals.bbox[neg_c_n]), dim=0)
                tmp_cls = torch.ones((neg_c_n.shape), dtype=torch.long, device=device) * c.add(1).item()
                gt_classes = torch.cat((gt_classes, tmp_cls.unsqueeze(1)), dim=0)
                gt_scores = torch.cat((gt_scores, cls_prob[neg_c_n].unsqueeze(1)), dim=0)

                ### same TODO neg_c_n fill(0)?
                '''
                score_weight = torch.max(cls_prob)
                neg_thres = cos_sim(triplet_feature[max_index].view(1,-1), triplet_feature[cal_iou(proposals, max_index)]).mean()
                neg_cos = cos_sim(triplet_feature, triplet_feature[max_index].view(1,-1)).mean(dim=1)
                neg_close = torch.nonzero(torch.ge(neg_cos, score_weight*neg_thres))
                neg_cluster = easy_cluster(proposals, neg_close.view(-1), n_cluster=5, c_iou=0.5)
                neg_c_n = easy_nms(proposals, neg_cluster, cls_prob, nms_iou=0.1)

                gt_boxes = torch.cat((gt_boxes, proposals.bbox[neg_c_n]), dim=0)
                tmp_cls = torch.ones((neg_c_n.shape), dtype=torch.long, device=device) * c.add(1).item()
                gt_classes = torch.cat((gt_classes, tmp_cls.unsqueeze(1)), dim=0)
                gt_scores = torch.cat((gt_scores, cls_prob[neg_c_n].unsqueeze(1)), dim=0)
                '''
                '''
                gt_boxes = torch.cat((gt_boxes, proposals.bbox[max_index].view(1, -1)), dim=0)
                gt_classes = torch.cat((gt_classes, c.add(1).view(1, 1)), dim=0)
                gt_scores = torch.cat((gt_scores, cls_prob[max_index].view(1, 1)), dim=0)
                _prob[max_index].fill_(0)
                max_ind = torch.cat((max_ind, max_index.view(1,-1)))
                '''

        if gt_boxes.shape[0]  == 0:
            num_rois = len(source_score)
            pseudo_labels = torch.zeros(num_rois, dtype=torch.long, device=device)
            loss_weights = torch.zeros(num_rois, dtype=torch.float, device=device)
        else:
            gt_boxes = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
            overlaps = boxlist_iou(proposals, gt_boxes)

            #max_overlaps, gt_assignment = overlaps.max(dim=1)
            max_overlaps  = torch.tensor(overlaps.cpu().numpy().max(axis=1), device=device)
            gt_assignment = torch.tensor(overlaps.cpu().numpy().argmax(axis=1), device=device)

            pseudo_labels = gt_classes[gt_assignment, 0]
            loss_weights = gt_scores[gt_assignment, 0]

            # Select background RoIs as those with <= FG_IOU_THRESHOLD
            fg_inds = max_overlaps.gt(0.5).nonzero(as_tuple=False)[:,0]
            bg_inds = max_overlaps.le(0.5).nonzero(as_tuple=False)[:,0] ### 0.6 -> more bg
            core_inds = max_overlaps.eq(1).nonzero(as_tuple=False)[:,0]
            ### labeling trick ###
            '''fg_inds = max_overlaps.gt(0.6).nonzero(as_tuple=False)[:,0]
            bg_inds = torch.bitwise_and(max_overlaps.ge(0.1),
                                        max_overlaps.lt(0.4)).nonzero(as_tuple=False)[:,0]
            ignore_inds = torch.bitwise_or(torch.bitwise_and(max_overlaps.le(0.6), max_overlaps.ge(0.4)),
                                           max_overlaps.lt(0.1)).nonzero(as_tuple=False)[:,0]
            loss_weights[ignore_inds] = 0
            '''
            ### labeling trick ###
            pseudo_labels[bg_inds] = 0
            ###
            #loss_weights[fg_inds] *= 1.2
            #loss_weights[core_inds] *= 0.2
            ###
            if return_targets:
                matched_targets = gt_boxes[gt_assignment]
                regression_targets = self.box_coder.encode(
                     matched_targets.bbox, proposals.bbox
                )
                return pseudo_labels, loss_weights, regression_targets, core_inds
        return pseudo_labels, loss_weights


class mist_cbs_layer(object):
    def __init__(self, p, iou=0.2):
        self.portion = p
        self.iou_th = iou
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)

    @torch.no_grad()
    def __call__(self, proposals, source_score, labels, device, pgt_instance, return_targets=False):
        gt_boxes = torch.zeros((0, 4), dtype=torch.float, device=device)
        gt_classes = torch.zeros((0, 1), dtype=torch.long, device=device)
        gt_scores = torch.zeros((0, 1), dtype=torch.float, device=device)

        num_rois = len(proposals)
        k = int(num_rois * self.portion)
        num_gt_cls = labels[1:].sum()
        if num_gt_cls != 0 and num_rois != 0:
            cls_prob = source_score[:, 1:]
            gt_cls_inds = labels[1:].nonzero(as_tuple=False)[:, 0]
            sorted_scores, max_inds = cls_prob[:, gt_cls_inds].sort(dim=0, descending=True)
            sorted_scores = sorted_scores[:k]
            max_inds = max_inds[:k]

            _boxes = proposals.bbox[max_inds.t().contiguous().view(-1)].view(num_gt_cls.int(), -1, 4)
            _boxes = BatchBoxList(_boxes, proposals.size, mode=proposals.mode)
            ious = batch_boxlist_iou(_boxes, _boxes)
            k_ind = torch.zeros(num_gt_cls.int(), k, dtype=torch.bool, device=device)
            k_ind[:, 0] = 1 # always take the one with max score
            for ii in range(1, k):
                max_iou, _ = torch.max(ious[:,ii:ii+1, :ii], dim=2)
                k_ind[:, ii] = (max_iou < self.iou_th).byte().squeeze(-1)

            mist_gt_boxes = _boxes.bbox[k_ind]
            mist_gt_cls_id = gt_cls_inds + 1
            mist_temp_cls = torch.ones((_boxes.bbox.shape[:2]), device=device) * mist_gt_cls_id.view(-1, 1).float()
            mist_gt_classes = mist_temp_cls[k_ind].view(-1, 1).long()
            mist_gt_scores = sorted_scores.t().contiguous()[k_ind].view(-1, 1)

            _labels = labels[1:]
            positive_classes = _labels.eq(1).nonzero(as_tuple=False)[:, 0]
            _prob = source_score[:, 1:].clone()

            for c in positive_classes:
                cls_prob = _prob[:, c]
                max_index = torch.argmax(cls_prob)

                sim_box = pgt_instance[c]

                if sim_box.nelement() == 0:
                    gt_boxes = torch.cat((gt_boxes, proposals.bbox[max_index].view(1, -1)), dim=0)
                    gt_classes = torch.cat((gt_classes, c.add(1).view(1, 1)), dim=0)
                    gt_scores = torch.cat((gt_scores, cls_prob[max_index].view(1, 1)), dim=0)
                    _prob[max_index].fill_(0)
                    #_prob[sim_box].fill_(0)
                else:
                    gt_boxes = torch.cat((gt_boxes, proposals.bbox[sim_box]), dim=0)
                    tmp_cls = torch.ones((sim_box.shape), dtype=torch.long, device=device) * c.add(1).item()
                    gt_classes = torch.cat((gt_classes, tmp_cls.unsqueeze(1)), dim=0)
                    gt_scores = torch.cat((gt_scores, cls_prob[sim_box].unsqueeze(1)), dim=0)
                    _prob[max_index].fill_(0)
                    #_prob[sim_box].fill_(0)


            '''for i, c in enumerate(positive_classes):
                if c.add(1) == duplicate:
                    cls_prob = _prob[:, c]
                    mist_pgt = max_inds[:,i][torch.nonzero(k_ind)[:,1][torch.where(mist_gt_classes == c.add(1))[0]]]
                    pgt_update = torch.cat((close_box, mist_pgt)).unique()

                    gt_boxes = torch.cat((gt_boxes, proposals.bbox[pgt_update]), dim=0)
                    tmp_cls = torch.ones((pgt_update.shape), dtype=torch.long, device=device) * c.add(1)
                    gt_classes = torch.cat((gt_classes, tmp_cls.unsqueeze(1)), dim=0)
                    gt_scores = torch.cat((gt_scores, cls_prob[pgt_update].unsqueeze(1)), dim=0)
            '''
            if gt_boxes.shape[0] != 0:
                gt_boxes = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
                overlaps = boxlist_iou(proposals, gt_boxes)

                # TODO: pytorch and numpy argmax perform differently
                # max_overlaps, gt_assignment = overlaps.max(dim=1)
                max_overlaps  = torch.tensor(overlaps.cpu().numpy().max(axis=1), device=device)
                gt_assignment = torch.tensor(overlaps.cpu().numpy().argmax(axis=1), device=device)

                pseudo_labels = gt_classes[gt_assignment, 0]
                loss_weights = gt_scores[gt_assignment, 0]

                # fg_inds = max_overlaps.ge(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0]
                # Select background RoIs as those with <= FG_IOU_THRESHOLD
                bg_inds = max_overlaps.lt(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0]
                pseudo_labels[bg_inds] = 0

                # compute regression targets
                if return_targets:
                    matched_targets = gt_boxes[gt_assignment]
                    regression_targets = self.box_coder.encode(
                        matched_targets.bbox, proposals.bbox
                    )
                    return pseudo_labels, loss_weights, regression_targets

                return pseudo_labels, loss_weights

        # corner case
        pseudo_labels = torch.zeros(num_rois, dtype=torch.long, device=device)
        loss_weights = torch.zeros(num_rois, dtype=torch.float, device=device)
        if return_targets:
            regression_targets = torch.zeros(num_rois, 4, dtype=torch.float, device=device)
            return pseudo_labels, loss_weights, regression_targets
        return pseudo_labels, loss_weights


