from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async, boxlist_nms_index, batch_boxlist_iou
from wetectron.structures.bounding_box import BoxList, BatchBoxList
import torch
from wetectron.modeling.utils import cat
from torch.nn import functional as F
import numpy as np
from wetectron.utils.utils import clustering, to_boxlist, cal_iou, one_line_nms, cos_sim, get_share_class, easy_nms, easy_cluster, generate_img_label

@torch.no_grad()
def get_max_index(final_score_list, ref_scores, num_refs, duplicate, proposals, b1_neg, b2_neg):
    anchor, positive, b1_n, b2_n = [],[],[],[]
    b1_n_cat = torch.zeros((0), dtype=torch.long, device=duplicate.device)
    b2_n_cat = torch.zeros((0), dtype=torch.long, device=duplicate.device)
    for idx, final_score_per_im in enumerate(final_score_list):
        for i in range(num_refs):
            source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
            if idx == 0:
                a_max = source_score[:,duplicate].topk(1)[1]
                anchor.append(a_max)
                if b1_neg.nelement() != 0:
                    b1_n_max = source_score[:,b1_neg].argmax(dim=0)
                    b1_n.append(b1_n_max)
                    #b1_n_cat = torch.cat((b1_n_cat, b1_n_max.view(1,-1)))
            if idx == 1:
                p_max = source_score[:,duplicate].topk(1)[1]
                positive.append(p_max)
                if b2_neg.nelement() != 0:
                    b2_n_max = source_score[:,b2_neg].argmax(dim=0)
                    b2_n.append(b2_n_max)
                    #b2_n_cat = torch.cat((b2_n_cat, b2_n_max.view(1,-1)))
    '''
    if b1_neg.nelement() != 0:
        for b1_n_c in b1_neg:
            for i in range(num_refs):
                source_score = final_score_list[0] if i == 0 else F.softmax(ref_scores[i-1][0], dim=1)
                import IPython; IPython.embed()
    if b2_neg.nelement() != 0:
        b2_n_temp = []
        for b2_n_c in b2_neg:
            for i in range(num_refs):
                source_score = final_score_list[1] if i == 0 else F.softmax(ref_scores[i-1][1], dim=1)
                b2_n_temp.append(source_score[:,b2_n_c].argmax(dim=0))
            import IPython; IPython.embed()
            b2_n_cat = torch.cat((b2_n_cat, torch.stack(b2_n_temp).view(1,-1)))
            import IPython; IPython.embed()
    '''
    return anchor, positive, b1_n, b2_n

@torch.no_grad()
def close_sampling(sim_feature, source_score, anchor, positive, p_size, proposals, duplicate, n_cluster, nms_iou, balance):
    #v_storage[duplicate-1] = torch.cat((v_storage[duplicate-1].view(1,-1), sim_feature[anchor], sim_feature[positive + p_size])).mean(0)
    #b1_cos = cos_sim(sim_feature, sim_feature[anchor]).mean(dim=1)
    #b2_cos = cos_sim(sim_feature, sim_feature[positive + p_size]).mean(dim=1)
    #thres_dist = cos_sim(sim_feature[anchor], sim_feature[positive + p_size]).mean()
    score_weight = (source_score[anchor, duplicate] + source_score[positive + p_size, duplicate])/2

    b1_cos = torch.mm(sim_feature, sim_feature[anchor].T).mean(dim=1)
    b2_cos = torch.mm(sim_feature, sim_feature[positive + p_size].T).mean(dim=1)
    thres_dist = torch.mm(sim_feature[anchor], sim_feature[positive + p_size].T).mean()
    #thres_dist = (torch.mm(sim_feature[anchor], v_storage[duplicate-1].view(1,-1).T).mean() + torch.mm(sim_feature[positive+p_size], v_storage[duplicate-1].view(1,-1).T).mean())/2

    if balance == 'max':
        cos_close = torch.bitwise_or(torch.ge(b1_cos, thres_dist), torch.ge(b2_cos, thres_dist))
    elif balance == 'score_weight':
        cos_close = torch.bitwise_or(torch.ge(b1_cos, score_weight*thres_dist), torch.ge(b2_cos, score_weight*thres_dist))
    elif 'balance' in balance:
        cos_close = torch.bitwise_or(torch.ge(b1_cos * source_score[anchor, duplicate], score_weight*thres_dist),
                                     torch.ge(b2_cos * source_score[positive + p_size, duplicate], score_weight*thres_dist))

    b1_pgt = torch.nonzero(cos_close[:p_size]).view(-1)
    b2_pgt = torch.nonzero(cos_close[p_size:]).view(-1)
    b1_pgt, b2_pgt = clustering(proposals, b1_pgt, b2_pgt, n_cluster=n_cluster, c_iou=0.5)
    b1_pgt, b2_pgt = one_line_nms(proposals, b1_pgt, b2_pgt, source_score, p_size, duplicate, nms_iou=nms_iou)

    #b1_cbs_pgt = torch.cat((anchor, b1_pgt)) if anchor not in b1_pgt else b1_pgt
    #b2_cbs_pgt = torch.cat((positive, b2_pgt)) if positive not in b2_pgt else b2_pgt
    b1_cbs_pgt = torch.cat((cal_iou(proposals[0],anchor, 0.5)[0], b1_pgt)) if anchor not in b1_pgt else b1_pgt
    b2_cbs_pgt = torch.cat((cal_iou(proposals[1], positive, 0.5)[0], b2_pgt)) if positive not in b2_pgt else b2_pgt
    return b1_cbs_pgt, b2_cbs_pgt

@torch.no_grad()
def neg_sampling(sim_feature, source_score, b1_neg, b2_neg, proposals, p_size, b1_n, b2_n, n_cluster, nms_iou, balance, k, i):
    b1_n_pgt, b2_n_pgt = [], []
    b1_n_pgt_cat = torch.zeros((0), dtype=torch.long, device=sim_feature.device)
    b2_n_pgt_cat = torch.zeros((0), dtype=torch.long, device=sim_feature.device)

    if b1_neg.nelement() != 0:
        for ind_c, b1_n_c in enumerate(b1_neg):
            max_index = torch.argmax(source_score[:p_size,b1_n_c])
            max_score = torch.max(source_score[:p_size,b1_n_c])
            #overlap_score = source_score[cal_iou(proposals[0],max_index),b1_n_c].mean()
            topk_score = source_score[:p_size,b1_n_c].topk(k)[0].mean()

            #v_storage[b1_n_c-1] = torch.cat((sim_feature[max_index].view(1,-1), sim_feature[b1_n_c-1].view(1,-1))).mean(0)
            #neg_cos = cos_sim(sim_feature[:p_size], sim_feature[max_index].view(1,-1)).mean(dim=1)
            #neg_thres = cos_sim(sim_feature[max_index].view(1,-1), sim_feature[source_score[:p_size,b1_n_c].topk(k)[1]]).mean()
            neg_cos = torch.mm(sim_feature[:p_size], sim_feature[max_index].view(1,-1).T).mean(1)
            #neg_thres = torch.mm(sim_feature[max_index].view(1,-1), v_storage[b1_n_c-1].view(1,-1).T).mean()
            #neg_thres = torch.mm(sim_feature[max_index].view(1,-1), sim_feature[source_score[:p_size,b1_n_c].topk(k)[1]].T).mean()
            neg_thres = (torch.mm(sim_feature[max_index].view(1,-1), sim_feature[cal_iou(proposals[0],max_index,0.5)[0]].T) * cal_iou(proposals[0],max_index,0.5)[1]).mean()
            #neg_thres = cos_sim(sim_feature[max_index].view(1,-1), sim_feature[cal_iou(proposals[0],max_index)]).mean()

            if balance == 'max':
                neg_close = torch.nonzero(torch.ge(neg_cos, neg_thres)).view(-1)
            elif balance == 'score_weight':
                neg_close = torch.nonzero(torch.ge(neg_cos, neg_thres * max_score)).view(-1)
            elif balance == 'balance':
                neg_close = torch.nonzero(torch.ge(neg_cos * max_score, neg_thres * topk_score)).view(-1)

            neg_cluster = easy_cluster(proposals[0], neg_close, n_cluster=n_cluster, c_iou=0.5)
            neg_pgt = easy_nms(proposals[0], neg_cluster, source_score[:p_size, b1_n_c], nms_iou=nms_iou)
            #neg_pgt = torch.cat((max_index.view(-1), neg_pgt)) if max_index not in neg_pgt else neg_pgt
            neg_pgt = torch.cat((cal_iou(proposals[0],max_index.view(-1), 0.5)[0], neg_pgt)) if max_index not in neg_pgt else neg_pgt

            b1_n_pgt_cat = torch.cat((b1_n_pgt_cat, neg_pgt))
            b1_n_pgt.append(neg_pgt)

    if b2_neg.nelement() != 0:
        for ind_c, b2_n_c in enumerate(b2_neg):
            max_index = torch.argmax(source_score[p_size:,b2_n_c])
            max_score = torch.max(source_score[p_size:,b2_n_c])
            #overlap_score = source_score[cal_iou(proposals[1], max_index) + p_size, b2_n_c].mean()
            topk_score = source_score[p_size:,b2_n_c].topk(k)[0].mean()

            #v_storage[b2_n_c-1] = torch.cat((sim_feature[max_index + p_size].view(1,-1), sim_feature[b2_n_c-1].view(1,-1))).mean(0)
            #neg_cos = cos_sim(sim_feature[p_size:], sim_feature[max_index + p_size].view(1,-1)).mean(dim=1)
            #neg_thres = cos_sim(sim_feature[max_index + p_size].view(1,-1), sim_feature[source_score[p_size:,b2_n_c].topk(k)[1]]).mean()

            neg_cos = torch.mm(sim_feature[p_size:], sim_feature[max_index + p_size].view(1,-1).T).mean(dim=1)
            #neg_thres = torch.mm(sim_feature[max_index + p_size].view(1,-1), sim_feature[source_score[p_size:,b2_n_c].topk(k)[1] + p_size].T).mean()
            neg_thres = (torch.mm(sim_feature[max_index + p_size].view(1,-1), sim_feature[cal_iou(proposals[1],max_index,0.5)[0] + p_size].T) * cal_iou(proposals[1],max_index,0.5)[1]).mean()
            #neg_thres = cos_sim(sim_feature[max_index + p_size].view(1,-1), sim_feature[cal_iou(proposals[1],max_index) + p_size]).mean()
            #neg_thres = torch.mm(sim_feature[max_index + p_size].view(1,-1), v_storage[b2_n_c-1].view(1,-1).T).mean()

            if balance == 'max':
                neg_close = torch.nonzero(torch.ge(neg_cos, neg_thres)).view(-1)
            elif balance == 'score_weight':
                neg_close = torch.nonzero(torch.ge(neg_cos, neg_thres * max_score)).view(-1)
            elif balance == 'balance':
                neg_close = torch.nonzero(torch.ge(neg_cos * max_score, neg_thres * topk_score)).view(-1)

            neg_cluster = easy_cluster(proposals[1], neg_close, n_cluster=n_cluster, c_iou=0.5)
            neg_pgt = easy_nms(proposals[1], neg_cluster, source_score[p_size:, b2_n_c], nms_iou=nms_iou)
            #neg_pgt = torch.cat((max_index.view(-1), neg_pgt)) if max_index not in neg_pgt else neg_pgt
            neg_pgt = torch.cat((cal_iou(proposals[1],max_index.view(-1),0.5)[0], neg_pgt)) if max_index not in neg_pgt else neg_pgt
            b2_n_pgt_cat = torch.cat((b2_n_pgt_cat, neg_pgt))
            b2_n_pgt.append(neg_pgt)
    return b1_n_pgt, b2_n_pgt, b1_n_pgt_cat, b2_n_pgt_cat

@torch.no_grad()
def mist_sampling(proposals, source_score, targets, duplicate, device):
    portion = 0.2
    iou_th = 0.15
    mist_pgt_share = []
    mist_pgt_neg = []
    source_score = source_score.split([len(p) for p in proposals])
    b1_mist_pgt_n = torch.zeros((0), dtype=torch.long, device=device)
    b2_mist_pgt_n = torch.zeros((0), dtype=torch.long, device=device)
    for idx, (proposals_per_im, score_per_im, targets_per_im) in enumerate(zip(proposals, source_score, targets)):
        num_rois = len(proposals_per_im)
        labels = targets_per_im.get_field('labels').unique()
        labels = generate_img_label(score_per_im.shape[1], labels, device)
        k = int(num_rois * portion)
        num_gt_cls = labels[1:].sum()
        if num_gt_cls != 0 and num_rois != 0:
            cls_prob = score_per_im[:, 1:]
            gt_cls_inds = labels[1:].nonzero(as_tuple=False)[:, 0]
            sorted_scores, max_inds = cls_prob[:, gt_cls_inds].sort(dim=0, descending=True)
            sorted_scores = sorted_scores[:k]
            max_inds = max_inds[:k]

            _boxes = proposals_per_im.bbox[max_inds.t().contiguous().view(-1)].view(num_gt_cls.int(), -1, 4)
            _boxes = BatchBoxList(_boxes, proposals_per_im.size, mode=proposals_per_im.mode)
            ious = batch_boxlist_iou(_boxes, _boxes)
            k_ind = torch.zeros(num_gt_cls.int(), k, dtype=torch.bool, device=device)
            k_ind[:, 0] = 1 # always take the one with max score
            for ii in range(1, k):
                max_iou, _ = torch.max(ious[:,ii:ii+1, :ii], dim=2)
                k_ind[:, ii] = (max_iou < iou_th).byte().squeeze(-1)

            mist_gt_boxes = _boxes.bbox[k_ind]
            mist_gt_cls_id = gt_cls_inds + 1
            mist_temp_cls = torch.ones((_boxes.bbox.shape[:2]), device=device) * mist_gt_cls_id.view(-1, 1).float()
            mist_gt_classes = mist_temp_cls[k_ind].view(-1, 1).long()
            mist_gt_scores = sorted_scores.t().contiguous()[k_ind].view(-1, 1)

            _labels = labels[1:]
            positive_classes = _labels.eq(1).nonzero(as_tuple=False)[:, 0]
            _prob = score_per_im[:, 1:].clone()

            for i, c in enumerate(positive_classes):
                if c.add(1) == duplicate:
                    if idx == 0:
                        b1_mist_pgt_s = max_inds[:,i][torch.nonzero(k_ind)[:,1][torch.where(mist_gt_classes == c.add(1))[0]]]
                    elif idx == 1:
                        b2_mist_pgt_s = max_inds[:,i][torch.nonzero(k_ind)[:,1][torch.where(mist_gt_classes == c.add(1))[0]]]
                else:
                    mist_pgt_nega = max_inds[:,i][torch.nonzero(k_ind)[:,1][torch.where(mist_gt_classes == c.add(1))[0]]]
                    if idx == 0:
                        b1_mist_pgt_n = torch.cat((b1_mist_pgt_n, mist_pgt_nega))
                    elif idx == 1:
                        b2_mist_pgt_n = torch.cat((b2_mist_pgt_n, mist_pgt_nega))
                    #mist_pgt_n = torch.cat((mist_pgt_n, mist_pgt_nega))
                    #mist_pgt_neg.append(mist_pgt_n)

            #mist_pgt.append([gt_boxes, gt_classes, gt_scores, torch.nonzero(k_ind), k_ind])

    return b1_mist_pgt_s, b2_mist_pgt_s, b1_mist_pgt_n, b2_mist_pgt_n

