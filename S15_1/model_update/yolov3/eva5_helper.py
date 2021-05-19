import random
import math

import numpy as np
import torch.nn.functional as F
import torch
from .utils.utils import compute_loss, non_max_suppression, clip_coords, xywh2xyxy, box_iou, ap_per_class

class YoloTrainer:
    def __init__(self, model, hyp, opt, nb, nc):
        self.nb = nb
        self.nc = nc
        self.n_burn = max(3 * nb, 500)
        self.optimizer = None
        self.model = model
        self.batch_size = opt.batch_size
        self.accumulate = max(round(64 / opt.batch_size), 1) 

        self.gs = 32 # (pixels) grid size
        imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)
        opt.multi_scale |= imgsz_min != imgsz_max
        if opt.multi_scale:
            if imgsz_min == imgsz_max:
                imgsz_min //= 1.5
                imgsz_max //= 0.667
            self.grid_min, self.grid_max = imgsz_min // self.gs, imgsz_max // self.gs
            imgsz_min, imgsz_max = int(self.grid_min *self.gs), int(self.grid_max * self.gs)
        self.img_size = imgsz_max

        hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

        self.hyp = hyp
        self.opt = opt

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        self.lf = lambda x: (((1 + math.cos(x * math.pi / opt.epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        assert math.fmod(imgsz_min, self.gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, self.gs)

        # validation stuff
        self.stats = []
        self.seen = 0

        iouv = torch.linspace(0.5, 0.95, 10).to("cuda" if torch.cuda.is_available() else "cpu")  # iou vector for mAP@0.5:0.95
        self.iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
        self.niou = iouv.numel()


    def pre_train_step(self, batch, batch_idx, epoch):
        if self.optimizer is None:
            raise Exception("Set optimizer for the yolo trainer.")

        imgs, _, _, _, _ = batch
        ni = self.calc_ni(batch_idx, epoch)  # number integrated batches (since train start)
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

        # Burn-in
        if ni <= self.n_burn:
            xi = [0, self.n_burn]  # x interp
            self.opt.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(epoch)])
                x['weight_decay'] = np.interp(ni, xi, [0.0, self.hyp['weight_decay'] if j == 1 else 0.0])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.9, self.hyp['momentum']])

        # Multi-Scale
        if self.opt.multi_scale:
            if ni / self.accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                self.img_size = random.randrange(self.grid_min, self.grid_max + 1) * self.gs
            sf = self.img_size / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        return imgs

    def post_train_step(self, outputs, batch, batch_idx, epoch):
        _, targets, _, _, _ = batch
       
        # Loss
        loss, loss_items = compute_loss(outputs, targets, self.model)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss_items)
            exit(-1)

        loss *= self.batch_size / 64  # scale loss

        return loss, loss_items

    def validation_epoch_start(self):
        self.seen = 0
        self.stats = []

    def validation_step(self, opt, outputs, batch, batch_idx, epoch):
        imgs, targets, paths, shapes, pad = batch
        _, _, height, width = imgs.shape
        
        inf_out, train_out = outputs
        whwh = torch.Tensor([width, height, width, height]).to(imgs.device)

        losses = compute_loss(train_out, targets, self.model)[1][:3]  # GIoU, obj, cls
        output = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, multi_label=self.calc_ni(batch_idx, epoch) > self.n_burn)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            self.seen += 1

            if pred is None:
                if nl:
                    self.stats.append((torch.zeros(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool, device=imgs.device)

            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > self.iouv[0].to(ious.device)).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        return losses

    def validation_epoch_end(self):
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if self.niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=self.nc)  # number of targets per class

        maps = np.zeros(self.nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        # Return results
        return (mp, mr, map, mf1), maps

    def calc_ni(self, batch_idx, epoch):
        return batch_idx + self.nb * epoch

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer