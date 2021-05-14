import torch
from .models.model import (
    compute_losses,
    l1LossMask,
    l2LossMask,
    unmoldDetections,
    calcXYZModule,
    invertDepth,
    warpModuleDepth,
    l2NormLossMask,
)


class PlaneRCNNTrainer:
    def __init__(self, config, refine_model):
        self.config = config
        self.options = config.options
        self.refine_model = refine_model

    def train_step_on_batch(self, batches, outputs, device="cpu"):
        losses = []
        for i in range(len(batches)):
            losses.append(self.train_step(batches[i], outputs[i], device=device))

        return losses

    def train_step(self, batch, outputs, device="cpu"):
        input_pair = []
        detection_pair = []
        losses = []

        (
            images,
            image_metas,
            rpn_match,
            rpn_bbox,
            gt_class_ids,
            gt_boxes,
            gt_masks,
            gt_parameters,
            gt_depth,
            extrinsics,
            gt_segmentation,
            camera,
        ) = batch
        input_pair.append(
            {
                "image": images,
                "depth": gt_depth,
                "mask": gt_masks,
                "bbox": gt_boxes,
                "extrinsics": extrinsics,
                "segmentation": gt_segmentation,
                "camera": camera,
            }
        )

        (
            rpn_class_logits,
            rpn_pred_bbox,
            target_class_ids,
            mrcnn_class_logits,
            target_deltas,
            mrcnn_bbox,
            target_mask,
            mrcnn_mask,
            target_parameters,
            mrcnn_parameters,
            detections,
            detection_masks,
            detection_gt_parameters,
            detection_gt_masks,
            rpn_rois,
            roi_features,
            roi_indices,
            depth_np_pred,
        ) = outputs

        (
            rpn_class_loss,
            rpn_bbox_loss,
            mrcnn_class_loss,
            mrcnn_bbox_loss,
            mrcnn_mask_loss,
            mrcnn_parameter_loss,
        ) = compute_losses(
            self.config,
            rpn_match,
            rpn_bbox,
            rpn_class_logits,
            rpn_pred_bbox,
            target_class_ids,
            mrcnn_class_logits,
            target_deltas,
            mrcnn_bbox,
            target_mask,
            mrcnn_mask,
            target_parameters,
            mrcnn_parameters,
        )

        losses += [
            rpn_class_loss
            + rpn_bbox_loss
            + mrcnn_class_loss
            + mrcnn_bbox_loss
            + mrcnn_mask_loss
            + mrcnn_parameter_loss
        ]

        if self.config.PREDICT_NORMAL_NP:
            normal_np_pred = depth_np_pred[0, 1:]
            depth_np_pred = depth_np_pred[:, 0]
            gt_normal = gt_depth[0, 1:]
            gt_depth = gt_depth[:, 0]
            depth_np_loss = l1LossMask(
                depth_np_pred[:, 80:560],
                gt_depth[:, 80:560],
                (gt_depth[:, 80:560] > 1e-4).float(),
            )
            normal_np_loss = l2LossMask(
                normal_np_pred[:, 80:560],
                gt_normal[:, 80:560],
                (torch.norm(gt_normal[:, 80:560], dim=0) > 1e-4).float(),
            )
            losses.append(depth_np_loss)
            losses.append(normal_np_loss)
        else:
            depth_np_loss = l1LossMask(
                depth_np_pred[:, 80:560],
                gt_depth[:, 80:560],
                (gt_depth[:, 80:560] > 1e-4).float(),
            )
            losses.append(depth_np_loss)
            normal_np_pred = None
            pass

        if len(detections) > 0:
            detections, detection_masks = unmoldDetections(
                self.config,
                camera,
                detections,
                detection_masks,
                depth_np_pred,
                normal_np_pred,
                debug=False,
            )
            if "refine_only" in self.options.suffix:
                detections, detection_masks = (
                    detections.detach(),
                    detection_masks.detach(),
                )
                pass
            XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(
                self.config,
                camera,
                detections,
                detection_masks,
                depth_np_pred,
                return_individual=True,
            )
            detection_mask = detection_mask.unsqueeze(0)
        else:
            XYZ_pred = torch.zeros(
                (3, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
            ).to(device)
            detection_mask = torch.zeros(
                (1, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
            ).to(device)
            plane_XYZ = torch.zeros(
                (1, 3, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
            ).to(device)
            pass

        input_pair.append(
            {
                "image": images,
                "depth": gt_depth,
                "mask": gt_masks,
                "bbox": gt_boxes,
                "extrinsics": extrinsics,
                "segmentation": gt_segmentation,
                "parameters": detection_gt_parameters,
                "camera": camera,
            }
        )
        # detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'feature_map': feature_map[0], 'plane_XYZ': plane_XYZ, 'depth_np': depth_np_pred})
        detection_pair.append(
            {
                "XYZ": XYZ_pred,
                "depth": XYZ_pred[1:2],
                "mask": detection_mask,
                "detection": detections,
                "masks": detection_masks,
                "plane_XYZ": plane_XYZ,
                "depth_np": depth_np_pred,
            }
        )

        if (
            len(detection_pair[0]["detection"]) > 0
            and len(detection_pair[0]["detection"]) < 30
        ) and "refine" in self.options.suffix:
            ## Use refinement network
            camera = camera.unsqueeze(0)
            c = 0
            detection_dict, input_dict = detection_pair[c], input_pair[c]
            detections = detection_dict["detection"]
            detection_masks = detection_dict["masks"]
            image = (
                input_dict["image"] + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))
            ) / 255.0 - 0.5
            image_2 = (
                input_pair[1 - c]["image"]
                + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))
            ) / 255.0 - 0.5
            depth_gt = input_dict["depth"].unsqueeze(1)

            masks_inp = torch.cat(
                [detection_masks.unsqueeze(1), detection_dict["plane_XYZ"]], dim=1
            )

            segmentation = input_dict["segmentation"]
            plane_depth = detection_dict["depth"]
            depth_np = detection_dict["depth_np"]
            if "large" not in self.options.suffix:
                ## Use 256x192 instead of 640x480
                detection_masks = torch.nn.functional.interpolate(
                    detection_masks[:, 80:560].unsqueeze(1),
                    size=(192, 256),
                    mode="nearest",
                ).squeeze(1)
                image = torch.nn.functional.interpolate(
                    image[:, :, 80:560], size=(192, 256), mode="bilinear"
                )
                image_2 = torch.nn.functional.interpolate(
                    image_2[:, :, 80:560], size=(192, 256), mode="bilinear"
                )
                masks_inp = torch.nn.functional.interpolate(
                    masks_inp[:, :, 80:560], size=(192, 256), mode="bilinear"
                )
                depth_gt = torch.nn.functional.interpolate(
                    depth_gt[:, :, 80:560], size=(192, 256), mode="nearest"
                )
                segmentation = (
                    torch.nn.functional.interpolate(
                        segmentation[:, 80:560].unsqueeze(1).float(),
                        size=(192, 256),
                        mode="nearest",
                    )
                    .squeeze()
                    .long()
                )
                plane_depth = torch.nn.functional.interpolate(
                    plane_depth[:, 80:560].unsqueeze(1).float(),
                    size=(192, 256),
                    mode="bilinear",
                ).squeeze(1)
                depth_np = torch.nn.functional.interpolate(
                    depth_np[:, 80:560].unsqueeze(1), size=(192, 256), mode="bilinear"
                ).squeeze(1)
            else:
                detection_masks = detection_masks[:, 80:560]
                image = image[:, :, 80:560]
                image_2 = image_2[:, :, 80:560]
                masks_inp = masks_inp[:, :, 80:560]
                depth_gt = depth_gt[:, :, 80:560]
                segmentation = segmentation[:, 80:560]
                plane_depth = plane_depth[:, 80:560]
                depth_np = depth_np[:, 80:560]

            depth_inv = invertDepth(depth_gt)
            depth_inv_small = depth_inv[:, :, ::4, ::4].contiguous()

            ## Generate supervision target for the refinement network
            segmentation_one_hot = (
                segmentation
                == torch.arange(segmentation.max() + 1).to(device).view((-1, 1, 1, 1))
            ).long()
            intersection = (
                (torch.round(detection_masks).long() * segmentation_one_hot)
                .sum(-1)
                .sum(-1)
            )
            max_intersection, segments_gt = intersection.max(0)
            mapping = intersection.max(1)[1]
            detection_areas = detection_masks.sum(-1).sum(-1)
            valid_mask = (
                mapping[segments_gt] == torch.arange(len(segments_gt)).to(device)
            ).float()

            masks_gt_large = (segmentation == segments_gt.view((-1, 1, 1))).float()
            masks_gt_small = masks_gt_large[:, ::4, ::4]

            ## Run the refinement network
            results = self.refine_model(
                image,
                image_2,
                camera,
                masks_inp,
                detection_dict["detection"][:, 6:9],
                plane_depth,
                depth_np,
            )

            plane_depth_loss = torch.zeros(1).to(device)
            depth_loss = torch.zeros(1).to(device)
            plane_loss = torch.zeros(1).to(device)
            mask_loss = torch.zeros(1).to(device)
            flow_loss = torch.zeros(1).to(device)
            flow_confidence_loss = torch.zeros(1).to(device)
            for resultIndex, result in enumerate(results[1:]):
                if "mask" in result:
                    masks_pred = result["mask"][:, 0]
                    if masks_pred.shape[-1] == masks_gt_large.shape[-1]:
                        masks_gt = masks_gt_large
                    else:
                        masks_gt = masks_gt_small
                        pass

                    all_masks_gt = torch.cat(
                        [1 - masks_gt.max(dim=0, keepdim=True)[0], masks_gt], dim=0
                    )
                    segmentation = all_masks_gt.max(0)[1].view(-1)
                    masks_logits = (
                        masks_pred.squeeze(1)
                        .transpose(0, 1)
                        .transpose(1, 2)
                        .contiguous()
                        .view((segmentation.shape[0], -1))
                    )
                    detection_areas = all_masks_gt.sum(-1).sum(-1)
                    detection_weight = detection_areas / detection_areas.sum()
                    detection_weight = -torch.log(
                        torch.clamp(detection_weight, min=1e-4, max=1 - 1e-4)
                    )
                    if "weight" in self.options.suffix:
                        mask_loss += torch.nn.functional.cross_entropy(
                            masks_logits, segmentation, weight=detection_weight
                        )
                    else:
                        mask_loss += torch.nn.functional.cross_entropy(
                            masks_logits,
                            segmentation,
                            weight=torch.cat([torch.ones(1).to(device), valid_mask], dim=0),
                        )
                    masks_pred = (
                        masks_pred.max(0, keepdim=True)[1]
                        == torch.arange(len(masks_pred)).to(device).long().view((-1, 1, 1))
                    ).float()[1:]
                continue
            losses += [mask_loss + depth_loss + plane_depth_loss + plane_loss]

            masks = results[-1]["mask"].squeeze(1)
            all_masks = torch.softmax(masks, dim=0)
            masks_small = all_masks[1:]
            all_masks = torch.nn.functional.interpolate(
                all_masks.unsqueeze(1), size=(480, 640), mode="bilinear"
            ).squeeze(1)
            all_masks = (
                all_masks.max(0, keepdim=True)[1]
                == torch.arange(len(all_masks)).to(device).long().view((-1, 1, 1))
            ).float()
            masks = all_masks[1:]
            detection_masks = torch.zeros(detection_dict["masks"].shape).to(device)
            detection_masks[:, 80:560] = masks
            detection_dict["masks"] = detection_masks
            results[-1]["mask"] = masks_small

            camera = camera.squeeze(0)

            if "refine_after" in self.options.suffix:
                ## Build the warping loss upon refined results
                XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(
                    self.config,
                    camera,
                    detections,
                    detection_masks,
                    detection_dict["depth_np"],
                    return_individual=True,
                )
                detection_dict["XYZ"] = XYZ_pred
        else:
            losses += [torch.zeros(1).to(device)]

        ## The warping loss
        for c in range(1, 2):
            if "warping" not in self.options.suffix:
                break

            detection_dict = detection_pair[1 - c]
            neighbor_info = torch.cat(
                [
                    detection_dict["XYZ"],
                    detection_dict["mask"],
                    input_pair[1 - c]["image"][0],
                ],
                dim=0,
            ).unsqueeze(0)
            warped_info, valid_mask = warpModuleDepth(
                self.config,
                camera,
                input_pair[c]["depth"][0],
                neighbor_info,
                input_pair[c]["extrinsics"][0],
                input_pair[1 - c]["extrinsics"][0],
                width=self.config.IMAGE_MAX_DIM,
                height=self.config.IMAGE_MIN_DIM,
            )

            XYZ = warped_info[:3].view((3, -1))
            XYZ = torch.cat([XYZ, torch.ones((1, int(XYZ.shape[1]))).to(device)], dim=0)
            transformed_XYZ = torch.matmul(
                input_pair[c]["extrinsics"][0],
                torch.matmul(input_pair[1 - c]["extrinsics"][0].inverse(), XYZ),
            )
            transformed_XYZ = transformed_XYZ[:3].view(detection_dict["XYZ"].shape)
            warped_depth = transformed_XYZ[1:2]
            warped_images = warped_info[4:7].unsqueeze(0)
            warped_mask = warped_info[3]

            with torch.no_grad():
                valid_mask = valid_mask * (input_pair[c]["depth"] > 1e-4).float()
                pass

            warped_depth_loss = l1LossMask(
                warped_depth, input_pair[c]["depth"], valid_mask
            )
            losses += [warped_depth_loss]

            if "warping1" in self.options.suffix or "warping3" in self.options.suffix:
                warped_mask_loss = l1LossMask(
                    warped_mask,
                    (input_pair[c]["segmentation"] >= 0).float(),
                    valid_mask,
                )
                losses += [warped_mask_loss]
                pass

            if "warping2" in self.options.suffix or "warping3" in self.options.suffix:
                warped_image_loss = l2NormLossMask(
                    warped_images, input_pair[c]["image"], dim=1, valid_mask=valid_mask
                )
                losses += [warped_image_loss]
                pass

            input_pair[c]["warped_depth"] = (
                warped_depth * valid_mask + (1 - valid_mask) * 10
            ).squeeze()
            continue
        loss = sum(losses)

        return loss

    @staticmethod
    def get_input_pair(batch):
        (
            images,
            _, # image_metas,
            _, # rpn_match,
            _, # rpn_bbox,
            _, # gt_class_ids,
            gt_boxes,
            gt_masks,
            _, # gt_parameters,
            gt_depth,
            extrinsics,
            gt_segmentation,
            camera,
        ) = batch
        return [
            {
                "image": images,
                "depth": gt_depth,
                "mask": gt_masks,
                "bbox": gt_boxes,
                "extrinsics": extrinsics,
                "segmentation": gt_segmentation,
                "camera": camera,
            }
        ]

    def get_input_pair_for_batch(self, batch):
        input_pair = []
        for data in batch:
            input_pair.append(self.get_input_pair(data)[0])

        return input_pair

    def get_detection_pair_for_batch(self, batch, output, device="cuda"):
        detection_pair = []
        for i in range(len(batch)):
            detection_pair.append(self.get_detection_pair(batch[i], output[i], device)[0])

        return detection_pair

    def get_detection_pair(self, batch, output, device="cuda"):
        (
            _, # rpn_class_logits,
            _, # rpn_pred_bbox,
            _, # target_class_ids,
            _, # mrcnn_class_logits,
            _, # target_deltas,
            _, # mrcnn_bbox,
            _, # target_mask,
            _, # mrcnn_mask,
            _, # target_parameters,
            _, # mrcnn_parameters,
            detections,
            detection_masks,
            _, # detection_gt_parameters,
            _, # detection_gt_masks,
            _, # rpn_rois,
            _, # roi_features,
            _, # roi_indices,
            depth_np_pred,
        ) = output

        if len(detections) > 0:
            detections, detection_masks = unmoldDetections(
                self.config,
                batch[-1], # camera
                detections,
                detection_masks,
                depth_np_pred,
                None, # normal_np_pred,
                debug=False,
            )
            if "refine_only" in self.options.suffix:
                detections, detection_masks = (
                    detections.detach(),
                    detection_masks.detach(),
                )
                pass
            XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(
                self.config,
                batch[-1], # camera
                detections,
                detection_masks,
                depth_np_pred,
                return_individual=True,
            )
            detection_mask = detection_mask.unsqueeze(0)
        else:
            XYZ_pred = torch.zeros(
                (3, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
            ).to(device)
            detection_mask = torch.zeros(
                (1, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
            ).to(device)
            plane_XYZ = torch.zeros(
                (1, 3, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
            ).to(device)

        return [
            {
                "XYZ": XYZ_pred,
                "depth": XYZ_pred[1:2],
                "mask": detection_mask,
                "detection": detections,
                "masks": detection_masks,
                "plane_XYZ": plane_XYZ,
                "depth_np": depth_np_pred,
            }
        ]
