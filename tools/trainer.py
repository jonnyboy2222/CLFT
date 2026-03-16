#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from clfcn.fusion_net import FusionNet
from utils.metrics import find_overlap
from utils.metrics import find_overlap_1
from clft.clft import CLFT
from utils.helpers import EarlyStopping
from utils.helpers import save_model_dict
from utils.helpers import adjust_learning_rate_clft
from utils.helpers import adjust_learning_rate_clfcn
from torch.amp import autocast, GradScaler

import os, csv
from pathlib import Path

from tools.clft_gdmp_gt import GDMPBuilderGT
from tools.clft_gdmp_loss import GDMPLoss
from tools.clft_sdup_loss import _build_sdup_target
from tools.clft_gdmp_loss import dice_loss_softmax


writer = SummaryWriter()


class Trainer(object):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.finished_epochs = 0
        self.device = torch.device(self.config['General']['device']
                                   if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        # gdmp gt&loss
        self.gdmp_gt_builder = GDMPBuilderGT(refine=True)
        self.gdmp_loss_fn = GDMPLoss()

        # self.pfim_lambda = 0.1
        self.sdup_lambda = 0.05
        self.gdmp_lambda = 0.1

        # human-only dice
        self.dice_lambda = 0.1  
        self.human_class_id = 2

        # 메모리 절약
        self.use_amp = False
        self.scaler = GradScaler(enabled=self.use_amp)

        # ===== Logs directory (relative to trainer.py) =====
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logs_dir = os.path.abspath(os.path.join(base_dir, "../logs"))
        self.log_path = os.path.join(self.logs_dir, "gs_diagnostics.csv")

        self.gs_csv_header = [
            "epoch",
            "sdup_loss",
            "gdmp_total",
            "u_mean",
            "u_p95",
            "u_p95_max",
            "p0_mean",
            "p0_p95",
            "p0_p95_max",
            "delta_ratio",
            "pix_ratio_h",
            "pix_ratio_v",
        ]

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(self.gs_csv_header)

        if args.backbone == 'clfcn':
            self.model = FusionNet()
            print(f'Using backbone {args.backbone}')
            self.optimizer_clfcn = torch.optim.Adam(self.model.parameters(), lr=config['CLFCN']['clfcn_lr'])

        elif args.backbone == 'clft':
            resize = config['Dataset']['transforms']['resize']
            self.model = CLFT(RGB_tensor_size=(3, resize, resize),
                              XYZ_tensor_size=(3, resize, resize),
                              patch_size=config['CLFT']['patch_size'],
                              emb_dim=config['CLFT']['emb_dim'],
                              resample_dim=config['CLFT']['resample_dim'],
                              read=config['CLFT']['read'],
                              hooks=config['CLFT']['hooks'],
                              reassemble_s=config['CLFT']['reassembles'],
                              nclasses=len(config['Dataset']['classes']),
                              type=config['CLFT']['type'],
                              model_timm=config['CLFT']['model_timm'],)
            print(f'Using backbone {args.backbone}')
            self.optimizer_clft = torch.optim.Adam(self.model.parameters(), lr=config['CLFT']['clft_lr'])

        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")

        self.model.to(self.device)

        self.nclasses = len(config['Dataset']['classes'])
        weight_loss = torch.Tensor(self.nclasses).fill_(0)
        # define weight of different classes, 0-background, 1-car, 2-people.
        # weight_loss[3] = 10
        weight_loss[0] = 1
        weight_loss[1] = 4
        weight_loss[2] = 10
        self.criterion = nn.CrossEntropyLoss(weight=weight_loss).to(self.device)

        if self.config['General']['resume_training'] is True:
            print('Resume training...')
            model_path = self.config['General']['resume_training_model_path']
            checkpoint = torch.load(model_path, map_location=self.device)

            if self.config['General']['reset_lr'] is True:
                print('Reset the epoch to 0')
                self.finished_epochs = 0
            else:
                self.finished_epochs = checkpoint['epoch']
                print( f"Finished epochs in previous training: {self.finished_epochs}")

            if self.config['General']['epochs'] <= self.finished_epochs:
                print('Current epochs amount is smaller than finished epochs!!!')
                print(f"Please setting the epochs bigger than {self.finished_epochs}")
                sys.exit()
            else:
                print('Loading trained model weights...')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print('Loading trained optimizer...')
                self.optimizer_clft.load_state_dict(checkpoint['optimizer_state_dict'])
                self.optimizer_clfcn.load_state_dict(checkpoint['optimizer_state_dict'])

        else:
            print('Training from the beginning')


    def train_clft(self, train_dataloader, valid_dataloader, modal):
        """
        The training of one epoch
        """
        epochs = self.config['General']['epochs']
        modality = modal
        early_stopping = EarlyStopping(self.config)
        self.model.train()

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.finished_epochs, epochs):
            lr = adjust_learning_rate_clft(self.config, self.optimizer_clft, epoch)
            print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, lr))
            print('Training...')
            train_loss = 0.0
            overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
            progress_bar = tqdm(train_dataloader)

            # ===== PFIM/GDMP epoch diagnostics accumulator =====
            diag_sum = {k: 0.0 for k in self.gs_csv_header if k != "epoch"}
            diag_cnt = 0

            # epoch 내 runaway/포화를 놓치지 않기 위한 max tracker
            diag_max = {
                "u_p95_max": 0.0,
                "p0_p95_max": 0.0,
            }

            def _sf(x, default=0.0):
                # safe float
                try:
                    if x is None:
                        return default
                    if torch.is_tensor(x):
                        return float(x.detach().item())
                    return float(x)
                except Exception:
                    return default

            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                self.optimizer_clft.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp, device_type="cuda"):
                    _, output_seg, extras_rgb = self.model(batch['rgb'], batch['lidar'], modality)

                # 1xHxW -> HxW
                # output_seg = output_seg.squeeze(1)
                anno = batch['anno']
                if anno.ndim == 4 and anno.size(1) == 1:
                    anno = anno.squeeze(1)   # (B,1,H,W)->(B,H,W)


                with torch.no_grad():
                    pred_for_metric = output_seg.detach().float()
                    batch_overlap, batch_pred, batch_label, batch_union = find_overlap(self.nclasses, pred_for_metric, anno)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                with autocast(enabled=self.use_amp, device_type="cuda"):
                    ce_loss = self.criterion(output_seg, anno)

                    # GDMP GT & loss
                    gt = self.gdmp_gt_builder(anno)              # {"M_gt","weights","valid"}

                    gdmp_loss_rgb, gdmp_logs_rgb = self.gdmp_loss_fn(
                        extras_rgb["p2_rgb"], extras_rgb["p1_rgb"], extras_rgb["p0_rgb"], gt
                    )                                           # returns (total, logs)

                    # SDUP loss
                    u_map_rgb = extras_rgb["u_map_rgb"]   # (B,1,H0,W0)

                    target_u_rgb = _build_sdup_target(
                        output_seg=output_seg,
                        p0_rgb=extras_rgb["p0_rgb"],
                        u_map_rgb=u_map_rgb
                    )

                    sdup_loss_rgb = F.l1_loss(u_map_rgb, target_u_rgb)

                    gdmp_applied = self.gdmp_lambda * gdmp_loss_rgb
                    sdup_applied = self.sdup_lambda * sdup_loss_rgb


                    # total

                    loss = ce_loss + gdmp_applied + sdup_applied

                # dbg
                with torch.no_grad():
                    pred = output_seg.argmax(dim=1)  # (B,H,W)
                    gt = anno                          # (B,H,W)

                    gt_h = (gt == 2).sum().item()
                    pr_h = (pred == 2).sum().item()

                    gt_v = (gt == 1).sum().item()
                    pr_v = (pred == 1).sum().item()

                    ratio_v = pr_v / (gt_v+1)
                    ratio_h = pr_h / (gt_h+1)

                    inter_h = ((pred == 2) & (anno == 2)).sum().item()

                    if i % 20 == 0:
                        print(f"[PIX] gt_v={gt_v} pred_v={pr_v} | gt_h={gt_h} pred_h={pr_h} | "
                              f"ratio_v={ratio_v:.3f} ratio_h={ratio_h:.3f}")

                        # weighted shares (모달 평균으로 스케일 맞춤)
                        # ce_w = float(ce_loss.detach().item())

                        # gdmp_w = float(self.gdmp_lambda * 0.5 * (
                        #     gdmp_loss_rgb.detach().item() #+
                        #     # gdmp_loss_xyz.detach().item()
                        # ))

                        # tot = ce_w + gdmp_w + 1e-12

                        # ---- RGB scalar logs from extras ----
                        p0_m_rgb = float(extras_rgb.get("p0_mean_rgb", 0.0))
                        p0_p95_rgb = float(extras_rgb.get("p0_p95_rgb", 0.0))

                        u_mean_rgb = float(u_map_rgb.detach().mean().item())
                        u_p95_rgb = float(torch.quantile(u_map_rgb.detach().flatten(), 0.95).item())
                        tgt_mean_rgb = float(target_u_rgb.detach().mean().item())
                        tgt_p95_rgb = float(torch.quantile(target_u_rgb.detach().flatten(), 0.95).item())

                        # GDMP (rgb/xyz) : p0 fg stats는 extras에 들어있음
                        print(f"[GDMP_RGB] loss={float(gdmp_loss_rgb.detach().item()):.4f} | "
                            f"p0_fg(m,p95)={p0_m_rgb:.3f},{p0_p95_rgb:.3f}")
                        
                        # SDUP
                        print(f"[SDUP_RGB] loss={float(sdup_loss_rgb.detach().item()):.4f} | "
                            f"u(m,p95)={u_mean_rgb:.3f},{u_p95_rgb:.3f} | "
                            f"tgt(m,p95)={tgt_mean_rgb:.3f},{tgt_p95_rgb:.3f}")

                        # print(f"[GDMP_XYZ] loss={float(gdmp_loss_xyz.detach().item()):.4f} | "
                        #     f"p0_fg(m,p95)={p0_m_xyz:.3f},{p0_p95_xyz:.3f}")
                            
                # ===== accumulate PP diagnostics per batch (epoch average) =====
                with torch.no_grad():
                    # pfim_loss_mean = pfim_loss_rgb
                    gdmp_loss_mean = gdmp_loss_rgb

                    p0_mean = extras_rgb.get("p0_mean_rgb", 0.0)
                    p0_p95  = extras_rgb.get("p0_p95_rgb", 0.0)

                    delta_ratio = extras_rgb.get("delta_ratio_rgb", 0.0)

                    # CSV에 맞는 필드만 채움 (없는 필드는 0 유지)
                    diag_sum["gdmp_total"] += _sf(gdmp_loss_mean)

                    diag_sum["p0_mean"] += _sf(p0_mean)
                    diag_sum["p0_p95"]  += _sf(p0_p95)

                    diag_sum["delta_ratio"] += _sf(delta_ratio)

                    # PIX ratio
                    diag_sum["pix_ratio_h"] += _sf(ratio_h)
                    diag_sum["pix_ratio_v"] += _sf(ratio_v)

                    # max tracker도 평균 기준으로
                    diag_max["p0_p95_max"]   = max(diag_max["p0_p95_max"], _sf(p0_p95))

                    u_mean_rgb = u_map_rgb.detach().mean()
                    u_p95_rgb = torch.quantile(u_map_rgb.detach().flatten(), 0.95)

                    diag_sum["sdup_loss"] += _sf(sdup_loss_rgb)
                    diag_sum["u_mean"] += _sf(u_mean_rgb)
                    diag_sum["u_p95"]  += _sf(u_p95_rgb)

                    diag_max["u_p95_max"] = max(diag_max["u_p95_max"], _sf(u_p95_rgb))

                    diag_cnt += 1

                train_loss += loss.item()
                # loss.backward()
                # self.optimizer_clft.step()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_clft)
                self.scaler.update()
                progress_bar.set_description(f'CLFT train loss:{loss:.4f}')
            
            # The IoU of one epoch
            train_epoch_IoU = overlap_cum / union_cum
            print(f'Training vehicles IoU for Epoch: {train_epoch_IoU[0]:.4f}')
            print(f'Training human IoU for Epoch: {train_epoch_IoU[1]:.4f}')
            # The loss_rgb of one epoch
            train_epoch_loss = train_loss / (i + 1)
            print(f'Average Training Loss for Epoch: {train_epoch_loss:.4f}')

            valid_epoch_loss, valid_epoch_IoU = self.validate_clft(valid_dataloader, modality)


            # ===== finalize epoch diagnostics (avg + max) =====
            if diag_cnt > 0:
                diag_avg = {k: (diag_sum[k] / diag_cnt) for k in diag_sum.keys()}
            else:
                diag_avg = {k: 0.0 for k in diag_sum.keys()}

            # max 컬럼 채우기
            diag_avg["u_p95_max"] = diag_max["u_p95_max"]
            diag_avg["p0_p95_max"] = diag_max["p0_p95_max"]

            # CSV append: header 순서대로 저장
            with open(self.log_path, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([epoch] + [diag_avg[k] for k in self.gs_csv_header if k != "epoch"])



            # Plot the train and validation loss in Tensorboard
            writer.add_scalars('Loss', {'train': train_epoch_loss,
                                        'valid': valid_epoch_loss}, epoch)
            # Plot the train and validation IoU in Tensorboard
            writer.add_scalars('Vehicle_IoU', {'train': train_epoch_IoU[0],
                                               'valid': valid_epoch_IoU[0]}, epoch)
            writer.add_scalars('Human_IoU', {'train': train_epoch_IoU[1],
                                             'valid': valid_epoch_IoU[1]}, epoch)

            # early_stop_index = round(valid_epoch_IoU[0].item(), 4) # vehicle iou 넘겨주는 부분
            veh_iou = round(valid_epoch_IoU[0].item(), 4)   # vehicle IoU
            human_iou = round(valid_epoch_IoU[1].item(), 4) # human IoU

            early_stopping(veh_iou, human_iou, epoch, self.model, modality, self.optimizer_clft)

            save_epoch = self.config['General']['save_epoch']
            if (epoch + 1) % save_epoch == 0 and epoch > 0:
                print(f'Saving model for every {save_epoch} epochs...')
                save_model_dict(self.config, epoch, self.model, modality, self.optimizer_clft, True)
                print('Saving Model Complete')
            if early_stopping.early_stop_trigger is True:
                break
        writer.close()
        print('Training Complete')

    def validate_clft(self, valid_dataloader, modal):
        """
            The validation of one epoch
        """
        self.model.eval()
        print('Validating...')
        valid_loss = 0.0
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        with torch.no_grad():
            progress_bar = tqdm(valid_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp, device_type="cuda"):
                    _, output_seg, extras_rgb = self.model(batch['rgb'], batch['lidar'], modal)

                # 1xHxW -> HxW
                # output_seg = output_seg.squeeze(1)
                anno = batch['anno']
                if anno.ndim == 4 and anno.size(1) == 1:
                    anno = anno.squeeze(1)

                with torch.no_grad():
                    pred_for_metric = output_seg.detach().float()
                    batch_overlap, batch_pred, batch_label, batch_union = find_overlap(self.nclasses, pred_for_metric, anno)

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                with autocast(enabled=self.use_amp, device_type="cuda"):
                    ce_loss = self.criterion(output_seg, anno)

                    # gdmp GT & loss
                    gt = self.gdmp_gt_builder(anno)              # {"M_gt","weights","valid"}

                    gdmp_loss_rgb, gdmp_logs_rgb = self.gdmp_loss_fn(
                        extras_rgb["p2_rgb"], extras_rgb["p1_rgb"], extras_rgb["p0_rgb"], gt
                    )                                           # returns (total, logs)

                    u_map_rgb = extras_rgb["u_map_rgb"]

                    target_u_rgb = _build_sdup_target(
                        output_seg=output_seg,
                        p0_rgb=extras_rgb["p0_rgb"],
                        u_map_rgb=u_map_rgb
                    )

                    sdup_loss_rgb = F.l1_loss(u_map_rgb, target_u_rgb)

                    # pfim_applied = self.pfim_lambda * pfim_loss_rgb
                    gdmp_applied = self.gdmp_lambda * gdmp_loss_rgb
                    sdup_applied = self.sdup_lambda * sdup_loss_rgb



                    # total
                    # loss = ce_loss + dice_applied + pfim_applied + gdmp_applied

                    # loss = ce_loss + gdmp_applied
                    loss = ce_loss + gdmp_applied + sdup_applied

                valid_loss += loss.item()
                progress_bar.set_description(f'valid fusion loss: {loss:.4f}')
        # The IoU of one epoch
        valid_epoch_IoU = overlap_cum / union_cum
        print(f'Validation vehicles IoU for Epoch: {valid_epoch_IoU[0]:.4f}')
        print(f'Validation human IoU for Epoch: {valid_epoch_IoU[1]:.4f}')
        # The loss_rgb of one epoch
        valid_epoch_loss = valid_loss / (i + 1)
        print(f'Average Validation Loss for Epoch: {valid_epoch_loss:.4f}')

        return valid_epoch_loss, valid_epoch_IoU

    def train_clfcn(self, train_dataloader, valid_dataloader, modal):
        """
        The training of one epoch
        """
        epochs = self.config['General']['epochs']
        modality = modal
        early_stopping = EarlyStopping(self.config)
        self.model.train()
        for epoch in range(self.finished_epochs, epochs):
            lr = adjust_learning_rate_clfcn(self.config, self.optimizer_clfcn, epoch)
            print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, lr))
            print('Training...')
            train_loss = 0.0
            overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
            progress_bar = tqdm(train_dataloader)
            for i, batch in enumerate(progress_bar):
                train_loss = 0.0
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                self.optimizer_clfcn.zero_grad()
                outputs = self.model(batch['rgb'], batch['lidar'], modality)

                output = outputs[modality]
                annotation = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = find_overlap(self.nclasses, output, annotation)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                if modality == 'rgb':
                    loss_rgb = self.criterion(outputs['rgb'], batch['anno'])
                    train_loss += loss_rgb.item()
                    loss_rgb.backward()
                    self.optimizer_clfcn.step()
                    progress_bar.set_description(f'train rgb loss:{loss_rgb:.4f}')

                elif modality == 'lidar':
                    loss_lidar = self.criterion(outputs['lidar'], batch['anno'])
                    train_loss += loss_lidar.item()
                    loss_lidar.backward()
                    self.optimizer_clfcn.step()
                    progress_bar.set_description(f'train lidar loss:{loss_lidar:.4f}')

                elif modality == 'cross_fusion':
                    loss_rgb = self.criterion(outputs['rgb'], batch['anno'])
                    loss_lidar = self.criterion(outputs['lidar'], batch['anno'])
                    loss_fusion = self.criterion(outputs['cross_fusion'], batch['anno'])
                    loss_all = loss_rgb + loss_lidar + loss_fusion
                    train_loss += loss_all.item()
                    loss_all.backward()
                    self.optimizer_clfcn.step()
                    progress_bar.set_description(f'train fusion loss:{loss_all:.4f}')

            # The IoU of one epoch
            train_epoch_IoU = overlap_cum / union_cum
            print( f'Training IoU of vehicles for Epoch: {train_epoch_IoU[0]:.4f}')
            print(f'Training IoU of human for Epoch: {train_epoch_IoU[1]:.4f}')
            # The loss_rgb of one epoch
            train_epoch_loss = train_loss / (i+1)
            print(f'Average Training Loss for Epoch: {train_epoch_loss:.4f}')

            valid_epoch_loss, valid_epoch_IoU = self.validate_clfcn(valid_dataloader, modality)

            # Plot the train and validation loss in Tensorboard
            writer.add_scalars('Loss', {'train': train_epoch_loss,
                                        'valid': valid_epoch_loss}, epoch)
            # Plot the train and validation IoU in Tensorboard
            writer.add_scalars('Vehicle_IoU', {'train': train_epoch_IoU[0],
                                               'valid': valid_epoch_IoU[0]}, epoch)
            writer.add_scalars('Human_IoU', {'train': train_epoch_IoU[1],
                                             'valid': valid_epoch_IoU[1]}, epoch)
            writer.close()

            early_stop_index = round(valid_epoch_IoU[0].item(), 4)
            early_stopping(early_stop_index, epoch, self.model, modality, self.optimizer_clfcn)
            save_epoch = self.config['General']['save_epoch']
            if (epoch + 1) % save_epoch == 0 and epoch > 0:
                print(f'Saving model for every {save_epoch} epochs...')
                save_model_dict(self.config, epoch, self.model, modality, self.optimizer_clfcn, True)
                print('Saving Model Complete')
            if early_stopping.early_stop_trigger is True:
                break
        print('Training Complete')

    def validate_clfcn(self, valid_dataloader, modal):
        """
        The validation of one epoch
        """
        self.model.eval()
        print('Validating...')
        modality = modal
        valid_loss = 0.0
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        with torch.no_grad():
            progress_bar = tqdm(valid_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True).squeeze(1)

                outputs = self.model(batch['rgb'], batch['lidar'], modality)

                output = outputs[modality]
                annotation = batch['anno']
                batch_overlap, batch_pred, batch_label, batch_union = find_overlap(self.nclasses, output, annotation)

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                if modality == 'rgb':
                    loss_rgb = self.criterion(outputs['rgb'], batch['anno'])
                    valid_loss += loss_rgb.item()
                    progress_bar.set_description(f'valid rgb loss:{loss_rgb:.4f}')

                elif modality == 'lidar':
                    loss_lidar = self.criterion(outputs['lidar'], batch['anno'])
                    valid_loss += loss_lidar.item()
                    progress_bar.set_description(f'valid lidar loss:{loss_lidar:.4f}')

                elif modality == 'cross_fusion':
                    loss_rgb = self.criterion(outputs['rgb'], batch['anno'])
                    loss_lidar = self.criterion(outputs['lidar'], batch['anno'])
                    loss_fusion = self.criterion(outputs['cross_fusion'], batch['anno'])
                    loss_all = loss_rgb + loss_lidar + loss_fusion
                    valid_loss += loss_all.item()
                    progress_bar.set_description(f'valid fusion loss:{loss_all:.4f}')
        # The IoU of one epoch
        valid_epoch_IoU = overlap_cum / union_cum
        print(f'Validatoin IoU of vehicles for Epoch: {valid_epoch_IoU[0]:.4f}')
        print(f'Validatoin IoU of human for Epoch: {valid_epoch_IoU[1]:.4f}')
        # The loss_rgb of one epoch
        valid_epoch_loss = valid_loss / (i+1)
        print(f'Average Validation Loss for Epoch: {valid_epoch_loss:.4f}')

        return valid_epoch_loss, valid_epoch_IoU
