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

from tools.clft_pgdp_gt import PGDPBuilderGT
from tools.clft_pgdp_loss import PGDPLoss
from tools.clft_pgdp_loss import dice_loss_softmax


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

        # pgdp gt&loss
        self.pgdp_gt_builder = PGDPBuilderGT(refine=True)
        self.pgdp_loss_fn = PGDPLoss()

        self.pfim_lambda = 0.01
        self.pgdp_lambda = 0.1

        # human-only dice
        self.dice_lambda = 0.1  
        self.human_class_id = 2

        # 메모리 절약
        self.use_amp = False
        self.scaler = GradScaler(enabled=self.use_amp)

        # ===== Logs directory (relative to trainer.py) =====
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logs_dir = os.path.abspath(os.path.join(base_dir, "../logs"))
        self.log_path = os.path.join(self.logs_dir, "pp_diagnostics.csv")

        self.pp_csv_header = [
            "epoch",
            "pfim_loss",
            "pgdp_total",
            "info_mean",
            "info_std",
            "info_p95",
            "info_p95_max",
            "pfim_gain_mean",
            "pfim_gain_p95",
            "pfim_gain_p95_max",
            "pfim_p_mean",
            "pfim_clamp_ratio",
            "pfim_mae_mu_x",
            "p0_mean",
            "p0_p95",
            "p0_p95_max",
            "pgdp_p0_fg_min",
            "pgdp_p0_fg_max",
            "delta_ratio",
            "y1_ratio",
            "y2_ratio",
            "pix_ratio_h",
            "pix_ratio_v",
        ]

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(self.pp_csv_header)

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
        for epoch in range(self.finished_epochs, epochs):
            lr = adjust_learning_rate_clft(self.config, self.optimizer_clft, epoch)
            print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, lr))
            print('Training...')
            train_loss = 0.0
            overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
            progress_bar = tqdm(train_dataloader)

            # ===== PFIM/PGDP epoch diagnostics accumulator =====
            diag_sum = {k: 0.0 for k in self.pp_csv_header if k != "epoch"}
            diag_cnt = 0

            # epoch 내 runaway/포화를 놓치지 않기 위한 max tracker
            diag_max = {
                "info_p95_max": 0.0,
                "pfim_gain_p95_max": 0.0,
                "p0_p95_max": 0.0,
                "pgdp_p0_fg_max": 0.0,
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
                    _, output_seg, extras = self.model(batch['rgb'], batch['lidar'], modality)

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

                    # human-only Dice
                    # dice_loss = dice_loss_softmax(
                    #     logits=output_seg,
                    #     target=anno,
                    #     class_id=self.human_class_id,
                    #     ignore_index=4,
                    # )

                    # dice_applied = self.dice_lambda * dice_loss

                    # PFIM loss (scalar)
                    pfim_loss = extras["pfim_loss"]

                    # PGDP GT & loss
                    gt = self.pgdp_gt_builder(anno)              # {"M_gt","weights","valid"}
                    pgdp_loss, pgdp_logs = self.pgdp_loss_fn(
                        extras["p2"], extras["p1"], extras["p0"], gt
                    )                                           # returns (total, logs)

                    pfim_applied = self.pfim_lambda * pfim_loss
                    pgdp_applied = self.pgdp_lambda * pgdp_loss


                    # total
                    # loss = ce_loss + dice_applied + pfim_applied + pgdp_applied

                    loss = ce_loss + pfim_applied + pgdp_applied

                    # # ===== Human stabilization losses (FP suppression + Dice) =====
                    # HUMAN_ID = 2

                    # lambda_absent = 0.05   # GT에 human 없는 이미지에서 human FP 억제 (0.1~0.5 추천)
                    # lambda_dice   = 0.1   # GT에 human 있는 이미지에서 recall 올리기 (0.3~1.0 추천)
                    # eps = 1e-6

                    # # (B,) human 존재 여부
                    # with torch.no_grad():
                    #     has_human = (anno == HUMAN_ID).view(anno.size(0), -1).any(dim=1)

                    # # softmax prob
                    # prob = torch.softmax(output_seg, dim=1)            # (B,C,H,W)
                    # p_h  = prob[:, HUMAN_ID]                           # (B,H,W)

                    # # 1) Absent penalty: human 없는 이미지에서 human 확률 평균을 줄임 (FP 폭주 방지)
                    # if (~has_human).any():
                    #     p_abs = p_h[~has_human]                       # (Nabs,H,W)
                    #     absent_penalty = torch.quantile(p_abs.reshape(p_abs.size(0), -1), 0.95, dim=1).mean()
                    # else:
                    #     absent_penalty = output_seg.new_tensor(0.0)

                    # # 2) Human Dice: human 있는 이미지에서만 dice로 맞추게 함 (pred=0 방지)
                    # g_h = (anno == HUMAN_ID).float()                   # (B,H,W)

                    # if has_human.any():
                    #     p = p_h[has_human].reshape(has_human.sum(), -1)
                    #     g = g_h[has_human].reshape(has_human.sum(), -1)
                    #     inter = (p * g).sum(dim=1)
                    #     dice = (2.0 * inter + eps) / (p.sum(dim=1) + g.sum(dim=1) + eps)
                    #     dice_loss = 1.0 - dice.mean()
                    # else:
                    #     dice_loss = output_seg.new_tensor(0.0)

                    # # 최종 loss 합치기
                    # loss = ce_loss + pfim_loss + pgdp_loss \
                    #     + lambda_absent * absent_penalty \
                    #     + lambda_dice   * dice_loss



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

                    if i % 50 == 0:
                        print(f"[PIX] gt_v={gt_v} pred_v={pr_v} | gt_h={gt_h} pred_h={pr_h} | "
                              f"ratio_v={ratio_v:.3f} ratio_h={ratio_h:.3f}")
                        # print(f"CE={ce_loss.item():.4f} "
                        # f"PFIM={(self.pfim_lambda * pfim_loss).item():.4f} "
                        # f"PGDP={(self.pgdp_lambda * pgdp_loss).item():.4f}")
                        # print(f"[HSTAT] gt={gt_h} pred={pr_h} inter={inter_h} "
                        #     f"absent_pen={float(absent_penalty):.6f} dice_loss={float(dice_loss):.6f}")

                    if i % 50 == 0:
                        # weighted shares
                        ce_w = ce_loss.item()
                        pfim_w = self.pfim_lambda * pfim_loss.item()
                        pgdp_w = self.pgdp_lambda * pgdp_loss.item()
                        # dice_w = self.dice_lambda * dice_loss.item()

                        # tot = ce_w + pfim_w + pgdp_w + dice_w + 1e-12
                        tot = ce_w + pfim_w + pgdp_w + 1e-12

                        dpf = getattr(self.model.pfim, "diag", None)
                        dpg = getattr(self.model.pgdp, "diag", None)

                        # print(f"[LOSS] ce={ce_w:.4f} pfim_w={pfim_w:.4f} pgdp_w={pgdp_w:.4f} dice_w={dice_w:.4f} | "
                        #     f"pfim_share={pfim_w/tot:.2%} pgdp_share={pgdp_w/tot:.2%} dice_share={dice_w/tot:.2%}")

                        print(f"[LOSS] ce={ce_w:.4f} pfim_w={pfim_w:.4f} pgdp_w={pgdp_w:.4f} | "
                            f"pfim_share={pfim_w/tot:.2%} pgdp_share={pgdp_w/tot:.2%}")

                        if dpf is not None:
                            print(f"[PFIM] info(m,p95,min)={dpf['info_mean']:.3f},{dpf['info_p95']:.3f},{dpf['info_min']:.3f} | "
                                f"gain(m,p95)={dpf['gain_mean']:.3f},{dpf['gain_p95']:.3f} | "
                                f"z={dpf['z_abs_mean']:.2f} p(m)={dpf['p_mean']:.3f} clamp={dpf['p_clamp_ratio']:.2%} | "
                                f"mae_mu_x={dpf['mae_mu_x']:.3f}")

                        if dpg is not None:
                            print(f"[PGDP] fg(m,p95)={dpg['p0_fg_mean']:.3f},{dpg['p0_fg_p95']:.3f} | "
                                f"fg(min,max)={dpg['p0_fg_min']:.3f},{dpg['p0_fg_max']:.3f}")
                            
                # ===== accumulate PP diagnostics per batch (epoch average) =====
                with torch.no_grad():
                    dpf = getattr(self.model.pfim, "diag", None)  # PFIM diag dict (if exists)
                    dpg = getattr(self.model.pgdp, "diag", None)  # PGDP diag dict (if exists)

                    diag_sum["pfim_loss"] += _sf(pfim_loss)
                    diag_sum["pgdp_total"] += _sf(pgdp_loss)

                    # extras 기반(없으면 0)
                    diag_sum["delta_ratio"] += _sf(extras.get("delta_ratio", 0.0))
                    diag_sum["y1_ratio"]    += _sf(extras.get("y1_ratio", 0.0))
                    diag_sum["y2_ratio"]    += _sf(extras.get("y2_ratio", 0.0))

                    # PIX ratio (너가 이미 계산한 값)
                    diag_sum["pix_ratio_h"] += _sf(ratio_h)
                    diag_sum["pix_ratio_v"] += _sf(ratio_v)

                    if dpf is not None:
                        info_mean = _sf(dpf.get("info_mean", 0.0))
                        info_std  = _sf(dpf.get("info_std", 0.0))
                        info_p95  = _sf(dpf.get("info_p95", 0.0))
                        gain_mean = _sf(dpf.get("gain_mean", 0.0))
                        gain_p95  = _sf(dpf.get("gain_p95", 0.0))
                        p_mean    = _sf(dpf.get("p_mean", 0.0))
                        clamp_r   = _sf(dpf.get("p_clamp_ratio", 0.0))
                        mae_mu_x  = _sf(dpf.get("mae_mu_x", 0.0))

                        diag_sum["info_mean"] += info_mean
                        diag_sum["info_std"]  += info_std
                        diag_sum["info_p95"]  += info_p95
                        diag_sum["pfim_gain_mean"] += gain_mean
                        diag_sum["pfim_gain_p95"]  += gain_p95
                        diag_sum["pfim_p_mean"]    += p_mean
                        diag_sum["pfim_clamp_ratio"] += clamp_r
                        diag_sum["pfim_mae_mu_x"]  += mae_mu_x

                        diag_max["info_p95_max"] = max(diag_max["info_p95_max"], info_p95)
                        diag_max["pfim_gain_p95_max"] = max(diag_max["pfim_gain_p95_max"], gain_p95)

                    if dpg is not None:
                        p0_mean = _sf(dpg.get("p0_fg_mean", 0.0))
                        p0_p95  = _sf(dpg.get("p0_fg_p95", 0.0))
                        p0_min  = _sf(dpg.get("p0_fg_min", 0.0))
                        p0_max  = _sf(dpg.get("p0_fg_max", 0.0))

                        diag_sum["p0_mean"] += p0_mean
                        diag_sum["p0_p95"]  += p0_p95
                        diag_sum["pgdp_p0_fg_min"] += p0_min
                        diag_sum["pgdp_p0_fg_max"] += p0_max

                        diag_max["p0_p95_max"] = max(diag_max["p0_p95_max"], p0_p95)
                        diag_max["pgdp_p0_fg_max"] = max(diag_max["pgdp_p0_fg_max"], p0_max)

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
            diag_avg["info_p95_max"] = diag_max["info_p95_max"]
            diag_avg["pfim_gain_p95_max"] = diag_max["pfim_gain_p95_max"]
            diag_avg["p0_p95_max"] = diag_max["p0_p95_max"]
            diag_avg["pgdp_p0_fg_max"] = diag_max["pgdp_p0_fg_max"]

            # CSV append: header 순서대로 저장
            with open(self.log_path, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([epoch] + [diag_avg[k] for k in self.pp_csv_header if k != "epoch"])



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
                    _, output_seg, extras = self.model(batch['rgb'], batch['lidar'], modal)

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

                    # human-only Dice
                    # dice_loss = dice_loss_softmax(
                    #     logits=output_seg,
                    #     target=anno,
                    #     class_id=self.human_class_id,
                    #     ignore_index=4,
                    # )

                    # dice_applied = self.dice_lambda * dice_loss

                    # PFIM loss (scalar)
                    pfim_loss = extras["pfim_loss"]

                    # PGDP GT & loss
                    gt = self.pgdp_gt_builder(anno)              # {"M_gt","weights","valid"}
                    pgdp_loss, pgdp_logs = self.pgdp_loss_fn(
                        extras["p2"], extras["p1"], extras["p0"], gt
                    )                                           # returns (total, logs)

                    pfim_applied = self.pfim_lambda * pfim_loss
                    pgdp_applied = self.pgdp_lambda * pgdp_loss

                    # total
                    # loss = ce_loss + dice_applied + pfim_applied + pgdp_applied

                    loss = ce_loss + pfim_applied + pgdp_applied

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
