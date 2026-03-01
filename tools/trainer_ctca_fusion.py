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
from clft_ctca_fusion.clft import CLFT
from utils.helpers import EarlyStopping
from utils.helpers import save_model_dict
from utils.helpers import adjust_learning_rate_clft
from utils.helpers import adjust_learning_rate_clfcn
from torch.amp import autocast, GradScaler

import os, csv
from pathlib import Path
from tools.clft_ctca_reg_utils import cwt_reg_losses


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

        # 메모리 절약
        self.use_amp = True
        self.scaler = GradScaler(enabled=self.use_amp)

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

        
        # ctca gate ep당 평균 로깅용
        self.ctca_csv_path = Path(self.config["General"].get("log_dir", "./logs")) / "ctca_gate_epoch.csv"
        self.ctca_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 헤더 한 번만 쓰기
        if not self.ctca_csv_path.exists():
            with open(self.ctca_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch", "lr",
                    "train_loss", "train_iou_vehicle", "train_iou_human",
                    "val_loss", "val_iou_vehicle", "val_iou_human",
                    # CTCA gate stats
                    "S2_g_mean", "S2_g_std", "S2_d_norm", "S2_upd_norm",
                    "S0_g_mean", "S0_g_std", "S0_d_norm", "S0_upd_norm",
                    "S2_g_min","S2_g_max","S2_gstd_min","S2_gstd_max","S2_d_min","S2_d_max","S2_upd_min","S2_upd_max",
                    "S0_g_min","S0_g_max","S0_gstd_min","S0_gstd_max","S0_d_min","S0_d_max","S0_upd_min","S0_upd_max",
                ])


    # cam uncertainty용
    # def _token_aux_ce(self, aux_logits_tok, anno, Hp, Wp):
    #     """
    #     aux_logits_tok: (B,N,K)
    #     anno: (B,H,W)  (long)
    #     return: scalar CE
    #     """
    #     if aux_logits_tok is None:
    #         return None

    #     B, N, K = aux_logits_tok.shape
    #     assert N == Hp * Wp, f"N({N}) != Hp*Wp({Hp*Wp})"

    #     # (B,N,K) -> (B,K,Hp,Wp)
    #     aux_logits_hw = aux_logits_tok.transpose(1, 2).contiguous().view(B, K, Hp, Wp)

    #     # anno downsample to (B,Hp,Wp)
    #     anno_ds = anno.unsqueeze(1).float()  # (B,1,H,W)
    #     anno_ds = F.interpolate(anno_ds, size=(Hp, Wp), mode="nearest").squeeze(1).long()

    #     return self.criterion(aux_logits_hw, anno_ds)

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

            # ep당 gate 평균 로깅용
            # g_sum = [0.0] * len(self.model.fusions)
            # s_sum = [0.0] * len(self.model.fusions)
            # cnt   = [0]   * len(self.model.fusions)

            # ep당 ctca gate 평균 로깅용
            g_sum = {"S2": 0.0, "S0": 0.0}
            s_sum = {"S2": 0.0, "S0": 0.0}
            d_sum = {"S2": 0.0, "S0": 0.0}   # delta norm sum
            u_sum = {"S2": 0.0, "S0": 0.0}   # update norm sum
            cnt   = {"S2": 0,   "S0": 0}

            # ep당 ctca gate min/max용
            INF = 1e9
            mins = {
            "S2": {"g": INF, "gs": INF, "d": INF, "u": INF},
            "S0": {"g": INF, "gs": INF, "d": INF, "u": INF},
            }
            maxs = {
            "S2": {"g": -INF, "gs": -INF, "d": -INF, "u": -INF},
            "S0": {"g": -INF, "gs": -INF, "d": -INF, "u": -INF},
            }

            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                self.optimizer_clft.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp, device_type="cuda"):
                    # _, output_seg = self.model(batch['rgb'], batch['lidar'], modality)

                    # cls token 정규화용 extras
                    _, output_seg, extras = self.model(batch['rgb'], batch['lidar'], modality, return_extras=True)

                    # cam uncertainty용
                    # _, output_seg, extras = self.model(batch['rgb'], batch['lidar'], modality)

                # 1xHxW -> HxW
                # output_seg = output_seg.squeeze(1)
                anno = batch['anno']
                if anno.ndim == 4 and anno.size(1) == 1:
                    anno = anno.squeeze(1)   # (B,1,H,W)->(B,H,W)
                
                # print("output_seg : ", output_seg.shape)
                # print("batch['anno'] : ", batch["anno"].shape)

                with torch.no_grad():
                    pred_for_metric = output_seg.detach().float()
                    batch_overlap, batch_pred, batch_label, batch_union = find_overlap(self.nclasses, pred_for_metric, anno)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                with autocast(enabled=self.use_amp, device_type="cuda"):
                    # loss = self.criterion(output_seg, anno)
                    loss_ce = self.criterion(output_seg, anno)

                    # cls token slot 정규화항
                    if extras is not None and 'HpWp' in extras and 'tok_cam_s2' in extras:
                        Hp, Wp = extras['HpWp']
                        gt_hw = anno            # (B,H,W)
                        class_ids = [2, 1]      # slot0=person, slot1=car (no reorder; just a target mapping)

                        # stage2 reg
                        reg_s2 = cwt_reg_losses(
                            tok_cam=extras['tok_cam_s2'], w_cam=extras['w_cam_s2'],
                            tok_lidar=extras['tok_xyz_s2'], w_lidar=extras['w_xyz_s2'],
                            gt_hw=gt_hw, Hp=Hp, Wp=Wp,
                            class_ids=class_ids,
                            ignore_id=3,
                            lambda_purity=1.0,
                            lambda_infonce=1.0,
                            temperature=0.07,
                        )

                        # stage0 reg
                        reg_s0 = cwt_reg_losses(
                            tok_cam=extras['tok_cam_s0'], w_cam=extras['w_cam_s0'],
                            tok_lidar=extras['tok_xyz_s0'], w_lidar=extras['w_xyz_s0'],
                            gt_hw=gt_hw, Hp=Hp, Wp=Wp,
                            class_ids=class_ids,
                            ignore_id=3,
                            lambda_purity=1.0,
                            lambda_infonce=1.0,
                            temperature=0.07,
                        )

                        loss_reg = 0.01 * (reg_s2["loss_reg_total"] + reg_s0["loss_reg_total"])
                        loss = loss_ce + loss_reg

                        if i % 100 == 0:
                            print(f"[LOSS] ce={loss_ce.item():.4f} reg={loss_reg.item():.4f} total={loss.item():.4f}")

                    # cam uncertainty loss
                    # Hp, Wp = extras["HpWp"]

                    # aux2 = self._token_aux_ce(extras.get("aux_logits_s2", None), anno, Hp, Wp)
                    # aux0 = self._token_aux_ce(extras.get("aux_logits_s0", None), anno, Hp, Wp)

                    # lam_aux = 0.05
                    # aux_sum = 0.0
                    # if aux2 is not None: aux_sum = aux_sum + aux2
                    # if aux0 is not None: aux_sum = aux_sum + aux0

                    # loss = loss + lam_aux * aux_sum
                # w_rgb = 1.1
                # w_lid = 0.9
                # loss = w_rgb*loss_rgb + w_lid*loss_lidar + loss_fusion

                # amp dbg
                # if not torch.isfinite(loss):
                #     print("non-finite loss!", loss.item())
                #     break


                train_loss += loss.item()
                # loss.backward()
                # self.optimizer_clft.step()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_clft)
                self.scaler.update()
                progress_bar.set_description(f'CLFT train loss:{loss:.4f}')

                # amp dbg
                # if i % 100 == 0:
                #     print("scale:", self.scaler.get_scale())

                # gate logging
                # for si, fus in enumerate(self.model.fusions):   # model이 CLFT 인스턴스
                #     d = getattr(fus, "last_log", None)
                #     if d is None or d["g"] is None:
                #         continue
                #     # 스텝별 확인용
                #     if i % 200 == 0:
                #         # stage별 fusion 통계를 보고 싶으면 fusions[i] 각각 찍기
                #         print(f"[FUSION] | S{si}: g={d['g']:.3f} scale={d['scale']:.3f}")
                #     # ep당 한번 평균 출력용
                #     g_sum[si] += d["g"]
                #     s_sum[si] += d["scale"]
                #     cnt[si] += 1

                # ctca gate logging ep당 평균용
                tg = getattr(self.model, "last_tokgate", None)
                if tg is not None:
                    for st in ["S2", "S0"]:
                        d = tg.get(st, None)
                        if d is None: 
                            continue
                        gm = d.get("g_mean", None)
                        gs = d.get("g_std", None)
                        dn = d.get("d_norm", None)
                        un = d.get("upd_norm", None)
                        if gm is None or gs is None:
                            continue
                        g_sum[st] += float(gm)
                        s_sum[st] += float(gs)
                        d_sum[st] += float(dn)
                        u_sum[st] += float(un)
                        mins[st]["g"]  = min(mins[st]["g"],  float(gm));  maxs[st]["g"]  = max(maxs[st]["g"],  float(gm))
                        mins[st]["gs"] = min(mins[st]["gs"], float(gs));  maxs[st]["gs"] = max(maxs[st]["gs"], float(gs))
                        mins[st]["d"]  = min(mins[st]["d"],  float(dn));  maxs[st]["d"]  = max(maxs[st]["d"],  float(dn))
                        mins[st]["u"]  = min(mins[st]["u"],  float(un));  maxs[st]["u"]  = max(maxs[st]["u"],  float(un))
                        cnt[st]   += 1

                # ctca gate logging
                if (i % 10) == 0:
                    tg = getattr(self.model, "last_tokgate", None)
                    if tg is not None:
                        s2 = tg.get("S2", None)
                        s0 = tg.get("S0", None)

                        def fmt(stage_dict, name):
                            if stage_dict is None:
                                return f"{name}: None"
                            gm = stage_dict.get("g_mean", None)
                            gs = stage_dict.get("g_std", None)
                            dn = stage_dict.get("d_norm", None)
                            un = stage_dict.get("upd_norm", None)
                            if gm is None or gs is None:
                                return f"{name}: (empty)"
                            if dn is None or un is None:
                                return f"{name}: g_mean={gm:.6f} g_std={gs:.6f}"
                            return f"{name}: g_mean={gm:.6f} g_std={gs:.6f} | d_norm={dn:.6f} upd_norm={un:.6f}"

                        print("[CTCA_STEP]", fmt(s2, "S2"), "|", fmt(s0, "S0"))
                    
            # 평균 출력
            # logs = []
            # for si in range(len(self.model.fusions)):
            #     if cnt[si] > 0:
            #         logs.append(f"S{si}: g={g_sum[si]/cnt[si]:.3f} scale={s_sum[si]/cnt[si]:.3f}")
            # print(f"[FUSION_EPOCH_AVG][alpha={self.model.fusions[0].alpha}]", " | ".join(logs))

            # 평균 출력
            # logs = []
            # for st in ["S2", "S0"]:
            #     if cnt[st] > 0:
            #         logs.append(f"{st}: g_mean={g_sum[st]/cnt[st]:.6f} g_std={s_sum[st]/cnt[st]:.6f} (n={cnt[st]})")
            # print("[CTCA_EPOCH_AVG]", " | ".join(logs))
            
            # The IoU of one epoch
            train_epoch_IoU = overlap_cum / union_cum
            print(f'Training vehicles IoU for Epoch: {train_epoch_IoU[0]:.4f}')
            print(f'Training human IoU for Epoch: {train_epoch_IoU[1]:.4f}')
            # The loss_rgb of one epoch
            train_epoch_loss = train_loss / (i + 1)
            print(f'Average Training Loss for Epoch: {train_epoch_loss:.4f}')

            valid_epoch_loss, valid_epoch_IoU = self.validate_clft(valid_dataloader, modality)

            # 평균출력
            def avg(st, arr_sum, arr_cnt):
                return (arr_sum[st] / arr_cnt[st]) if arr_cnt[st] > 0 else float("nan")

            S2_gm = avg("S2", g_sum, cnt); S2_gs = avg("S2", s_sum, cnt)
            S2_dn = avg("S2", d_sum, cnt); S2_un = avg("S2", u_sum, cnt)
            S0_gm = avg("S0", g_sum, cnt); S0_gs = avg("S0", s_sum, cnt)
            S0_dn = avg("S0", d_sum, cnt); S0_un = avg("S0", u_sum, cnt)

            print(
                f"[CTCA_EPOCH_AVG] "
                f"S2 g={S2_gm:.6f}±{S2_gs:.6f} d={S2_dn:.6f} upd={S2_un:.6f} | "
                f"S0 g={S0_gm:.6f}±{S0_gs:.6f} d={S0_dn:.6f} upd={S0_un:.6f}"
            )

            # --- CSV append (after we have everything) ---
            veh_tr = float(train_epoch_IoU[0].item())
            hum_tr = float(train_epoch_IoU[1].item())
            veh_va = float(valid_epoch_IoU[0].item())
            hum_va = float(valid_epoch_IoU[1].item())

            row = [
                epoch, float(lr),
                float(train_epoch_loss), veh_tr, hum_tr,
                float(valid_epoch_loss), veh_va, hum_va,
                float(S2_gm), float(S2_gs), float(S2_dn), float(S2_un),
                float(S0_gm), float(S0_gs), float(S0_dn), float(S0_un),
            ]

            row += [
            mins["S2"]["g"],  maxs["S2"]["g"], mins["S2"]["gs"], maxs["S2"]["gs"], 
            mins["S2"]["d"],  maxs["S2"]["d"], mins["S2"]["u"],  maxs["S2"]["u"],
            mins["S0"]["g"],  maxs["S0"]["g"], mins["S0"]["gs"], maxs["S0"]["gs"], 
            mins["S0"]["d"],  maxs["S0"]["d"], mins["S0"]["u"],  maxs["S0"]["u"],
            ]

            with open(self.ctca_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"[EPOCH_CSV] appended -> {self.ctca_csv_path}")

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
                    _, output_seg = self.model(batch['rgb'], batch['lidar'], modal)

                    # cam uncertainty용
                    # _, output_seg, extras = self.model(batch['rgb'], batch['lidar'], modal)

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
                    loss = self.criterion(output_seg, anno)

                    # cam uncertainty loss
                    # Hp, Wp = extras["HpWp"]

                    # aux2 = self._token_aux_ce(extras.get("aux_logits_s2", None), anno, Hp, Wp)
                    # aux0 = self._token_aux_ce(extras.get("aux_logits_s0", None), anno, Hp, Wp)

                    # lam_aux = 0.1
                    # aux_sum = 0.0
                    # if aux2 is not None: aux_sum = aux_sum + aux2
                    # if aux0 is not None: aux_sum = aux_sum + aux0

                    # loss = loss + lam_aux * aux_sum

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
