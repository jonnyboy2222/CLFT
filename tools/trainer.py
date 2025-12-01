#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
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

# focal loss
# import torch.nn.functional as F
# class FocalLoss(nn.Module):
#     # increasing gamma reduces more contribution of good ones
#     # alpha balances the importance of different classes
#     # reduction specifies how to aggregate the loss
#     def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # inputs: BCHW
#         # targets: BHW
#         ce_loss = F.cross_entropy(inputs, targets, reduction="none")
#         pt = torch.exp(-ce_loss)  # pt = softmax 결과의 정답 클래스 확률

#         focal_loss = (1 - pt) ** self.gamma * ce_loss

#         if self.alpha is not None:
#             at = self.alpha[targets]
#             focal_loss = focal_loss * at

#         return focal_loss.mean() if self.reduction == "mean" else focal_loss


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

        # focal loss
        # alpha_vec = torch.ones(self.nclasses, device=self.device)
        # alpha_vec[2] = 2.0 # human class에 2배 가중치
        # self.focal_loss = FocalLoss(
        #     gamma=2.0,
        #     alpha=alpha_vec,
        #     reduction="none" # per pixel로 받은 후 human만 골라서 평균낼 예정
        # )
        # self.lambda_focal = 0.5 # L = CE + 0.5 * focal

        # # --- Aux human head용 focal loss 설정 ---
        # # human class index (0:bg, 1:vehicle, 2:human 이라고 가정)
        # self.human_class_idx = 2

        # # aux head는 binary (non-human vs human)
        # # alpha[0]=1.0 (non-human), alpha[1]=2.0 (human에 약간 더 가중)
        # aux_alpha = torch.tensor([1.0, 2.0], device=self.device)
        # self.aux_focal_loss = FocalLoss(
        #     gamma=2.0,
        #     alpha=aux_alpha,
        #     reduction="mean"
        # )

        # # aux head loss 비율 (메인 CE 대비)
        # self.lambda_aux = 0.5  # 나중에 0.3~1.0 사이에서 튜닝 가능

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
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                self.optimizer_clft.zero_grad()
                # 보조헤드 -> aux_human_logits 추가
                out_depth, output_seg = self.model(batch['rgb'], batch['lidar'], modality)

                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = find_overlap(self.nclasses, output_seg, anno)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                # # --- 메인 CE loss ---
                # ce_loss = self.criterion(output_seg, anno)

                # # --- Aux human focal loss ---
                # aux_loss = 0.0
                # if aux_human_logits is not None:
                #     # aux head 출력: [B, 2, H, W]
                #     # target: 0 = non-human, 1 = human
                #     target_aux = (anno == self.human_class_idx).long()
                #     aux_loss = self.aux_focal_loss(aux_human_logits, target_aux)

                #     total_loss = ce_loss + self.lambda_aux * aux_loss
                # else:
                #     total_loss = ce_loss

                # train_loss += total_loss.item()
                # total_loss.backward()
                # self.optimizer_clft.step()
                # progress_bar.set_description(
                #     f'CLFT train loss:{total_loss:.4f}'
                # )

                # 기존
                loss = self.criterion(output_seg, batch['anno'])
                
                # 원래 주석
                # w_rgb = 1.1
                # w_lid = 0.9
                # loss = w_rgb*loss_rgb + w_lid*loss_lidar + loss_fusion

                # # focal loss
                # # 1) 기본 ce loss
                # ce = self.criterion(output_seg, anno)
                
                # # 2) focal loss 전체 픽셀에 대해 계산
                # focal_map = self.focal_loss(output_seg, anno)

                # # 3) human class만 골라내기
                # human_idx = 2
                # human_mask = (anno == human_idx) # [B, H, W] bool

                # if human_mask.any():
                #     focal_human = focal_map[human_mask].mean()
                # else:
                #     # human 픽셀이 없는 batch에서는 focal을 0으로
                #     focal_human = 0.0 * ce

                # # 4) 최종 loss: CE + lambda * Focal(human 전용)
                # loss = ce + self.lambda_focal * focal_human
                
                # 기존
                train_loss += loss.item()
                loss.backward()
                self.optimizer_clft.step()
                progress_bar.set_description(f'CLFT train loss:{loss:.4f}')

            # The IoU of one epoch
            train_epoch_IoU = overlap_cum / union_cum
            print(f'Training vehicles IoU for Epoch: {train_epoch_IoU[0]:.4f}')
            print(f'Training human IoU for Epoch: {train_epoch_IoU[1]:.4f}')
            # The loss_rgb of one epoch
            train_epoch_loss = train_loss / (i + 1)
            print(f'Average Training Loss for Epoch: {train_epoch_loss:.4f}')

            valid_epoch_loss, valid_epoch_IoU = self.validate_clft(valid_dataloader, modality)

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
            early_stopping(early_stop_index, epoch, self.model, modality, self.optimizer_clft)
            save_epoch = self.config['General']['save_epoch']
            if (epoch + 1) % save_epoch == 0 and epoch > 0:
                print(f'Saving model for every {save_epoch} epochs...')
                save_model_dict(self.config, epoch, self.model, modality, self.optimizer_clft, True)
                print('Saving Model Complete')
            if early_stopping.early_stop_trigger is True:
                break
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

                # 보조헤드 추가 -> aux_human_logits
                out_depth, output_seg = self.model(batch['rgb'], batch['lidar'], modal)
                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = find_overlap(self.nclasses, output_seg, anno)

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                # 기존
                loss = self.criterion(output_seg, batch['anno'])

                # # --- 메인 CE loss ---
                # ce_loss = self.criterion(output_seg, anno)

                # # --- Aux human focal loss ---
                # if aux_human_logits is not None:
                #     target_aux = (anno == self.human_class_idx).long()
                #     aux_loss = self.aux_focal_loss(aux_human_logits, target_aux)
                #     total_loss = ce_loss + self.lambda_aux * aux_loss
                # else:
                #     total_loss = ce_loss

                # valid_loss += total_loss.item()
                # progress_bar.set_description(f'valid fusion loss: {total_loss:.4f}')

                # # focal loss
                # # 1) 기본 ce loss
                # ce = self.criterion(output_seg, anno)
                
                # # 2) focal loss 전체 픽셀에 대해 계산
                # focal_map = self.focal_loss(output_seg, anno)

                # # 3) human class만 골라내기
                # human_idx = 2
                # human_mask = (anno == human_idx) # [B, H, W] bool

                # if human_mask.any():
                #     focal_human = focal_map[human_mask].mean()
                # else:
                #     # human 픽셀이 없는 batch에서는 focal을 0으로
                #     focal_human = 0.0 * ce

                # # 4) 최종 loss: CE + lambda * Focal(human 전용)
                # loss = ce + self.lambda_focal * focal_human

                # 기존
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
