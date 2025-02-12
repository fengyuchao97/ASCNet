import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.networks import *

import torch
import torch.optim as optim

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy,FocalLoss,softmax_focalloss,annealing_softmax_focalloss
import models.losses as losses
from models.losses2 import Hybrid,Dice
import torch.nn.functional as F

from misc.logger_tool import Logger, Timer
from spikingjelly.activation_based import functional
from utils import de_norm
import time

def BCEDiceLoss(inputs, targets):
    # # print(inputs.shape, targets.shape)
    # if inputs.dim() == 4:
    #     inputs = torch.squeeze(inputs, dim=1)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print('fyc:', bce.item(), (1-dice).item()) # inter.item(), inputs.sum().item(), 
    return bce + 1 - dice

def BCELoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    # if inputs.dim() == 4:
    #     inputs = torch.squeeze(inputs, dim=1)
    bce = F.binary_cross_entropy(inputs, targets)
    return bce

class CDTrainer():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        # print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "Adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
        elif args.optimizer == "SGD":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4) # default: 5e-4
        elif args.optimizer == "AdamW":
            self.optimizer_G = torch.optim.AdamW(self.net_G.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
        else:
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=0.01,
                                    momentum=0.9,
                                    weight_decay=5e-4) # default: 5e-4

        # self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.class_pre = None
        self.G_pred = None
        self.G_pred_v = None
        self.G_pred_v1 = None
        self.G_pred_v2 = None
        self.G_pred1 = None
        self.G_pred2 = None
        self.G_pred3 = None
        self.latent_loss = None
        self.attention_loss = None
        self.loss_res = None 
        self.loss_att = None
        self.loss_lat = None

        self.Edge1 = None
        self.Edge2 = None

        self.d6_out = None
        self.d5_out = None
        self.d4_out = None
        self.d3_out = None
        self.d2_out = None

        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # define the loss functions
        self.loss = args.loss
        if args.loss == 'ce':
            hybrid_loss = Hybrid()
            dice = Dice()
            self._pxl_loss = cross_entropy
            # self.criteria = torch.nn.BCELoss()
        elif args.loss == 'focal':
            # self._pxl_loss = softmax_focalloss
            self._pxl_loss = annealing_softmax_focalloss
            # focal = FocalLoss(gamma=2.0, alpha=0.25)
            # self._pxl_loss = cross_entropy
            # self._pxl_loss2 = focal
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'cd_loss':
            self._pxl_loss = losses.cd_loss
        elif args.loss == 'hybrid':
            hybrid_loss = Hybrid()
            self._pxl_loss = cross_entropy
            self._pxl_loss2 = hybrid_loss
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        # pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred = torch.where(self.G_pred > 0.5, torch.ones_like(self.G_pred), torch.zeros_like(self.G_pred)).long()
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        # G_pred = self.G_pred.detach()
        # G_pred = torch.argmax(G_pred, dim=1)

        G_pred = torch.where(self.G_pred > 0.5, torch.ones_like(self.G_pred), torch.zeros_like(self.G_pred)).long()

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)


        if np.mod(self.batch_id, 500) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            # plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, batch):
        self.batch = batch
        
        imgs = batch['aug'].to(self.device)
        gt = self.batch['L'].to(self.device).float()
        self.G_pred, self.G_pred_middle1, self.G_pred_middle2, self.G_pred_v, self.loss_att, self.loss_res = self.net_G(imgs, gt) 

    def _backward_G(self):
        gt = self.batch['L'].to(self.device).float()
        self.G_loss = BCEDiceLoss(self.G_pred, gt) + (BCEDiceLoss(self.G_pred_middle1, F.interpolate(gt, size=self.G_pred_middle1.shape[2:])) + BCEDiceLoss(self.G_pred_middle2, F.interpolate(gt, size=self.G_pred_middle2.shape[2:]))) + BCEDiceLoss(self.G_pred_v, F.interpolate(gt, size=self.G_pred_v.shape[2:])) + 0.01* (self.loss_att.mean() + self.loss_res.mean()) 
        self.G_loss.backward()


    def train_models(self):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True

            starttime = time.time()
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()
                # functional.reset_net(self.net_G) 

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            endtime = time.time()-starttime
            self.logger.write('epoch time: %0.3f \n' % endtime)

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()

