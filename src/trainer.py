import glob
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
from dataloader import make_data_loader
from dataloader import PointCloudDataset
from dataprocess.inout_points import load_points
from dataprocess.inout_points import points2voxels
from dataprocess.inout_points import save_points
from dataprocess.inout_points import select_voxels
from loss import get_bce_loss
from loss import get_classify_metrics
from pcc_model import PCCModel
from torch.utils.tensorboard import SummaryWriter
from utils.file_util import read_yaml
from utils.general import init_log
from utils.log import init_torch_seeds
from utils.timer import Timer

logger = logging.getLogger(__name__)


init_torch_seeds(89)


class Trainer:
    def __init__(self, path_config, model):
        self.path_config = path_config
        self.exp_dir = Path(str(os.path.dirname(path_config)))
        self.config = read_yaml(path_config)
        self.device = torch.device("cuda:" + self.config['GPU']) if torch.cuda.is_available() else torch.device("cpu")
        self.timer = Timer()
        self.network = model.to(self.device)
        self.prepare()

    def prepare_dataset(self):
        pass

    def prepare(self):
        self.checkpoint_dir = self.exp_dir.joinpath('checkpoint')
        self.tensorboard_dir = self.exp_dir.joinpath('tensorboard')
        shutil.rmtree(self.tensorboard_dir, ignore_errors=True)
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)  # Delete old checkpoint folder
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.tensorboard_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.load_state_dict()
        logger.info(self.network)
        self.optimizer = self.set_optimizer()

    def update_lr(self, lr):
        self.config['lr'] = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return

    def set_optimizer(self):
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config['lr'],
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        # params_lr_list = []
        # for module_name in self.network._modules.keys():
        #     logger.info("module_name: {}".format(module_name))
        #     params_lr_list.append(
        #         {
        #             "params": self.network._modules[module_name].parameters(),
        #             'lr': self.config['lr']
        #         }
        #     )
        # optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        if self.config["init_checkpoint"] != '':
            ckpt = torch.load(self.config['init_checkpoint'], map_location=self.device)
            if 'optimizer' in ckpt.keys():
                optimizer.load_state_dict(ckpt['optimizer'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.config['lr']

        return optimizer

    def load_state_dict(self):
        if self.config["init_checkpoint"] == '':
            logger.info('Random initialization.')
        else:
            checkpoint = torch.load(self.config['init_checkpoint'], map_location=self.device)
            if isinstance(self.network, torch.nn.DataParallel):
                self.network.module.load_state_dict(checkpoint['model'])
            else:
                self.network.load_state_dict(checkpoint['model'])
            logger.info('Load checkpoint from ' + self.config["init_checkpoint"])

    def save_checkpoint(self, filename):
        save_dir = os.path.join(self.checkpoint_dir, filename)
        state = self.network.module.state_dict() if isinstance(self.network, torch.nn.DataParallel) else self.network.state_dict()
        torch.save({'model': state, 'optimizer': self.optimizer.state_dict()}, save_dir)
        logger.info('Saving model at {}'.format(str(save_dir)))

    def train(self, dataloader):
        logger.info('alpha: ' + str(round(self.config['alpha'], 2)) + '\tbeta: ' + str(round(self.config['beta'], 2)))
        logger.info('learning rate: ' + str([params['lr'] for params in self.optimizer.param_groups]))
        logger.info('Training Files length: ' + str(len(dataloader)))

        train_bpp_ae_sum = 0.
        train_bpp_hyper_sum = 0.
        train_IoU_sum = 0.
        num = 0.

        for batch_step, points in enumerate(tqdm.tqdm(dataloader)):
            self.optimizer.zero_grad()
            # Data
            x_np = points2voxels(points, 64).astype('float32')
            x_cpu = torch.from_numpy(x_np).permute(0, 4, 1, 2, 3)  # (8, 64, 64, 64, 1) -> (8, 1, 64, 64, 64)

            x = x_cpu.to(self.device)
            # Forward
            out_set = self.network(x, training=True)

            # Loss function
            num_points = torch.sum(torch.gt(x, 0).float())
            train_bpp_ae = torch.sum(torch.log(out_set['likelihoods'])) / (-np.log(2) * num_points)
            train_bpp_hyper = torch.sum(torch.log(out_set['likelihoods_hyper'])) / (-np.log(2) * num_points)
            train_zeros, train_ones = get_bce_loss(out_set['x_tilde'], x)

            train_distortion = self.config['beta'] * train_zeros + 1.0 * train_ones
            train_loss = (self.config['alpha'] * train_distortion
                          + self.config['delta'] * train_bpp_ae
                          + self.config['gamma'] * train_bpp_hyper)

            # Backward & Optimize
            train_loss.backward()
            self.optimizer.step()
            global_step = self.current_epoch * len(dataloader) + batch_step + 1
            if (batch_step + 1) % self.config['DISPLAY_STEP'] == 0:
                logger.info('Train_zeros: ' + str(train_zeros.item()))
                logger.info('Train_ones: ' + str(train_ones.item()))
                logger.info('Train_distortion: ' + str(train_distortion.item()))
                logger.info('Train_loss: ' + str(train_loss.item()))
                self.writer.add_scalar('Train_loss', train_loss.item(), global_step)

            del train_loss

            with torch.no_grad():
                # Post-process: classification.
                points_nums = torch.sum(x_cpu, dim=(1, 2, 3, 4)).int()
                x_tilde = out_set['x_tilde'].cpu().numpy()  # (8, 1, 64, 64, 64)
                output = select_voxels(x_tilde, points_nums, 1.0)  # (8, 1, 64, 64, 64)
                output = torch.from_numpy(output)  # CPU
                _, _, IoU = get_classify_metrics(output, x_cpu)

                train_bpp_ae_sum += train_bpp_ae.item()
                train_bpp_hyper_sum += train_bpp_hyper.item()
                train_IoU_sum += IoU.item()
                num += 1

                # Display
                if (batch_step + 1) % self.config['DISPLAY_STEP'] == 0:
                    train_bpp_ae_sum /= num
                    train_bpp_hyper_sum /= num
                    train_IoU_sum /= num
                    train_bpp = train_bpp_ae_sum + train_bpp_hyper_sum

                    logger.info("Iteration {0:}:".format(global_step))
                    logger.info("Bpps (AE): {0:.4f}".format(train_bpp_ae_sum))
                    logger.info("Bpps (Hyper): {0:.4f}".format(train_bpp_hyper_sum))
                    logger.info("Bpps (All): {0:.4f}".format(train_bpp))
                    logger.info("IoU: {0:.4f}".format(train_IoU_sum))

                    self.writer.add_scalar('Train/bpps_ae', train_bpp_ae_sum, global_step)
                    self.writer.add_scalar('Train/bpps_hyper', train_bpp_hyper_sum, global_step)
                    self.writer.add_scalar('Train/bpps', train_bpp, global_step)
                    self.writer.add_scalar('Train/IoU', train_IoU_sum, global_step)

                    # record
                    # self.record_set['bpp_ae'].append(train_bpp_ae_sum)
                    # self.record_set['bpp_hyper'].append(train_bpp_hyper_sum)
                    # self.record_set['bpp'].append(train_bpp_ae_sum+train_bpp_hyper_sum)
                    # self.record_set['IoU'].append(train_IoU_sum)
                    # self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)

                    num = 0.
                    train_bpp_ae_sum = 0.
                    train_bpp_hyper_sum = 0.
                    train_IoU_sum = 0.
                    train_bpp = 0.

                # global_step = self.current_epoch * len(dataloader) + batch_step + 1
                # Save checkpoints.
                if global_step % self.config['SAVE_STEP'] == 0:
                    # logger.info('Iteration ' + str(global_step) + " save model!")
                    self.save_checkpoint('model_' + str(self.current_epoch) + '_'
                                         + str(global_step) + '.pt')
                    self.save_checkpoint('final_model.pt')

            # torch.cuda.empty_cache() # empty cache.

    @torch.no_grad()
    def test(self, dataloader):
        bpps_ae = 0.
        bpps_hyper = 0.
        IoUs = 0.
        logger.info('Testing Files length: ' + str(len(dataloader)))

        for _, points in enumerate(tqdm.tqdm(dataloader)):
            # Data
            x_np = points2voxels(points, 64).astype('float32')
            x_cpu = torch.from_numpy(x_np).permute(0, 4, 1, 2, 3)  # (8, 64, 64, 64, 1) -> (8, 1, 64, 64, 64)
            x = x_cpu.to(self.device)
            # Forward.
            out_set = self.network(x, training=False)

            num_points = torch.sum(torch.gt(x, 0).float())
            train_bpp_ae = torch.sum(torch.log(out_set['likelihoods'])) / (-np.log(2) * num_points)
            train_bpp_hyper = torch.sum(torch.log(out_set['likelihoods_hyper'])) / (-np.log(2) * num_points)

            points_nums = torch.sum(x_cpu, dim=(1, 2, 3, 4)).int()
            x_tilde = out_set['x_tilde'].cpu().numpy()  # (8, 1, 64, 64, 64)
            output = select_voxels(x_tilde, points_nums, 1.0)  # (8, 1, 64, 64, 64)
            output = torch.from_numpy(output)
            _, _, IoU = get_classify_metrics(output, x_cpu)
            bpps_ae = bpps_ae + train_bpp_ae.item()
            bpps_hyper = bpps_hyper + train_bpp_hyper.item()
            IoUs = IoUs + IoU.item()

            # torch.cuda.empty_cache()# empty cache.

        bpps_ae = bpps_ae / len(dataloader)
        bpps_hyper = bpps_hyper / len(dataloader)
        IoUs = IoUs / len(dataloader)
        bpps = bpps_ae + bpps_hyper

        logger.info("Bpps (AE): {0:.4f}".format(bpps_ae))
        logger.info("Bpps (Hyper): {0:.4f}".format(bpps_hyper))
        logger.info("Bpps (All): {0:.4f}".format(bpps))
        logger.info("IoU: {0:.4f}".format(IoUs))

        self.writer.add_scalar('Test/bpps_ae', bpps_ae, self.current_epoch)
        self.writer.add_scalar('Test/bpps_hyper', bpps_hyper, self.current_epoch)
        self.writer.add_scalar('Test/bpps', bpps, self.current_epoch)
        self.writer.add_scalar('Test/IoU', IoUs, self.current_epoch)

    def run(self):
        RATIO_EVAL = 9
        self.current_epoch = -1
        filedirs = sorted(glob.glob(self.config['dataset'] + '*.h5'))
        try:
            self.network = self.network.to(self.device)
            self.timer.start('Training Process', verbal=True)
            for _ in range(self.config["epoch"]):
                self.current_epoch += 1
                # if self.current_epoch > 0:
                #     self.update_lr(lr=max(self.config['lr'] / 2, 1e-5))

                self.timer.start('Epoch {}/{}'.format(self.current_epoch, self.config["epoch"]), verbal=True)
                random_list = {
                    'train': random.sample(
                        filedirs[len(filedirs) // RATIO_EVAL:],
                        self.config['TRAINING_FILE'] * self.config['batch_size']
                    ),
                    'test': random.sample(
                        filedirs[:len(filedirs) // RATIO_EVAL],
                        self.config['TESTING_FILE'] * self.config['batch_size']
                    )
                }

                for mode in ['train', 'test']:
                    dataset = PointCloudDataset(random_list[mode])
                    dataloader = make_data_loader(
                        dataset,
                        batch_size=self.config['batch_size'],
                        shuffle=True if mode == 'train' else False,
                        num_workers=6,
                        repeat=False
                    )
                    if mode == 'train':
                        self.train(dataloader)
                    elif mode == 'test':
                        self.test(dataloader)

                self.timer.stop()

            self.timer.stop()

        except KeyboardInterrupt:
            logger.info('Training interrupted')
            self.save_checkpoint('final_model.pt')

        self.timer.stop()
