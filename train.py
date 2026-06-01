import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os

class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, prediction, target):
        # Calculate frequency domain loss
        pred_fft = torch.fft.rfft2(prediction)
        target_fft = torch.fft.rfft2(target)
        
        loss = F.l1_loss(torch.real(pred_fft), torch.real(target_fft)) + \
               F.l1_loss(torch.imag(pred_fft), torch.imag(target_fft))
        return loss
    
class EMACallback(pl.Callback):
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.ema_weights is None:
            self.ema_weights = {k: v.clone().detach() for k, v in pl_module.state_dict().items()}
        
        for k, v in pl_module.state_dict().items():
            self.ema_weights[k].copy_(self.decay * self.ema_weights[k] + (1 - self.decay) * v)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['ema_state_dict'] = self.ema_weights

    

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        self.fft_loss = FFTLoss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # 1. Unpack data
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        
        # 2. Native Pixel-Level Similarity Loss (128x128 scale)
        l1_loss = self.loss_fn(restored, clean_patch)
        
        # 3. Multi-Scale Frequency Domain Loss
        # Scale 1: Full resolution frequency loss (128x128)
        f_loss_128 = self.fft_loss(restored, clean_patch)
        
        # Scale 2: Mid-resolution frequency loss (Downsampled to 64x64)
        # avg_pool2d extracts the lower frequency background structures
        restored_64 = F.avg_pool2d(restored, kernel_size=2)
        clean_64 = F.avg_pool2d(clean_patch, kernel_size=2)
        f_loss_64 = self.fft_loss(restored_64, clean_64)
        
        # Scale 3: Low-resolution frequency loss (Downsampled to 32x32)
        restored_32 = F.avg_pool2d(restored_64, kernel_size=2)
        clean_32 = F.avg_pool2d(clean_64, kernel_size=2)
        f_loss_32 = self.fft_loss(restored_32, clean_32)
        
        # 4. Total Balanced Loss
        # We weigh the sub-scales so frequency doesn't overpower the vital L1 loss
        total_fft_loss = f_loss_128 + 0.5 * f_loss_64 + 0.25 * f_loss_32
        loss = l1_loss + (0.1 * total_fft_loss)
        
        # Logging to tracking panels
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=5, max_epochs=opt.epochs)
        return [optimizer], [scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir, filename='model', every_n_epochs = 1, monitor="train_loss",  mode="min", save_top_k=1)
    ema_callback = EMACallback(decay=0.999)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback, ema_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)
    #trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path='/content/model.ckpt')

if __name__ == '__main__':
    main()



