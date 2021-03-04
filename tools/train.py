from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from dataset import MyDataset
from config import Config
from dataset import Pipline
from torch.utils.data import DataLoader
from utils import accuracy, AverageMeter
from utils import BuildNet
import torch
import torch.optim as optim


class Trainer:
    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        self.backbone = None
        self.pipline = Pipline()
        self.writer = SummaryWriter('../log')    # create tensorboard


        self.__init_train_status()
        self.__init_model()
        self.__init_optimizer()
        self.__init_lr_scheduler()
        self.__init_data()

    def __init_train_status(self):
        torch.cuda.empty_cache()
        self.device = torch.device(Config.device)
        self.criterion = self.__get_loss_fuc(Config.loss)
        self.backbone = Config.backbone

    def __get_loss_fuc(self, loss_name):
        if loss_name == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss().to(self.device)

    def __init_model(self):
        model_builder = BuildNet(self.backbone, Config.num_classes)
        self.model = model_builder()

    def __init_optimizer(self):
        if Config.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)

        if Config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=Config.lr,
                weight_decay=Config.weight_decay,
                amsgrad=False
            )

    def __init_lr_scheduler(self):
        if Config.lr_scheduler == 'CosineAnnealingWarmRestarts':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=Config.t_0,
                T_mult=Config.t_mult,
                eta_min=Config.eta_min,
                last_epoch=-1
            )
        if Config.lr_scheduler == 'ReduceLROnPlateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.2,
                patience=5,
                verbose=False
            )
    def __init_data(self):
        self.train_dataset = MyDataset(
            Config.train_root,
            Config.train_label_root,
            transforms=self.pipline(mode='train'),
            mode='train',
            input_size=Config.input_size
        )
        self.val_dataset = MyDataset(
            Config.train_root,
            Config.train_label_root,
            transforms=self.pipline(mode='val'),
            mode='val',
            input_size=Config.input_size
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers
        )

    def train_one_epoch(self):
        self.model.train()          # train mode
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        for i, (data, label) in pbar:
            data = data.cuda()
            label = label.cuda()

            output = self.model(data)
            loss = self.criterion(output, label)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label.data, topk=(1, 3))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # print(f'train losses: {loss}')
            log_info = 'train loss: ' + str(losses.avg)
            pbar.set_description(log_info)
            self.writer.add_scalar('train_loss',loss.item(), global_step=i)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

        return losses.avg, top1.avg, top5.avg

    @torch.no_grad()
    def val_one_epoch(self):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        pbar = tqdm(enumerate(self.train_dataloader, len(self.train_dataloader)))

        for i, (data, label) in pbar:
            data, label = data.cuda(), label.cuda()

            output = self.model(data)
            loss = self.criterion(output, label)

            prec1, prec5 = accuracy(output.data, label.data, topk=(1, 3))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))
            log_info = 'val loss: ' + str(losses.avg)
            pbar.set_description(log_info)

        return losses.avg, top1.avg, top5.avg

    def run_train(self):
        for epoch in range(Config.max_epoch):
            train_loss, top1_acc, top5_acc = self.train_one_epoch()
            print(f'train_loss: {train_loss}, top1_acc: {top1_acc}, top5_acc: {top5_acc}')
            if epoch % 4 == 0:
                self.save_model(
                    epoch=epoch,
                    model=self.model,
                    optimizer = self.optimizer,
                    model_name=f'epoch{epoch}.pth'
                )
            val_loss, val_top1, val_top5 = self.val_one_epoch()
            print(f'val_loss: {val_loss}, val_top1: {val_top1}, val_top5: {val_top5}')
        self.writer.close()

    def save_model(self, epoch, model, optimizer, model_name):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            },
            model_name
        )

if __name__ == '__main__':
    t = Trainer()
    t.run_train()