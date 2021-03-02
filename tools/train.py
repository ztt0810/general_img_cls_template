from dataset import MyDataset
from config import Config
from dataset import Pipline
from torch.utils.data import DataLoader
import torch
import torch.optim as optim


class Trainer:
    def __init__(self, opt):
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.backbone = None
        self.pipline = Pipline()

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
            return torch.nn.CrossEntropyLoss.to(device=self.device)

    def __init_model(self):
        if self.backbone == 'efficientnet-b6':
            from model import EfficientNet
            self.model = EfficientNet.from_pretrained(model_name=self.backbone)

    def __init_optimizer(self):
        if Config.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)

    def __init_lr_scheduler(self):
        pass
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