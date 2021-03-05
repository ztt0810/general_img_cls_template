from tqdm import tqdm
from torch.utils.data import DataLoader
from config import Config
from dataset import MyDataset
from utils import BuildNet
from dataset import Pipline
import pandas as pd
import torch
import os


class Inference:
    def __init__(self, ckpt):
        self.model = None
        self.ckpt = ckpt
        self.test_dataset = None
        self.test_dataloader = None
        self.device = torch.device(Config.device)
        self.pipline = Pipline()

        self.__init_data()
        self.__init_model()

    def __init_data(self):
        self.test_dataset = MyDataset(
            data_root=Config.test_root,
            label_root=Config.test_label_root,
            transforms=self.pipline(mode='test'),
            mode='test',
            input_size=Config.input_size
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=Config.num_workers
        )

    def __init_model(self):
        builder = BuildNet(Config.backbone, 4)
        self.model = builder()
        self.model.load_state_dict(torch.load(self.ckpt)['state_dict'])

    def predict(self):
        pbar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))

        test_preds = []
        for i, img in pbar:
            img = img.to(self.device)
            with torch.no_grad():
                outputs = self.model(img)
                _, pred_label = torch.max(outputs.data, 1)
                test_preds.append(pred_label.cpu().data.numpy()[0])
        sub_df = pd.read_csv(Config.test_label_root)
        sub_df['label'] = test_preds
        sub_df.to_csv(os.path.join(Config.res_output_dir, Config.output_fie), header=False, index=False)

if __name__ == '__main__':
    i = Inference(Config.load_model)
    i.predict()