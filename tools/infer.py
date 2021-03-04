from tqdm import tqdm
from torch.utils.data import DataLoader
from config import Config
from dataset import MyDataset
from dataset import Pipline
import pandas as pd
import torch


class Inference:
    def __init__(self, model):
        self.model = model
        self.test_dataset = None
        self.test_dataloader = None
        self.device = torch.device(Config.device)
        self.pipline = Pipline()

        self.__init_data()

    def __init_data(self):
        self.test_dataset = MyDataset(
            data_root=Config.test_root,
            # label_root=Config.
            transforms=self.pipline(mode='test'),
            mode='test',
            input_size=Config.input_size
        )
        print(len(self.test_dataset))
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=Config.val_batch_size,
            shuffle=True,
            num_workers=Config.num_workers
        )

    def predict(self):
        pbar = tqdm(enumerate(self.test_dataloader), len(self.test_dataloader))

        test_preds = []
        for i, img in pbar:
            img = img.to(self.device)
            with torch.no_grad():
                outputs = self.model(img).detach().cpu().numpy()
                _, pred_label = torch.max(outputs.data, 1)
                test_preds.append(pred_label)
        sub_df = pd.read_csv('../data/test_images.csv')
        sub_df['num_class'] = test_preds
        sub_df.to_csv('./output/submission.csv', header=False, index=False)

if __name__ == '__main__':
    from utils import BuildNet
    b = BuildNet(Config.backbone, 4)
    model = b()
    model.load_state_dict(torch.load('../tools/epoch16.pth')['state_dict'])
    i = Inference(model)
    i.predict()