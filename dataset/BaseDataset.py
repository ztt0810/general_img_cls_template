from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
from config import Config as config
import pandas as pd
import numpy as np
import os


class MyDataset(Dataset):

    def __init__(self,
                 data_root,
                 label_root,
                 transforms=None,
                 mode='train',
                 input_size=640):

        self.__data_root = data_root
        self.__label_root = label_root
        self.__train_data = None
        self.__img_name_list = list()
        self.__label_list = list()
        self.__img_column_name, self.__label_column_name = str, str
        self.__transforms = transforms
        self.__mode = mode
        self.__input_size = input_size

        self.__init_dataset()


    def __get_column_name(self, data_frame):
        """
        get image column name and label column name
        :param data_frame: train csv
        :return: (column name, label column name)
        """

        for column_name in list(data_frame):
            try:
                if type(int(data_frame[column_name][0])) == int:
                    self.__label_column_name = column_name
            except:
                self.__img_column_name = column_name

    def __init_dataset(self):
        """
        init train images and train labels
        :return: none
        """

        self.__train_data = pd.read_csv(self.__label_root)
        self.__get_column_name(self.__train_data)
        if self.__mode == 'train' and config.train_proportion == 1.0:
            # 全数据训练
            self.__img_name_list = [os.path.join(self.__data_root, img_name) for img_name in self.__train_data[self.__img_column_name].values]
            self.__label_list = self.__train_data[self.__label_column_name].values
        else:
            # 8/2分
            pass

    def __getitem__(self, idx):
        if self.__mode == 'train':
            print(self.__mode)
            self.__get_train_data(idx)
        if self.__mode == 'val':
            self.__get_val_data(idx)

    def __len__(self):
        return len(self.__img_name_list)

    def __get_train_data(self, idx):
        img_name = self.__img_name_list[idx]
        img_data = Image.open(img_name)
        img_data = img_data.convert('RGB')

        if self.__transforms:
            self.__transforms(img_data)

        label = np.int32(self.__label_list[idx])
        return img_data.float(), label

    def __get_val_data(self, idx):
        pass


if __name__ == '__main__':
    b = MyDataset('../data/train_images/', '../data/train_images.csv')
    print(b.__len__())
