import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from config import Config as config
import pandas as pd
import numpy as np
import os


class MyDataset(Dataset):

    def __init__(self,
                 data_root=None,
                 label_root=None,
                 transforms=None,
                 mode='train',
                 input_size=640):

        self.__data_root = data_root
        self.__label_root = label_root
        self.__train_data = None
        self.__test_data = None
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
        if self.__mode == 'train':
            self.__train_data = pd.read_csv(self.__label_root)
            self.__get_column_name(self.__train_data)

        # TODO: data 0.8 / 0.2 split
        if self.__mode == 'train' and config.train_proportion == 1.0:
            # 全数据训练
            self.__img_name_list = [os.path.join(self.__data_root, img_name) for img_name in self.__train_data[self.__img_column_name].values]
            self.__label_list = self.__train_data[self.__label_column_name].values
        else:
            # 8/2分
            pass
        if self.__mode == 'test':
            self.__test_data = pd.read_csv(self.__label_root)
            self.__get_column_name(self.__test_data)
            self.__img_name_list = [os.path.join(self.__data_root, img_name) for img_name in self.__test_data[self.__img_column_name].values]

    def __getitem__(self, idx):
        if self.__mode == 'train':
            return self.__get_train_data(idx)
        if self.__mode == 'val':
            return self.__get_val_data(idx)
        if self.__mode == 'test':
            return self.__get_test_data(idx)

    def __len__(self):
        return len(self.__img_name_list)

    def __get_train_data(self, idx):
        img_name = self.__img_name_list[idx]
        img_data = cv2.imread(img_name)
        # img_data = img_data.convert('RGB')
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        if self.__transforms:
            img_data = self.__transforms(image=img_data)['image']

        label = int(self.__label_list[idx])
        return img_data.float(), label

    def __get_val_data(self, idx):
        img_name = self.__img_name_list[idx]
        img_data = cv2.imread(img_name)
        # img_data = img_data.convert('RGB')
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        if self.__transforms:
            img_data = self.__transforms(image=img_data)['image']

        label = int(self.__label_list[idx])
        return img_data.float(), label

    def __get_test_data(self, idx):
        img_name = self.__img_name_list[idx]

        img_data = cv2.imread(img_name)
        # img_data = img_data.convert('RGB')
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        if self.__transforms:
            img_data = self.__transforms(image=img_data)['image']

        return img_data.float()


if __name__ == '__main__':
    b = MyDataset('../data/train_images/', '../data/train_images.csv')
    print(b.__len__())
