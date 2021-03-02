import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config as config


class Pipline(object):

    def __call__(self, mode='train'):
        if mode == 'train':
            return self.__get_train_transform()
        if mode == 'test' or 'val':
            return self.__get_test_transform()

    def __get_train_transform(self):
        """
        training time data augmentation
        :return: transform
        """
        aug_list = [
            A.RandomResizedCrop(int(config.input_size * 1.2), int(config.input_size * 1.2)),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.OneOf(
            # [
            #   A.GaussNoise(p=1),
            #   A.GaussianBlur(p=1),
            # ], p=0.3),
            # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3),
            # A.HueSaturationValue(hue_shift_limit=5, val_shift_limit=5, p=0.3),
            A.Resize(config.input_size, config.input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        return A.Compose(aug_list, p=1.0)

    def __get_test_transform(self):
        aug_list = [
            A.Resize(config.input_size, config.input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        return A.Compose(aug_list)



if __name__ == '__main__':
    p = Pipline()

    print(p())