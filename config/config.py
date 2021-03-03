class Config:
    backbone = 'efficientnet-b6'   # resnext/efficient is alternative
    num_classes = 4
    train_root = '../data/train_images/'
    train_label_root = '../data/train_images.csv'
    test_root = '../data/test_images/'
    train_proportion = 1.0  # proportion of training set,
    input_size = 480
    max_epoch = 20
    train_batch_size = 16
    val_batch_size = 16
    loss = 'CrossEntropyLoss'
    optimizer = 'sgd'       # adam is alternative with lr:1e-5
    lr = 1e-3
    momentum = 0.9
    device = "cuda"         # cuda  or cpu
    gpu_id = [0]
    num_workers = 0         # how many workers for loading data
    weight_decay = 5e-4
    t_0 = 3
    t_mult = 2
    eta_min = 1e-6
    lr_scheduler = 'cosine' # MultiStepLR,
    lr_decay_epoch = 10
    lr_decay = 0.95         # when val_loss increase, lr = lr*lr_decay
    model_output_dir = 'ckpt/'
    res_output_dir = 'output/'