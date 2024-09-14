import torch
import os
class Config:
    # paths and dirs
    train_csv = './amazonML/dataset/train.csv'
    test_csv = './amazonML/dataset/test.csv'
    new_train_csv = './amazonML/dataset/new_train.csv'
    image_save_dir = './dataset/Images'
    os.mkdir(image_save_dir)

