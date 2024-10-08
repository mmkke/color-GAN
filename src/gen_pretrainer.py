
"""
Nelson Farrell & Michael Massone
Image Enhancement: Colorization - cGAN
CS 7180 Advanced Perception
Bruce Maxwell, PhD.
09-28-2024

This file contains a class that will create and train U-Net for the task of colorization.
The U-Net will created will have pretrained ResNet18 backbone.
"""
##################################################### Packages ###################################################################
import glob
from utils.pretrain_utils import *

def main():

    # params
    checkpoint_path = None
    load_previous_state = False
    data_dir = "/Users/nelsonfarrell/.fastai/data/coco_sample/train_sample"
    paths = glob.glob(data_dir + "/*.jpg")
    num_images = 10000
    size = 256
    batch_size = 32
    epochs = 101
    lr = 0.0002 
    beta1 = 0.5
    beta2 = 0.999
    l1_loss = nn.L1Loss()
    run = "Res_full_data_3"
    start_epoch = 0

    # train model
    model = PretrainGenerator(size, batch_size, epochs, lr, beta1, beta2, l1_loss, run, start_epoch)
    model.set_train_and_val_paths(paths, num_images)
    model.set_data_loaders()
    model.set_model()
    if load_previous_state:
        model.load_state(checkpoint_path)
    model.train_model()

if __name__ == "__main__":
    main()
    
