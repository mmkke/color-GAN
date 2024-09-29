#!/usr/bin/python3

"""
Nelson Farrell & Michael Massone
Image Enhancement: Colorization - cGAN
CS 7180 Advanced Perception
Bruce Maxwell, PhD.
09-28-2024

This is the training module for the GAN, run with hyperparameters from params.yml. 
"""

import os
import glob
import yaml
import torch
import logging

import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

from models.discriminator import PatchDiscriminator
from models.generator import UNet
from utils.gan_utils import *

import warnings
# Suppress specific warning related to CIE-LAB conversion
warnings.filterwarnings("ignore", message=".*negative Z values that have been clipped to zero.*")

################################################################################################################################
# FUNCTIONS
################################################################################################################################
def load_config(config_path='params.yaml'):
    '''Function to load configuration from YAML file'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

################################################################################################################################
def get_optimizer(optimizer_config, model_params):
    """
    Returns the optimizer based on the configuration.

    Args:
        optimizer_config (dict): Dictionary containing optimizer configuration such as type, learning rate, and hyperparameters.
        model_params (iterable): The model's parameters that the optimizer will update.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    
    opt_type = optimizer_config.get('type', 'Adam')  # Default to 'Adam' if type is not specified
    lr = optimizer_config.get('lr', 0.001)  # Default learning rate if not provided

    if opt_type == "Adam":
        beta1 = optimizer_config.get('beta1', 0.9)
        beta2 = optimizer_config.get('beta2', 0.999)
        weight_decay = optimizer_config.get('weight_decay', 0)  # Default to no weight decay
        optimizer = torch.optim.Adam(model_params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    elif opt_type == "SGD":
        momentum = optimizer_config.get('momentum', 0.9)  # Default momentum for SGD
        weight_decay = optimizer_config.get('weight_decay', 0)  # Default to no weight decay for SGD
        optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError(f"Optimizer type '{opt_type}' not recognized. Please choose 'Adam' or 'SGD'.")

    return optimizer

################################################################################################################################
def initialize_models(generator, discriminator, device, config):
    ''' 
    Initialize the generator and discriminator models based on the config settings.
    
    Args:
        generator (torch.nn.Module): Generator model to be initialized or loaded from checkpoint.
        discriminator (torch.nn.Module): Discriminator model to be initialized or loaded from checkpoint.
        device (torch.device): The device on which the models are initialized (e.g., CPU or CUDA).
        config (dict): Configuration dictionary containing settings for model initialization and checkpoints.
    
    Returns:
        generator (torch.nn.Module): Initialized or checkpoint-loaded generator model.
        discriminator (torch.nn.Module): Initialized or checkpoint-loaded discriminator model.
    '''
    
    if config['pretrained']['checkpoint_path']:
        checkpoint_path = config['pretrained']['checkpoint_path']
        checkpoint = torch.load(checkpoint_path)

    # Load or initialize the generator model
    if config['pretrained']['load_gen_state'] and config['pretrained']['checkpoint_path']:
        try:
            # Load the model checkpoints
            generator.load_state_dict(checkpoint["generator_state_dict"])
            logging.info(f"Successfully loaded generator state from {checkpoint_path}")
        except FileNotFoundError as e:
            logging.error(f"Checkpoint file not found at {checkpoint_path}. Exception: {e}")
        except KeyError as e:
            logging.error(f"Key error when loading generator state from checkpoint: {e}")
    else:
        # Initialize the generator
        init_G = ModelInitializer(device, init_type=config['init_G']['init_type'], gain=config['init_G']['gain'])
        generator = init_G.init_model(generator)
        logging.info("Generator initialized with custom weights.")
    
    # Load or initialize the discriminator model
    if config['pretrained']['load_disc_state'] and config['pretrained']['checkpoint_path']:
        try:
            # Load the model checkpoints
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            logging.info(f"Successfully loaded discriminator state from {checkpoint_path}")
        except FileNotFoundError as e:
            logging.error(f"Checkpoint file not found at {checkpoint_path}. Exception: {e}")
        except KeyError as e:
            logging.error(f"Key error when loading discriminator state from checkpoint: {e}")
    else:
        # Initialize the discriminator
        init_D = ModelInitializer(device, init_type=config['init_D']['init_type'], gain=config['init_D']['gain'])
        discriminator = init_D.init_model(discriminator)
        logging.info("Discriminator initialized with custom weights.")

    return generator, discriminator

################################################################################################################################   
def load_optimizer_states(optimizer_G, optimizer_D, config):
    """
    Loads the optimizer states for generator and discriminator from the checkpoint
    if they exist and if the config specifies to do so.

    Args:
        optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        checkpoint (dict): The loaded checkpoint containing model and optimizer states.
        config (dict): Configuration dictionary specifying whether to load optimizer states.

    Returns:
        None
    """
    if config['pretrained']['checkpoint_path']:
        checkpoint_path = config['pretrained']['checkpoint_path']
        checkpoint = torch.load(checkpoint_path)
        
    if config['pretrained']['load_optim_states']:
        try:
            if 'optimizer_gen_state_dict' in checkpoint and 'optimizer_disc_state_dict' in checkpoint:
                optimizer_G.load_state_dict(checkpoint['optimizer_gen_state_dict'])
                optimizer_D.load_state_dict(checkpoint['optimizer_disc_state_dict'])
                logging.info("Optimizer states loaded successfully!")
            else:
                logging.warning("Optimizer state dictionaries not found in checkpoint.")
        except Exception as e:
            logging.error(f"Error loading optimizer states: {e}")
    return optimizer_G, optimizer_D

################################################################################################################################
def get_content_loss(config):
    """
    Returns the content loss based on the configuration.
    
    Args:
        config (dict): Dictionary containing the content loss configuration.
        feature_extractor (torch.nn.Module, optional): Pretrained model for perceptual loss (e.g., VGG).
    
    Returns:
        loss_fn (callable): The content loss function.
    """
    
    loss_type = config['loss']['content']  # Read loss type from config
    
    if loss_type == "L1":
        loss_fn = nn.L1Loss()
        logging.info("Content Loss: L1 Loss")
    
    elif loss_type == "L2":
        loss_fn = nn.MSELoss() 
        logging.info("Content Loss: L2 Loss")
    
    elif loss_type == "Entropy":
        loss_fn = EntropyLoss()
        logging.info("Content Loss: Entropy Loss")
    
    elif loss_type == "PerceptualLoss":
        loss_fn = PerceptualLoss()
        logging.info("Selected: Perceptual Loss")
    
    else:
        raise ValueError(f"Loss type '{loss_type}' not recognized. Please choose 'L1', 'L2', 'Entropy', or 'PerceptualLoss'.")
    
    return loss_fn

################################################################################################################################
def get_adversarial_loss(config):
    """
    Returns the adversarial loss based on the configuration.
    
    Args:
        config (dict): Dictionary containing the content loss configuration.
        feature_extractor (torch.nn.Module, optional): Pretrained model for perceptual loss (e.g., VGG).
    
    Returns:
        loss_fn (callable): The adversarial loss function.
    """
    
    loss_type = config['loss']['adversarial']  # Read loss type from config
    
    if loss_type == "BCE":
        loss_fn = nn.BCE()
        logging.info("Adversarial loss: Binary Cross Entropy")
    
    elif loss_type == "BCEWithLogitsLoss":
        loss_fn = nn.BCEWithLogitsLoss()  # L2 Loss is the same as Mean Squared Error Loss
        logging.info("Adversarial loss: BCE With Logits")

    else:
        raise ValueError(f"Loss type '{loss_type}' not recognized. Please choose 'L1', 'L2', 'Entropy', or 'PerceptualLoss'.")
    
    return loss_fn
################################################################################################################################
# MAIN
################################################################################################################################
def main():
    
    # Load the configuration from YAML
    config = load_config()
    
    # Define the directory path based on the config (run_dir inside base_dir)
    output_dir = os.path.join(config['output']['base_dir'], config['output']['run_dir'])
    print(f"Output_dir: {output_dir}")
    
    # Create the directories if they don't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging: Save the log file inside the run_dir
    log_filename = f"{config['output']['run_dir']}.log"
    logging_filepath = os.path.join(output_dir, log_filename)
    logging.basicConfig(filename=logging_filepath, level=logging.INFO, format='%(asctime)s %(message)s')
    
    # Save the configuration dictionary to a YAML file inside the run_dir
    yaml_filepath = os.path.join(output_dir, 'config.yml')
    with open(yaml_filepath, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
        
    print(f"Configuration saved to {yaml_filepath}")
    print(f"Logging to {logging_filepath}")

    # Setup device (GPU/CPU)
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using GPU.")
        device = torch.device("cuda")
    else:
        logging.info("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    # File path from YAML
    coco_path = config['data']['coco_path']
    paths = glob.glob(coco_path + "/*.jpg")  # Grabbing all the image file names

    # Load number of images from config
    num_imgs = config['data']['num_imgs']
    split = config['data']['split']
    train_paths, val_paths = select_images(paths, num_imgs, split)
    logging.info(f"Training set: {len(train_paths)} images")
    logging.info(f"Validation set: {len(val_paths)} images")

    # Image size from YAML
    size = config['data']['image_size']
    train_ds = ColorizationDataset(size, paths=train_paths, split="train")
    val_ds = ColorizationDataset(size, paths=val_paths, split="val")

    # Batch size from YAML
    batch_size = config['training']['batch_size']
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Check Tensor Size
    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    assert Ls.shape == torch.Size([batch_size, 1, size, size]) and abs_.shape == torch.Size([batch_size, 2, size, size])

    # Model parameters
    generator = UNet()
    discriminator = PatchDiscriminator()

    # Initialize the models
    generator, discriminator = initialize_models(generator, discriminator, device, config)
  
    # Move models to device (GPU/CPU)
    generator.to(device)
    discriminator.to(device)

    # Get optimizer from YAML configuration for both generator and discriminator
    optimizer_G = get_optimizer(config['optimizer_G'], generator.parameters())
    optimizer_D = get_optimizer(config['optimizer_D'], discriminator.parameters())

    # Load states if present
    optimizer_G, optimizer_D = load_optimizer_states(optimizer_G, optimizer_D, config)

    # Learning rate scheduler
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 
                                                       mode=config['scheduler_G']['mode'], 
                                                       factor=config['scheduler_G']['factor'], 
                                                       patience=config['scheduler_G']['patience'],
                                                       verbose=config['scheduler_G']['verbose'])
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode=config['scheduler_D']['mode'], 
                                                       factor=config['scheduler_D']['factor'], 
                                                       patience=config['scheduler_D']['patience'],
                                                       verbose=config['scheduler_D']['verbose'])

    # Adversarial Loss function
    adversarial_loss = get_adversarial_loss(config)
    adversarial_loss = adversarial_loss.to(device) 

    # Content Loss function
    content_loss = get_content_loss(config)
    content_loss= content_loss.to(device)
    lambda_l1 = config['training']['lambda_l1']

    # Number of epochs from YAML
    epochs = config['training']['epochs']

    # Flags for showing and saving images
    show_fig = config['training']['show_fig']
    save_images = config['training']['save_images']

    # Initialize GANDriver with all parameters from YAML
    driver = GANDriver(
        generator=generator,
        discriminator=discriminator,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        adversarial_loss=adversarial_loss,
        content_loss=content_loss,
        lambda_l1=lambda_l1,
        device=device,
        epochs=epochs,
        scheduler_D=scheduler_D,
        scheduler_G=scheduler_G,
        run_dir=config['output']['run_dir'],
        base_dir=config['output']['base_dir']
    )

    # Run the GAN training and save metrics to CSV after each epoch
    train_results = driver.run(show_fig=show_fig, save_images=save_images)

    # Save training results to CSV
    results_df = pd.DataFrame(train_results)
    result_path = f"{config['output']['base_dir']}/{config['output']['run_dir']}/{config['output']['training_results_csv']}"
    results_df.to_csv(result_path, index=False)
    logging.info(f"Training complete. Results saved to {result_path}.")

################################################################################################################################
if __name__ == "__main__":
    main()