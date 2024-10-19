import os
import wandb
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from genie.config import Config
from genie.utils.model_io import load_model

from genie.data.data_module import GenieDataModule


def main(args):

	# Load configuration
	config = Config(filename=args.config)

	# Initial random seeds
	seed_everything(config.training['seed'], workers=True)

	# Data module
	dm = GenieDataModule(
		**config.io,
		batch_size=config.training['batch_size']
	)

	# Model
	model = load_model(config.io['rootdir'], config.io['name'])

	# Trainer
	trainer = Trainer(accelerator="gpu", devices=1)    
	predictions = trainer.predict(model, dm)
	breakpoint()


if __name__ == '__main__':

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
	args = parser.parse_args()

	# Run
	main(args)