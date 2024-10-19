import os
import argparse
from pytorch_lightning.trainer import Trainer, seed_everything

import torch

from tqdm import tqdm

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

    # Save
    names = dm._load_names(os.path.join(dm.rootdir, dm.name, 'index.txt'))
    output_path = os.path.join(dm.rootdir, dm.name, "features")
    os.makedirs(output_path, exist_ok=True)
    idx = 0
    for pred in tqdm(predictions):
        pred.pop('z')
        for i in range(pred[0].shape[0]):
            output = {'label': names[idx]}
            output['mean_representations'] = {}
            for k in pred:
                output['mean_representations'][k] = pred[k][i]
            torch.save(output, os.path.join(output_path, names[idx]+".pt"))
            idx += 1


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
    args = parser.parse_args()

    # Run
    main(args)
