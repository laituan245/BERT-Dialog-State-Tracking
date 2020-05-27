import os
import torch
import argparse

from models import Model
from pathlib import Path
from utils import create_dir_if_not_exists, read_json, get_n_params
from dataset import Dataset, Ontology

def create_parser():
    # Creates a parser for command-line arguments.
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--bert_model', type=str, required=True,
                        choices=['bert-base-uncased', 'bert-large-uncased'])
    parser.add_argument('--output_dir', type=Path, required=True)

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--random_oversampling', action='store_true')

    # Other Parameters
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run evaluation.')

    return parser

def load_dataset(base_path):
    dataset = {}
    dataset['train'] = Dataset.from_dict(read_json(base_path / 'train.json'))
    dataset['dev'] = Dataset.from_dict(read_json(base_path / 'dev.json'))
    dataset['test'] = Dataset.from_dict(read_json(base_path / 'test.json'))
    ontology = Ontology.from_dict(read_json(base_path / 'ontology.json'))

    return dataset, ontology


def main(opts):
    if not opts.do_train and not opts.do_eval:
        raise ValueError('At least one of `do_train` or `do_eval` must be True.')
    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir) and opts.do_train:
        raise ValueError('Output directory ({}) already exists and is not empty.'.format(opts.output_dir))

    # Create the output directory (if not exists)
    create_dir_if_not_exists(opts.output_dir)

    # Device device type
    opts.device = torch.device('cuda' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')
    opts.n_gpus = torch.cuda.device_count() if str(opts.device) == 'cuda' else 0
    print('Device Type: {} | Number of GPUs: {}'.format(opts.device, opts.n_gpus), flush=True)

    # Load Datasets and Ontology
    dataset, ontology = load_dataset(opts.data_dir)
    print('Loaded Datasets and Ontology', flush=True)
    print('Number of Train Dialogues: {}'.format(len(dataset['train'])), flush=True)
    print('Number of Dev Dialogues: {}'.format(len(dataset['dev'])), flush=True)
    print('Number of Test Dialogues: {}'.format(len(dataset['test'])), flush=True)

    if opts.do_train:
        # Load model from scratch
        model = Model.from_scratch(opts.bert_model)
        model.move_to_device(opts)
        print('Number of model parameters is: {}'.format(get_n_params(model)))

        # Start Training
        print('Start Training', flush=True)
        model.run_train(dataset, ontology, opts)

        # Free up all memory pytorch is taken from gpu memory
        del model
        torch.cuda.empty_cache()

    if opts.do_eval:
        if not (os.path.exists(opts.output_dir) and os.listdir(opts.output_dir)):
            raise ValueError('Output directory ({}) is empty. Cannot do evaluation'.format(opts.output_dir))

        # Load trained model
        model = Model.from_model_path(opts.output_dir)
        model.move_to_device(opts)
        print('Number of model parameters is: {}'.format(get_n_params(model)))

        # Start evaluating
        print('Start Evaluating', flush=True)
        print(model.run_dev(dataset, ontology, opts),flush=True)
        print(model.run_test(dataset, ontology, opts),flush=True)


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()

    main(opts)
