import argparse

import torch

from network.net import CharacterRNN
from utils.data import Data
from utils.model import train_model, load_model

torch.manual_seed(1)


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Model learning rate  default: 0.01')
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  False: Train model default: True')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data = Data(['hey how are you', 'good i am fine', 'have a nice day'])
    model = CharacterRNN(in_dim=data.dict_size,
                         hidden_dim=12,
                         layer_dim=1,
                         out_dim=data.dict_size)
    if args.load:
        model_name = 'model-8zi3p.pkl'
        load_model(model, 'weights/{}'.format(model_name))
        while True:
            word = input('type the word ... ')
            predicted = data.sample(model, word)
            print(predicted, len(predicted))
    else:
        input_seq, target_seq = data.preprocess()
        train_model(model, input_seq=input_seq, target_seq=target_seq,
                    learning_rate=args.lr, n_epochs=200,
                    save_model=True)
