import argparse


def get_config(parser=argparse.ArgumentParser()):
    parser.add_argument(nargs='--batch_size',
                        type=int,
                        default=4,
                        help='input batch size, default=64')

    parser.add_argument(nargs='--epoch',
                        type=int,
                        default=10,
                        help='number of epochs to train for, default=10')

    parser.add_argument(nargs='--lr',
                        type=float,
                        default=3e-5,
                        help='select the learning rate, default=1e-3')

    opt = parser.parse_args()

    if opt.output:
        print(f'batch_size: {opt.batch_size}')
        print(f'epochs: {opt.niter}')
        print(f'learning rate: {opt.lr}')

    return opt
