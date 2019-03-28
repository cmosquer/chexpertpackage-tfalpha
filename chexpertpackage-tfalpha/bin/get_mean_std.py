import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import math

import pickle

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import chexpertpackage.bin  # noqa: F401
    __package__ = "chexpertpackage.bin"


from ..labels.custom_generators import  MeanGenerator, VarianceGenerator


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Training script for CheXpert database.')

    def csv_list(string):
        return string.split(',')

    #subparsers = parser.add_subparsers(help='Arguments for specific dataset.', dest='dataset_type')
    #subparsers.required = True
    #csv_parser = subparsers.add_parser('csv')
    #csv_parser.add_argument('--dataset', help='Path to CSV file containing dataset for training.', type=str, default= '//lxestudios.hospitalitaliano.net/pacs/CheXpert-v1.0-small/train_renamed.csv')
    #csv_parser.add_argument('--val-dataset', help='Path to a CSV file containing dataset for validation.')

    parser.add_argument('outputfile',   help='Path to txt file to save mean and std', type=str)
    parser.add_argument('--dataset', help='Path to CSV file containing the paths to images that will be evaluated for mean and std.', type=str, default= '//lxestudios.hospitalitaliano.net/pacs/CheXpert-v1.0-small/train_renamed.csv')
    parser.add_argument('--experiment-ID', help='ID for registering results and tables.', type=str, default=str(int(100*np.random.rand(1))))
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', default = '00000000:18:00.0', type=str)
    parser.add_argument('--gpu-percentage',   help='Percentage of GPU usage to assign.', default = 0.5, type=float)

    # Fit generator arguments
    parser.add_argument('--workers', help='Use of multiprocessing workers. To disable multiprocessing, set workers to false', default=1,type=int)
    parser.add_argument('--use-multiprocessing', help='Use of multiprocessing workers. To disable multiprocessing, set workers to false', default=False)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=5)

    return parser.parse_args(args)


def get_session(gpu_perc):
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_perc
    #config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def main(args=None):

    #------------Prepare environment---------#
    #Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    #Optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session(args.gpu_percentage))
    csv_path = args.dataset
    print("Dataset de train: " + csv_path)

    labels_df = pd.read_csv(csv_path)
    paths = labels_df["Path"]
    N = paths.shape[0]
    #Determine image size. Used for loading images and fo
    HEIGHT = 320
    WIDTH = 320
    channels = 3


    if not os.path.exists(args.outputfile):
        os.makedirs(args.outputfile)

    mean_generator = MeanGenerator(paths, (HEIGHT, WIDTH))
    current_mean_sum=0.
    sub_means = np.zeros(int(N/1000)+1)
    for j in range(1,N+1):
        img_mean = next(mean_generator)
        current_mean_sum += img_mean
        if (j%1000) == 0:
            print("Evaluando imagen " + str(j) + " de " + str(N))
            idx = int(j/1000)-1
            sub_means[idx] = current_mean_sum/N
            current_mean_sum = 0

    sub_means[idx+1] = current_mean_sum/N
    tot_mean = np.sum(sub_means)
    print("La media es ", tot_mean)
    var_generator = VarianceGenerator(paths, tot_mean, (HEIGHT, WIDTH))
    sum_var=0.
    sub_vars = np.zeros(int(N/1000)+1)

    for j in range(1,N+1):
        img_var = next(var_generator)
        sum_var += img_var
        if (j%1000) == 0:
            print("Evaluando imagen " + str(j) + " de " + str(N))
            idx = int(j/1000)-1
            sub_vars[idx] = sum_var/N
            sum_var = 0

    sub_vars[idx+1] = sum_var/N
    tot_std = math.sqrt((np.sum(sub_vars)))
    print("El desvio es ", tot_std)

    final = np.array([tot_mean,tot_std])

    np.savetxt(os.path.join(args.outputfile, "mean_std_train.txt"), final,fmt="%28.25f")


if __name__ == '__main__':
    main()