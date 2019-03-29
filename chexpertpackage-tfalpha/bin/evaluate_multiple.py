import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import keras
from keras.models import load_model
import csv

from matplotlib import pyplot as plt

IMAGEPATH = '//lxestudios.hospitalitaliano.net/pacs/CheXpert-v1.0-small/train/'
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import chexpertpackage_tfalpha.bin  # noqa: F401
    __package__ = "chexpertpackage_tfalpha.bin"



from ..labels.createLabels import proccessLabels
from ..labels.custom_generators import  TestGenerator
from ..models.losses import my_binary_crossentropy, dice_coef
from ..models.createModels import getModelAUC

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Training script for CheXpert database.')

    def csv_list(string):
        return string.split(',')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ignore',          help='Ignore uncertain labels.',action='store_const', const=True, default=False)
    group.add_argument('--zero',  help='Set uncertain labels as negative labels',action='store_const', const=True, default=False)
    group.add_argument('--one',           help='Initialize the model with weights from a file.',action='store_const', const=True, default=False)
    group.add_argument('--regression',        help='Don\'t initialize the model with any weights.',action='store_const', const=True, default=False)

    parser.add_argument('--dataset', help='Path to CSV file containing dataset for training.', type=str, default= '//lxestudios.hospitalitaliano.net/pacs/CheXpert-v1.0-small/valid_renamed.csv')
    parser.add_argument('--models-path' , help='Path where models are saved', type=str)
    parser.add_argument('--outputs-dir', help= 'Path to save csv file and figure', type=str)
    # Fit generator arguments
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', default = '00000000:18:00.0', type=str)
    parser.add_argument('--gpu-percentage',   help='Percentage of GPU usage to assign.', default = 0.5, type=float)

    parser.add_argument('--workers', help='Use of multiprocessing workers. To disable multiprocessing, set workers to false', default=1,type=int)
    parser.add_argument('--use-multiprocessing', help='Use of multiprocessing workers. To disable multiprocessing, set workers to false', default=False)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=5)

    return parser.parse_args(args)


def get_session(gpu_perc):
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_perc
    return tf.Session(config=config)




def main(args=None):

    #------------Prepare environment---------#
    #Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session(args.gpu_percentage))

    #Determine the labeling strategy from command argument
    if args.ignore:
        #Images with one or more uncertain labels will be ignored.
        uncertainMode = 'ignore'

    if args.zero:
        #All uncertain labels will be treated as negative labels (zeros)
        uncertainMode='zero'
    if args.one:
        #All uncertain labels will be treated as positive labels (ones)
        uncertainMode = 'one'
    if args.regression:
        #Labels will be treated as continuous variables. The values are set from function uRegression in createLabels.py
        # zero labels will be mapped to constant SEGURO_NO
        # ones labels will be mapped to constant SEGURO_SI
        # uncertain labels will be mapped to constant POSIBLE_SI
        # empty labels will be mapped to constant POSIBLE_NO
        uncertainMode='regression'

    #Determine image size. Used for loading images and fo
    HEIGHT = 320
    WIDTH = 320


    print("Procesando etiquetas...")
    labels_df, labels, mlb = proccessLabels(args.dataset, uncertainMode)
    labels_df = labels_df.reset_index()
    N = labels.shape[0]
    N_real_classes = len(labels[0][:])
    main_aucs=[]
    main_aucs_names = ["Atelectasis", "Edema", "Consolidation", "PleuralEfussion","Cardiomegaly"]

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    with open(os.path.join(args.outputs_dir, "output.csv"), 'w') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        for file in os.listdir(args.models_path):
            if file.endswith(".hdf5"):
                model_path = os.path.join(args.models_path, file)
                print("Cargando modelo " + os.path.basename(model_path))
                model = load_model(model_path, custom_objects={'my_binary_crossentropy': my_binary_crossentropy, 'dice_coef': dice_coef})
                test_generator = TestGenerator(labels_df["Path"], labels, (HEIGHT, WIDTH))
                print("Comenzando evaluacion...")
                aucs=getModelAUC(N, test_generator, model, N_real_classes)
                sum=0
                for i,clss in enumerate(mlb.classes_):
                    print(clss)
                    print(aucs[i])
                    if clss in main_aucs_names:
                        sum += aucs[i]
                aucs.insert(0, file)
                aucs.append(sum/len(main_aucs_names))
                wr.writerow(map(lambda x: str(x), aucs))

                main_aucs.append(sum/len(main_aucs_names))


    x = np.arange(len(main_aucs))
    plt.plot(x, main_aucs)
    plt.title("Promedio de AUC entre las 5 patologías principales")
    plt.xlabel('Modelos')
    plt.ylabel('AUC promedio')
    fig_path = os.path.join(args.outputs_dir, "mainAUCS.jpg")
    plt.savefig(fig_path)


if __name__ == '__main__':
    main()