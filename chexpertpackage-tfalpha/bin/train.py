# coding=utf-8

import argparse
import os
import sys
import pickle
import warnings

import keras
import keras.preprocessing.image
#from keras.engine.training_generator import fit_generator_cross_validation,fit_generator_valid_batch
from keras import backend as K


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
# Allow relative imports when being executed as script.
#####
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import chexpertpackage_tfalpha.bin  # noqa: F401
    #import chexpertpackage
    __package__ = "chexpertpackage_tfalpha.bin"
    #__package__ = "chexpertpackage"



from ..labels.createLabels import proccessLabels, loadImages
from ..models.createModels import buildCustomModel, buildDenseNet, ModelCheckpointByIteration, NBatchAUC, TrainValTensorBoard
from ..models.losses import dice_coef_loss,my_binary_crossentropy, dice_coef, AUC, AUC2, CategoricalTruePositives
from ..labels.custom_generators import DataGenerator, create_K_generators

IMAGEPATH = '//lxestudios.hospitalitaliano.net/pacs/CheXpert-v1.0-small/train/'



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

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ignore',          help='Ignore uncertain labels.',action='store_const', const=True, default=False)
    group.add_argument('--zero',  help='Set uncertain labels as negative labels',action='store_const', const=True, default=False)
    group.add_argument('--one',           help='Initialize the model with weights from a file.',action='store_const', const=True, default=False)
    group.add_argument('--regression',        help='Don\'t initialize the model with any weights.',action='store_const', const=True, default=False)

    validation = parser.add_mutually_exclusive_group()
    validation.add_argument('--validation-set', help='Path to CSV file containing dataset for validation.', type=str, default=None)
    validation.add_argument('--K-folds', help='Integer number of K-folds to split data in dataset for cross validation', type=int, default=0)
    validation.add_argument('--percentage-split',help='Integer percentage of dataset that will be used as validation data', type = int, default=0)

    parser.add_argument('--dataset', help='Path to CSV file containing dataset for training.', type=str, default= '//lxestudios.hospitalitaliano.net/pacs/CheXpert-v1.0-small/train_renamed.csv')
    parser.add_argument('--normalize-file', help='Path to txt file containing mean in first line and std in second line.', type=str, default= None)
    parser.add_argument('--dataset_labels', help='Path to txt pickle file containing labels for training.', type=str, default= '-1')
    parser.add_argument('--dataset_paths', help='Path to txt pickle file containing paths for training.', type=str, default= '-1')
    parser.add_argument('--experiment-ID', help='ID for registering results and tables.', type=str, default=str(int(100*np.random.rand(1))))
    parser.add_argument('--architecture',         help='Type of architecture', default='densenet121', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', default = '00000000:18:00.0', type=str)
    parser.add_argument('--gpu-percentage',   help='Percentage of GPU usage to assign.', default = 0.5, type=float)
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=5)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=None)
    parser.add_argument('--initial-lr',               help='Learning rate.', type=float, default=1e-2)
    parser.add_argument('--adaptative-lr',
                        help='Learning rate adapts during training. It reduces on this factor on every loss plateau.',default=0,type=float)
    parser.add_argument('--checkpoints-path',
                        help='Path to store checkpoints of models during training (defaults to \'./checkpoints\')', default='./checkpoints')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
#    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
#    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
#    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
#    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
#    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
#    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

#   parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')

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
    sess= tf.Session(config=config)
    #config.gpu_options.allow_growth = True
    return sess


def main(args=None):

    #------------Prepare environment---------#
    #Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    #Optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    K.tensorflow_backend.set_session(get_session(args.gpu_percentage))

    tf.enable_eager_execution()
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
    channels = 3

    #Optionally choose a limit to the number of images.
    #set_limit=None
    set_limit=None
    print("Usando trainset " + args.dataset)
    #Process CSV
    print("Procesando etiquetas...")
    classes_names=None
    if args.dataset_labels!="-1" and args.dataset_paths != "-1":
        print("Se usarán las labels y paths especificados por comando")
        with open(args.dataset_labels, "rb") as fp:  # Pickling

            labels_list = pickle.load(fp)
            if isinstance(labels_list,list) and len(labels_list)>1:
                labels=labels_list[1]
                classes_names=labels_list[0]
            else:
                labels=labels_list
        with open(args.dataset_paths, "rb") as fp:  # Pickling
            paths = pickle.load(fp)
    else:
        print("Se procesarán las labels y paths.")
        labels_df, labels = proccessLabels(args.dataset, uncertainMode, limit = set_limit)
        labels_df = labels_df.reset_index()
        paths=labels_df["Path"]


    if set_limit is not None:
        labels = labels[0:set_limit]
        paths=paths[0:set_limit]

    CLASSES=len(labels[0][:])
    print("Se encontraron " + str(CLASSES) + " clases.")
    print("Se usarán  " + str(len(paths)) + " imagenes.")

    #Define architecture
    print("Construyendo modelo...")
    if args.architecture == 'densenet121':
        model = buildDenseNet(HEIGHT, WIDTH, channels, CLASSES)
    if args.architecture == 'custom':
        model = buildCustomModel(HEIGHT, WIDTH, channels , CLASSES)

    #Define optimization algorithm
    print("Compilando modelo...")
    optim = keras.optimizers.Adam(lr = args.initial_lr, beta_1=0.9, beta_2=0.999)

    if uncertainMode is not 'regression':
        model.compile(loss=my_binary_crossentropy, optimizer=optim,
                      metrics=["categorical_accuracy", dice_coef, AUC, AUC2, CategoricalTruePositives()])
    else:
        model.compile(loss="mean_squared_error", optimizer=optim,
                      metrics=["mae", "acc"])

    #CALLBACKSsEBioMedicine. 2018 Aug;34:27-34.

    print("Cargando callbacks...")
    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)

    iterations = 300  # How many iterations between saving checkpoints

    ckp_name_epoch = args.checkpoints_path + "\model-Epoch{epoch:02d}-{loss:.4f}.hdf5"
    ckp_name_iterations = args.checkpoints_path + "\model-Iter{epoch:02d}-{loss:.4f}.hdf5"

    checkpoint_epoch = ModelCheckpoint(ckp_name_epoch, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=False, mode='max')
    checkpoint_batch = ModelCheckpointByIteration(ckp_name_iterations, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False,
                                 mode='min', period = iterations)
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    tbCallBack = TrainValTensorBoard(log_dir=args.tensorboard_dir, histogram_freq=0, write_graph=False,
                                     write_images=False, update_freq='epoch')
    #tbCallBack = TensorBoard(log_dir=args.tensorboard_dir, histogram_freq=0, write_graph=False, write_images=False, update_freq=4800)
    callbacks_list = [checkpoint_epoch, checkpoint_batch, tbCallBack]

    if args.adaptative_lr != 0:
        lr_reducer=ReduceLROnPlateau(monitor='loss', factor=args.adaptative_lr, cooldown=3, patience=3, min_lr=0.00001)
        callbacks_list.append(lr_reducer)

        #callbacks_list = [checkpoint_epoch, tbCallBack]
    #imagegen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
    #                              zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    #Cargar valores de normalizacion
    if args.normalize_file is not None:
        with open(args.normalize_file) as f:
            listnorm = []
            for line in f:
                listnorm.append(line)
        arraynorm = np.asarray(listnorm,dtype=np.float64)
        print("Se normaliza con media " + listnorm[0] +" y desvío " + listnorm[1])
        normalizer = {'mean': arraynorm[0], 'std': arraynorm[1]}
    else:
        normalizer = None

    #Evaluo opciones de validacion
    if args.validation_set is not None:
        print("La validación se realizará con los datos de " + args.validation_set)
        val_labels_df, val_labels, mlb_val = proccessLabels(args.validation_set, uncertainMode, classes=classes_names)
        val_labels_df = val_labels_df.reset_index()
        val_paths = val_labels_df["Path"]
        if args.steps is None:
            steps = int(labels.shape[0] / args.batch_size)
        else:
            steps = args.steps

        val_steps = int(val_labels.shape[0] / args.batch_size)
        # Parameters
        train_params = {'dim': (HEIGHT, WIDTH),
                        'batch_size': args.batch_size,
                        'n_classes': CLASSES,
                        'n_channels': channels,
                        'shuffle': True,
                        'steps': steps,
                        'normalize': normalizer}

        val_params = {'dim': (HEIGHT, WIDTH),
                      'batch_size': args.batch_size,
                      'n_classes': CLASSES,
                      'n_channels': channels,
                      'shuffle': False,
                      'steps': val_steps,
                      'normalize': normalizer}
        # Generators
        training_generator = DataGenerator(paths, labels, **train_params)
        validation_generator = DataGenerator(val_paths, val_labels, **val_params)
        AUC_Callback = NBatchAUC(iterations, val_paths, val_labels, args.checkpoints_path, classes_names=mlb_val.classes_)
        callbacks_list.append(AUC_Callback)
    if args.K_folds != 0:
        print("La validación se realizará como un cross-validation de " + str(args.K_folds) + " folds de los datos")
        if args.steps is None:
            steps = int(((args.K_folds-1)*labels.shape[0]/args.K_folds) / args.batch_size)
        else:
            steps = args.steps
        params =    {'dim': (HEIGHT, WIDTH),
                      'batch_size': args.batch_size,
                      'n_classes': CLASSES,
                      'n_channels': channels,
                      'shuffle': False,
                      'steps': steps,
                      'normalize': normalizer}
        generators = create_K_generators(paths, labels, args.K_folds, **params)
        val_steps = int(int(labels.shape[0]/args.K_folds) / args.batch_size)

    if args.percentage_split != 0:
        val_split = args.percentage_split
        input_idxs = np.random.permutation(labels.shape[0])
        split_idx = int(labels.shape[0]*(1-val_split))
        indexes_train = input_idxs[:split_idx]
        indexes_val=input_idxs[split_idx:]
        if args.steps is None:
            steps = int(split_idx / args.batch_size)
        else:
            steps = args.steps

        val_steps = int((labels.shape[0] - split_idx) / args.batch_size)
        # Parameters
        train_params = {'dim': (HEIGHT, WIDTH),
                        'batch_size': args.batch_size,
                        'n_classes': CLASSES,
                        'n_channels': channels,
                        'shuffle': True,
                        'steps': steps,
                        'normalize': normalizer}

        val_params = {'dim': (HEIGHT, WIDTH),
                      'batch_size': args.batch_size,
                      'n_classes': CLASSES,
                      'n_channels': channels,
                      'shuffle': False,
                      'steps': val_steps,
                      'normalize': normalizer}
        # Generators
        training_generator = DataGenerator(paths, labels, list_IDs = indexes_train,  **train_params)
        validation_generator = DataGenerator(paths, labels, list_IDs = indexes_val, **val_params)


    #-----------Entrenar----------------------------------#
    print("Preparandose para entrenar...")
    K.get_session().run(tf.local_variables_initializer())
    if args.K_folds ==0:
        if args.validation_set is not None:
            hist = model.fit_generator(training_generator,
                                                   steps_per_epoch=steps,
                                                   validation_data=validation_generator,
                                                   validation_steps=val_steps,
                                                   epochs=args.epochs,
                                                   callbacks=callbacks_list,
                                                   max_queue_size=args.max_queue_size,
                                                   workers=args.workers,
                                                   use_multiprocessing=args.use_multiprocessing,
                                                   verbose=1)

    #     else:
    #         #hist = model.fit_generator_valid_batch(training_generator,
    #         hist = fit_generator_valid_batch(model, training_generator,
    #                                     steps_per_epoch=steps,
    #                                     validation_data=validation_generator,
    #                                     validation_steps=val_steps,
    #                                     epochs=args.epochs,
    #                                     callbacks=callbacks_list,
    #                                     max_queue_size=args.max_queue_size,
    #                                     workers=args.workers,
    #                                     use_multiprocessing=args.use_multiprocessing,
    #                                     verbose=1)
    # else:
    #     #hist = model.fit_generator_cross_validation(generators,
    #     hist=fit_generator_cross_validation(model, generators,
    #                                 steps_per_epoch=steps,
    #                                 validation_steps=val_steps,
    #                                 epochs=args.epochs,
    #                                 callbacks=callbacks_list,
    #                                 max_queue_size=args.max_queue_size,
    #                                 workers=args.workers,
    #                                 use_multiprocessing=args.use_multiprocessing,
    #                                 verbose=1)
    model.save(os.path.join(args.checkpoints_path, 'final_model_' + args.experiment_ID + '.h5'))
    with open(os.path.join(args.checkpoints_path, 'hist_' + args.experiment_ID + '.npy', "wb")) as fp:   #Pickling
        pickle.dump(hist, fp)


if __name__ == '__main__':
    main()