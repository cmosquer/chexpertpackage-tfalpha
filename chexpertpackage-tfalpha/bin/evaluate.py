import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import keras
import csv
import pickle
from keras.models import load_model

from vis.visualization import visualize_cam

from matplotlib import pyplot as plt
from sklearn import metrics

IMAGEPATH = '//lxestudios.hospitalitaliano.net/pacs/CheXpert-v1.0-small/train/'
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import chexpertpackage_tfalpha.bin  # noqa: F401
    __package__ = "chexpertpackage_tfalpha.bin"

#Agus

from ..labels.createLabels import proccessLabels, loadImages
from ..labels.custom_generators import image_generator, DataGenerator, TestGenerator
from ..models.losses import dice_coef_loss,my_binary_crossentropy, dice_coef

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

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--one-model' , help='Define --model-path as full path for model to load', action='store_const', const=True, default=False)
    group2.add_argument('--ensamble', help='Define --model-path as full path to CSV where each row contains a full path to a model and optionally the class its strong at', action='store_const', const=True, default=False)

    parser.add_argument('--model-path',help='path to one model o to ensamble CSV')
    parser.add_argument('--dataset', help='Path to CSV file containing dataset for test.', type=str)
    parser.add_argument('--dataset_labels', default='C:/ImageAnalytics/Torax/chexpertpackage/chexpertpackage/labels/zero_labels.txt')
    parser.add_argument('--experiment-ID', help='ID for registering results and tables.', type=str, default=str(int(100*np.random.rand(1))))
    parser.add_argument('--normalize-file', help = 'TXT file with mean and standard deviation for normalization. Should be the same used during training', default ='C:/ImageAnalytics/Torax/chexpertpackage/chexpertpackage/labels/mean_std_train.txt')

    parser.add_argument('--heatmaps-dir', type=str, default=None)
    parser.add_argument('--roc', action = 'store_const',const=True,default=False)
    parser.add_argument('--roc-dir', type=str, default='C:/Users/UsuarioHI/Desktop/rocs')

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
    #config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def get_heatmap(img,filter_index,model,last_layer_idx):
    return visualize_cam(model, last_layer_idx, filter_index, seed_input=img)


def translate_to_0_1(arr):
    # Figure out how 'wide' each range is
    Max=arr.max()
    Min=arr.min()
    Span = Max - Min
    norm=np.subtract(arr.astype(np.float64),Min)
    norm=np.divide(norm,Span)
    return norm


def save_heatmap(img, img_orig, hm, save_dir):
    # img=array_to_img(im)
    plt.figure(1, figsize=(18, 10))

    plt.subplot(1, 2, 1)
    plt.grid(False)

    plt.imshow(translate_to_0_1(np.squeeze(img_orig)))

    plt.subplot(1, 2, 2)
    plt.grid(False)
    normhm = translate_to_0_1(np.squeeze(hm))
    normimg = translate_to_0_1(np.squeeze(img))
    plt.imshow(normimg, cmap='gray')
    plt.imshow(normhm, cmap='jet', alpha=0.5)

    plt.savefig(save_dir)
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')


def main(args=None):
    STRONG_WEIGHT = 2
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
    channels = 3
    PROB_THRESHOLD = 0.1
    PROB_THRESHOLD_NO_FINDING=0.5

    classes_names = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices']
    print("Procesando etiquetas...")
    labels_df, labels, mlb = proccessLabels(args.dataset, uncertainMode, classes=classes_names)
    labels_df = labels_df.reset_index()
    CLASSES=len(labels[0][:])
    print(len(labels[0][:]))
    th = [PROB_THRESHOLD] * CLASSES
    nf = classes_names.index("No Finding")
    th[nf] = PROB_THRESHOLD_NO_FINDING
    print(mlb.classes_)
    models_strengths={} #Diccionario donde las keys son los full paths a modelos y los values son una lista de pesos por clase (1 si no es fuerte en esa clase, STRONG_WEIGHT si es fuerte)
    if args.one_model:
        models_strengths[args.model_path] = [1]*CLASSES
    if args.ensamble:
        with open(args.model_path, "r") as modelcsv:
            readCSV = csv.reader(modelcsv, delimiter=',')
            for line in readCSV:
                has_strong=False
                weights = [1]*CLASSES
                print(line[0])
                for strong_class in line[1:]:
                    if strong_class in list(mlb.classes_):
                        has_strong=True
                        weights[list(mlb.classes_).index(strong_class.strip('/'))] = STRONG_WEIGHT
                if has_strong and os.path.exists(line[0]):
                    models_strengths[line[0]] = weights
                else:
                    print('No se usa el modelo ' + line[0] + ' porque su clase fuerte no existe en el set de testeo o porque no existe el path')

    models={}
    for curr_model_path in models_strengths:

        print("Cargando modelo " + os.path.basename(curr_model_path) +"...")
        models[curr_model_path]=load_model(curr_model_path, custom_objects={'my_binary_crossentropy': my_binary_crossentropy, 'dice_coef': dice_coef})


    if args.normalize_file is not None:
        with open(args.normalize_file) as f:
            listnorm = []
            for line in f:
                listnorm.append(line)

    arraynorm = np.asarray(listnorm,dtype=np.float64)
    print("Se normaliza con media " + listnorm[0] +" y desvío " + listnorm[1])
    normalizer = {'mean': arraynorm[0], 'std': arraynorm[1]}

    test_generator = TestGenerator(labels_df["Path"],labels, (HEIGHT,WIDTH),normalizer=normalizer)

    N = labels.shape[0]
    TP = [0] * CLASSES
    FP = [0] * CLASSES
    FN = [0] * CLASSES
    TN = [0] * CLASSES
    if args.heatmaps_dir is not None:
        for cl in mlb.classes_:
            if not os.path.exists(args.heatmaps_dir + '/' + cl):
                os.makedirs(args.heatmaps_dir + '/' + cl)
    if args.roc:
        all_probs = []
        all_gt = []

    found_images=[]
    print("Comenzando evaluacion...")

    for j in range(N):  #Itero sobre las imagenes
        probs = [0] * CLASSES
        sum_weights_per_class = [0] * CLASSES
        print("Evaluando imagen " + str(j) + " de " + str(N))
        pred_classes = []
        gt_classes = []
        img, gt = next(test_generator)
        #Calcular predicciones
        if not np.isnan(img).any():
            found_images.append(j)
            for path,model in models.items(): #Itero sobre los modelos
                curr_probs = list(model.predict(img)[0])
                weighted_curr_probs = [a*b for a, b in zip(curr_probs,models_strengths[path])]
                probs = [sum(x) for x in zip(probs, weighted_curr_probs)]

                sum_weights_per_class = [sum(x) for x in zip(sum_weights_per_class, models_strengths[path])]

            probs = [a/b for a,b in zip(probs, sum_weights_per_class)]

            for i, prob in enumerate(probs): #Itero sobre las clases
                if i >= gt.shape[0]:
                    break
                if gt[i] == 1 and prob >= th[i]:
                    TP[i] += 1
                    pred_classes.append(mlb.classes_[i])
                    gt_classes.append(mlb.classes_[i])
                    if args.heatmaps_dir is not None and i!=nf:
                        hm = get_heatmap(img, i, model, len(model.layers) - 1)
                        save_dir = args.heatmaps_dir + '/' + mlb.classes_[i] + '/' + str(j) + '-orig_' + "-".join(
                            c[:3] for c in gt_classes) + '__pred_' + "-".join(c[:3] for c in pred_classes)
                        print("Guardando heatmap en " + save_dir)
                        save_heatmap(img, img, hm, save_dir)

                if gt[i] == 0 and prob >= th[i]:
                    FP[i] += 1
                    pred_classes.append(mlb.classes_[i])

                if gt[i] == 1 and prob < th[i]:
                    FN[i] += 1
                    gt_classes.append(mlb.classes_[i])

                if gt[i] == 0 and prob < th[i]:
                    TN[i] += 1


            if args.roc:
                all_probs.append(probs)
                all_gt.append(gt)
        else:
            print("Se excluyó una imagen.")

    N_per_class = np.sum(labels[found_images],axis=0)
    print('\n'+"Ocurrencias de cada patología en el set de testeo")
    for i, oc in enumerate(N_per_class):
        if N_per_class[i] != 0:
            print(mlb.classes_[i] + ": " + str(oc))

    print("Resultados del test:")
    print('\n' + "Verdaderos positivos")
    for i, tp in enumerate(TP):
        if N_per_class[i] != 0:
            print(mlb.classes_[i] + ": " + str(tp))
    print('\n' + "Falsos positivos")
    for i, fp in enumerate(FP):
        if N_per_class[i] != 0:
            print(mlb.classes_[i] + ": " + str(fp))
    print('\n' + "Verdaderos negativos")
    for i, tn in enumerate(TN):
        if N_per_class[i] != 0:
            print(mlb.classes_[i] + ": " + str(tn))
    print('\n' + "Falsos negativos")
    for i, fn in enumerate(FN):
        if N_per_class[i]!= 0:
            print(mlb.classes_[i] + ": " + str(fn))
    print('\n' + "Sensibilidad")
    for i, tp in enumerate(TP):
        if N_per_class[i]!= 0:
            print(mlb.classes_[i] + ": " +"%.2f" %(tp / (FN[i]+TP[i])))
    print('\n' + "Especificidad")
    for i, tn in enumerate(TN):
        if N_per_class[i] != 0:
            print(mlb.classes_[i] + ": " + "%.2f" % (tn/(FP[i]+TN[i])))

    if args.roc:
        print("Generando curvas ROC...")
        if not os.path.exists(args.roc_dir):
            os.makedirs(args.roc_dir)
        for i,clss in enumerate(mlb.classes_):
            if N_per_class[i] != 0:
                probs_clss = [p[i] for p in all_probs]
                gt_clss = [g[i] for g in all_gt]

                fpr_keras, tpr_keras, thresholds = metrics.roc_curve(gt_clss, probs_clss)
                #auc_keras = metrics.auc(fpr_keras, tpr_keras)
                auc_keras = metrics.roc_auc_score(gt_clss,probs_clss)

                fig = plt.figure(i, figsize=(15, 15))
                ax = fig.add_subplot(111)  # add a subplot to the new figure, 111 means "1x1 grid, first subplot"
                ax.set_title(clss, fontsize=50)
                ax.axis([0, 1, 0, 1])
                ax.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
                ax.set_xlabel('1-Especificidad', fontsize=30)
                ax.set_ylabel('Sensibilidad', fontsize=30)
                ax.legend(loc='best', fontsize=30)
                fig_path  = args.roc_dir + '/' + clss + '.jpg'
                plt.savefig(fig_path)

                fig_all = plt.figure(CLASSES+1, figsize=(15, 15))
                ax_all = fig_all.add_subplot(111)  # add a subplot to the new figure, 111 means "1x1 grid, first subplot"
                ax_all.set_title('Curvas ROC', fontsize=50)
                ax_all.axis([0, 1, 0, 1])
                ax_all.plot(fpr_keras, tpr_keras, label=clss[:4])
                ax_all.set_xlabel('1-Especificidad', fontsize=30)
                ax_all.set_ylabel('Sensibilidad', fontsize=30)
                ax_all.legend(loc='best', fontsize=30)
                print("Roc " + str(i))
            if i == mlb.classes_.shape[0]-1:
                fig_path = args.roc_dir + '/todas.jpg'
                plt.savefig(fig_path)

    #print("Calculando predicciones")
    #predictions = model.predict_generator(test_generator, steps=None, max_queue_size=args.max_queue_size,
    #                                 workers=args.workers, use_multiprocessing=args.use_multiprocessing, verbose=0)

    #for i, pred in enumerate(predictions):
    #    print(str(i)+": ",predictions[i][:])


if __name__ == '__main__':
    main()