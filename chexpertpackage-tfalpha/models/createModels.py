from keras.models import Sequential
from keras import layers
from keras import models
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.python.eager import context

import warnings
import numpy as np
from sklearn import metrics
import os

from ..labels.custom_generators import  TestGenerator
#Test SVN Pycharm


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

def getModelAUC(N, test_generator, model, N_real_classes):
    all_probs = []
    all_gt = []
    for j in range(N):
        print("Evaluando imagen de validación " + str(j) + " de " + str(N))
        img, gt = next(test_generator)
        probs = list(model.predict(img)[0])
        all_probs.append(probs)
        all_gt.append(gt)
    auc = N_real_classes*[0]
    for i in range(N_real_classes):
        probs_clss = [p[i] for p in all_probs]
        gt_clss = [g[i] for g in all_gt]
        if np.count_nonzero(gt_clss) != 0:
            auc[i] = metrics.roc_auc_score(gt_clss, probs_clss)
        else:
            print("Un AUC se dejó en cero")
    return auc


class NBatchAUC(Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, iterations, img_paths, labels, file, classes_names=None):
        self.step = 0
        self.iterations = iterations
        self.img_paths = img_paths
        self.labels = labels
        self.aucs_file=os.path.join(file,'aucs.csv')
        self.sens_file = os.path.join(file, 'sensibilidad.csv')
        self.espe_file = os.path.join(file, 'especificidad.csv')
        if classes_names is not None:
            with open(self.aucs_file, mode='a') as f:
                str_names = ','.join(classes_names) + '\n'
                f.write(str_names)
            with open(self.sens_file, mode='a') as f:
                str_names = ','.join(classes_names) + '\n'
                f.write(str_names)
            with open(self.espe_file, mode='a') as f:
                str_names = ','.join(classes_names) + '\n'
                f.write(str_names)
    def on_batch_end(self, batch, logs={}):
        self.step += 1
        if self.step % self.iterations == 0:
            generator = TestGenerator(self.img_paths,self.labels, (320,320))
            aucs = getModelAUC(self.labels.shape[0], generator, self.model, self.labels.shape[1])
            print("AUCS: ", aucs)
            with open(self.aucs_file,mode='a') as f:
                straucs = ','.join(str(a) for a in aucs) + '\n'
                f.write(straucs)


class ModelCheckpointByIteration(Callback):
     def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        #super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

     def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nBatch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nBatch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


def buildCustomModel(height, width, channels, classes):
    model=Sequential()
    inputShape=(height, width, channels)

    #First convolution block: 64 filters * 2
    model.add(Conv2D(64, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(64, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))

    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.3))

    #Second convolution block: 128 filters
    model.add(Conv2D(128, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(128, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))

    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.3))

    # Flatten output into fc layers

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))
    model.summary()
    return model


def buildDenseNet(height, width, channels, classes):
    print("Creando densenet")
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(height, width, channels))
    for layer in base_model.layers[-4:]:
        layer.trainable = False
    x = base_model.layers[-1].output  # es la salida del ultimo activation despues del add
    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.GlobalMaxPool2D()(x)
    # output layer
    #---1) NO LINEAL + LINEAL
    prepredictions = Dense(256, activation='relu')(x)
    predictions = Dense(classes, activation='sigmoid')(prepredictions)

    #---2) LINEAL
    predictions = Dense(classes, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    return model