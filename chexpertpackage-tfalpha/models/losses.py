from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import roc_auc_score

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def AUC(y_true,y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
    return value

def AUC2(y_true, y_pred):
    value, update_op = tf.metrics.auc(y_true, y_pred)
    return update_op

def AUC_loss(y_true, y_pred):
    return -AUC(y_true, y_pred)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def my_binary_crossentropy(y_true, y_pred):
    bce = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return bce





class CategoricalTruePositives(keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred)
        values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        return self.true_positives.assign_add(tf.reduce_sum(values))  # TODO: fix

    def result(self):
        return tf.identity(self.true_positives)  # TODO: fix

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)