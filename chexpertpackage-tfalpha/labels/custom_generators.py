import numpy as np
import cv2
import keras
import random
import matplotlib.pyplot as plt
import urllib.request

def load_image(path, height=128, width=128, normalize=None):
    #if gs:
    if not (path.lower().endswith(('.png', '.jpg', '.jpeg'))):
        path = path + '.jpg'
    img = cv2.imread(path)
    if type(img) == np.ndarray:
        img = cv2.resize(img, (height, width))
        img = np.array(img, dtype="float64")
        if normalize is not None:
            if normalize is 'imagenet':
                img = img / 255.0
                means = [0.485, 0.456, 0.406]
                stds = [0.229, 0.224, 0.225]
            else:
                means = [normalize['mean']]*3
                stds = [normalize['std']]*3
            img = img - means
            img = img / stds
            plt.imshow(img[:,:,0],cmap='gray')
    else:
        print("No se encontró la imagen " + path)
        img = None
    return img


def load_label(inputs, path):
    for i,current_path in enumerate(inputs[0]):
        if current_path == path:
            return inputs[1][i]


def image_generator(inputs, batch_size, height=320, width=320):
    while True:

        batch_paths = np.random.choice(a=inputs[0], size=batch_size)
        batch_images=[]
        batch_labels=[]

        for path in batch_paths:
            image = load_image(path, height, width)
            label = load_label(inputs, path)
            batch_images.append(image)
            batch_labels.append(label)

        batch_x=np.array(batch_images)
        batch_y=np.array(batch_labels)

        yield(batch_x,batch_y)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paths,  labels, list_IDs=None,batch_size=5, dim=(320, 320), n_channels=3, steps=200,
                 n_classes=14, shuffle=False, normalize='imagenet'):
        'Initialization'
        self.dim = dim
        self.paths = paths
        self.batch_size = batch_size
        self.labels = labels
        if list_IDs is not None:
            self.list_IDs = list_IDs
        else:
            self.list_IDs = np.arange(labels.shape[0])
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.steps = steps
        self.normalize = normalize
        self.indexes = np.arange(len(self.list_IDs))
    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.list_IDs) / self.batch_size))
        return int(self.steps)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            print(("Epoch end..,Shuffling data"))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype='int') #np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = load_image(self.paths[ID], self.dim[0], self.dim[1], normalize=self.normalize)

            # Store class

            y[i, ] = self.labels[ID]
            #print("Path")
            #print(self.paths[ID])
            #print("Labels")
            #print(self.labels[ID])

        return X, y


def TestGenerator(paths, labels, dim=(320, 320), normalizer='imagenet'):
    X = np.empty((1, *dim, 3))
    #y = np.empty((labels.shape[0], n_classes), dtype='int')  # np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, path in enumerate(paths):
        # Store sample
        X[0, ] = load_image(path, dim[0], dim[1], normalize=normalizer)
        y = labels[i]
        yield X, y


def MeanGenerator(paths, dim=(320, 320)):
    X = np.empty((1, *dim, 3))
    #y = np.empty((labels.shape[0], n_classes), dtype='int')  # np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, path in enumerate(paths):
        X[0, ] = load_image(path, dim[0], dim[1])
        img_mean = np.mean(X)
        yield img_mean


def VarianceGenerator(paths, mean, dim=(320, 320)):
    X = np.empty((1, *dim, 3))
    # y = np.empty((labels.shape[0], n_classes), dtype='int')  # np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, path in enumerate(paths):
        X[0,] = load_image(path, dim[0], dim[1])
        substractmean = X - mean
        squared = np.square(substractmean)
        sum = np.sum(squared)
        img_var = np.sum(np.square(X-mean))/X.size
        yield img_var


def create_K_generators(paths,labels,K, **params):

    c = list(zip(paths, labels))

    random.shuffle(c)

    paths, labels = zip(*c)

    N_indexes = int(len(labels) / K)
    paths = np.asarray(paths)
    labels = np.asarray(labels)
    generators = []
    for k in range(K):
        if k==K-1:
            indexes=np.arange(k*N_indexes, labels.shape[0])
        else:
            indexes = np.arange(k * N_indexes, (k + 1) * N_indexes)
        curr_paths = paths[indexes]
        curr_labels = labels[indexes]
        generator = DataGenerator(curr_paths, curr_labels, **params)
        generators.append(generator)

    return generators