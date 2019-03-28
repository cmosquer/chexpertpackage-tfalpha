import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def rename(csv_path, train_dir, new_name):
    data = pd.read_csv(csv_path)
    patients = os.listdir(train_dir)
    paths = []
    for patient in patients:
        estudios = os.listdir(train_dir + patient)
        for estudio in estudios:
            imagenes = os.listdir(train_dir + patient + '/' + estudio)
            for imagen in imagenes:
                nuevo_nombre = patient + '_' + estudio + '_' + imagen
                os.rename(train_dir + patient + '/' + estudio + '/' + imagen, train_dir + patient + '/' + estudio+'/' + nuevo_nombre)
                paths.append(train_dir + patient + '/' + estudio + '/' + nuevo_nombre)
    paths.sort()
    if len(paths) != len(data.index):
        paths = paths[:(abs(len(paths)-len(data.index)))]

    data['Path'] = paths
    data.to_csv(new_name)


def uZero(labels_df,limit=None,classes_names=None):
    labels = list([""] * len(labels_df.index))
    column_names = labels_df.columns.values
    for i, row in labels_df.iterrows():
        label_i = []
        if limit is not None:
            if i > limit:
                break
        for col_name in column_names:
            if row[col_name] == 1.0:
                label_i.append(str(col_name))
        labels[i] = label_i
    if classes_names is not None:
        mlb = MultiLabelBinarizer(classes=classes_names)
    else:
        mlb = MultiLabelBinarizer()
    multihotencoded = mlb.fit_transform(labels)

    return mlb, multihotencoded


def uOne(labels_df,limit=None,classes_names=None):
    labels = list([""] * len(labels_df.index))
    column_names = labels_df.columns.values
    for i, row in labels_df.iterrows():
        label_i = []
        if limit is not None:
            if i > limit:
                break
        for col_name in column_names:
            if row[col_name] == -1.0:
                # print("Era -1")
                label_i.append(str(col_name))
            if row[col_name] == 1.0:
                # print("era 1")
                label_i.append(str(col_name))
        # print(label_i)
        labels[i] = label_i

    if classes_names is not None:
        mlb = MultiLabelBinarizer(classes=classes_names)
    else:
        mlb = MultiLabelBinarizer()
    multihotencoded = mlb.fit_transform(labels)

    return mlb, multihotencoded


def uIgnore(labels_df,limit=None,classes_names=None):
    orig_limit=limit
    labels = list([""] * len(labels_df.index))
    column_names = labels_df.columns.values
    original_N =str(len(labels_df.index))
    j=0
    flag=1
    for i, row in labels_df.iterrows():
        label_i = []
        if i%1000==0:
            print("Cargando imagen " + str(i) + " de " + original_N)
        if limit is not None:
            if i > limit-1:
                break
        for col_name in column_names:
            if row[col_name] == 1.0:
                # print("era 1")
                label_i.append(str(col_name))
            if row[col_name] == -1.0:
                # print("Era -1")
                label_i =[]
                flag=0
                #print("Se elimina la fila ", str(i))
                break
        # print(label_i)
        if flag==1 and label_i is None:
            labels_df.drop([i], inplace=True)
            j+=1
        if not label_i and limit is not None:
                limit += 1
        if not label_i:
            j+=1
            labels_df.drop([i], inplace=True)
        labels[i] = label_i
        flag=1

    print("Antes: " + str(len(labels)))
    labels = list(filter(None, labels))
    print("Despues: " + str(len(labels)))
    print("Se dropearon " + str(j) + " imagenes.")
    if classes_names is not None:
        mlb = MultiLabelBinarizer(classes=classes_names)
    else:
        mlb = MultiLabelBinarizer()

    multihotencoded = mlb.fit_transform(labels)
    print(mlb.classes_)
    return mlb, multihotencoded, labels_df



def uRegression(labels_df,limit=None):
    SEGURO_SI = 1
    SEGURO_NO = 0
    POSIBLE_NO = 0.1
    POSIBLE_SI = 0.6
    classes=14
    labels = list([""] * len(labels_df.index))
    column_names = labels_df.columns.values[-classes:]

    for i, row in labels_df.iterrows():
        label_i = []
        if limit is not None:
            if i > limit:
                break
        for col_name in column_names:
            # Todas las columnas de patologia tienen que ser un append al label_i
            if row[col_name] == 0.0:
                label_i.append(SEGURO_NO)
                # print("era cero")
            if math.isnan(row[col_name]):
                label_i.append(POSIBLE_NO)
                # print("Era nan")
            if row[col_name] == -1.0:
                label_i.append(POSIBLE_SI)
                # print("era -1")
            if row[col_name] == 1.0:
                # print("era 1")
                label_i.append(SEGURO_SI)
        # print(label_i)
        labels[i] = label_i

    return labels


def proccessLabels(csv_path,uncertainMode,limit=None, classes=None):

    #new_name = os.path.dirname(csv_path) + '/' + IDexperim + '_renamed.csv'
    #if not os.path.isfile((images_path+'renamedFlag.txt')):
    #    f = open(images_path+"renamedFlag.txt", "w+")
    #    f.close()
    #    rename(csv_path, images_path, new_name)
    #labels_df = pd.read_csv(new_name)
    labels_df = pd.read_csv(csv_path)

    if uncertainMode != 'regression':
        if uncertainMode=='ignore':
            mlb, multihotencoded, labels_df = uIgnore(labels_df,limit,classes_names=classes)
        if uncertainMode=='zero':
            mlb, multihotencoded = uZero(labels_df,limit,classes_names=classes)
        if uncertainMode == 'one':
            mlb, multihotencoded = uOne(labels_df,limit,classes_names=classes)

        if limit is not None:
            labels = multihotencoded[0:limit]
            labels_df = labels_df[0:limit]
        else:
            labels= multihotencoded


        print("Cantidad de imagenes de validación: " + str(int(multihotencoded.size / multihotencoded[0].size)))

    if uncertainMode=='regression':
        labels = uRegression(labels_df,limit)
        return labels_df, labels
    return labels_df, labels, mlb


def loadImages(labels_df_final, limit=None, height = 128, width = 128):

    paths = labels_df_final['Path']

    if limit is not None:
        paths= paths[0:limit]

    i = 0
    data = []
    for p in paths:
        img = cv2.imread(p)
        if i % 1000 == 0:
            print('Reading image {} from {}'.format(i + 1, p))
        if type(img) == np.ndarray:
            # resize image and add to data set
            img = cv2.resize(img, (height, width))
            data.append(img)
        else:
            print(type(img))


        #if i > limit:
        #    break
        i += 1


    data = np.array(data, dtype="float64") /255.0

    return data

