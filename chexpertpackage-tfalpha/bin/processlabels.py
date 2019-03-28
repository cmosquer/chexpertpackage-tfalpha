import os
import sys
import pickle

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import chexpertpackage.bin  # noqa: F401
    __package__ = "chexpertpackage.bin"

from ..labels.createLabels import proccessLabels

#uncertainMode = "ignore" #Opciones: "ignore", "zero", "one", "regression"

dataset_path= '//lxestudios.hospitalitaliano.net/pacs/CheXpert-v1.0-small/train_custom.csv'
modes = ["ignore", "zero", "one"]
#modes = ["regression"]
for uncertainMode in modes:

    output_path_labels = "C:/ImageAnalytics/Torax/chexpertpackage/chexpertpackage/labels/" + uncertainMode + "__labels_custom.txt"
    output_path_paths ="C:/ImageAnalytics/Torax/chexpertpackage/chexpertpackage/labels/" + uncertainMode + "_paths_custom.txt"
    if uncertainMode=='regression':
        labels_df, labels= proccessLabels(dataset_path, uncertainMode)
    else:
        labels_df, labels, mlb = proccessLabels(dataset_path, uncertainMode)
        labels_df = labels_df.reset_index()

    print("Se uso el enfoque " + uncertainMode +". Se conservaron " + str(len(labels_df.index)) + " imágenes.")
    labels_list = [mlb.classes_,labels]
    with open(output_path_labels, "wb") as fp:   #Pickling
        pickle.dump(labels_list, fp)

    with open(output_path_paths, "wb") as fp:   #Pickling
        pickle.dump(labels_df["Path"], fp)

    print("Se guardo el dataset en " + output_path_labels + " y en " + output_path_paths)