seed = 2022

from scipy.io import loadmat

# Load the .mat file to a variable
patient_1_dataset = loadmat(r"Brain_SVM/data/dataset/ID0065C01_dataset.mat")

# Understand what loadmat returns
print(type(patient_1_dataset))

# See what elements are contained in dataset
print(patient_1_dataset.keys())

data = patient_1_dataset['data']
labels = patient_1_dataset['label']

# What are the data and labels variables?
print(f"'data' type: {type(data)}")
print(f"'labels' type: {type(labels)}")

# See the dimensions of "data" and "labels"
print(f"Samples array size: {data.shape}")
print(f"Labels array size: {labels.shape}")

# Load datasets from other patients and mix all together
import numpy as np
patient_2_dataset = loadmat(r"Brain_SVM/data/dataset/ID0067C01_dataset.mat")
patient_3_dataset = loadmat(r"Brain_SVM/data/dataset/ID0070C02_dataset.mat")
data = np.concatenate(
[
patient_1_dataset["data"],
patient_2_dataset["data"],
patient_3_dataset["data"],
],
axis=0,
)
labels = np.concatenate(
[
patient_1_dataset["label"],
patient_2_dataset["label"],
patient_3_dataset["label"],
],
axis=0,
)
# See the dimensions of "data" and "labels"
print(f"Samples array size: {data.shape}")
print(f"Labels array size: {labels.shape}")

# How many labels do we have?
print(f"Unique labels: {np.unique(labels, return_counts=True)}")
print(f"Different number of labels: {len(np.unique(labels))}")

from sklearn.svm import SVC
# Create an instance of the model
model = SVC(kernel="linear", probability=True, random_state=seed)

# To change the shape of labels from (X, 1) to (X, ), we do:
labels = labels.ravel()
print(f"New shape of labels array: {labels.shape}")
# Then we can fit the model without any warning
model.fit(X=data, y=labels)

predictions = model.predict(data)
print(predictions)

print(model.predict_proba(data))

from sklearn.metrics import accuracy_score
# Compute the accuracy of the predictions
acc = accuracy_score(y_true=labels, y_pred=predictions)
print(f"ACCURACY: {100*acc:.2f}%")

# Exercice 3
patient_4_dataset = loadmat(r"Brain_SVM/data/dataset/ID0071C02_dataset.mat")
data   = patient_4_dataset["data"]
labels = patient_4_dataset["label"]

predictions = model.predict(data)

acc = accuracy_score(y_true=labels, y_pred=predictions)
print(f"ACCURACY: {100*acc:.2f}%")

## Generate visualizations

# Load entire hyperspectral cube
patient_id = "ID0071C02"
preprocessed_mat = loadmat(rf"Brain_SVM/data/cubes/SNAPimages{patient_id}_cropped_Pre-processed.mat")
cube = preprocessed_mat["preProcessedImage"]

# Predict method needs an input array of shape (n_samples n_features)
# but we have (width, height, bands). Therefore, we need to reshape the
# 3D cube into a 2D cube and then predict.
cube_reshaped = cube.reshape((cube.shape[0] * cube.shape[1]),
cube.shape[2])

pred_map = model.predict_proba(X=cube_reshaped)

# Generate a classification map
from classification_maps import ClassificationMap
cls_map = ClassificationMap(
map=pred_map,
cube_shape=cube.shape,
unique_labels=np.unique(labels),
)

# * Save the classification map
cls_map.plot(
title=f"Patient classified with {type(model).__name__}",
show_axis=False,
path_="./outputs/",
file_suffix=f"{patient_id}",
file_format="png",
)

# Generate the ground-truth map image
from ground_truth_maps import GroundTruthMap

gt = GroundTruthMap(r"Brain_SVM/data/ground-truth", patient_id)

# Save the ground truth map
gt.plot(
title=f"Ground truth from patient {patient_id}",
show_axis=False,
path_="./outputs/",
file_suffix=f"GT_{patient_id}",
file_format="png",
)

## Optimize a ML model by tuning its hyperparameters.

from sklearn.model_selection import GridSearchCV

hyperparameters = {"kernel": ("linear", "rbf"), "C": [1], "gamma":[1]}

model = GridSearchCV(estimator=SVC(probability=True,random_state=seed), param_grid=hyperparameters, verbose=4)

# Perform 5-fold cross validation to obtain the best parameters
# With those, automatically train the model with the same data passed
model.fit(X=data, y=labels)
print(f"Best parameters found: {model.best_params_}")

# Predict data from the patient not used during training (as done before)
new_predictions = model.predict(patient_new_dataset["data"])

acc = accuracy_score(y_true=patient_new_dataset["label"],y_pred=new_predictions)
print(f"ACCURACY (on new data with optimized SVM): {100*acc:.2f}%")

# Classify image with optimized SVM
pred_map = model.predict_proba(X=cube_reshaped)

cls_map = ClassificationMap(
map=pred_map,
cube_shape=cube.shape,
unique_labels=np.unique(labels),
)

# Save the classification map
cls_map.plot(
    title=f"Patient classified with optimized {type(model.estimator).__name__}",
    show_axis=False,
    path_="./outputs/",
    file_suffix=f"{patient_id}_optimized",
    file_format="png",
)