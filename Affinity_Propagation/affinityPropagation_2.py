
import numpy as np

csv_file = open('Affinity_Propagation/air-quality-cyl.csv')

train_data = np.empty((0, 3),dtype=int)
test_data  = np.empty((0, 3),dtype=int)

for csv_line in csv_file:

    csv_fields = csv_line.split(',')

    csv_date   =  csv_fields[0].strip()

    if csv_date.find('2020-1') != -1 :
        csv_NO2_field  = csv_fields[3].strip()
        csv_PM10_filed = csv_fields[5].strip()
        csv_SO2_field  = csv_fields[7].strip()

        if (csv_NO2_field != '') and (csv_PM10_filed != '') and (csv_SO2_field != ''):
            csv_fields = [int(csv_NO2_field), int(csv_PM10_filed), int(csv_SO2_field)]
            train_data = np.append(train_data, [csv_fields], axis=0)

    if csv_date.find('2018-05-03') != -1 :
        csv_NO2_field  = csv_fields[3].strip()
        csv_PM10_filed = csv_fields[5].strip()
        csv_SO2_field  = csv_fields[7].strip()

        if (csv_NO2_field != '') and (csv_PM10_filed != '') and (csv_SO2_field != ''):
            csv_fields = [int(csv_NO2_field), int(csv_PM10_filed), int(csv_SO2_field)]
            test_data = np.append(test_data, [csv_fields], axis=0)

from sklearn.cluster import AffinityPropagation

ap = AffinityPropagation(damping=0.90,
                        affinity='euclidean',
                         preference=-8000.0,
                         max_iter=3000)

print('Numero de muestras para entrenamiento: ' + str(len(train_data)))
print('Numero de muestras para clasificar: ' + str(len(test_data)))

import time

start_time = time.time()

preds_training = ap.fit_predict(train_data)

end_time = time.time()
print('Fitting time: ' + str(end_time - start_time) + ' s')

start_time = time.time()

preds_test = ap.predict(test_data)

end_time = time.time()
print('Predicting time: ' + str(end_time - start_time) + ' s')

print('Number of clusters: ', len(ap.cluster_centers_indices_))

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection="3d")
ax.scatter3D(train_data[:,0], train_data[:,1],  train_data[:,2], c=preds_training, cmap='Accent')
plt.savefig('outputs/affinityPropagationOutput_TrainClustering.png')

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection="3d")
ax.scatter3D(test_data[:,0], test_data[:,1],  test_data[:,2], c=preds_test, cmap='Accent')
plt.savefig('outputs/affinityPropagationOutput_TestClustering.png')