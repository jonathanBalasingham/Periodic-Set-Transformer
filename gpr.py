"""
This is a modifed version of the code from:
Jakob Ropers, Marco M. Mosca, Olga Anosova, Vitaliy Kurlin, and Andrew I. Cooper. Fast pre-
dictions of lattice energies by continuous isometry invariants of crystal structures. In Alexei
Pozanenko, Sergey Stupnikov, Bernhard Thalheim, Eva Mendez, and Nadezhda Kiselyova (eds.),
Data Analytics and Management in Data Intensive Domains, pp. 178–192, Cham, 2022. Springer
International Publishing. ISBN 978-3-031-12285-9.
"""

import amd
import numpy as np
import pickle
import math
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RationalQuadratic, RBF, ConstantKernel, Matern, ExpSineSquared

files = ["./data/P2_Predicted_Structures.cif",
         "./data/P1_Predicted_Structures.cif",
         "./data/P1M_Predicted_Structures.cif",
         "./data/P2M_Predicted_Structures.cif"]

sizes = []

def create_data(k =1000):
    import os
    print("creating data...")
    for file in files:
        print(file)
        r = amd.CifReader(file)
        periodic_sets = [i for i in r]
        sizes.append(len(periodic_sets))
        energies = [float(ps.name.split("_")[0]) for ps in periodic_sets]
        amds = np.vstack([amd.AMD(ps, k=k) for ps in periodic_sets])
        energies = np.array(energies).reshape((-1,1))
        data = np.hstack([amds, energies])
        with open("./data/amds_" + os.path.basename(file).split(".")[0], "wb") as f:
            pickle.dump(data, f)


create_data(100)

def read_data(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


data_files = ["./data/amds_P2_Predicted_Structures",
              "./data/amds_P1_Predicted_Structures",
              "./data/amds_P1M_Predicted_Structures",
              "./data/amds_P2M_Predicted_Structures"]


data = [read_data(file) for file in data_files]
data = np.vstack(data)

feature_data = data[:, :-1]
label_data = data[:, -1]


def data(feature_data, label_data):
    feature_data = np.nan_to_num(feature_data)
    feature_data = feature_data[:, :100]
    ## MinMax Scaler
    feature_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_scaled = feature_scaler.fit_transform(feature_data)

    label_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    y_scaled = label_scaler.fit_transform(label_data.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=sizes[-1] / np.sum(sizes), shuffle=False)

    return X_train, y_train, X_test, y_test, label_scaler


X_train, y_train, X_test, y_test, label_scaler = data(feature_data, label_data)

kernel = RationalQuadratic()

gpr = GaussianProcessRegressor(kernel=kernel)

print("training...")
gpr.fit(X_train, y_train)
mean_predictions, std_predictions = gpr.predict(X_test, return_std=True)
std_predictions = std_predictions.reshape(-1,1)
scaler = np.divide(std_predictions, mean_predictions)

mean_predictions = label_scaler.inverse_transform(mean_predictions)
std_predictions = np.multiply(scaler,mean_predictions)
y_test = label_scaler.inverse_transform(y_test)

average_loss = 0
average_loss_percentage = 0
average_loss_percentage_rel_range = 0
counter = 0
rms = 0

error_ranges = np.array((0, 0, 0, 0, 0, 0))

max_value = -999999.99
min_value = 999999.99

for label in y_test:
    if (label > max_value):
        max_value = label

    if (label < min_value):
        min_value = label

label_range = abs(max_value - min_value)

for i, prediction in enumerate(mean_predictions):
    percentage_difference = abs((abs(prediction - y_test[i]) / y_test[i]) * 100)
    percentage_difference2 = abs((abs(prediction - y_test[i]) / label_range) * 100)
    loss = abs(prediction - y_test[i])
    average_loss += loss

    rms += loss ** 2

    if (loss <= 1.0):
        error_ranges[0] += 1
    elif (loss <= 2.0):
        error_ranges[1] += 1
    elif (loss <= 4.0):
        error_ranges[2] += 1
    elif (loss <= 8.0):
        error_ranges[3] += 1
    elif (loss <= 10.0):
        error_ranges[4] += 1
    else:
        error_ranges[5] += 1

    average_loss_percentage += percentage_difference
    average_loss_percentage_rel_range += percentage_difference2
    counter += 1

rms = math.sqrt(rms / counter)

print()
print("SUMMARY:")
print()
print("Root Mean Squared Error: " + str(rms))
print("Mean Absolute Error: " + str(average_loss / counter))
print("Mean Absolute Percentage Error: " + str(average_loss_percentage / counter) + "%")
print(
    "Mean Absolute Percentage Error relative to Label Range: " + str(average_loss_percentage_rel_range / counter) + "%")
print("Accuracy: " + str(100 - (average_loss_percentage / counter)) + "%")
print()
print("BREAKDOWN:")
print("   Error <= 1.0 kJ/mol: " + str(error_ranges[0]) + " or " + str(
    (error_ranges[0] / counter) * 100) + "% of Test Set")
print("   Error <= 2.0 kJ/mol: " + str(error_ranges[1]) + " or " + str(
    (error_ranges[1] / counter) * 100) + "% of Test Set")
print("   Error <= 4.0 kJ/mol: " + str(error_ranges[2]) + " or " + str(
    (error_ranges[2] / counter) * 100) + "% of Test Set")
print("   Error <= 8.0 kJ/mol: " + str(error_ranges[3]) + " or " + str(
    (error_ranges[3] / counter) * 100) + "% of Test Set")
print("   Error <= 10.0.0 kJ/mol: " + str(error_ranges[4]) + " or " + str(
    (error_ranges[4] / counter) * 100) + "% of Test Set")
print("   Error > 10.0 kJ/mol: " + str(error_ranges[5]) + " or " + str(
    (error_ranges[5] / counter) * 100) + "% of Test Set")
print("----------------------------------------------------------------------------------------------")