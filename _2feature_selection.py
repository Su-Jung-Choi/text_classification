"""
//********************************************************************
// CSC790: Information Retrieval and Web Search
// Project Title: Research Paper Classification
// by Team 1 (Sujung Choi, Jeniya Sultana)
// May 2, 2024
//
// Description: This file is used to perform feature selection using mutual information, chi-square, and f_classif (ANOVA F-Value) methods.
//
//********************************************************************
"""
import pandas as pd
import os    # for parsing documents
from nltk.tokenize import word_tokenize  # for tokenization
from nltk.stem import PorterStemmer  # for stemming 
import numpy as np
import multiprocessing #for parallel processing
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
import random
from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
import pickle
import time

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


def f1_score_metric(y_true, y_pred):
    """
    f1_score_metric function calculates the F1 score metric for the model.
    """
    y_pred = tf.round(y_pred)
    true_positives = tf.keras.backend.sum(tf.round(y_true) * tf.round(y_pred))
    predicted_positives = tf.keras.backend.sum(tf.round(y_pred))
    possible_positives = tf.keras.backend.sum(tf.round(y_true))
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
     
    return f1


def NN_model(input_sz, num_classes, learning_rate):
    """
    NN_model function creates a neural network model with the specified input size, number of classes, and learning rate.
    """
    inputs = Input(shape=(input_sz,))
    
    L1  = Dense(400, activation = 'relu')(inputs)
    L2  = Dense(200, activation = 'relu')(L1)
    L3  = Dense(100, activation = 'relu')(L2)

    L4  = Dense(num_classes, activation='softmax')(L3)
        
    nn_model = Model(inputs=inputs, outputs=L4)
    
    # compile the model with the specified loss function, optimizer, and metrics
    nn_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate), 
                     metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score_metric])
    nn_model.summary() 
        
    return nn_model


def feature_selection(X, y, num_classes):
    """
    # feature_selection function performs feature selection by finding the best k features 
    # using mutual information, chi-square, and f_classif (ANOVA F-Value) methods.
    """
    # set the hyperparameters
    batch_size = 32
    epochs = 30
    learning_rate=0.0005

    # Convert the one-hot encoded labels to 1D arrays
    y_1d = np.argmax(y, axis=1)

    # Define a range of values for k (number of selected features)
    # k_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    start = 100
    end = 27000
    num_values = 20
    step = (end - start) / (num_values - 1)
    k_values = [int(start + step * i) for i in range(num_values)]

    # Initialize lists to store results
    num_features_selected = { 'mutual_info': [], 'chi2': [], 'f_classif': [] }
    accuracy_scores = { 'mutual_info': [], 'chi2': [], 'f_classif': [] }

    # Split the data into training and testing sets for mutual information
    X_train_mi, X_test_mi, y_train_mi, y_test_mi = train_test_split(X, y_1d, test_size=0.2, random_state=42)
    
    # iterate over different k values to find best k
    for k in k_values:
        # Perform mutual information feature selection
        mi = mutual_info_classif(X_train_mi, y_train_mi)
        top_k_indices = np.argsort(mi)[::-1][:k]
        X_train_mi_selected = X_train_mi[:, top_k_indices]
        X_test_mi_selected = X_test_mi[:, top_k_indices]
        
        # Convert the 1D arrays back to one-hot encoded format
        y_train_mi_onehot = to_categorical(y_train_mi, num_classes)
        y_test_mi_onehot = to_categorical(y_test_mi, num_classes)

        # Train a classifier for mutual information
        nn_mi = NN_model(k, num_classes, learning_rate)
        nn_mi.fit(X_train_mi_selected, y_train_mi_onehot, batch_size=batch_size, epochs=epochs, verbose=1)

        # save
        #with open('nn_mi.pkl', 'wb') as f_mi:
            #pickle.dump(nn_mi, f_mi)

        mi_results = nn_mi.evaluate(X_test_mi_selected, y_test_mi_onehot, verbose=1)
        print("Mutual Information results:", mi_results)
        accuracy_mi = mi_results[1]

        #--------------------------mutual information end-----------------------------------------

        # Perform chi-square feature selection
        chi2_selector = SelectKBest(chi2, k=k)
        X_selected_chi2 = chi2_selector.fit_transform(X, y_1d)

        # Split the data into training and testing sets for chi-square
        X_train_chi2, X_test_chi2, y_train_chi2, y_test_chi2 = train_test_split(X_selected_chi2, y_1d, test_size=0.2, random_state=42)
        
        # Convert the 1D arrays back to one-hot encoded format
        y_train_chi2_onehot = to_categorical(y_train_chi2, num_classes)
        y_test_chi2_onehot = to_categorical(y_test_chi2, num_classes)

        # Train a classifier for chi-square
        nn_chi2 = NN_model(k, num_classes, learning_rate)
        nn_chi2.fit(X_train_chi2, y_train_chi2_onehot, batch_size=batch_size, epochs=epochs, verbose=1)

        # save
        #with open('nn_chi2.pkl', 'wb') as f_chi2:
            #pickle.dump(nn_chi2, f_chi2)

        chi2_results = nn_chi2.evaluate(X_test_chi2, y_test_chi2_onehot, verbose=1)
        accuracy_chi2 = chi2_results[1]

        #-----------------------------chi2 end-----------------------------------------
        
        # Perform ANOVA f-value feature selection
        f_selector = SelectKBest(f_classif, k=k)
        X_selected_f = f_selector.fit_transform(X, y_1d)

        # Split the data into training and testing sets for f_classif
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_selected_f, y_1d, test_size=0.2, random_state=42)
        
        # Convert the 1D arrays back to one-hot encoded format
        y_train_f_onehot = to_categorical(y_train_f, num_classes)
        y_test_f_onehot = to_categorical(y_test_f, num_classes)

        # Train a classifier for f_classif
        nn_f = NN_model(k, num_classes, learning_rate)
        nn_f.fit(X_train_f, y_train_f_onehot, batch_size=batch_size, epochs=epochs, verbose=1)

        # save
        #with open('nn_f.pkl', 'wb') as f_f_classif:
            #pickle.dump(nn_f, f_f_classif)

        f_results = nn_f.evaluate(X_test_f, y_test_f_onehot, verbose=1)
        accuracy_f = f_results[1]

        #-------------------------------------------------------------------
        # Store results in the dictionaries
        num_features_selected['mutual_info'].append(k)
        accuracy_scores['mutual_info'].append(accuracy_mi)

        num_features_selected['chi2'].append(k)
        accuracy_scores['chi2'].append(accuracy_chi2)

        num_features_selected['f_classif'].append(k)
        accuracy_scores['f_classif'].append(accuracy_f)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_features_selected['mutual_info'], accuracy_scores['mutual_info'], marker='o', label='Mutual Information')
    plt.plot(num_features_selected['chi2'], accuracy_scores['chi2'], marker='o', label='Chi-Square')
    plt.plot(num_features_selected['f_classif'], accuracy_scores['f_classif'], marker='o', label='ANOVA F-Value')
    plt.xlabel('Number of Selected Features (k)')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs. Number of Selected Features')
    plt.legend()
    plt.grid(True)
    plt.savefig('feature_selection.png')
    plt.show()

def main():

    # Load the data
    X = np.load('X_4car.npy')
    y = np.load('y_4car.npy')

    start_time = time.perf_counter()

    num_classes = 4

    feature_selection(X, y, num_classes)

    finish_time = time.perf_counter()
    elapsed_time = (finish_time - start_time) / 60
    print(f'Finished in {round(elapsed_time, 2)} minute(s)')


if __name__ == "__main__":
    main()
