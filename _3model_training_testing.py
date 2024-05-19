"""
//********************************************************************
// CSC790: Information Retrieval and Web Search
// Project Title: Research Paper Classification
// by Sujung Choi, Jeniya Sultana
// Description: This file is used to train the neural network model using the k selected features and evaluates the model on the test set.
// It also includes training the model using the entire features to compare the performance.
// It generates plots for the progress of the training and validation loss, accuracy, and F1 score for both models.
//
//********************************************************************
"""
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
import random
from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras import regularizers
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

def NN_model(input_size, num_classes, learning_rate):
    """
    NN_model function creates a neural network model with the specified input size, number of classes, and learning rate.
    """
    dropout_rate = 0.5
    l2_reg = 0.001
    
    inputs = Input(shape=(input_size,))

    
    L1  = Dense(400, activation = 'relu', kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    L1_dropout = Dropout(dropout_rate)(L1)

    L2  = Dense(200, activation = 'relu', kernel_regularizer=regularizers.l2(l2_reg))(L1_dropout)
    L2_dropout = Dropout(dropout_rate)(L2)

    L3  = Dense(100, activation = 'relu', kernel_regularizer=regularizers.l2(l2_reg))(L2_dropout)
    L3_dropout = Dropout(dropout_rate)(L3)

    L4  = Dense(num_classes, activation='softmax')(L3_dropout)
    
    nn_model = Model(inputs=inputs, outputs=L4)


    # compile the model and calculate the accuracy
    nn_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate), 
                     metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score_metric])
    nn_model.summary() 
        
    return nn_model


def train_all(X, y, num_classes):
    """
    # train_all function trains the neural network model using all the features and evaluates the model on the test set.
    # It also generates plots for the progress of the training and validation loss, accuracy, and F1 score.
    # It is used to compare the performance of the model trained on all features with the model trained on selected features.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # set the hyperparameters
    learning_rate=0.0005
    batch_size = 32
    num_epochs = 30

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    nn_model = NN_model(X_train.shape[1], num_classes, learning_rate)

    history = nn_model.fit(X_train, y_train, validation_split=0.2, epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping])
    
    # list all data in history
    #print(history.history.keys()) 

    # Plot the training and validation F1 score
    plt.plot(history.history['f1_score_metric'], label='train')
    plt.plot(history.history['val_f1_score_metric'], label='validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.title(f'Training and Validation F1 score')
    plt.savefig(f'train_all/F1.png')
    #plt.show()   
    
    # Plot the training and validation accuracy
    plt.clf() # clear the plot
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy score')
    plt.title(f'Training and Validation Accuracy score')
    plt.savefig(f'train_all/accuracy.png')
    #plt.show() 

    # Plot the training and validation loss
    plt.clf() # clear the plot
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss')
    plt.savefig(f'train_all/loss.png')  
    #plt.show()       

    # Evaluate the model on the test set
    test_eval = nn_model.evaluate(X_test, y_test, verbose=1)
    print("\nError on test set:", round(test_eval[0], 3))
    print("Accuracy on test set:", round(test_eval[1] * 100, 3))
    print("Precision on test set:", round(test_eval[2] * 100, 3))
    print()


def save_models(trained_model, file_name):
    """
    save_models function saves the trained model to a file.
    """
    trained_model.save(file_name)


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    plot_confusion_matrix function generates and plots a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


def main():
    """
    # main function trains the neural network model using the k selected features and evaluates the model on the test set.
    # It also generates plots for the progress of the training and validation loss, accuracy, and F1 score.
    """
    
    # set the number of classes
    num_classes = 4

    # load the data
    X = np.load('X_4car.npy')
    y = np.load('y_4car.npy')

    # Train the model on all features
    train_all(X, y, num_classes)

    # convert the 2D array to 1D array to be suitable for the model
    y_1d = np.argmax(y, axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_1d, test_size=0.2, random_state=42)

    # set the best k number of features chosen by the chi2 method
    K = 10000

    # Create a selector object that will use the chi2 metric to select the best K features
    selector = SelectKBest(score_func=chi2, k=K) 

    # Fit the selector to training data
    selector.fit(X_train, y_train)

    # Transform training data to select only the top k features
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Convert the 1D arrays back to one-hot encoded format
    y_train_onehot = to_categorical(y_train, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)

    # set the hyperparameters
    learning_rate=0.0005
    batch_size = 32
    num_epochs = 30

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    nn_model = NN_model(K, num_classes, learning_rate)
    # Save the trained model
    save_models(nn_model, "models/nn_model_four.keras")


    history = nn_model.fit(X_train_selected, y_train_onehot, validation_split=0.2, epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping])
    
    # list all data in history
    #print(history.history.keys()) 

    # Plot the training and validation F1 score
    plt.plot(history.history['f1_score_metric'], label='train')
    plt.plot(history.history['val_f1_score_metric'], label='validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.title(f'Training and Validation F1 score')
    plt.savefig(f'F1.png')
    #plt.show()   
    
    # Plot the training and validation accuracy
    plt.clf() # clear the plot
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy score')
    plt.title(f'Training and Validation Accuracy score')
    plt.savefig(f'accuracy.png')
    #plt.show() 

    # Plot the training and validation loss
    plt.clf() # clear the plot
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss')
    plt.savefig(f'loss.png')  
    #plt.show()       

    # Evaluate the model on the test set
    test_eval = nn_model.evaluate(X_test_selected, y_test_onehot, verbose=1)
    print("\nError on test set:", round(test_eval[0], 3))
    print("Accuracy on test set:", round(test_eval[1] * 100, 3))
    print("Precision on test set:", round(test_eval[2] * 100, 3))
    print()


    # Predict labels on the test set
    y_pred = nn_model.predict(X_test_selected)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Plot confusion matrix
    class_names = ['Computer_Science', 'Physics', 'Mathematics', 'Statistics']  
    plot_confusion_matrix(y_test, y_pred_classes, class_names)
    


if __name__ == "__main__":
    main()

