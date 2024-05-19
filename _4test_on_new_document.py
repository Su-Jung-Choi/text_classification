"""
//********************************************************************
// CSC790: Information Retrieval and Web Search
// Project Title: Research Paper Classification
// by Sujung Choi, Jeniya Sultana
// Description: This file is used to test the model on a new document.
//
//********************************************************************
"""
import pandas as pd
from nltk.tokenize import word_tokenize  # for tokenization
from nltk.stem import PorterStemmer  # for stemming 
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
import random
from keras.models import load_model
import json
from sklearn.feature_selection import SelectKBest, chi2
from _1tf_idf_save import process_text_file, load_stopwords, load_special_characters
from _3model_training_testing import f1_score_metric

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# this function find the set of unique terms in the collection
def finding_unique_terms(term_frequency_dataset):
    unique_terms = set()
    for doc_id, frequencies in term_frequency_dataset.items():
        unique_terms.update(frequencies['content'].keys())
    return unique_terms


# this function reads new text data
def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

# this function process new data
def preprocessing_new_data(file_path, stopwords, special_chars, unique_terms, idf_vector):
    text = read_text_from_file(file_path)
    doc_id = 1
    tf_data = {}
    tf = process_text_file(text, stopwords, special_chars)
    tf_data[doc_id] = {'content': tf,  'label': None}
    new_doc_unique_terms = finding_unique_terms(tf_data)
    common_terms = set(new_doc_unique_terms).intersection(unique_terms)
    # Initialize TF vector for the new document
    tf_vector = np.zeros(len(unique_terms))
    unique_terms_list = list(unique_terms)
    # Calculate TF for each word in the new document
    for term in common_terms:
        tf_vector[unique_terms_list.index(term)] = tf[term] / sum(tf.values())
    
    # Compute TF-IDF
    tf_idf_vector =  tf_vector * idf_vector
    

    return tf_idf_vector



# load idf_vector
def load_vector_from_file(file_path):

    return np.load(file_path)


# load unique terms
def load_unique_terms_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        unique_terms = [line.strip() for line in f]
    return unique_terms


def main():
    new_document = 'new_research_paper.txt'
    stopwords_file = "stopwords/stopwords.txt"
    special_chars_file = "special_chars/special-chars.txt"
    custom_stopwords = load_stopwords(stopwords_file)
    special_chars = load_special_characters(special_chars_file)

    unique_terms = load_unique_terms_from_file('unique_terms_4_classes.txt')
 
    idf_vector = load_vector_from_file('idf_vector_4_classes.npy')

    tf_idf_vector_new_data = preprocessing_new_data(new_document, custom_stopwords, special_chars, unique_terms, idf_vector)
    tf_idf_vector_new_data = np.array(tf_idf_vector_new_data)


    # load the data
    X = np.load('X_4car.npy')
    y = np.load('y_4car.npy')

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

    X_new_selected = tf_idf_vector_new_data.reshape(1, -1)
    X_new_selected = selector.transform(X_new_selected)  

    trained_model = load_model("models/nn_model_four.keras", custom_objects={'f1_score_metric': f1_score_metric})
    predictions = trained_model.predict(X_new_selected)

    class_names = ['Computer_Science', 'Physics', 'Mathematics', 'Statistics']  

    # Loop through each prediction
    for prediction in predictions:
        # Get the predicted class index
        predicted_class_index = np.argmax(prediction)
        
        # Get the corresponding class name
        predicted_class_name = class_names[predicted_class_index]
        
        # Get the confidence score for the predicted class
        confidence_score = prediction[predicted_class_index]
        
        # Print the class name and confidence score
        print(f"\nPredicted Class: {predicted_class_name}, Confidence Score: {confidence_score}")


if __name__ == "__main__":
    main()
