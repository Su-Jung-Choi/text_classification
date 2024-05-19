"""
//********************************************************************
// CSC790: Information Retrieval and Web Search
// Project Title: Research Paper Classification
// by Team 1 (Sujung Choi, Jeniya Sultana)
// May 2, 2024
//
// Description: this file is to save numpy arrays for X(tf-idf) and y(labels) to a file for future use.
//
//********************************************************************
"""
import pandas as pd
from nltk.tokenize import word_tokenize  # for tokenization
from nltk.stem import PorterStemmer  # for stemming 
import numpy as np
import multiprocessing #for parallel processing
import tensorflow as tf
import random
import json

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def read_file(file_path):
    """
    # read_file function reads the csv file and returns the data
    """
    data = pd.read_csv(file_path)
    return data

def load_stopwords(stopwords_file):
    """
    # load_stopwords function loads the stopwords from the given stopwords file
    """
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        return set(file.read().splitlines())
    
def load_special_characters(special_char_file):
    """
    # load_special_characters function loads the special characters from the given special characters file
    """
    with open(special_char_file, 'r', encoding='utf-8') as file:
        punctuation_chars = file.read().strip()
    return punctuation_chars

def tokenize(text):
    """
    # tokenize function tokenizes the text
    """
    tokens = word_tokenize(text)
    return tokens

def lowercase(tokens):
    """
    # lowercase function converts the tokens to lowercase
    """
    return [token.lower() for token in tokens]

def remove_punctuation(tokens, special_chars):
    """
    # remove_punctuation function removes the punctuations from the tokens
    """
     # Create a translation table to remove punctuations
    translation_table = str.maketrans("", "", special_chars)
    
    # Remove punctuations from each token
    return [token.translate(translation_table) for token in tokens]

def remove_stopwords(tokens, custom_stopwords):
    """
    # remove_stopwords function removes the stopwords from the tokens
    """
    return [token for token in tokens if token not in custom_stopwords]

def stemming(tokens):
    """
    # stemming function stems the tokens using PorterStemmer
    """
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]

def calculate_term_frequency(tokens):
    """
    # calculate_term_frequency function calculates the term frequency for each file
    """
    term_frequency = {}
    for token in tokens:
        term_frequency[token] = term_frequency.get(token, 0) + 1
    return term_frequency

def process_text_file(data, custom_stopwords, special_chars):
    """
    # process_text_file function merges all the previous functions to process the text file
    # it includes tokenization, lowercasing, removing punctuations, removing stopwords, stemming and calculating term frequency
    """
    tokens = tokenize(data)
    lower_case_tokens = lowercase(tokens)
    punctuations_removed = remove_punctuation(lower_case_tokens, special_chars)
    filtered_tokens = remove_stopwords(punctuations_removed, custom_stopwords)
    stemmed_tokens = stemming(filtered_tokens)
    term_frequency = calculate_term_frequency(stemmed_tokens)
    return term_frequency

def process_file(title_abstract, custom_stopwords, special_chars):
    """
    # process_file function processes the file by calling the process_text_file function"""
    term_frequency_title_abstract = process_text_file(title_abstract, custom_stopwords, special_chars)
    return term_frequency_title_abstract

def process_text_dataset(file_path, custom_stopwords, special_chars):
    """
    # process_text_dataset function processes the text dataset by calling the process_file function
    # it reads the file, concatenates the title and abstract, and calculates the term frequency for each file
    # it uses multiprocessing to process the files in parallel
    """
    dataset = {} 
    data = read_file(file_path)
    num_of_documents = data.shape[0]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_file, [((row['TITLE'] + " " + row['ABSTRACT']), custom_stopwords, special_chars) for _, row in data.iterrows()])

    for term_frequency, row in zip(results, data.itertuples()):
        try:
            doc_id = row.ID
            label = [getattr(row, column) for column in ['Computer_Science', 'Physics', 'Mathematics', 'Statistics']]
            dataset[doc_id] = {
                'content': term_frequency,  # Storing the term frequency for concatenated title and abstract
                'label': label
            }
        except Exception as e:
            print(f"Error processing {row.Index}: {e}")

    return num_of_documents, dataset

def calculate_document_frequency(term_frequency_dataset):
    """
    # calculate_document_frequency function calculates the document frequency for each term
    """
    document_frequency = {}
    
    # initialize document frequency counts for each term to 0
    for doc_id, frequencies in term_frequency_dataset.items():
        for term in frequencies['content'].keys():
            document_frequency[term] = 0
    
    # count the number of documents containing each term
    for doc_id, frequencies in term_frequency_dataset.items():
        for term in frequencies['content'].keys():
            document_frequency[term] += 1
    
    return document_frequency

def idf(num_of_documents, document_frequency):
    """
    # idf function calculates the inverse document frequency (IDF) for each term
    """
    idf_vector = np.log10(num_of_documents / (np.array(list(document_frequency.values())) + 1))

    return idf_vector

def calculate_tf_idf(num_of_documents, tf_vectors, document_frequency):
    """
    # calculate_tf_idf function calculates the term frequency-inverse document frequency (tf-idf) for each document
    """
    # calculate idf
    idf_vector = idf(num_of_documents, document_frequency)

    # calculate tf-idf
    tf_idf_tuples = []
    for _, tf_vector in tf_vectors:  
        tf_idf_vector = tf_vector * idf_vector
        # Flatten the tf-idf vector and convert it to a list
        tf_idf_flat = tf_idf_vector.flatten().tolist()
        tf_idf_tuples.append(tf_idf_flat)

    return tf_idf_tuples, idf_vector

def finding_unique_terms(term_frequency_dataset):
    """
    # finding_unique_terms function finds the unique terms in the dataset
    """
    unique_terms = set()
    for doc_id, frequencies in term_frequency_dataset.items():
        unique_terms.update(frequencies['content'].keys())
    return unique_terms


def process_tf_for_each_document(doc_number, frequencies, unique_terms, max_length):
    """
    # process_tf_for_each_document function generates the term frequency vector for each document
    """
    vector = np.zeros(max_length)
    for term, freq in frequencies.items():
        if term in unique_terms:
            index = list(unique_terms).index(term)
            vector[index] = freq
    return doc_number, vector

def term_frequency_vector_generate(term_frequency_dataset, unique_terms):
    """
    # term_frequency_vector_generate function is used for multiprocessing of the process_tf_for_each_document function
    """
    max_length = len(unique_terms)
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_tf_for_each_document, [(doc_id, data['content'], unique_terms, max_length) 
                                                              for doc_id, data in term_frequency_dataset.items()])

    return results


def get_labels(term_frequency_dataset):
    """
    # get_labels function extracts the labels from the term frequency dataset
    """
    labels = [entry['label'] for entry in term_frequency_dataset.values()]

    return labels


def get_y(labels):
    """
    # get_y function converts the labels to binary matrix representation
    """
    labels_binary = np.array([[1 if label == 1 else 0 for label in sample_labels] for sample_labels in labels])
    return labels_binary


def save_np_array(array, name):
    """
    # save_np_array function saves the numpy array to a file
    """
    np.save(name, array)


def save_unique_terms_to_file(unique_terms, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for term in unique_terms:
            f.write(term + '\n')



def save_dict(dict, file_path):
    with open(file_path, "w") as json_file:
        json.dump(dict, json_file)



def main():
    """
    # main function reads the text dataset, processes the text dataset, generates tf vector for each document,
    # calculates document frequency, tf-idf vector, and extracts labels.
    # Then, it saves the X and y to numpy files for future use.
    """
    file_path = 'four_category_dataset.csv'
    stopwords_file = "stopwords/stopwords.txt"
    special_chars_file = "special_chars/special-chars.txt"
    custom_stopwords = load_stopwords(stopwords_file)
    special_chars = load_special_characters(special_chars_file)

    # process the text dataset
    num_of_documents, term_frequency_dataset = process_text_dataset(file_path, custom_stopwords, special_chars)
    unique_terms = finding_unique_terms(term_frequency_dataset)
    save_unique_terms_to_file(unique_terms, 'unique_terms_4_classes.txt')
    save_dict(term_frequency_dataset, 'term_frequency_dataset.json')

    # generates tf vector for each document
    term_frequency_vector = term_frequency_vector_generate(term_frequency_dataset, unique_terms)    

    # calculate document frequency and tf-idf vector
    document_frequencies = calculate_document_frequency(term_frequency_dataset)
    tf_idf_vector, idf_vector = calculate_tf_idf(num_of_documents, term_frequency_vector, document_frequencies) 

    save_np_array(tf_idf_vector, 'tf_idf_vector_4_classes.npy')
    save_np_array(idf_vector, 'idf_vector_4_classes.npy')

    # extract labels
    labels = get_labels(term_frequency_dataset)

    # set the input size to the number of unique terms
    input_size = len(unique_terms)

    # Convert the tf-idf vector to a numpy array
    X = np.array(tf_idf_vector)
    y = get_y(labels)
    
    # Save the X and y to numpy files for future use
    save_np_array(X, 'X_4car.npy')
    save_np_array(y, 'y_4car.npy')
    


if __name__ == "__main__":
    main()