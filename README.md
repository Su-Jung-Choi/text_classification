# IR project - Research Paper Classification: A Neural Network Approach with Feature Selection and TF-IDF

This project was completed in Spring 2024 as a final group project for the Information Retrieval course. The objective was to build a research paper classification system. 

We utilized a multi-label classification dataset obtained from Kaggle. It comprises 20,972 data entries and includes fields such as ID, Title, Abstract, and six labels representing six different categories: Computer Science, Physics, Mathematics, Statistics, Quantitative Biology, and Quantitative Finance. Papers within the dataset may belong to multiple categories, and if the paper belongs to a category, it is represented by ‘1’ for that label and ‘0’ otherwise. In this project, since we decided to focus on multi-class classification, the dataset is filtered to have only the papers that belong to a single category. Then, to address data imbalance while maintaining enough data, we dropped the two categories that have the least amount of papers. Thus, we sampled 6,544 papers in total, containing 1,636 papers per category in four categories: Computer Science, Physics, Mathematics, and Statistics.

By utilizing the feature selection technique, we reduced the number of features, which led to efficient training. Then, we produced TF-IDF vectors for text representation and used a neural network model for the classifier. The model achieved a classification accuracy of 77.78% and a precision of 81% on the test set. Such results leave room for improvement in
future work, which underscores the importance of continued exploration and refinement using different classification methodologies as well as incorporating advanced text representation techniques.



*Dataset Link: https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset

1. sampling.ipynb : This file is used to conduct preliminary analysis on data distribution through visualizations and to sample the data according to our project's purpose. It generates the 'four_category_dataset.csv,' which is used throughout the subsequent processes.
2. "_1tf_idf_save.py" : This file is used to execute preprocessing steps for text representation.
3. "_2feature_selection.py": This file is used to find best k for feature selection and to generate the graph of k features for mutual information, chi-squared and ANOVA f value. 
** Execution of this file may take hours based on the processing power of machine. 
4. "_3model_training_testing.py": This file is used to train the neural network model and generate plots for the progress of the training and validation loss, accuracy, and F1 and confusion matrix. 
5. "_4test_on_new_document.py" : This file is used to test the model on new data. 

![image](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/779e470d-e50c-4fde-8f3c-b3ffc29ac397)

Fig 1. Proposed framework for research paper classification


![image](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/d0373fae-6398-450f-9ff2-ddc0c87f9ad5)

Fig 2. Original Data Distribution


![image](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/9523e101-925d-451b-94b8-ed0a82571fdd)
Fig 3. Sampled Data Distribution

![image](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/d171081f-7d57-4304-89d2-4682d5f69132)

Fig 4. Feature Selection



![image](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/f009d5a2-02e7-4d22-8f73-d3d3f63f1f3c)

Fig 5. Model Summary


![image](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/ece080d9-f344-4708-9294-bfb3d34540a4)

Fig 6. Training and validation Accuracy Performance


![image](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/328123ea-2f35-4652-84d1-3ddf3d4db414)

Fig 7. Training and validation F1 Performance


![image](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/2d55f22d-7fbb-4882-bcc6-cff4db3ba24c)

Fig 8. Training and validation Loss Performance


![result_test](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/52e71286-9424-4d7d-9c1e-252a224afc3a)

Fig 9. Performance Results on Test Set with Selected Features


![result_with_entire_features](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/dc45d48c-a90d-45c0-a2da-f082a3720f33)

Fig 10. Performance Results on Test Set with all features


![image](https://github.com/Su-Jung-Choi/text_classification/assets/88897881/8c95f317-03b0-423f-ac95-d796c1909650)

Fig 11. Confusion Matrix for true labels vs. predicted labels









