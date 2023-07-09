# Email Classification

This is a simple email classifier application built using Streamlit. It classifies emails as either spam or non-spam (ham) using a machine learning model trained on a custom dataset.

## Dataset

The dataset used for training and testing the classifier contains a collection of sample emails. Each email is labeled as either spam or non-spam. The dataset includes the subject and body of the emails.

## Features

- Preprocessing: The text data is preprocessed by converting it to lowercase, removing stop words, and applying word stemming.
- Vectorization: The preprocessed text is transformed into numerical feature vectors using TF-IDF vectorization.
- Classification: A Random Forest classifier is trained on the vectorized features to predict the email labels.

Link to my application: [Email Classification](https://email-classification.streamlit.app/). 

