# Next Word Prediction 

This project implements a **Next Word Prediction** system using  deep learning model trained on *Shakespeare's Hamlet* from the NLTK Gutenberg corpus.  
The trained model predicts the next word in a sequence of text and is deployed with a **Streamlit** web interface.

## Features
- **Text preprocessing**: tokenization, sequence generation, and padding.
- **Deep Learning model**: Embedding + GRU layers [or LSTM can be added in MODEL section of experiments.ipynb file in place of GRU].
- **EarlyStopping** to prevent overfitting.
- **Interactive web app** built with Streamlit.
- **Model and tokenizer persistence** for reuse without retraining.

## Tech Stack
- **Python 3.11**
- **TensorFlow / Keras**
- **NLTK**
- **NumPy / Pandas**
- **Scikit-learn**
- **Streamlit**

##  Project Structure
 - hamlet.txt # Training corpus
 - next_word_lstm.h5 # Saved model
 - tokenizer.pickle # Saved tokenizer
 - experiments.ipynb # Model training script
 - app.py # Streamlit app
 - requirements.txt # Dependencies
 - README.md # Project documentation


