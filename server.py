from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
# Common imports
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Data processing and visualization imports
import string
import pandas as pd
import plotly.express as px
import tensorflow.data as tfd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Model building imports
from sklearn.utils import class_weight
from tensorflow.keras import callbacks
from tensorflow.keras import Model, layers

# variables 
vocab_size = 10000
max_seq_len = 40

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# load pre trained model
model = tf.keras.models.load_model('./trained_model/my_model.tf')
model.summary()


# Specify the path to the SPAM text message dataset
data_path = './data/bangla_emails.csv'
data_frame = pd.read_csv(data_path)
X = data_frame['Email'].tolist()
y = data_frame['Status'].tolist()

def decode_tokens(tokens):
    """
    This function takes in a list of tokenized integers and returns the corresponding text based on the provided vocabulary.
    
    Args:
    - tokens: A list of integers representing tokenized text.
    - vocab: A list of words in the vocabulary corresponding to each integer index.
    
    Returns:
    - text: A string of decoded text.
    """
    text = " ".join(VOCAB[int(token)] for token in tokens).strip()
    return text

# Define a function to preprocess the text
def preprocess_text(text: str) -> str:
    """
    Preprocesses the text by removing punctuation, lowercasing, and stripping whitespace.
    """
    # Replace punctuation with spaces
    text = tf.strings.regex_replace(text, f"[{string.punctuation}]", " ")
    
    # Strip leading/trailing whitespace
    text = tf.strings.strip(text)
    
    return text
    

# Create a TextVectorization layer
text_vectorizer = layers.TextVectorization(
    max_tokens=vocab_size,                       # Maximum vocabulary size
    output_sequence_length=max_seq_len,          # Maximum sequence length
    standardize=preprocess_text,                 # Custom text preprocessing function
    pad_to_max_tokens=True,                      # Pad sequences to maximum length
    output_mode='int'                            # Output integer-encoded sequences
)

# Adapt the TextVectorization layer to the data
text_vectorizer.adapt(X)

# Get the vocabulary
VOCAB = text_vectorizer.get_vocabulary()

# Initialize label encoder
label_encoder = LabelEncoder()
label_encoder.fit_transform(y)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Assuming the input is a string and we need to preprocess it for the model
        input_string = data['input_string']

        tokens = text_vectorizer([input_string])
        print('tokens ', tokens)
        print('decode_tokens ', decode_tokens(tokens[0]))

        proba = 1 if model.predict(tokens, verbose=0)[0][0]>0.5 else 0
        print('proba ', proba)
        pred = label_encoder.inverse_transform([proba])
        print(f"Message: '{input_string}' | Prediction: {pred[0].title()}")
        

        return jsonify({'prediction': pred[0].title()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the app
if __name__ == '__main__':
    app.run(debug=True)