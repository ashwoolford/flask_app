from flask import Flask, request, jsonify
from flask_cors import CORS
# Common imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Data processing and visualization imports
import string
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Model building imports
from transformer_layer import TransformerLayer
from token_and_positional_embedding import TokenAndPositionalEmbedding

from utils import bangla_words_percentage

# variables 
vocab_size = 10000
max_seq_len = 40
embed_dim = 256
num_heads = 4
ff_dim = 128

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# load pre trained model
# Model
InputLayer = layers.Input(shape=(max_seq_len,), name="InputLayer")

# Embedding Layer
embeddings = TokenAndPositionalEmbedding(embed_dim, vocab_size, max_seq_len, name="EmbeddingLayer")(InputLayer)

# Transformer Layer
encodings = TransformerLayer(num_heads=num_heads, embedding_dims=embed_dim, ff_dim=ff_dim, dropout_rate=0.1, name="TransformerLayer")(embeddings)

# Classifier
gap = layers.GlobalAveragePooling1D(name="GlobalAveragePooling")(encodings)
drop = layers.Dropout(0.5, name="Dropout")(gap)
OutputLayer = layers.Dense(1, activation='sigmoid', name="OutputLayer")(drop)
model = keras.Model(InputLayer, OutputLayer, name="TransformerNet")
model.load_weights('./trained_model/my_model_weights.h5')
# model = tf.keras.models.load_model('./trained_model/my_model.tf')
model.summary()


# Specify the path to the SPAM text message dataset
data_path = './data/bangla_emails.csv'
data_frame = pd.read_csv(data_path)
X = data_frame['Email'].tolist()
y = data_frame['Status'].tolist()


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

        if bangla_words_percentage(input_string) >= 90:
            tokens = text_vectorizer([input_string])
            
            proba = 1 if model.predict(tokens, verbose=0)[0][0]>0.5 else 0
            pred = label_encoder.inverse_transform([proba])

            return jsonify({'prediction': pred[0].title()}), 200
        else:
            return jsonify({'error': 'Input has not enough bangla words'}), 422

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the app
if __name__ == '__main__':
    app.run(debug=True)