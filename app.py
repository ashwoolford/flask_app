from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
# Common imports
import tensorflow as tf
from tensorflow import keras

# Data processing and visualization imports
import string
import pandas as pd
import tensorflow.data as tfd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Model building imports
from sklearn.utils import class_weight
from tensorflow.keras import callbacks
from tensorflow.keras import Model, layers

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

class TransformerLayer(layers.Layer):
    
    def __init__(self, num_heads: int, dropout_rate: float, embedding_dims: int, ff_dim: int, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        
        # Initialize Parameters
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.embedding_dims = embedding_dims
        self.ff_dim = ff_dim
        
        # Initialize Layers
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dims, dropout=dropout_rate)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(embedding_dims)
        ])
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        """Forward pass of the Transformer Layer.
        
        Args:
            inputs: Tensor with shape `(batch_size, seq_len, embedding_dims)` representing the input sequence.
        
        Returns:
            Tensor with shape `(batch_size, seq_len, embedding_dims)` representing the output sequence after applying the Transformer Layer.
        """
        
        # Multi-Head Attention
        attention = self.mha(inputs, inputs, inputs)
        
        # Layer Normalization and Residual Connection
        normalized1 = self.ln1(attention + inputs)
        
        # Feedforward Network
        ffn_out = self.ffn(normalized1)
        
        # Layer Normalization and Residual Connection
        normalized2 = self.ln2(ffn_out + normalized1)
        
        return normalized2
    
    def get_config(self):
        """Get the configuration of the Transformer Layer.
        
        Returns:
            Dictionary with the configuration of the layer.
        """
        config = super(TransformerLayer, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "embedding_dims": self.embedding_dims,
            "ff_dim": self.ff_dim
        })
        return config

class TokenAndPositionalEmbedding(layers.Layer):
    
    def __init__(self, embedding_dims, vocab_size, seq_len, **kwargs):
        super(TokenAndPositionalEmbedding, self).__init__(**kwargs)
        
        # Initialize parameters
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.embed_scale = tf.math.sqrt(tf.cast(embedding_dims, tf.float32))
        
        # Define layers
        self.token_embedding = layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dims,
            name="token_embedding"
        )
        
        self.positional_embedding = layers.Embedding(
            input_dim=seq_len, 
            output_dim=embedding_dims,
            name="positional_embedding"
        )
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        
        # Token Embedding
        token_embedding = self.token_embedding(inputs)
        token_embedding *= self.embed_scale
        
        # Positional Embedding
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positional_embedding = self.positional_embedding(positions)
        
        # Add Token and Positional Embedding
        embeddings = token_embedding + positional_embedding
        
        return embeddings
        
    
    def get_config(self):
        config = super(TokenAndPositionalEmbedding, self).get_config()
        config.update({
            'embedding_dims': self.embedding_dims,
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
        })
        return config

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