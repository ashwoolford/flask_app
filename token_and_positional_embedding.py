# Model building imports
import tensorflow as tf
from tensorflow.keras import layers

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
