from tensorflow.keras import layers
from tensorflow import keras


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
