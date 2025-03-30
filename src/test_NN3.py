from neural_network3 import NeuralNetwork as nn
from transformers import AutoTokenizer
# Setup tokenizer and model parameters
tokenizer_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
vocab_size = len(tokenizer)
embed_dim = 128
hidden_dim = 64

# Layer sizes for language model
layer_sizes = [vocab_size, embed_dim, hidden_dim, vocab_size]

model = nn(
    layer_sizes=layer_sizes,
    use_embedding=True,
    vocab_size=vocab_size,
    embed_dim=512,
    max_seq_length=1024,
    use_attention=True,
    num_attention_heads=8,
    attention_dropout=0.1,
    use_positional_embedding=True,
    positional_embedding_type='sinusoidal'
)


