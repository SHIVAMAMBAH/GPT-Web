### <div align = "center">GPT-Web</div>
- GPT's working and architecture is explained and displayed on the web step by step 
- To know the working of the GPT model, visit the repository [GPT](https://github.com/SHIVAMAMBAH/GPT-Model)
- To see the code [click here](https://github.com/SHIVAMAMBAH/GPT-Web/blob/main/model.py)

##
- We have implemented the **GPT-2 Small Model with 117M** parameters
- We have used the dash librarby of the python for web application
- To install the dash librarby, open command prompt and type :
```
pip install dash[html,core,bootstrap]
```
- We have used the **transformers library** to access the parameters of th Model.
- To install the transformers library, open command prompt and type : 
```
pip install transformers
```
- Check the version of the transformers library
```
transformers.__version__
```
- The transformers library contains the GPT2 model
- To access the model from transformers, import
```
from transformers import GPT2Model
```
- After accessing the model load the model
```
model_name = "gpt2"
model = GPT2Model.from_pretrained(model_name)
```
- The number of dimension in the model is 768 :
```
d_model = model.config.n_embd
```
- Also load GPT2Tokenizer for tokenization
```
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
```
- The first step is to pass the text and then tokenize it.
- The second step is to convert them into IDs
```
#Inside the update_output function
inputs = tokenizer(input_text, return_tensors ='pt', add_special_tokens = False)
tokens = tokenizer.tokenize(input_text)
token_ids = inputs['input_ids'][0].tolist()
```
- Tokens and their IDs are stored in the [file](https://github.com/SHIVAMAMBAH/GPT-Web/blob/main/vocab.json).
- The third step is to convert these IDs into their corresponding embedding matrix and add positional encodings.
```
def get_embeddings_and_positional_encodings(input_text):
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract token embeddings and positional encodings
    embeddings = outputs.last_hidden_state  # Token embeddings
    positional_encodings = model.wpe.weight[:embeddings.size(1)]  # Positional encodings

    return embeddings, positional_encodings

# inside update_output function
embeddings, positional_encodings = get_embeddings_and_positional_encodings(input_text)
sum_emb_pos = embeddings + positional_encodings
```
- The next step is to pass the sum of embeddings and positional_encodings throught the encoder layers.
- GPT-2 Small contains 12 encoder layers.
- In each encoder layer there are **Multi-Head Self-Attention (MHSA) layer** and **Feed-Forward Neural-Network (FFNN) Layer**
- In each MHSA, there are 12 heads.
- So when we pass sum of embeddings and positional_encodings through the encoder layers to the encoder, it will pass through the 12 heads of the MHSA of the first encoder layer.
- In each head, we calculate the *Query (Q)*, *Key (K)* and *Value (V)*, *attention_scores* and *attention_weights* and then *context_vector*
- The following class calculates these values of a particular head of a particular layera and returns those values.
```
class MultiHeadSelfAttention:
    def __init__(self,layer_number:int ,head_number: int):
        self.model = GPT2Model.from_pretrained("gpt2")  # Using GPT-2 small directly
        self.head_number = head_number
        self.layer_number = layer_number
        
        #validate number of layers in the model (12 layers)
        n_layers = 12
        if layer_number<0 or layer_number>=n_layers:
            raise ValueError(f"Layer number must be between 0 and {n_layers-1}")
        
        # Validate head number (12 in each MHSA)
        n_heads = self.model.config.n_head
        if head_number < 0 or head_number >= n_heads:
            raise ValueError(f"Head number must be between 0 and {n_heads - 1}")

    def get_attention_values(self, embeddings: torch.Tensor):
        # Access the specified layer of the model
        layer = self.model.h[self.layer_number]
        mhsa = layer.attn

        # Get the combined Q, K, V weight matrix and split them
        c_attn_weight = mhsa.c_attn.weight  # Shape: (3 * d_model, d_model)
        d_model = self.model.config.n_embd
        n_heads = self.model.config.n_head
        head_dim = d_model // n_heads

        # Reshape to split into Q, K, and V matrices for all heads
        qkv_weight = c_attn_weight.view(3, n_heads, head_dim, d_model)  # Shape: (3, n_heads, head_dim, d_model)

        #   Extract weights for the specified attention head
        q_weight = qkv_weight[0, self.head_number]  # Q weight for specified head
        k_weight = qkv_weight[1, self.head_number]  # K weight for specified head
        v_weight = qkv_weight[2, self.head_number]  # V weight for specified head

        # Project embeddings into Q, K, and V matrices
        Q = torch.matmul(embeddings, q_weight.T)  # Shape: (batch_size, seq_length, head_dim)
        K = torch.matmul(embeddings, k_weight.T)
        V = torch.matmul(embeddings, v_weight.T)

        # Ensure Q and K have compatible shapes for matrix multiplication
        Q = Q.unsqueeze(1)  # Add an extra dimension if needed
        K = K.unsqueeze(1)

        # Calculate attention scores (dot product of Q and K)
        attention_scores = torch.matmul(Q, K.mT) / (head_dim ** 0.5)  # Use mT for transposition

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Calculate the context vector (weighted sum of V)
        context_vector = torch.matmul(attention_weights, V)

        return {
            "q-weights": q_weight,
            "k-weights": k_weight,
            "v-weights": v_weight,
            "Q": Q,
            "K": K,
            "V": V,
            "attention_scores": attention_scores,
            "attention_weights": attention_weights,
            "context_vector": context_vector
        }
```
- Here is the formulas for the Q, K, V, attention scores, attention weights and context vector

![MHSA (1)](https://github.com/user-attachments/assets/fda5a066-397d-4728-8cd1-7f8e7f573780)

- After getting context vector from each head they are concatenated
```
torch.cat((context_vector_11,context_vector_12,context_vector_13,context_vector_14,context_vector_15,context_vector_16,context_vector_17,context_vector_18,context_vector_19,context_vector_110,context_vector_111,context_vector_112),dim=-1)
```
where context_vector11 denotes the context vector of first head of the first layer.
