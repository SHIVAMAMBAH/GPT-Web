import torch
from dash import Dash, dcc, html, Input, Output
from transformers import GPT2Tokenizer, GPT2Model
# import torch.nn.functional as F
# import math

# Load the GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)


# In a real model, these would be learned parameters
d_model = model.config.n_embd  # GPT-2 embedding size (768 for 'gpt2')


# Function to get the embeddings and positional encodings for the input text
def get_embeddings_and_positional_encodings(input_text):
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract token embeddings and positional encodings
    embeddings = outputs.last_hidden_state  # Token embeddings
    positional_encodings = model.wpe.weight[:embeddings.size(1)]  # Positional encodings

    return embeddings, positional_encodings


class MultiHeadSelfAttention:
    def __init__(self,layer_number:int, head_number: int):
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
    
layer_1_head_1 = MultiHeadSelfAttention(0, 0)
layer_1_head_2 = MultiHeadSelfAttention(0, 1)
layer_1_head_3 = MultiHeadSelfAttention(0, 2)
layer_1_head_4 = MultiHeadSelfAttention(0, 3)
layer_1_head_5 = MultiHeadSelfAttention(0, 4)
layer_1_head_6 = MultiHeadSelfAttention(0, 5)
layer_1_head_7 = MultiHeadSelfAttention(0, 6)
layer_1_head_8 = MultiHeadSelfAttention(0, 7)
layer_1_head_9 = MultiHeadSelfAttention(0, 8)
layer_1_head_10 = MultiHeadSelfAttention(0, 9)
layer_1_head_11 = MultiHeadSelfAttention(0, 10)
layer_1_head_12 = MultiHeadSelfAttention(0, 11)

# layer_2_head_1 = MultiHeadSelfAttention(1,0)
# layer_2_head_2 = MultiHeadSelfAttention(1,1)
# layer_2_head_3 = MultiHeadSelfAttention(1,2)
# layer_2_head_4 = MultiHeadSelfAttention(1,3)
# layer_2_head_5 = MultiHeadSelfAttention(1,4)
# layer_2_head_6 = MultiHeadSelfAttention(1,5)
# layer_2_head_7 = MultiHeadSelfAttention(1,6)
# layer_2_head_8 = MultiHeadSelfAttention(1,7)
# layer_2_head_9 = MultiHeadSelfAttention(1,8)
# layer_2_head_10 = MultiHeadSelfAttention(1,9)
# layer_2_head_11 = MultiHeadSelfAttention(1,10)
# layer_2_head_12 = MultiHeadSelfAttention(1,11)


app = Dash(__name__)

app.layout = html.Div([
    
    html.Div([
      html.H1("GPT-2 Small with 117M parameters and 12 attention heads.")  
    ]),
    
    # Input Text Area
    dcc.Textarea(
        id='input-text',
        placeholder='Enter text...',
        style={'width': '100%', 'height': 100, 'marginBottom': 20},
    ),
    # Token ID Output Box
    html.Div([
        html.H4('IDs', style={'textAlign': 'center'}),  # Centered Heading for ID
        html.Div(
            id='id-output',
            style={
                'whiteSpace': 'pre-line',
                'border': '2px solid #000000',  # Blue border for ID output
                'padding': '10px',
                'width': '100%',  # Same width as text input
                'height': '200px',  # Fixed height for ID output box
                'overflowY': 'scroll',  # Scrollable
                'marginBottom': 20,
                'font-family':'Courier New'
            }
        )
    ]),
    # Embedding and Positional Encoding Boxes Side by Side
    html.Div([
        html.Div([
            html.H4('Embedding Matrix', style={'textAlign': 'center'}),  # Heading for Embedding Matrix
            html.Div(
                id='embedding-output',
                style={
                    'whiteSpace': 'pre-line',
                    'border': '2px solid #000000',  # Green border for embedding output
                    'padding': '10px',
                    'width': '100%',  # Full width inside container div
                    'height': '400px',  # Fixed height
                    'overflowY': 'scroll',  # Scrollable
                    'wordBreak': 'break-all',  # Word wrapping
                    'font-family':'Courier New'
                }
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),
        html.Div([
            html.H4('Positional Encodings', style={'textAlign': 'center'}),  # Heading for Positional Encoding
            html.Div(
                id='positional-encoding-output',
                style={
                    'whiteSpace': 'pre-line',
                    'border': '2px solid #000000',  # Red border for positional encodings
                    'padding': '10px',
                    'width': '100%',  # Full width inside container div
                    'height': '400px',  # Fixed height
                    'overflowY': 'scroll',  # Scrollable
                    'wordBreak': 'break-all',  # Word wrapping
                    'font-family': 'Courier New'
                }
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ]),
    # Sum Box Below Embeddings and Positional Encodings
    html.Div([
        html.H4('Sum of Embeddings and Positional Encodings', style={'textAlign': 'center'}),  # Heading for Sum Box
        html.Div(
            id='sum-output',
            style={
                'whiteSpace': 'pre-line',
                'border': '2px solid #000000',  # Orange border for sum output
                'padding': '10px',
                'width': '100%',  # Full width
                'height': '200px',  # Fixed height for sum output box
                'overflowY': 'scroll',  # Scrollable
                'marginTop': 20,  # Add space between positional encoding and sum
                'wordBreak': 'break-all',  # Word wrapping
                'font-family': 'Courier New'
            }
        )
    ]),
    
    html.Div([
      html.H1("GPT-2 Encoder"),
      html.H4("There are 12 layers of Encoder in GPT-2 small with 117 parameters. Each Layer consists of two parts : "),
      html.H4("(i) Multi-Head Self-attention Mechanism (MHSA)"),
      html.H4("(ii) Feed-Forward Neural Network (FFNN)"),
      html.H4("There are 12 heads in the MHSA")
        
    ],style = {'text-align': 'center'}),
    
    html.Div([
      html.H1("Multi-Head Self-Attention Mechanism"),
      html.H4("Below is the calculation of all heads of the first encoder layer of the first.")
        
    ],style = {'text-align': 'center'}),
    
    html.Div([
    html.Div([
            html.H3("first_layer_first_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_first_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                    'wordBreak': 'break-all',  # Word wrapping
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_second_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_second_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    
    html.Div([
            html.H3("first_layer_third_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_third_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_fourth_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_fourth_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_fifth_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_fifth_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_sixth_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_sixth_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_seventh_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_seventh_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_eighth_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_eighth_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_ninth_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_ninth_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_tenth_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_tenth_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_eleventh_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_eleven_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
            html.H3("first_layer_twefth_head", style={'textAlign': 'center'}),
            html.Div(
                id='first_layer_twelve_head',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family': 'Courier New',
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    ],style = {'width': '100%', 'display': 'inline-block','height':'500px','overflowY':'scroll','border': '2px solid #000000', }),
    
    html.Div([
            html.H3("final_context_vector_first_layer", style={'textAlign': 'center'}),
            html.Div(
                id='final_context_vector_first_layer',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family':'Courier New',
                    'overflowX': 'scroll'
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
    
    html.Div([
            html.H3("final_context_vector_second_layer", style={'textAlign': 'center'}),
            html.Div(
                id='final_context_vector_second_layer',
                style={
                    'border': '2px solid #000000', 
                    'height': '200px', 
                    'overflowY': 'scroll',
                    'font-family':'Courier New',
                    'overflowX': 'scroll'
                }
                )
            ],style={'width': '100%', 'display': 'inline-block'}),
])

@app.callback(
    Output('id-output', 'children'),
    Output('embedding-output', 'children'),
    Output('positional-encoding-output', 'children'),
    Output('sum-output', 'children'),
    Output('first_layer_first_head', 'children'),
    Output('first_layer_second_head', 'children'),
    Output('first_layer_third_head', 'children'),
    Output('first_layer_fourth_head', 'children'),
    Output('first_layer_fifth_head', 'children'),
    Output('first_layer_sixth_head', 'children'),
    Output('first_layer_seventh_head', 'children'),
    Output('first_layer_eighth_head', 'children'),
    Output('first_layer_ninth_head', 'children'),
    Output('first_layer_tenth_head', 'children'),
    Output('first_layer_eleven_head', 'children'),
    Output('first_layer_twelve_head', 'children'),
    Output('final_context_vector_first_layer', 'children'),
    # Output('final_context_vector_second_layer','children'),
    Input('input-text', 'value')    
)
def update_output(input_text):
    if not input_text:
        return "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""

    # Tokenize the input using the GPT-2 tokenizer
    inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=False)
    tokens = tokenizer.tokenize(input_text)
    token_ids = inputs['input_ids'][0].tolist()

    # Retrieve embeddings and positional encodings
    embeddings, positional_encodings = get_embeddings_and_positional_encodings(input_text)
    sum_emb_pos = embeddings+positional_encodings
    
    id_output = []
    embedding_output = []
    positional_output = []
    sum_output = []
    
    last_index_context_vector = list(layer_1_head_1.get_attention_values(sum_emb_pos).keys())[-1]
    
    first_layer_first_head = layer_1_head_1.get_attention_values(sum_emb_pos)
    context_vector_11 =  first_layer_first_head[last_index_context_vector]
    first_layer_first_head_output = []
    
    first_layer_second_head = layer_1_head_2.get_attention_values(sum_emb_pos)
    context_vector_12 =  first_layer_second_head[last_index_context_vector]
    first_layer_second_head_output = []
    
    first_layer_third_head = layer_1_head_3.get_attention_values(sum_emb_pos)
    context_vector_13 =  first_layer_third_head[last_index_context_vector]
    first_layer_third_head_output = []
    
    first_layer_fourth_head = layer_1_head_4.get_attention_values(sum_emb_pos)
    context_vector_14 =  first_layer_fourth_head[last_index_context_vector]
    first_layer_fourth_head_output = []
    
    first_layer_fifth_head = layer_1_head_5.get_attention_values(sum_emb_pos)
    context_vector_15 =  first_layer_fifth_head[last_index_context_vector]
    first_layer_fifth_head_output = []
    
    first_layer_sixth_head = layer_1_head_6.get_attention_values(sum_emb_pos)
    context_vector_16 =  first_layer_sixth_head[last_index_context_vector]
    first_layer_sixth_head_output = []
    
    first_layer_seventh_head = layer_1_head_7.get_attention_values(sum_emb_pos)
    context_vector_17 =  first_layer_seventh_head[last_index_context_vector]
    first_layer_seventh_head_output = []
    
    first_layer_eighth_head = layer_1_head_8.get_attention_values(sum_emb_pos)
    context_vector_18 =  first_layer_eighth_head[last_index_context_vector]
    first_layer_eighth_head_output = []
    
    first_layer_ninth_head = layer_1_head_9.get_attention_values(sum_emb_pos)
    context_vector_19 =  first_layer_ninth_head[last_index_context_vector]
    first_layer_ninth_head_output = []
    
    first_layer_tenth_head = layer_1_head_10.get_attention_values(sum_emb_pos)
    context_vector_110 =  first_layer_tenth_head[last_index_context_vector]
    first_layer_tenth_head_output = []
    
    first_layer_eleven_head = layer_1_head_11.get_attention_values(sum_emb_pos)
    context_vector_111 =  first_layer_eleven_head[last_index_context_vector]
    first_layer_eleven_head_output = []
    
    first_layer_twelve_head = layer_1_head_12.get_attention_values(sum_emb_pos)
    context_vector_112 =  first_layer_twelve_head[last_index_context_vector]
    first_layer_twelve_head_output = []
    
    
    final_context_vector_first_layer = torch.cat((context_vector_11,context_vector_12,context_vector_13,context_vector_14,context_vector_15,context_vector_16,context_vector_17,context_vector_18,context_vector_19,context_vector_110,context_vector_111,context_vector_112),dim=-1)
    final_context_vector_first_layer_output = []
    
    # context_vector_21 = layer_2_head_1.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_22 = layer_2_head_2.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_23 = layer_2_head_3.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_24 = layer_2_head_4.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_25 = layer_2_head_5.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_26 = layer_2_head_6.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_27 = layer_2_head_7.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_28 = layer_2_head_8.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_29 = layer_2_head_9.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_210 = layer_2_head_10.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_211 = layer_2_head_11.get_attention_values(sum_emb_pos)[last_index_context_vector]
    # context_vector_212 = layer_2_head_12.get_attention_values(sum_emb_pos)[last_index_context_vector]
    
    # final_context_vector_second_layer = torch.cat((context_vector_21,context_vector_22,context_vector_23,context_vector_24,context_vector_25,context_vector_26,context_vector_27,context_vector_28,context_vector_29,context_vector_210,context_vector_211,context_vector_212),dim=-1)
    # final_context_vector_second_layer_output = []
    
    # Display the token IDs, embeddings, positional encodings, and their sum
    for idx, (token, token_id) in enumerate(zip(tokens, token_ids)):
        id_output.append(f"{token} -> {token_id}")
        embedding_output.append(f"{token}{embeddings[0, idx].detach().numpy().shape} = {embeddings[0, idx].detach().numpy()}")
        positional_output.append(f"{token}{embeddings[0, idx].detach().numpy().shape} = {positional_encodings[idx].detach().numpy()}")
        sum_result = embeddings[0, idx] + positional_encodings[idx]
        sum_output.append(f"{token}{embeddings[0, idx].detach().numpy().shape} = {sum_result.detach().numpy()}")
        
    first_layer_first_head_output.append(first_layer_first_head)
    first_layer_second_head_output.append(first_layer_second_head)
    first_layer_third_head_output.append(first_layer_third_head)
    first_layer_fourth_head_output.append(first_layer_fourth_head)
    first_layer_fifth_head_output.append(first_layer_fifth_head)
    first_layer_sixth_head_output.append(first_layer_sixth_head)
    first_layer_seventh_head_output.append(first_layer_seventh_head)
    first_layer_eighth_head_output.append(first_layer_eighth_head)
    first_layer_ninth_head_output.append(first_layer_ninth_head)
    first_layer_tenth_head_output.append(first_layer_tenth_head)
    first_layer_eleven_head_output.append(first_layer_eleven_head)
    first_layer_twelve_head_output.append(first_layer_twelve_head)
    
    final_context_vector_first_layer_output.append(f"{final_context_vector_first_layer.shape}{str(final_context_vector_first_layer.detach().numpy())}")
    
    # final_context_vector_second_layer_output.append(f"{final_context_vector_second_layer.shape}{str(final_context_vector_second_layer.detach().numpy())}")
    
    return '\n'.join(id_output), '\n'.join(embedding_output), '\n'.join(positional_output), '\n'.join(sum_output), '\n'.join(str(item) for item in first_layer_first_head_output), '\n'.join(str(item) for item in first_layer_second_head_output), '\n'.join(str(item) for item in first_layer_third_head_output), '\n'.join(str(item) for item in first_layer_fourth_head_output), '\n'.join(str(item) for item in first_layer_fifth_head_output), '\n'.join(str(item) for item in first_layer_sixth_head_output), '\n'.join(str(item) for item in first_layer_seventh_head_output), '\n'.join(str(item) for item in first_layer_eighth_head_output), '\n'.join(str(item) for item in first_layer_ninth_head_output), '\n'.join(str(item) for item in first_layer_tenth_head_output), '\n'.join(str(item) for item in first_layer_eleven_head_output), '\n'.join(str(item) for item in first_layer_twelve_head_output), '\n'.join(final_context_vector_first_layer_output)
# ,'\n'.join(final_context_vector_second_layer_output)

if __name__  == "__main__":
    app.run_server(debug = True)
