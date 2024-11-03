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
```
