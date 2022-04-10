# from modeling_gpt2 import GPT2Config
# from convlab.modules.e2e.multiwoz.Transformer.pytorch_transformers 
from modeling_gpt2 import GPT2DoubleHeadsModel, GPT2Tokenizer
from . import GPT2Tokenizer
GPT2Config

config = GPT2Config.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2DoubleHeadsModel(config)
choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]  
# Assume you've added [CLS] to the vocabulary
input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
mc_token_ids = torch.tensor([-1, -1]).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, mc_token_ids)
lm_prediction_scores, mc_prediction_scores = outputs[:2]
