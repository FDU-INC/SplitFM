import os
import sys
from transformers import AutoTokenizer
import transformers
import torch
from torch import nn

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from transformers import (
    AutoTokenizer,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM #, LlamaModel_Client
from modelsplit import LlamaModel_Client, LlamaModel_Server

from utils import load_pretrain, load_pretrain_split

sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from infer_adapter import SplitModelAdapter

class LlamaAdapter(SplitModelAdapter):
    def load(self, weights_path):
        self.configuration = LlamaConfig.from_pretrained(weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

        self.model_client = LlamaModel_Client(self.configuration)
        self.model_server = LlamaModel_Server(self.configuration)
        self.lm_head = nn.Linear(self.configuration.hidden_size, self.configuration.vocab_size, bias=False)
        self.model_client, self.model_server, self.lm_head = load_pretrain_split(self.model_client, self.model_server, self.lm_head, weights_path)

        self.model_client = self.model_client.half().cuda()
        self.model_server = self.model_server.half().cuda()
        self.lm_head = self.lm_head.half().cuda()
        self.model_client.eval()
        self.model_server.eval()
    
    def _generate_tokens(self, input_sentence, max_new_tokens, log_tokens):
        inputs = self.tokenizer(input_sentence, return_tensors='pt').to('cuda')
        if log_tokens:
            print("Split inference token by token:")
        with torch.no_grad():
            for i in range(max_new_tokens):
                hidden_states, causal_mask, position_ids = self.model_client(**inputs)
                outputs = self.model_server(hidden_states=hidden_states, causal_mask=causal_mask, position_ids=position_ids)
                logits = self.lm_head(outputs[0])
                last_token_logits = logits[:, -1, :]
                predicted_token_id = torch.argmax(last_token_logits, dim=-1).item()
                predicted_token = self.tokenizer.decode(predicted_token_id)
                if log_tokens:
                    print(predicted_token, end="", flush=True)
                yield predicted_token
                input_sentence = input_sentence + predicted_token
                inputs = self.tokenizer(input_sentence, return_tensors='pt').to('cuda')

    def infer(self, input_sentence, **kwargs):
        max_new_tokens = int(kwargs.get("max_new_tokens") or 150)
        return_full_text = kwargs.get("return_full_text", True)
        log_tokens = kwargs.get("log_tokens", True)
        generated_tokens = list(self._generate_tokens(input_sentence, max_new_tokens, log_tokens))
        if return_full_text:
            return input_sentence + "".join(generated_tokens)
        return "".join(generated_tokens)

    def stream_infer(self, input_sentence, **kwargs):
        max_new_tokens = int(kwargs.get("max_new_tokens") or 150)
        log_tokens = kwargs.get("log_tokens", False)
        yield from self._generate_tokens(input_sentence, max_new_tokens, log_tokens)
