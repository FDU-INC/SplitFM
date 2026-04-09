import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from transformers import Qwen2VLConfig, AutoProcessor,AutoTokenizer
from modelsplit import Qwen2VLForConditionalGeneration_Client, Qwen2VLForConditionalGeneration_Server
from utils import load_pretrained_Qwen2VL
from qwen_vl_utils import process_vision_info

sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from infer_adapter import SplitModelAdapter

class QwenAdapter(SplitModelAdapter):
    def load(self, weights_path):
        self.configuration = Qwen2VLConfig.from_pretrained(weights_path)
        configuration._attn_implementation = "flash_attention_2"
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(weights_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.tokenizer = AutoTokenizer.from_pretrained(weights_path)
        self.model_client = Qwen2VLForConditionalGeneration_Client(self.configuration)
        self.model_server = Qwen2VLForConditionalGeneration_Server(self.configuration)
        self.lm_head = nn.Linear(self.configuration.hidden_size, self.configuration.vocab_size, bias=False)
        self.model_client, self.model_server, self.lm_head = load_pretrained_Qwen2VL(weights_path, self.model_client, self.model_server, self.lm_head)
        self.model_client = self.model_client.half().cuda(0)
        self.model_server = self.model_server.half().cuda(1)
        self.lm_head = lm_head.half().cuda(1)

    def _generate_tokens(self, input_sentence, image_path, max_new_tokens):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path if image_path else "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": input_sentence },
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to('cuda:0')
        #============================inference ================================
        with torch.no_grad():
            for i in range(max_new_tokens): 
                inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to('cuda:0')
                hidden_states, causal_mask, position_ids = self.model_client(**inputs)
                hidden_states = hidden_states.cuda(0)
                position_ids = position_ids.cuda(0)               
                outputs = self.model_server(hidden_states=hidden_states, causal_mask=causal_mask, position_ids=position_ids)
                logits = self.lm_head(outputs[0])
                last_token_logits = logits[:, -1, :]
                softmax_logits = F.softmax(last_token_logits, dim = -1)
                predicted_token_id = torch.argmax(softmax_logits, dim=-1)
                predicted_token = self.tokenizer.decode(predicted_token_id)
                text += predicted_token
                yield predicted_token

    def infer(self, input_sentence, image_path=None, **kwargs):
        max_new_tokens = int(kwargs.get("max_new_tokens") or 64)
        generated_tokens = list(self._generate_tokens(input_sentence, image_path, max_new_tokens))
        final_answer = ''.join(generated_tokens)
        return final_answer

    def stream_infer(self, input_sentence, image_path=None, **kwargs):
        max_new_tokens = int(kwargs.get("max_new_tokens") or 64)
        yield from self._generate_tokens(input_sentence, image_path, max_new_tokens)
