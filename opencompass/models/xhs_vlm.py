import os
import re
import base64
import io
import PIL.Image

from typing import Optional, Dict, List, Union
from opencompass.utils.prompt import PromptList
from transformers import AutoModelForCausalLM, AutoConfig, AutoProcessor,  GenerationConfig
from opencompass.utils import get_logger

import torch
from .base import BaseModel, LMTemplateParser
from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser

PromptType = Union[PromptList, str]
VERBOSE = os.environ.get('VERBOSE', 0)

def _get_meta_template(meta_template):
    default_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='user'),
            # XXX: all system roles are mapped to human in purpose
            dict(role='SYSTEM', api_role='system'),
            dict(role='BOT', api_role='assistant', generate=True),
        ]
    )
    return APITemplateParser(meta_template or default_meta_template)


def _convert_chat_messages(inputs, merge_role=True, skip_empty_prompt=True):
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item['prompt']:
                    continue
                if item['role'] in ['HUMAN', 'BOT', 'SYSTEM']:
                    role = {
                        'HUMAN': 'user',
                        'BOT': 'assistant',
                        'SYSTEM': 'system',
                    }[item['role']]
                    messages.append({'role': role, 'content': item['prompt']})
                else:
                    messages.append({'role': item['role'], 'content': item['prompt']})
        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def is_image_path(path):
    # 定义常见的图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    
    # 获取文件扩展名
    _, ext = os.path.splitext(path.lower())
    
    # 检查扩展名是否在预定义的图片扩展名集合中
    if ext in image_extensions:
        # 如果需要，检查文件是否存在
        # return os.path.isfile(path)  # Uncomment this line to check for file existence
        return True
    return False

class XHSVLM(BaseModel):
    is_multimodal = True

    def __init__(self,
                path: str,
                max_seq_len: int = 4096,
                tokenizer_only: bool = False,
                meta_template: Optional[Dict] = None,
                **kwargs):   
        
        self.logger = get_logger()
        self.path = path
        self.max_seq_len = max_seq_len
        self.template_parser = _get_meta_template(meta_template)

        self._load_model(path, kwargs)
        self.processor = AutoProcessor.from_pretrained(path)
        self.tokenizer = self.processor.tokenizer

    def _load_model(self, path: str, kwargs: dict):
        from transformers import AutoModel, AutoModelForCausalLM
        import sys
        sys.path.insert(0, path)

        import modeling_agi

        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = self._set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f'using model_kwargs: {model_kwargs}')

        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

    def _set_model_kwargs_torch_dtype(self, model_kwargs):
        import torch
        if 'torch_dtype' not in model_kwargs:
            torch_dtype = torch.float16
        else:
            torch_dtype = {
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float': torch.float,
                'auto': 'auto',
                'None': None,
            }.get(model_kwargs['torch_dtype'])
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype
        return model_kwargs

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        # 这里只是为了绕过一些限制，多模模型
        return len(prompt.split(" "))
    
    def generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        No change.
        """
        messages = _convert_chat_messages(inputs)
        
        result = []
        for msg in messages:
            result.append(self._generate(msg, max_out_len, **kwargs))
        return result    

    def _generate(self, input: str, max_out_len: int, **kwargs) -> List[str]:
        """Generate results given a single input.
        
        1. find and extect img from str
        2. call self.model.generate/self.model.multi_generate/self.model.interleave_generate

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        prompt = self.tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False)
        model_inputs = self.processor(prompt=prompt)

        try:
            outputs = self.model.generate(
                input_ids=model_inputs.input_ids.unsqueeze(0).cuda(),
                max_new_tokens=max_out_len,
                # use_cache=False,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        except AssertionError as e:
            import pdb; pdb.set_trace()
            print(f"AssertionError caught: {e}")
            outputs = None
            # 处理AssertionError的逻辑
        answer = self.tokenizer.decode(
            outputs[0][len(model_inputs.input_ids):].cpu().tolist(), 
            skip_special_tokens=False)
        
        answer = answer.strip()

        if answer.endswith("<|endofassistant|>"):
            answer =  answer.replace('<|endofassistant|>', "")
        
        if VERBOSE:
            get_logger().info(f"\n###Response:\n{answer}")
        
        return answer

def get_image(content, is_path=True):
    if is_path:
        pil_img = PIL.Image.open(content)
    else:
        image_bytes = base64.b64decode(content)
        pil_img = PIL.Image.open(io.BytesIO(image_bytes))
    pil_img = pil_img.convert("RGB")
    return pil_img


def parse_string(s):
    if isinstance(s, str):
        parts = re.split(r'(<ImagePath>.*?</ImagePath>|<ImageBase64>.*?</ImageBase64>)', s)
        result = []
        for part in parts:
            if "<ImagePath>" in part:
                result.append({"imagepath": re.search('<ImagePath>(.*?)</ImagePath>', part).group(1)})
            elif "<ImageBase64>" in part:
                result.append({"imagebase64": re.search('<ImageBase64>(.*?)</ImageBase64>', part).group(1)})
            else:
                result.append({"text": part})
        return result
    elif isinstance(s, list):
        for line in s:
            if isinstance(line['content'], str):
                line['content'] = parse_string(line['content'])
        return s
    
