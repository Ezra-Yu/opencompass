from typing import List
import time
import os
import re
import base64
import requests
import random
import math

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from concurrent.futures import ThreadPoolExecutor
from opencompass.utils import get_logger
from .base_api import BaseAPIModel

from tqdm import tqdm

# 保存原始的 __init__ 方法
original_init_func = tqdm.__init__

def custom_tqdm_init(self, *args, **kwargs):
    kwargs.setdefault('ncols', 135) # 修改 tqdm 的默认进度条长度
    # kwargs.setdefault('disable', True)
    original_init_func(self, *args, **kwargs)

# 用自定义的方法替换 tqdm 的 __init__ 方法
tqdm.__init__ = custom_tqdm_init

VERBOSE = os.environ.get('VERBOSE', 0)

@MODELS.register_module('maas')
class MAASAPI(BaseAPIModel):
    def __init__(self,
                 base_url: str,
                 api_key: str = "demo_api_1",
                 model_name: str = "model",
                 max_seq_len: int = 4096,
                 temperature: float = 0.2,
                 top_p: float = 0.9,
                 top_k: int = 1,
                 meta_template: dict = None,
                 retry: int = 5,
                 max_concurrency_num: int = 2,
                 system_prompt: str = "",
                 disable_tqdm: bool = False,
                 time_out: int = 90):
        super().__init__(path="", max_seq_len=max_seq_len, retry=retry, meta_template=meta_template)
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.time_out = time_out
        self.system_prompt = system_prompt
        self.disable_tqdm = disable_tqdm
        self.max_concurrency_num = max_concurrency_num
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        assert self.top_k in [0, 1], "API 模型 `top_k` 只能为 0 或 1"
        self.model_name = model_name
        self.time_out = time_out
        self.TIME_OUT_MESSAGE="【【请求超时，无结果。】】\n 请尝试降低 `并发数` 或者增加 `等待时间`; " + \
                              f"当前MAX_CONCURRENCY_NUM={self.max_concurrency_num}; TIME_OUT={self.time_out}；" + \
                              "运行前设置环境变量 MAX_CONCURRENCY_NUM & TIME_OUT"
        
        self._check_connection()

    def _check_connection(self):
        """检查一下连接问题"""
        messages = [
            { "role": "system", "content": [{"type": "text", "text": "你是我的生活助手。"}] },
            { "role": "user", "content": [{"type": "text", "text": "你好，你是谁？"}] }
        ]
        res = self._generate(messages, self.max_seq_len, self.temperature, self.top_p, self.top_k)
        if res == self.TIME_OUT_MESSAGE or res.startswith("请求失败") or res.startswith("|<<请求错误>>|"):
            raise RuntimeError(
                "请检查你的 XDG 配置，当前发送的请求没有被回应；当前配置：\n"
                f"\tbase_url={self.base_url}; \n"
                f"\tmodel_name={self.model_name}; \n"
                f"\tmax_concurrency_num={self.max_concurrency_num}; \n"
                f"\ttime_out={self.time_out}; \n"
                f"\tSystem_Prompt={repr(self.system_prompt)} \n"
            )

    def _generate(self, messages, max_out_len, temperature, top_p, top_k): 
        max_num_retries = 0
        flag = False
        while max_num_retries < self.retry:
            headers = {
                "Content-Type" : "application/json" ,
                "Authorization" : f"Bearer {self.api_key}",
            }
            req_body={
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_out_len,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": False if top_k and top_k == 1 else True,
            }

            try:
                resp = requests.post(self.base_url, headers=headers, json=req_body)
                resp.encoding = "utf-8"
                response_content = resp.json()['choices'][0]['message']['content'][0]
                response_text = response_content[response_content['type']]     
                flag = True
                break
            except Exception as e:
                if max_num_retries == self.retry - 1:
                    return f"请求失败: {str(e)}"
                time.sleep(random.randint(
                    math.floor(2 ** max(0, max_num_retries - 1)),
                    math.floor(2 ** max_num_retries)
                ))
                max_num_retries += 1
        
        if flag:
            if VERBOSE: # log the messages
                for message in messages:
                    for i, content in enumerate(message['content']):
                        if content['type'] == 'image_url' and content['image_url']['url'].startswith('data:image/jpeg;base64,'):
                            content['image_url']['url'] = content['image_url']['url'][:50] + "......" + content['image_url']['url'][-50:]
                get_logger().info(
                    f"\n###Request_URL: {self.base_url}"
                    f"\n###Prompt: {messages}"
                    f"\n###Hyperparameter: max_out_len={max_out_len}, topk={top_k}; top_p={top_p}; temperature={temperature}"
                    f"\n###Response:\n{response_text}")
            return response_text
       
        return "请求失败，无结果。"

    def generate(
            self,
            inputs: List[str or PromptList],
            max_out_len: int = 512,
            batch_size: int = 8,
            temperature: float = None,
            top_p: float = None,
            top_k: int = None,
            **kwargs
        ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.

        Returns:
            List[str]: A list of generated strings.
        """
        inputs = [self._process_input(input) for input in inputs]
        # 如果参数为None，则 model 的默认值
        max_out_len = max_out_len if max_out_len is not None else self.max_seq_len
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k

        with ThreadPoolExecutor(max_workers=self.max_concurrency_num) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self._generate,
                        inputs,
                        [max_out_len] * len(inputs),
                        [temperature] * len(inputs),
                        [top_p] * len(inputs),
                        [top_k] * len(inputs),
                    ),
                    total=len(inputs),
                    disable=self.disable_tqdm
                )
            )
        return results
    
    def _process_input(self, input):
        def process_image_input(input_str):
            parts = re.split('(<ImagePath>.*?</ImagePath>|<ImageBase64>.*?</ImageBase64>)', input_str)
            process_input = []
            for part in parts:
                if "<ImagePath>" in part:
                    image_path = re.search('<ImagePath>(.*?)</ImagePath>', part).group(1)
                    image_base64 = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
                    process_input.append({
                        "type": "image_url",
                        "image_url" : {'url': f"data:image/jpeg;base64,{image_base64}"}
                    })
                elif "<ImageBase64>" in part:
                    image_base64 = re.search('<ImageBase64>(.*?)</ImageBase64>', part).group(1)
                    process_input.append({
                        "type": "image_url",
                        "image_url" : {'url': f"data:image/jpeg;base64,{image_base64}"}
                    })
                else:
                    process_input.append({"type": "text", "text": part})
        
            return process_input
     
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': process_image_input(input)}]
        else:
            messages = []
            for item in input:
                msg = {'content': process_image_input(item['prompt'])}
                
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        return messages