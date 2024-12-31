import os
from opencompass.models import MAASAPI
import re

system_prompt = os.environ.get('SYSTEM_PROMPT', "")
model_abbr = os.environ.get('MODEL_ABBR', None)
time_out = int(os.environ.get('TIME_OUT', 60))
retry = int(os.environ.get('RETRY', 3))
max_out_len = int(os.environ.get('MAX_OUT_LEN', 4096))

_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

max_concurrency_num = int(os.environ.get('MAX_CONCURRENCY_NUM', '2'))
model_name = os.environ.get('MODEL_NAME', 'model')
max_seq_length = int(os.environ.get('MAX_SEQ_LENGTH', '4096'))

base_url = os.environ.get('BASE_URL', None)
demo_hash = os.environ.get('DEMO_HASH', None)
if demo_hash is not None and base_url  is None:
    base_url_template = f"https://llm-pipeline.devops.xiaohongshu.com/modelzoos/quickdeploy/{demo_hash}/model/v1/chat/completions"
    base_url = base_url_template.format(demo_hash)
elif base_url is not None:
    pass
else:
    # 兼容之前的 xdg 行为，不了解不管这里
    base_urls = os.environ.get('XDG_URLS').split(',')
    if len(base_urls) == 1 and re.match("https://llm-pipeline.devops.xiaohongshu.com/modelzoos/quickdeploy/.*/model", base_urls[0]):
        base_url = base_urls[0] + "/v1/chat/completions"
    else:
        raise ValueError("BASE_URL or DEMO_HASH is required")

models = [
    dict(
        type=MAASAPI,
        abbr=model_abbr if model_abbr else f"{model_name}",
        base_url=base_url,
        retry=retry,
        meta_template=_meta_template,
        max_concurrency_num=max_concurrency_num,   # 服务的最大并发数目
        batch_size=2 ** 14,                        # 兼容 opencompass 属性，可以忽略这个参数           
        model_name=model_name,
        max_seq_len=max_seq_length,
        max_out_len=max_out_len,
        time_out=time_out
    )
]
