from opencompass.models import XHSVLM

models = [
    dict(
        type=XHSVLM,
        abbr='agi-vlm1.4.1.1-ChatV1.0-s2.2',
        path='/cpfs/user/qiaoyu/oc/agi-vlm1.4.1.1-ChatV1.0-s2.2',
        max_out_len=4096,
        torch_dtype='torch.bfloat16',
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
