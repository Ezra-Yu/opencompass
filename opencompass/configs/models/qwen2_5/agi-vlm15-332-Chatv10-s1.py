from opencompass.models import XHSVLM

models = [
    dict(
        type=XHSVLM,
        abbr='agi-vlm1.5-3.3.2-Chatv1.0-s1',
        path='/cpfs/user/qiaoyu/oc/agi-vlm1.5-3.3.2-Chatv1.0-s1',
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=4),
    )
]
