import os


files_to_compare = [
    "generate.py",
    "wan/text2video.py",
    "wan/utils/utils.py",
    "wan/utils/fm_solvers.py",
    "wan/utils/fm_solvers_unipc.py",
    "wan/utils/__init__.py",
    "wan/utils/prompt_extend.py",
    "wan/utils/qwen_vl_utils.py",
    "wan/utils/utils.py",
    "wan/modules/attention.py",
    "wan/modules/clip.py",
    "wan/modules/__init__.py",
    "wan/modules/model.py",
    "wan/modules/t5.py",
    "wan/modules/tokenizers.py",
    "wan/modules/vae.py",
    "wan/modules/xlm_roberta.py",
    "wan/distributed/fsdp.py",
    "wan/distributed/__init__.py",
    "wan/distributed/xdit_context_parallel.py",
    "wan/configs/__init__.py",
    "wan/configs/shared_config.py",
    "wan/configs/wan_i2v_14B.py",
    "wan/configs/wan_t2v_1_3B.py",
    "wan/configs/wan_t2v_14B.py"
]

for f in files_to_compare:
    print("="*50)
    print(f"\t Comparing {f}")
    print("="*50)

    os.system(f"diff {f} ../../arielshaulov/Wan2.1/{f}")
