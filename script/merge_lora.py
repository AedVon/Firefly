from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model():
    model_name_or_path = '/nvme_disk1/public/weights/Qwen2.5-32B-Instruct'
    adapter_name_or_path = '/nvme_disk1/tairu/weights/Qwen2.5-32B-Instruct_qa_xtop_qualib_openroad_v12_epoch_2_max_seq_3072_lora_rank_32_v1'
    save_path = '/nvme_disk1/tairu/weights/Qwen2.5-32B-Instruct_qa_xtop_qualib_openroad_v12_epoch_2_max_seq_3072_lora_rank_32_v1_merge_lora'

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
