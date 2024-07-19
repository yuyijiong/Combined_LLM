# Encoding: UTF-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import transformers
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name1 = "deepseek-ai/deepseek-math-7b-instruct"
model_name2 = "deepseek-ai/DeepSeek-V2-Lite-Chat"
combined_lm_head_weight_path ="combined_lm_head.pth"

# 准备模型
bnb_config = transformers.BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model1 = AutoModelForCausalLM.from_pretrained(model_name1,
                                              torch_dtype=torch.bfloat16,
                                              quantization_config=bnb_config,
                                              trust_remote_code=True,
                                              attn_implementation="flash_attention_2",
                                              device_map="auto"
                                              )
model2 = AutoModelForCausalLM.from_pretrained(model_name2,
                                              torch_dtype=torch.bfloat16,
                                              quantization_config=bnb_config,
                                              trust_remote_code=True,
                                              attn_implementation="flash_attention_2",
                                              device_map="auto"
                                              )

# 初始化联合模型
from combine_2_model import CombinedModelForCausalLM
model = CombinedModelForCausalLM(model1.model, model2.model)

# 加载联合模型的lm_head权重
if combined_lm_head_weight_path:
    print("加载联合模型的lm_head权重")
    combined_lm_head_weight = torch.load(combined_lm_head_weight_path)
    model.combined_lm_head.load_state_dict(combined_lm_head_weight)
else:
    print("拼接联合模型的lm_head权重")
    # 分别获取model1和2的lm_head的权重并拼接
    model.combined_lm_head.weight.data = torch.cat([model1.lm_head.weight.data, model2.lm_head.weight.data], dim=-1)

print('model.combined_lm_head.weight.data:', model.combined_lm_head.weight.data.shape)
model.eval().cuda()

# 准备tokenizer，默认为model1的tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name1)

# 输入问题
text="9.11和9.9哪个更大？"

messages=[{"role":"user","content":text}]
prompt=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,max_length=512)
inputs = tokenizer(prompt, return_tensors="pt").to(model1.device)
prompt_len=len(inputs["input_ids"][0])

with torch.autocast("cuda"):
    response = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=800,
        do_sample=False,
        temperature=0.8,
        use_cache=True,)

print(tokenizer.decode(response[0][prompt_len:], skip_special_tokens=True))

