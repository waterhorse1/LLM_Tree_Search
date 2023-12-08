from tsllm.llm.text_generation import llm_gen_with_logp_vllm
from tsllm.inference.value import _value_inference_fastchat

# a = llm_gen_with_logp_vllm("last_model_hf", None, None, "What Can you do?\n\n", 3)


x = _value_inference_fastchat("llama2-7b-game24-value-sft-ep3", None, ["YES! 1+1+1+1=24", "who are you", "fff"])
print(x)