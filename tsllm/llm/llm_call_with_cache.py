from typing import Optional, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from tsllm.model.modeling_prm import ValueHeadedLLM


@torch.inference_mode()
def llm_forward_fn_with_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
) -> [torch.FloatTensor, List[torch.FloatTensor]]:
    """ """
    inputs = tokenizer(prompt, return_tensors="pt")  # .to(model.device)
    past_key_values = None
    if past_key_values is not None:
        past_key_values = [
            [x.to(model.device) for x in past_key_value]
            for past_key_value in past_key_values
        ]
        past_context_len = past_key_values[0][0].shape[2]
        # get the last index item of inputs
        input_ids = inputs.input_ids[:, past_context_len:]
    else:
        input_ids = inputs.input_ids
    input_ids = input_ids.to(model.device)

    model_outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    logits2return = model_outputs.logits[:, -1]
    past_key_values = model_outputs.past_key_values
    # convert_past_key_values to cpu
    past_key_values = [
        [x.to("cpu") for x in past_key_value] for past_key_value in past_key_values
    ]
    return logits2return, past_key_values


@torch.inference_mode()
def token_value_fn_with_cache(
    critic: ValueHeadedLLM,
    tokenizer: PreTrainedTokenizer,
    input_str: str,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
):
    if isinstance(input_str, list):
        indices2pick = torch.LongTensor(
            [len(tokenizer.encode(txt)) - 1 for txt in input_str]
        )
    else:
        indices2pick = torch.LongTensor([len(tokenizer.encode(input_str)) - 1])

    inputs = tokenizer(input_str, return_tensors="pt", padding=True)
    if past_key_values is None:
        input_ids = inputs.input_ids
    else:
        past_context_len = past_key_values[0][0].shape[2]
        past_key_values = [
            [x.to(critic.device) for x in past_key_value]
            for past_key_value in past_key_values
        ]
        input_ids = input_ids[:, past_context_len:]
        indices2pick = indices2pick - past_context_len
    input_ids = input_ids.to(critic.device)

    critic_output = critic(
        input_ids=input_ids, use_cache=True, past_key_values=past_key_values
    )
    value = critic_output.value.cpu()
    value = value.gather(1, indices2pick.unsqueeze_(1)).squeeze_(1).float().numpy()
    past_key_values = critic_output.past_key_values
    past_key_values = [
        [x.to("cpu") for x in past_key_value] for past_key_value in past_key_values
    ]
    return value, past_key_values
