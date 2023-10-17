import requests
import torch


def llm_gen_ct2(
    generator, tokenizer, static_prompt, prompt, num_sequence, stop, **generation_config
):
    prompt_tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(prompt, add_special_tokens=False)
    )
    if static_prompt is not None:
        static_prompt_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(static_prompt)
        )
    else:
        static_prompt_tokens = None
        prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

    if isinstance(stop, int):
        stop = [stop]
    step_results = generator.generate_batch(
        [prompt_tokens],
        sampling_temperature=generation_config.get("temperature", 1.0),
        sampling_topp=generation_config.get("top_p", 1.0),
        sampling_topk=generation_config.get("top_k", 1),
        max_length=generation_config.get("max_new_tokens", 16),
        return_scores=True,
        include_prompt_in_result=False,
        end_token=stop,
        static_prompt=static_prompt_tokens,
        max_batch_size=generation_config.get("max_batch_size", 0),
        num_hypotheses=num_sequence,
    )

    results = list(step_results)
    texts = [tokenizer.decode(seq) for seq in results[0].sequences_ids]
    logps = results[0].scores

    return texts, logps


def llm_forward_ct2(generator, tokenizer, prompt):
    prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

    step_logits = generator.forward_batch(
        [prompt_tokens],
    )
    # Currently this only supports float32
    logits = torch.as_tensor(step_logits)[:, -1]
    return logits


# def llm_gen_ct2(generator, sp, static_prompt, prompt, num_sequence, stop,
#                 **generation_config):
#     prompt_tokens = sp.encode(prompt, out_type=str)
#     if static_prompt is not None:
#         static_prompt_tokens = ["<s>"] + sp.encode(static_prompt, out_type=str)
#     else:
#         static_prompt_tokens = None
#         prompt_tokens = ["<s>"] + prompt_tokens

#     if isinstance(stop, int):
#         stop = [stop]
#     step_results = generator.generate_batch(
#         [prompt_tokens] * num_sequence,
#         sampling_temperature=generation_config.get("temperature", 1.0),
#         sampling_topp=generation_config.get("top_p", 1.0),
#         sampling_topk=generation_config.get("top_k", 1),
#         max_length=generation_config.get("max_new_tokens", 16),
#         return_scores=True,
#         include_prompt_in_result=False,
#         end_token=stop,
#         static_prompt=static_prompt_tokens,
#         max_batch_size=generation_config.get("max_batch_size", 0),
#     )

#     results = list(step_results)
#     texts = [sp.decode(r.sequences_ids[0]) for r in results]
#     logps = [[r.scores[0]] for r in results]

#     return texts, logps


@torch.no_grad()
def llm_gen_with_logp_v1(
    model, tokenizer, static_prompt, prompt, num_sequence, stop, **generation_config
):
    state_all = [static_prompt + prompt for _ in range(num_sequence)]
    inputs = tokenizer(
        state_all,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(model.device)

    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    transition_scores = transition_scores.cpu().float().numpy()
    output_tokens = outputs.sequences[:, input_length:]

    split_list = []
    logprob_list = []
    # for ChatGLM 13 means <0x0A> for \n
    split_token_id = 13

    def find_index(tensor, element):
        equals_value = torch.eq(tensor, element)
        if sum(equals_value) == 0:
            return -1
        return torch.nonzero(equals_value)[0].item()

    for scores, ot in zip(transition_scores, output_tokens):
        assert len(ot) == len(scores)
        pos = find_index(ot, split_token_id)
        if pos != 0 and pos != -1:  # 0 means the first token is split str
            ot = ot[:pos]
            scores = scores[:pos]
        elif pos == -1:  # -1 means doesn't contain split str,
            #  just use the whole string as the action, for chatglm
            eos_token_id = tokenizer.eos_token_id
            eos_pos = find_index(ot, eos_token_id)
            ot = ot[:eos_pos]
            scores = scores[:eos_pos]
        else:
            continue
        os = tokenizer.decode(ot)
        if os in split_list:
            continue
        else:
            if len(scores) == 0:
                print(123, state_all[-1], os)
                exit()
            split_list.append(os)
            # todo determine np.sum or np.mean
            logprob_list.append(scores)

    return split_list, logprob_list
