import json

topk = 5

def load_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list
data_mcts = load_jsonl('') # your mcts rollout path here

for data in data_mcts:
    data['answer'] = sorted(data['answer'], key=lambda item: item['reward'], reverse=True)[:topk] # topk

# save policy data
with open('./rlhf_data_best5_mcts.jsonl', 'w') as file:
    for data in data_mcts:
        json_str = json.dumps(data)
        file.write(json_str + '\n')
