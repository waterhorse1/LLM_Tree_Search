import json
def load_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list
data_direct = load_jsonl('')# Your direct sampling data
data_mcts = load_jsonl('')# Your MCTS sampling data
sorted_direct = sorted(data_direct, key=lambda item: item['question'])
sorted_mcts =  sorted(data_mcts, key=lambda item: item['question'])
import random
random.seed(42)
data_mixed = []
for data_d, data_m in zip(sorted_direct, sorted_mcts):
    assert data_d['question'] == data_m['question']
    answer_d = random.sample(data_d['answer'], 40)
    answer_d.extend(data_m['answer'])
    random.shuffle(answer_d)
    data = {'question': data_d['question'], 'answer': answer_d}
    assert len(answer_d) == 50 # make sure the number of data equals to 50
    data_mixed.append(data)

# save critic training data
with open('./rlhf_data_mixed_value.jsonl', 'w') as file:
    for data in data_mixed:
        json_str = json.dumps(data)
        file.write(json_str + '\n')