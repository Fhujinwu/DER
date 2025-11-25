


import torch
import json
from transformers import DebertaV2Tokenizer, AutoTokenizer
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained('./opt-125m')
        self.max_length = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        State = "Question:{} Answer:{}"
        item = self.data[index]
        init_source = State.format(item['instruction'] + item['input'], " ")
        target = item['output'] if 'output' in item else None
        source = item['instruction'] + item['input']
        encoding = self.tokenizer(init_source, max_length=self.max_length, padding='max_length', truncation=True,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'source': source,
            'target': target,
        }

    def get_example(self, index):
        return self.data[index]


def load_data(data_path, args, max_size=None):
    # random.seed(args.seed)
    assert data_path, "data_path is not specified"
    print("Loading data from {}".format(data_path))
    if data_path.endswith('.jsonl'):
        with open(data_path) as f:
            data = [json.loads(line) for line in f.readlines()]
    elif data_path.endswith('.json0127'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    else:
        raise ValueError("Unknown data")
    if max_size is not None and max_size > 0:
        data = data[:max_size]
    examples = []
    for example in data:
        new_item = {}
        new_item['id'] = example['id']
        new_item['instruction'] = example['instruction']
        new_item['input'] = example['input']
        new_item['output'] = example['output']
        examples.append(new_item)
    return examples



if __name__ == '__main__':
    pass