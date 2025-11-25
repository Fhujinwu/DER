
import bert_score
from bert_score import BERTScorer
import torch
from utils.bart_score import BARTScorer
from bleurt import score
import os
class Bert_calculate:
    def __init__(self):
        self.tokenizer = None
        self.scorer = None
        self.max_tokens = 512

    def load_weight(self):
        print("loading pretrained bertscore...")
        self.tokenizer = bert_score.utils.get_tokenizer(model_type="bert-base-multilingual-cased")
        self.scorer = BERTScorer(lang="en", device='cuda:2',model_type="bert-base-multilingual-cased")

    def bertscore_calculate(self,candidates, references, batchsize):
        for i in range(len(candidates)):
            candidate = candidates[i]
            reference = references[i]
            candidate_tokens = self.tokenizer.tokenize(
                self.tokenizer.decode(self.tokenizer.encode(candidate, add_special_tokens=True)))
            reference_tokens = self.tokenizer.tokenize(
                self.tokenizer.decode(self.tokenizer.encode(reference, add_special_tokens=True)))
            if len(candidate_tokens) > self.max_tokens or len(reference_tokens) > self.max_tokens:
                candidate_tokens = candidate_tokens[:self.max_tokens]
                reference_tokens = reference_tokens[:self.max_tokens]
                candidate = self.tokenizer.convert_tokens_to_string(candidate_tokens)
                reference = self.tokenizer.convert_tokens_to_string(reference_tokens)
                min_len = min(len(candidate), len(reference))
                candidates[i] = candidate[:min_len]
                references[i] = reference[:min_len]
        P, R, F1 = self.scorer.score(candidates, references, verbose=False, batch_size=batchsize)
        return F1.numpy().tolist()

class Bart_calculate:
    def __init__(self):
        self.bart_scorer = None

    def load_weight(self):
        print("loading pretrained bartscore...")
        self.bart_scorer = BARTScorer(device='cuda:2', checkpoint='./bart-large-cnn')
        self.bart_scorer.load(path='./bart_score/bart_score.pth')
    def calculate_bart_score(self, candidate, reference):
        """
        :param batch_size:
        :return:
        """
        torch.cuda.empty_cache()
        results = []
        for i in range(len(candidate)):
            result = self.bart_scorer.score([candidate[i]], [reference[i]], batch_size=1)
            results.append(result[0])
        return results

class Bleurt_calculate:
    def __init__(self):
        self.scorer = None

    def load_weight(self):
        print("loading pretrained blerut...")
        checkpoint = "./bleurt/bleurt-base-128"
        self.scorer = score.BleurtScorer(checkpoint)

    def calculate_bleurt_score(self, candidate, reference):
        """
        :param candidate:
        :param reference:
        :return:
        """
        results = []
        for i in range(len(candidate)):
            scores = self.scorer.score(references=[reference[i]], candidates=[candidate[i]])
            assert isinstance(scores, list) and len(scores) == 1
            results.append(scores[0])
        return results

class Reward_calculate:
    def __init__(self):
        self.bert_calc = Bert_calculate()

    def load_checkpoint(self):
        self.bert_calc.load_weight()


    def reward_calc(self, candidate, reference):
        bert = self.bert_calc.bertscore_calculate(candidate, reference, batchsize=1)
        reward = bert
        return reward



if __name__ == '__main__':
    pass