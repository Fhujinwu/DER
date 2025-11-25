from utils.LLM_models import LLMs
from transformers import DebertaV2Tokenizer, AutoTokenizer

class Environment:
    def __init__(self, actor, reward_calculator):
        self.target = None
        self.state_s = None
        self.attention_masks_e = None
        self.actor = actor.to('cuda:0')
        self.state = None
        self.action_sequence = []
        self.ifinit = True
        self.tokenizer = AutoTokenizer.from_pretrained('./opt-125m')
        self.reward_calculator = reward_calculator

    def reset(self, initial_prompt, attention_masks, sources, target):
        """
        :param initial_prompt:
        :param attention_masks:
        :param sources:
        :return:
        """
        self.question = sources
        self.state_s = " "
        self.state = initial_prompt
        self.attention_masks_e = attention_masks
        self.target = target
        self.action_sequence = []
        self.ifinit = True

    def step(self, actions):
        # 采取动作，得到更新想状态
        model_name = ["koala-7B-HF", "Vicuna-13B", "alpaca-13B", "dolly-12B", "baize-13B", "stablelm-7B",
                      "mpt-7B", "OpenAssistant-12B", "t5-xxl", "moss", "chatglm-6B"]
        PROMPT = "{} \nThese are the answers to the question from other student: \n{} \nUsing other student's answers as additional advice, you need to give a more satisfactory answer directly. DO NOT mention other students."
        State = "Question:{} Answer:{}"
        if self.ifinit:
            inputs = self.question
        else:
            inputs = PROMPT.format(self.question, self.state_s)

        self.state_s = LLMs(model_name[actions], inputs)

        new_inputs = State.format(self.question, self.state_s)
        new_inputs = self.preprocess_text_data(new_inputs)
        self.state = new_inputs['input_ids'].to('cuda:3')
        self.attention_masks_e = new_inputs['attention_mask'].to('cuda:3')

        self.action_sequence.append(actions)
        self.ifinit = False

        score = self.reward_calculator.reward_calc([self.state_s], [self.target])
        if score[0] >= 0.73:
            stop = 1
        else:
            stop = 0
        return self.state_s, score, stop

    def preprocess_text_data(self, data):
        source = data

        inputs = self.tokenizer(source, max_length=512, padding='max_length', truncation=True,
                                return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }
