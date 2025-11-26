import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import (
    load_data,
    Dataset
)
from ppo_discreate import Actor, Critic, PPO
from environment import Environment
import logging
import threading
import numpy as np
import os
import json
def main(args):

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info(f"device: {device}, n_gpu: {n_gpu}")

    test_examples = load_data(args.test_data_path, args, max_size=args.max_test_data_size)
    test_dataset = Dataset(test_examples)

    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True)


    actor = Actor(config_path=args.actor_model, num_classes=args.LLMs_nums, mlp_hidden_size=256).to(device)
    critic = Critic(args.critic_model,256).to(device)

    actor.load_state_dict(torch.load(args.actor_model_bin))
    critic.load_state_dict(torch.load(args.critic_model_bin))
    ppo_agent = PPO(actor, critic)

    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        sources = batch['source']

        trajectorys = []

        i = 0
        while i < args.batch_size:
            thread_args = []
            for j in range(args.thread_nums):
                args_tuple = (
                    Environment(actor),
                    trajectorys,
                    ppo_agent,
                    input_ids[i+j],
                    attention_masks[i+j],
                    sources[i+j],
                )
                thread_args.append(args_tuple)

            threads = []
            for k, args_tuple in enumerate(thread_args, start=1):
                thread = threading.Thread(target=worker, args=(k, args_tuple))
                threads.append(thread)

            for thread in threads:
                thread.start()


            for thread in threads:
                thread.join()
            i += args.thread_nums
        kk = 0
        output_file_path = os.path.join(args.out_dir, 'Ours.jsonl')
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for traj in trajectorys:
                json.dump(traj, output_file)
                output_file.write('\n')
                print(kk, traj)
                kk = kk + 1


def worker(thread_id, collect_args):
    collect(*collect_args)
    print(f"Thread {thread_id} completed")

def collect(environment, trajectorys, agent, inputs_id, attention_mask, source):

    answers_outs = []
    trajectory = {}

    environment.reset(inputs_id, attention_mask, source)

    step = 0
    while True:
        action, action_pro = agent.get_action_eval(environment.state, environment.attention_masks)
        if action == 0 or step == 4:
            answers, _, _ = environment.step(action)
            answers_outs.append(answers)
            break
        else:
            answers, _, _ = environment.step(action)
            answers_outs.append(answers)
            step += 1
    trajectory["question"] = source
    trajectory['answers'] = answers_outs
    print(np.array([tensor.item() for tensor in environment.action_sequence]))
    trajectory['actions'] = np.array([tensor.item() for tensor in environment.action_sequence]).tolist()
    trajectorys.append(trajectory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("LLM-Synthesis！")

    #model config
    parser.add_argument("--actor_model", type=str, default="./pretrain_model/deberta-v3-large")
    parser.add_argument("--actor_model_bin", type=str, default="./model_pth2/actor_epoch_1.bin")
    parser.add_argument("--critic_model", type=str, default="./pretrain_model/deberta-v3-base")
    parser.add_argument("--critic_model_bin", type=str, default="./model_pth2/critic_epoch_1.bin")

    #data config
    parser.add_argument("--test_data_path", type=str, default="./mix-instruct/test_data_prepared.jsonl")

    # training hyperparameters
    parser.add_argument("--max_test_data_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--thread_nums", type=int, default=50)
    parser.add_argument("--LLMs_nums", type=int, default=12)

    #save path
    parser.add_argument("--out_dir", type=str, default="./ours_outputs/")
    args = parser.parse_args()
    main(args)
