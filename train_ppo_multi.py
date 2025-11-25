import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.Utils import (
    seed_everything,
    str2bool,
)
from utils.data import (
    load_data,
    Dataset
)
from ppo_discreate import Actor, Critic, PPO
from reward import Reward_calculate
from environment import Environment
import logging
import threading
import wandb
import os
import numpy as np
os.environ['WANDB_MODE'] = 'online'
wandb.init(project="XXX", name="XXX")

def main(args):

    seed_everything(args.seed)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info(f"device: {device}, n_gpu: {n_gpu}")

    train_dataset = None
    if args.do_train:
        train_examples = load_data(args.train_data_path, args, max_size=args.max_train_data_size)
        train_dataset = Dataset(train_examples)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)


    actor = Actor(config_path=args.actor_model, num_classes=args.LLMs_nums, mlp_hidden_size=256).to(device)
    critic = Critic(args.critic_model,256).to(device)

    ppo_agent = PPO(actor, critic)

    reward_calculator = Reward_calculate()
    reward_calculator.load_checkpoint()

    for epoch in range(args.epochs):

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            sources = batch['source']
            targets = batch['target']

            trajectorys = []
            i = 0
            while i < args.batch_size:
                thread_args = []
                for j in range(args.thread_nums):
                    args_tuple = (
                        Environment(actor, reward_calculator),
                        trajectorys,
                        ppo_agent,
                        input_ids[i+j],
                        attention_masks[i+j],
                        sources[i+j],
                        targets[i+j]
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

            print("traj number：", len(trajectorys))
            if len(trajectorys) < 4:
                print("Not enough trajectorys collected, skipping PPO update.")
                None
            else:
                rewards_mean, last_score_mean, score_mean = log_rewards(trajectorys)
                wandb.log({"rewards": rewards_mean, "last_score": last_score_mean, "score": score_mean})
                ppo_agent.update(trajectorys, args.per_batch_size, args.sample_size, device, wandb)
                save_model(actor, critic, epoch, args.save_dir)


def worker(thread_id, collect_args):
    collect(*collect_args)
    print(f"Thread {thread_id} completed")

def collect(environment, trajectorys, agent, inputs_id, attention_mask, source, target):
    rewards = []
    scores = []
    action_pros = []
    states = []
    old_values = []
    attention_masks_outs = []
    trajectory = {}
    environment.reset(inputs_id, attention_mask, source, target)
    states.append(inputs_id)
    attention_masks_outs.append(attention_mask)
    value = agent.get_value(inputs_id.unsqueeze(0), attention_mask.unsqueeze(0))
    old_values.append(value)
    step = 0
    penalise = [0.007, 0.013, 0.013, 0.012, 0.013, 0.007, 0.007, 0.012, 0.011, 0.016, 0.006]
    while True:

        action, action_pro = agent.get_action(environment.state.unsqueeze(0), environment.attention_masks_e.unsqueeze(0))  # 动作标号和对应的概率值
        action_pros.append(action_pro)

        answers, score, stop = environment.step(action)

        if step > 0:
            if score[0] > scores[-1][0]:
                reward = score[0] - penalise[action] + 5 * (score[0] - scores[-1][0])
            else:
                reward = score[0] - penalise[action] - 5 * (scores[-1][0] - score[0])
        else:
            reward = score[0] - penalise[action]
        scores.append(score)
        if stop == 1 or step == 4:
            if stop == 1 and step <= 4:

                reward = reward + 2.0
            elif stop == 0 and step == 4:
                reward = reward - 2.0
            rewards.append([reward])
            break
        else:
            rewards.append([reward])
            states.append(environment.state)
            attention_masks_outs.append(environment.attention_masks_e)
            value = agent.get_value(environment.state.unsqueeze(0), environment.attention_masks_e.unsqueeze(0))
            old_values.append(value)
            step += 1
    trajectory['states'] = states
    trajectory['rewards'] = rewards
    trajectory['scores'] = scores
    trajectory['old_action_pros'] = action_pros
    trajectory['old_values'] = old_values
    trajectory['actions'] = environment.action_sequence
    trajectory['attention_masks'] = attention_masks_outs
    print("actions:", environment.action_sequence, " ;scores:", scores)
    trajectorys.append(trajectory)

def save_model(actor, critic, epoch, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    actor_save_path = os.path.join(save_dir, f"actor_epoch_{epoch}.bin")
    torch.save(actor.state_dict(), actor_save_path)

    critic_save_path = os.path.join(save_dir, f"critic_epoch_{epoch}.bin")
    torch.save(critic.state_dict(), critic_save_path)
    print(f"Saved models at epoch {epoch}.")

def log_rewards(trajectorys):
    sum_mean = 0
    scores_mean = 0
    sum_last = 0
    toal = 0
    for traj in trajectorys:
        score = np.array(traj['scores'])
        scores_mean += np.mean(score)
        sum_last += traj['scores'][-1][0]
        reward = np.array(traj['rewards'])
        sum_mean += np.mean(reward)
        toal += 1
    return sum_mean/toal, sum_last/toal, scores_mean / toal

if __name__ == '__main__':
    parser = argparse.ArgumentParser("LLM-Synthesis！")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    #model config
    parser.add_argument("--actor_model", type=str, default="./opt-125m")
    parser.add_argument("--critic_model", type=str, default="./opt-125m")
    # training/evaluation/test
    parser.add_argument("--do_train", type=str2bool, default=True)

    #data config
    parser.add_argument("--train_data_path", type=str, default=".train_data_prepared.jsonl")


    # training hyperparameters
    parser.add_argument("--max_train_data_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--per_batch_size", type=int, default=16)
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--thread_nums", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--LLMs_nums", type=int, default=11)

    #save path
    parser.add_argument("--save_dir", type=str, default="./")
    args = parser.parse_args()
    main(args)
