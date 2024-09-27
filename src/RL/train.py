import argparse
import os
import torch
import numpy as np
import random
import gymnasium as gym
from tqdm import tqdm
from base_module import *
from pymahjong import MahjongPyWrapper as pm
from pymahjong.myEnv_pymahjong import myMahjongEnv

def make_env():
    env = myMahjongEnv()
    return env

def parse_args():
    parser = argparse.ArgumentParser(description="Mahjong AI Training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of environments")
    parser.add_argument("--num_games", type=int, default=10240, help="Number of games per epoch")
    parser.add_argument("--train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    # parser.add_argument("--train_device", type=int, default=3, help="Training device")
    # parser.add_argument("--inference_device", type=int, default=2, help="Inference device")
    return parser.parse_args()

def main():
    args = parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    
    inference_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print("inference_device:", inference_device)
    print("training_device:", training_device)

    env = gym.vector.AsyncVectorEnv([make_env for _ in range(args.num_envs)])
    obs, info = env.reset()
    done = np.array([False for _ in range(args.num_envs)])

    config = GPT2Config(n_embd=512, n_layer=8, n_head=8, n_positions=128)
    inference_model = Policy_Network(config).to(inference_device)
    inference_model.eval()

    training_model = Policy_Network(config).to(training_device)
    training_model.load_state_dict(inference_model.state_dict())
    training_model.train()

    collator = myCollator(device=training_device)
    inference_collator = inference_Collator(device=inference_device)
    optimizer = torch.optim.Adam(training_model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs)): 
        dataset = []
        while len(dataset) < args.num_games * 4:
            input = inference_collator(obs)
            with torch.no_grad():
                output = inference_model.inference(**input)
            prob = output["action_probs"].cpu().numpy()
            actions = np.array([np.random.choice(47, p=prob[i]) for i in range(args.num_envs)])
            try:
                obs, reward, done, _, info = env.step(actions)
            except IndexError as e:
                print(e)
                env.step_wait()
                obs, info = env.close()
                env = gym.vector.AsyncVectorEnv([make_env for _ in range(args.num_envs)])
                obs, info = env.reset()
                done = np.array([False for _ in range(args.num_envs)])
                continue

            if 'final_info' in info:
                for i in range(args.num_envs):
                    if info['final_info'][i] is not None:
                        if info['final_info'][i]['payoffs'].any():
                            dataset.extend(info['final_info'][i]['obs_with_return'])

        for train_epoch in tqdm(range(args.train_epochs)):
            random.shuffle(dataset)
            for batch_idx in range(0, len(dataset), args.batch_size):
                batch = dataset[batch_idx:batch_idx + args.batch_size]
                data = collator(batch)
                for key in data.keys():
                    if key == "info":
                        for sub_key in data[key].keys():
                            data[key][sub_key] = data[key][sub_key].to(training_device)
                    else:
                        data[key] = data[key].to(training_device)
                optimizer.zero_grad()
                output = training_model.forward(**data)
                loss = output["loss"]
                loss.backward()
                optimizer.step()

        inference_model.load_state_dict(training_model.state_dict())
        torch.save(inference_model.state_dict(), f"./checkpoints/epoch_{epoch}.pth")

if __name__ == "__main__":
    main()
