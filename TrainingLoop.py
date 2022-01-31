import gym
import os
import numpy as np

from tqdm import tqdm
from agents.DQNAgent import DQNAgent

# force CPU usage with TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Parameters definition
LEARNING_RATE = 0.00025
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
EPSILON_MAX = 1
DISCOUNT_RATE = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 200

N_EPISODE = 10
MAX_EPOCHS = 10000


def run_episode(env_, agent_, max_epochs: int, learning: bool = True, render: bool = False):
    ep_done = False
    cum_reward = 0
    epoch = 0
    previous_obs = env_.reset()
    while not ep_done or epoch < max_epochs:
        next_action = agent_.act(previous_obs) if learning else agent_.greedyAction(previous_obs)
        next_obs, reward, ep_done, info = env.step(next_action)
        if render:
            env_.render()
        if learning:
            agent_.addExperiences(obs=(previous_obs / 255).astype(np.float32), action=next_action, reward=reward,
                                  done=ep_done, next_obs=(next_obs / 255).astype(np.float32))
            if epoch > max_epochs / 50:
                agent_.learn()
        cum_reward += reward
        epoch += 1
        previous_obs = next_obs
    return agent_, cum_reward


# TODO : EVERYTHING

# TEST
if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    agent = DQNAgent(env.action_space.n, env.observation_space.shape, EPSILON_DECAY, EPSILON_MIN, EPSILON_MAX,
                     LEARNING_RATE, DISCOUNT_RATE, BATCH_SIZE, MEMORY_SIZE)
    for ep in tqdm(range(N_EPISODE), "Training Episodes"):
        agent, training_return = run_episode(env, agent, MAX_EPOCHS)
        print(training_return)
    # demo run
    agent, test_return = run_episode(agent, 100000, learning=False, render=True)
    print(test_return)
