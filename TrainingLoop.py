import gym
import os

from agents.DQNAgent import DQNAgent

# force CPU usage with TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Hyper-Parameters definition
LEARNING_RATE = 0.00025
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
EPSILON_MAX = 1
DISCOUNT_RATE = 0.99
BATCH_SIZE = 128
MEMORY_SIZE = 2000


def run_episode():
    pass


def training():
    pass


# TODO : EVERYTHING

# TEST
if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    agent = DQNAgent(env.action_space.n, env.observation_space.shape, EPSILON_DECAY, EPSILON_MIN, EPSILON_MAX,
                     LEARNING_RATE, DISCOUNT_RATE, BATCH_SIZE, MEMORY_SIZE)
    for step in range(10000):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        observation, reward, done, info = env.step(action)
        env.render()
