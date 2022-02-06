import gym
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from matplotlib import pyplot as plt
from agents.DQNAgent import DQNAgent
from collections import deque

# force CPU usage with TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

RUN_NAME = "TEST_STACKED_FRAMES_"
MODEL_PATH = f"data/weight{RUN_NAME}"

# Parameters definition
LEARNING_RATE = 0.001
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05
EPSILON_MAX = 1
DISCOUNT_RATE = 0.97
BATCH_SIZE = 32
MEMORY_SIZE = 1000
LEARNING_FREQUENCY = 8

N_EPISODE = 50
MAX_EPOCHS = 4000
TEST_FREQUENCY = 10
TEST_NUMBER = 1

STACK_SIZE = 4


def stack_frames(stacked_frames, frame, is_new_episode):
    """
    Frame Stacking technique from original DQN DeepMing paper, example here https://github.com/EXJUSTICE/Optimized_DQN_OpenAI_SpaceInvaders/blob/master/TF2_SpaceInvaders_GC.ipynb
    :param stacked_frames: existing deque of stacked frames
    :param frame: new frame to add to the queue
    :param is_new_episode: True of it's the first epoch of the episode
    :return: stacked frames in a 3d np matrix and the deque of current stacked frames
    """
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((120, 120), dtype=int) for i in range(STACK_SIZE)], maxlen=STACK_SIZE)
        # Because we're in a new episode, copy the same frame 4x, apply elementwise maxima
        maxframe = np.maximum(frame, frame)
        stacked_frames.append(maxframe)
        stacked_frames.append(maxframe)
        stacked_frames.append(maxframe)
        stacked_frames.append(maxframe)
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Since deque append adds t right, we can fetch rightmost element
        maxframe = np.maximum(stacked_frames[-1], frame)
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(maxframe)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.expand_dims(np.stack(stacked_frames, axis=2), axis=2)

    return stacked_state[:, :, :, 0], stacked_frames


def convert_obs(obs):
    """
    :param obs: RGB image [n x n x 3] describing the environment's state
    :return: a gray & rescaled image
    """
    return np.expand_dims(cv.resize((cv.cvtColor(obs, cv.COLOR_BGR2GRAY) / 255).astype(np.float32), (120, 120),
                                    interpolation=cv.INTER_AREA), axis=2)


def run_episode(env_, agent_, max_epochs: int, learning: bool = True, render: bool = False):
    """
    episode loop
    :param env_: game environment
    :param agent_: agent
    :param max_epochs: maximum number of epochs per episode
    :param learning: default = True, if False the learning of the agent is disabled
    :param render: default = False, if True will display each state of the episode
    :return: episode's updated agent and metrics
    """
    ep_done = False
    cum_reward = 0
    epoch = 0
    cum_imputation = 0
    previous_obs = convert_obs(env_.reset())
    previous_obs, stacked_frames = stack_frames(None, previous_obs, True)
    while not ep_done and epoch < max_epochs:
        next_action = agent_.act(previous_obs[None]) if learning else agent_.greedyAction(previous_obs[None])
        next_obs, reward, ep_done, info = env_.step(next_action)
        next_obs = convert_obs(next_obs)
        next_obs, stacked_frames = stack_frames(stacked_frames, next_obs, True)
        if reward == 0:
            cum_imputation += 1 / 2000
            reward -= min(cum_imputation, 0.025)
        else:
            cum_imputation = 0
        if render:
            cv.waitKey(1)
            env_.render()
        if learning:
            agent_.addExperiences(obs=previous_obs, action=next_action, reward=reward,
                                  done=ep_done, next_obs=next_obs)
            if epoch >= BATCH_SIZE and not epoch % LEARNING_FREQUENCY:
                agent_.learn()
        cum_reward += reward
        epoch += 1
        previous_obs = next_obs
    return agent_, cum_reward, epoch


def training():
    best_return = 0
    training_return_list = []
    test_return_list = []
    test_duration_list = []
    env = gym.make('SpaceInvaders-v0')
    obs = env.reset()
    agent = DQNAgent(env.action_space.n, (convert_obs(obs).shape[0], convert_obs(obs).shape[1], STACK_SIZE), EPSILON_DECAY,
                     EPSILON_MIN, EPSILON_MAX,
                     LEARNING_RATE, DISCOUNT_RATE, BATCH_SIZE, MEMORY_SIZE)
    for ep in tqdm(range(1, N_EPISODE + 1), "Training Episodes"):
        agent, training_return, _ = run_episode(env, agent, MAX_EPOCHS)
        agent.decayEpsilon()
        training_return_list.append(training_return)
        temp_test_store = []
        duration_test_store = []
        if not ep % TEST_FREQUENCY:
            print(f"\nTEST at episode {ep}\n")
            for i in range(TEST_NUMBER):
                _, test_return, epoch = run_episode(env, agent, MAX_EPOCHS, learning=False)
                temp_test_store.append(test_return)
                duration_test_store.append(epoch)
                if test_return > best_return:
                    agent.save(MODEL_PATH)
                    best_return = test_return
            test_return_list.append(np.mean(temp_test_store))
            test_duration_list.append(np.mean(duration_test_store))
    return training_return_list, test_return_list, test_duration_list


# TODO : EVERYTHING

# TEST
if __name__ == '__main__':
    training_list, test_list, duration_list = training()

    fig = plt.figure(figsize=(8, 6))
    plt.title("Agent Training Return")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(range(1, len(training_list) + 1), training_list)
    plt.savefig(f"data/agent_training_return_{RUN_NAME}.png")

    plt.figure(figsize=(8, 6))
    plt.title("Agent Average Test Return")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(range(TEST_FREQUENCY, N_EPISODE + 1, TEST_FREQUENCY), test_list)
    plt.savefig(f"data/agent_average_test_return_{RUN_NAME}.png")

    plt.figure(figsize=(8, 6))
    plt.title("Agent Average Test Duration")
    plt.xlabel("Episode")
    plt.ylabel("N_Epochs")
    plt.plot(range(TEST_FREQUENCY, N_EPISODE + 1, TEST_FREQUENCY), duration_list)
    plt.savefig(f"data/agent_average_test_duration_{RUN_NAME}.png")
