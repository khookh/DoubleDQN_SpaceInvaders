# From our previous work :
# https://gitlab.com/clengele/egt-presentation-commonpoolresources/-/blob/main/agent/dqn_agent.py
from agents.neuralnetworkutils import returnConvModel
from typing import Tuple
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np
import random


class DQNAgent:
    def __init__(self, nactions: int, obs_shape: Tuple[int, int], epsilonDecay: float, epsilonMin: float,
                 epsilonMax: float, alpha: float,
                 gamma: float, batch_size: int,
                 memorySize: int) -> None:

        """
        CONSTRUCTOR OF DQN AGENT
        :param nactions:
        :param obs_shape:
        :param epsilonDecay:
        :param epsilonMin:
        :param epsilonMax:
        :param alpha:
        :param gamma:
        :param batch_size:
        :param memorySize:
        """
        # ASSIGN THE PARAMETER PASSED AS ATTRIBUTES
        self.nactions = nactions
        self.obs_shape = obs_shape
        self.epsilon_max = epsilonMax
        self.epsilon_min = epsilonMin
        self.epsilon_decay = epsilonDecay
        self.epsilon = epsilonMax
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.batch_size = batch_size
        self.memory_size = memorySize
        self._memory_index = 0
        self._nbr_steps = 0

        # CONSTRUCT THE DEEP NEURAL NETWORK FOR THE MODEL AND THE TARGET
        self.model = returnConvModel(input_shape=obs_shape, output_shape=nactions)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=MeanSquaredError())
        self.model.build((None,) + obs_shape)

        # CONSTRUCT "MEMORY" OF AGENT
        self.obs = np.empty((self.memory_size,) + obs_shape, dtype=np.uint8)
        self.reward = np.empty((self.memory_size,), dtype=np.float32)
        self.actions = np.empty((self.memory_size,), dtype=np.uint8)
        self.next_obs = np.empty((self.memory_size,) + obs_shape, dtype=np.uint8)
        self.done = np.empty((self.memory_size,), dtype=np.bool)

        # DEBUG PURPOSED VARIABLES
        self.random_action_taken = 0
        self.greedy_action_taken = 0

    def save(self, path: str):
        """
        Save the weights of the network
        :param path: filepath where weights are saved
        """
        self.model.save_weights(path)

    def load(self, path: str):
        """
        Load the weights of the network into the model and the target model
        :param path: filepath where weights are saved
        """
        self.model.load_weights(path)

    def learn(self) -> None:
        """
        Update the Q-function estimator by updating the neural network following the Bellman equation. A batch of experiences is first collected. Then the experiences
        are replayed to the agent to approximate the Q function as best as possible
        :param training_epochs:
        :return:
        """
        if self._nbr_steps < self.batch_size:
            raise ValueError("Not enough experiences has been stored for the agent to learn")

        # CREATE BATCHES OF EXPERIENCES TO FEED THE NEURAL NETWORK
        memory_explored = min(self.memory_size, self._nbr_steps)
        item = random.randrange(memory_explored - self.batch_size)
        memory_sample = np.arange(start=item, stop=item + self.batch_size)
        # memory_sample = np.random.randint(low=0, high=memory_explored, size=self.batch_size)
        obs_batch = self.obs[memory_sample]
        reward_batch = self.reward[memory_sample]
        next_obs_batch = self.next_obs[memory_sample]
        done_batch = self.done[memory_sample]
        action_batch = self.actions[memory_sample]

        # ESTIMATE THE Q-VALUE VIA OUR FUNCTION APPROXIMATION
        target = self.model.predict(obs_batch)
        target_next = self.model.predict(next_obs_batch)

        # UPDATE THE TARGET Q VALUE FOLLOWING THE BELLMAN EQUATION
        for i in range(self.batch_size):
            if not done_batch[i]:
                target[i][action_batch[i]] = reward_batch[i] + self.discount_factor * (np.max(target_next[i]))
            else:
                target[i][action_batch[i]] = reward_batch[i]

        self.model.fit(obs_batch, target, batch_size=self.batch_size, verbose=0)

    def qvalues(self, obs):
        """
        Return the Q-values approximated by the neural network
        :param obs: state for which we need to approximate q values
        :return: q-values
        """
        return self.model.predict(obs)

    def randomAction(self) -> int:
        """
        Return a acceptable random action
        :return:
        """
        return np.random.randint(self.nactions)

    def greedyAction(self, obs) -> int:
        """
        Return the best action (the action with the highest q-value) for a certain state
        :param obs: current state
        :return:
        """
        return np.argmax(self.model.predict(obs))

    def act(self, observation):
        """
        Return the action for a certain state with an epsilon-greedy strategy
        :param observation: current state
        :return: the action chosen for the current state
        """
        if np.random.rand() < self.epsilon:
            action = self.randomAction()
            self.random_action_taken += 1
        else:
            action = self.greedyAction(observation)
            self.greedy_action_taken += 1
        return action

    def decayEpsilon(self) -> None:
        """
        Update the epsilon value with a simple greedy epsilon descent
        :return:
        """
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def resetMemory(self):
        """
        Reset the pointer of the memory and indicate that it is empty. This function does not erase the previous
        experiences in order to save time. However, it indicates which experiences are valid to use
        :return:
        """
        self._memory_index = 0
        self._nbr_steps = 0

    def addExperiences(self, obs, reward, done, next_obs, action) -> None:
        """
        add a new sample into the memory. Afterwards thoses samples will be used to update the Q-function estimator.
        It also increment the memory pointer to store the experience to a new position. The counter reset whenever
        the memory pointer reaches the size max of the memory.
        :return:
        """
        self.obs[self._memory_index] = obs
        self.reward[self._memory_index] = reward
        self.done[self._memory_index] = done
        self.next_obs[self._memory_index] = next_obs
        self.actions[self._memory_index] = action

        # UPDATE THE MEMORY POINTER AND AGENT STEP
        self._memory_index += 1
        self._nbr_steps += 1
        self._memory_index %= self.memory_size


if __name__ == "__main__":
    agent = DQNAgent(8, (11, 11, 3), .995, .01, .25, 0.0015, .99, 8, 200)
    agent.model.summary()
