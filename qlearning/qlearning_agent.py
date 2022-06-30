from typing import Any, List
import numpy as np

from mlagents_envs.base_env import ActionTuple


class QLearning():

    def __init__(self, choose_action_method: str, initial_epsilon: float,
                 final_epsilon: float, gamma: float, learning_rate: float,
                 num_actions: int, num_episodes: int):
        self.choose_action_method: str = choose_action_method
        self.epsilon: float = initial_epsilon
        self.epsilon_decay: float = (initial_epsilon -
                                     final_epsilon) / num_episodes
        self.gamma: float = gamma
        self.learning_rate: float = learning_rate
        self.num_actions: int = num_actions
        self.num_episodes: int = num_episodes
        self.q_table: dict = {}
        self.cumulative_rewards: List[float] = []

    def choose_action_randomly(self, _: Any) -> int:
        return np.random.choice(self.num_actions)

    def choose_action_epsilon_greedy(self, state: int) -> int:
        random_number = np.random.uniform(0,1)
        if random_number <= self.epsilon:
            return self.choose_action_randomly(state)
        else:
            return max(self.q_table[state])

    def choose_action(self, state: int) -> ActionTuple:
        action = self.choose_action_randomly(
            state) if self.choose_action_method.upper(
            ) == "RANDOM" else self.choose_action_epsilon_greedy(state)
        return ActionTuple(discrete=np.array([[action]]))

    def update_q_table(self, current_state: int, action_tuple: ActionTuple,
                       reward: float, next_state: int) -> None:
        action = action_tuple.discrete[0][0]

        if current_state not in self.q_table:
            self.q_table[current_state] = {}
        if action not in self.q_table[current_state]:
            self.q_table[current_state][action] = 0

        best_future_reward = max(self.q_table[next_state].values()) if next_state in self.q_table else 0
        self.q_table[current_state][action] += (self.learning_rate * (reward + (self.gamma * best_future_reward) - self.q_table[current_state][action]))

    def update_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay

    def add_cumulative_rewards(self, cum_rewards: float) -> None:
        self.cumulative_rewards.append(cum_rewards)

    def get_mean_cumulative_rewards(self) -> float:
        return np.array(self.cumulative_rewards).mean()
