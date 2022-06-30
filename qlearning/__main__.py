import os
from typing import Any, Tuple
import numpy as np

from mlagents_envs.registry import default_registry
from qlearning_agent import QLearning

ENV_NAME: str = "Basic"
NUM_ACTIONS: int = 3
STATE_NORMALIZATION: int = 8

action_mapping: dict = {0: "Nothing", 1: "Left", 2: "Right"}


def initialize_unity_env() -> Tuple[Any, Any]:
    env = default_registry["Basic"].make()
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    return env, behavior_name


def initialize_qlearning_agent() -> QLearning:
    choose_action_method = os.getenv("CHOOSE_ACTION_METHOD", "EPSILON_GREEDY")
    initial_epsilon = float(os.getenv("INITIAL_EPSILON", 1))
    final_epsilon = float(os.getenv("FINAL_EPSILON", 0.1))
    gamma = float(os.getenv("GAMMA", 0.99))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.1))
    num_episodes = int(os.getenv("NUM_EPISODES", 100))
    return QLearning(choose_action_method, initial_epsilon, final_epsilon,
                     gamma, learning_rate, NUM_ACTIONS, num_episodes)


def training_qlearning_agent(env: Any, behavior_name: Any,
                             qlearning_agent: QLearning) -> None:
    for episode in range(qlearning_agent.num_episodes):
        # Initializing Environment in the beginning of an episode
        env.reset() 
        decision_steps, terminal_steps = env.get_steps(
            behavior_name)
        tracked_agent = -1
        done = False 
        episode_rewards = episode_num_actions = 0

        while not done:

            # Getting the current state where the agent is located
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # Getting the current state number
            current_state = get_state(decision_steps[tracked_agent].obs)

            # Applying the agent's policy to select an action
            action = qlearning_agent.choose_action(current_state)

            # Executing the action in the current state
            env.set_actions(behavior_name, action)
            env.step()
            episode_num_actions += 1

            # Updating next_state and the immediate reward 
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            next_state = get_state(
                decision_steps[tracked_agent].obs
            ) if tracked_agent in decision_steps else get_state(
                terminal_steps[tracked_agent].obs)
            done = True if tracked_agent in terminal_steps else False
            reward = decision_steps[
                tracked_agent].reward if tracked_agent in decision_steps else terminal_steps[
                    tracked_agent].reward
            episode_rewards += reward

            # Updating Q Table
            qlearning_agent.update_q_table(current_state, action, reward,
                                           next_state)

        # Updating epsilon (considering Decaying Epsilon-Greedy)
        qlearning_agent.update_epsilon()

        # Updating cumulative rewards list
        qlearning_agent.add_cumulative_rewards(episode_rewards)

        # Episode's information
        print(
            f"Total rewards for episode {episode} is {episode_rewards} - {episode_num_actions} executed actions - Epsilon: {qlearning_agent.epsilon} - Mean Cumulative Rewards: {qlearning_agent.get_mean_cumulative_rewards()}"
        )

    print(f"Final Q Table: {qlearning_agent.q_table}")


def get_state(observation: Any) -> int:
    return np.where(observation[0] == 1)[0][0] - STATE_NORMALIZATION


if __name__ == "__main__":
    try:
        env, behavior_name = initialize_unity_env()
        qlearning = initialize_qlearning_agent()
        training_qlearning_agent(env, behavior_name, qlearning)
    finally:
        env.close()
        print("Closed environment")
