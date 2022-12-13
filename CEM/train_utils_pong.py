import numpy as np
import gym
from state_features import pong_features


def find_reward(weights, episodes):

    env = gym.make("Pong-v0")
    
    cum_reward = 0.0
    assert episodes!=0, "Episodes should not be zero"

    for _ in range(episodes):
        env.reset()
        is_terminal = False
        count = 0
        while not is_terminal: # until game ends - is_terminal!=True
            best_action = find_best_action(weights, env)
            _, rew, is_terminal, _ = env.step(best_action)
            cum_reward += rew
            count+=1
    
    avg_reward = cum_reward/episodes
    return avg_reward


def find_best_action(weights, env):
    """
    Possible Actions:
    0: NOOP
    1: -
    2: RIGHT
    3: LEFT
    4: -
    5: -

    """

    feature_fn = pong_features()

    max_val = float('-inf')
    best_action = None
    actions = [0, 2, 3]
    
    for action in actions:
        # cloning state from original game and setting equal to simulate into future
        old_state = env.clone_full_state()
        old_obs = env.unwrapped._get_image()

        # step (but restores to original using above lines in next iteration)
        env.step(action)
        new_obs = env.unwrapped._get_image()

        feature_vector = feature_fn.features(old_obs, new_obs)
        cur_val = np.dot(weights, feature_vector)
        if cur_val > max_val:
            best_action = action
        max_val = max(max_val, cur_val)

        env.restore_full_state(old_state)
    
    return best_action