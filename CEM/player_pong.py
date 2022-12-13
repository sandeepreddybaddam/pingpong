import numpy as np
import gym
from train_utils_pong import find_best_action
from gym.wrappers import Monitor

# Load the agent
with open('pong_elite_weights_4.npy', 'rb') as f:
    weights = np.load(f)

elite_player = weights[0]

episodes = 1
env = gym.make("Pong-v0")
# env = Monitor(gym.make("Pong-v0"), './video_random', force=True)
assert episodes!=0, "Episodes should not be zero."
total_score = 0
max_score = 0

for episode in range(episodes):
    env.reset()
    is_terminal = False
    print("-------- Episode " + str(episode) + " --------")
    status = 0
    score_per_episode = 0

    while is_terminal!=True: # until game ends
        env.render()
        best_action = find_best_action(elite_player, env)
        _, reward, is_terminal, _ = env.step(best_action)
        score_per_episode += reward
        # if (status%2000==0):
        #     print("Game ongoing...")
        status+=1
    
    print("Score in episode " + str(episode) \
           + ": " + str(score_per_episode) + "\n")

    total_score += score_per_episode
    max_score = max(score_per_episode, max_score)
    

print("Average score: ", (int)(total_score/episodes))
print("Maximum score among 20 episodes: ", max_score)