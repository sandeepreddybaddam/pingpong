import numpy as np
from train_utils_pong import find_reward


weight_cols = 6 # because of six hand-crafted features
mean = np.zeros(weight_cols)
cov = np.eye(weight_cols) * 100
percentile = 0.3 # for elite sample consideration
r_percentile = 0.1

batch_size = 20 # number of players
# batch_size_end = 10
episodes = 20 # 40 # number of episodes each player plays
episodes_end = 10

DESIRED_SCORE = 21

elite_percentile = int(percentile*batch_size)
retain_percentile = int(r_percentile*batch_size)
elite_retained = []

# Looping parameters
maxits = 100
iter_count = 0

while iter_count<maxits:

    # Initialize random weights (but we would be updating mean
    # and covariance)
    weights = np.random.multivariate_normal(mean, cov, batch_size)

    # consider the previously retained elite weights
    if len(elite_retained):
        batch_size += len(elite_retained)
        weights = np.concatenate([weights, elite_retained])

    # Initialize zero reward for each player
    # this stores average reward of each player for 'n' episodes
    reward = np.zeros(batch_size)
    
    for i in range(batch_size):
        reward[i] = find_reward(weights[i], episodes)
    
    # reward - shape[1, players]
    # weights - shape[players, features]

    # Sort weights according to the reward gained
    rew_inds = reward.argsort()
    sorted_rewards = reward[rew_inds[::-1]] # descending order
    sorted_weights = weights[rew_inds[::-1]]
    elite_rewards = sorted_rewards[0:elite_percentile]
    elite_weights = sorted_weights[0:elite_percentile]

    # Update parameters of sampling distribution
    mean = np.mean(elite_weights, axis=0)
    cov = np.cov(elite_weights, rowvar=False)
    # our varibles are along the columns and obs are along rows

    # adding decreasing noise to covariance to avoid
    # sub-optimal solution (Source: above paper)
    decreasing_factor = max((5-iter_count/10), 0)
    noise = np.eye(weight_cols) * decreasing_factor

    cov+=noise
    
    # Retaining few elite samples for next iteration
    elite_retained = elite_weights[0:retain_percentile]

    avg_elite_reward = np.mean(elite_rewards)
    print("Average elite reward: ", avg_elite_reward)


    # Explore initially and then exploit as the model improves
    # i.e., decreasing batch_size and increasing episodes with performance
    episodes_decision = (int) ( \
                        (DESIRED_SCORE-avg_elite_reward) * episodes/(2*DESIRED_SCORE) + \
                        (DESIRED_SCORE+avg_elite_reward) * episodes_end/(2*DESIRED_SCORE))
    episodes = max(episodes_decision, episodes_end)

    # batch_size_decision = (int) ( \
    #                     (DESIRED_SCORE-avg_elite_reward) * batch_size/(2*DESIRED_SCORE) + \
    #                     (DESIRED_SCORE+avg_elite_reward) * batch_size_end/(2*DESIRED_SCORE))
    # batch_size = max(batch_size_decision, batch_size_end)


    with open('pong_elite_weights_5.npy', 'wb') as f:
        np.save(f, elite_weights)
    
    iter_count+=1