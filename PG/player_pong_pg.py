import numpy as np
import pickle
import gym
from gym.wrappers import Monitor

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
model = pickle.load(open('save_0.001.p', 'rb'))

env = gym.make("Pong-v0")
# env = Monitor(gym.make("Pong-v0"), './video_lr4', force=True)
observation = env.reset()

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(float).ravel()


def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state


episodes = 20
total_score = 0
max_score = 0


for episode in range(episodes):
    env.reset()
    prev_x = None # used in computing the difference frame
    score_per_episode = 0

    is_terminal = False
    print("-------- Episode " + str(episode) + " --------")
    status = 0

    while is_terminal!=True: # until game ends
        # env.render()
        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

        # step the environment and get new measurements
        observation, reward, is_terminal, _ = env.step(action)
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
