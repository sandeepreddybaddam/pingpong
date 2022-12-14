### Video
[Presentation Video](https://drive.google.com/file/d/1kmHJZ692sxHUmiyb73w93Hen8Teo85ZM/view?usp=share_link)

### Abstract
The pong game could be a good example, to see how one can apply RL algorithms to games and make the AI agent play against humans. At any moment, all the information that is available to us is an image frame of the game (say 210x160x3), which is a snapshot of the game at that moment. We call this image the “state” of the game in terms of RL literature. Our task now is to get the best “action” available for that state so that when we take that action, we maximize our score over time. Here, the actions available to us are moving “UP”, moving “DOWN” , or “STAY” at the same position.

### Problem Statement
The goal of this project is to build a pong learning player that should be able to score 5 points(including -1 when misses the ball) on an average among 20 episodes. And, the constraint here is limiting the training time.

Here, we would be using two different algorithms to build a Ping-Pong playing learner.
- Noisy Cross Entropy Method
- Deep-Q Networks

Later, compare the performance obtained from both the methods with the Policy Gradients method (sourcing from Karpathy’s blog). We train our RL (Reinforcement Learning) agent by actually playing against a hard-coded opponent. Among these techniques, cross-entropy uses self-designed features or in other words, it does not use convolutions to extract features from game frames but whereas the others do. Moreover, Deep Q-Networks and Policy Gradients techniques use deep neural networks in learning action-value(Q) function and policy respectively.


#### Why Reinforcement Learning?
Reinforcement Learning (RL) makes agent learn to perform actions in an environment so as to maximize a reward. There are two main components:
1. Environment : Represents the problem to be solved
2. Agent: Represents the learning algorithm

And, these algorithms have proven record at complex games like AlphaGo and Dota. This was the main motivation to take up this project.


### Methodology
RL is a set of methods that learn how to optimally behave in an environment and MDP is formal representation of such an environment.
Given problem setup can be formulated as an Markov Decision Problem(MDP) because of the fact that it satisfies Markovian property.
Markovian property: Current state information is sufficient to predict the future i.e., no past information is required.

#### Components of MDP:**

- Agent: Trained Pong AI-model
- State: Sequence of frames. NOTE: State representation differs for different methods.
- Action: Discrete space whether the paddle has to move UP, DOWN, or IDLE
- Reward function shaping is crucial in any reinforcement learning algorithm as it characterizes the agent how to play. Sometimes, it is very difficult for complex games like Dota. In pong, the reward function is as follows: -1, if the agent misses the ball; 1, if the opponent misses the ball; 0, otherwise. **r(s,a)** = number of points gained in current turn

We used [OpenAI’s Gym](https://www.gymlibrary.dev/environments/atari/pong/) environment to develop and test pong learning player.

#### 1. Cross Entropy Method
Due to the discrete nature of the game, gradient-free methods are often the primary approach to reward optimization[1]. This includes methods such as the Cross Entropy Method and CMA-ES. These methods also perform better when reducing the state-space to a set of hand-crafted features[3] and learning a set of weights on these features through reinforcement learning episodes. These features are self-designed for this project and can capture higher-level properties of the game frame such as: (1) agent’s paddle position, (2) opponent’s paddle position, (3) ball x-position, (4) ball y-position, (5) ball direction, and (6) distance difference (euclidean distance (agent’s paddle to ball) difference between current frame and immediate next frame).

More recent work on Ping-pong used deep learning methods. A set of convolutional layers are first applied on the game frame, then fed into networks for feature extraction. Here in pong, the frames are pre-processed to gray out  the unwanted pixels. By doing so, most of the pixels become zero, which makes input data sparse. This may lead to inefficient learning and we will look at comparison at the end of this page.

##### Approach

We used a variant of the cross-entropy method i.e., noisy cross-entropy method[8] which is a black-box policy optimization technique. We call this because the algorithm tries to find the weights that map the states to best action. In this method, we have two repeating phases[9]. One, we draw a sample from a probability distribution. Two, minimize the cross-entropy between this distribution and a target distribution to produce a better sample in the next iteration. We implemented this technique to learn the weights for the six hand-crafted features. It is a linear model (weights multiplied by features) that gives value for each possible action given the state information. Then we choose the action that has the maximum value. Here, the algorithm analyzes many players over a set of episodes and selects weights as the mean of elite samples for the next iteration. Elite samples are the ones that score the maximum number of points on an average over episodes. Because of this fact, it is not a greedy approach.

After performing some iterations trying out the cross-entropy method without any noise to covariance, it showed that the learned weights reach sub-optimal values very early (also termed as local optima). This indicates that CE application to RL problems lead to the learned distribution concentrating to a single point too fast. To prevent this early convergence, [8] introduced a trick to add some extra noise (Z) to the distribution. Moreover to make the agent perform even more better, [8] mentions that decreasing noise turns out to be better than just adding constant noise.

#### 2. Policy Gradients
Source:[link](https://karpathy.github.io/2016/05/31/rl/)
As we are using the gym environment to learn a pong game, the input is an RGB frame showing the paddles, ball, and the score.
![pong_RGB](https://user-images.githubusercontent.com/100727983/207286374-6d0a1c4f-8b42-4c87-8b98-4499f7585b2d.png)

Here, much of the information is unwanted like borders, score when understanding the state of the game. So, a processing is being done on input frames of size (210x210x8). First, we crop, downsample by factor of 2, black out background pixels. Finally set the pixel values of paddles and ball to one and then flatten it. Overall, it results in a grayscale vector of size 6400. To understand game state better (like capturing ball direction), we would be sending the difference between current and previous frames as input.

Next, we put this into a neural network (architecture defined in below section). This has a single output unit that gives probability of taking action 2 (In gym, UP-2, DOWN-3).

Main step: We compute discounted reward for the entire episode. Here, we exponentially decay any reward over time. Then, this would be normalized (subtracted with mean and divided over standard deviation) and used as an advantage (we will see how it’s used later).
Given this is kind of logistic regression, loss can be defined as `L = ylogp + (1-y)log(1-p)`.

But, in policy gradients, we multiply log probabilities with advantage to weigh them correctly as per their performance. This can make learning faster as we are modulating the gradient with advantage.
We then backprop using the derivative of the [log probability of the taken action given this image] with respect ot the [output of the network(before sigmoid)]
Finally, we do RMSProp to update the weights.

![policy](https://user-images.githubusercontent.com/100727983/207290060-cb2219cf-9545-4a8a-a989-a5769c01d023.png)

Architecture: `Input layer(6400) -> RELU -> Hidden layer(200) -> Sigmoid -> Output(1 unit)`
    
#### Training

As mentioned before, training time was fixed for all the methods.
Here, we trained the models for `~36 hours CPU time on 3.5GHz x 24 machine`.
Below figures show the learning graphs i.e., running average reward with iterations/episodes.
Once the average reward crosses zero, then there is high probability for agent win.

In Cross Entropy Method, each iteration here corresponds to <=800 games(20 players x 40 episodes). It is `<=` because we introduced the concept of varying episodes ie., reduce the count as the model improves. The final average reward was **5.583**. So, it has **0.6329** probability of winning.

![CEM_training_](https://user-images.githubusercontent.com/100727983/207395749-cf52afb8-aea1-49d9-a142-63e78492a114.png)

For the given training time, Policy Gradients ran for 9756 episodes. The final average reward is **3.01** which indicates that the agent can beat hard-coded player with **0.5714** probability.
![PG_training](https://user-images.githubusercontent.com/100727983/207396771-fda02bc0-be2c-44f6-9593-673ccbb8b454.png)

**Hyperparmaters:**

1. CEM
    - weight_cols = 6 # because of six hand-crafted features
    - mean = zeros of size weight_cols # Initialization
    - cov = Identity matrix of size weight_cols * 100
    - percentile = 0.3 # for elite sample consideration
    - r_percentile = 0.1
    - batch_size = 20 # number of players
    - episodes = 40 # number of episodes each player plays
    - episodes_end = 10 (varying episodes concept)
    - maxits = 100
 
2. DQN
    - lr = 1e-4
    - batch_size = 32
    - GAMMA = 0.99

3. PG
    - batch_size = 10 # episodes frequency for parameter update
    - learning_rate = 1e-3
    - gamma = 0.99 # discount factor in reward computation
    - decay_rate = 0.99 # decay factor for RMSProp


#### Evaluation and Results

This can be considered as tesing step. In RL, especially when building AI player, the performance that we obtain during training phase is very much in-line with testing as the player actually evaluate itself during training.

**Metric:** Average reward over `20 episodes`(after ~36 hours training)

| Method | Average reward over 20 episodes | Maximum reward among 20 episodes | Win Probability |
| Noisy Cross Entropy Method | 4.75 | 14 | 0.6131 |
| Deep Q-Networks | -17.24 | -13.65 | 0.089 |
| Policy Gradient Method | 2.3 | 9 | 0.5547 |

Agent performance over `20 episodes` (Green indicates win and red indicates lose)


| **Cross Entopy Method**  |  **DQN**  | **Policy Gradients** |
| :-------------------------:|:-------------------------:|:-------------------------: |
| ![CEM_episodes](https://user-images.githubusercontent.com/100727983/207402450-ed42df4b-3878-49c7-a912-adef9c4a4299.png)  |  ![DQN_episodes](https://user-images.githubusercontent.com/100727983/207405271-f43d30d1-3282-4060-8035-bc520c3b256b.png)  |  ![PG_episodes](https://user-images.githubusercontent.com/100727983/207402720-3cc1554a-768b-414a-af34-b9b58b0e4f67.png) |

|-----------------+------------+-----------------+----------------|
| Default aligned |Left aligned| Center aligned  | Right aligned  |
|-----------------|:-----------|:---------------:|---------------:|
| First body part |Second cell | Third cell      | fourth cell    |
| Second line     |foo         | **strong**      | baz            |
| Third line      |quux        | baz             | bar            |
|-----------------+------------+-----------------+----------------|
| Second body     |            |                 |                |
| 2 line          |            |                 |                |
|=================+============+=================+================|
| Footer row      |            |                 |                |
|-----------------+------------+-----------------+----------------|
{: .tablelines}


Finally, let's see how our agent plays at different points during training (CEM):

| `0` hours  |  `12` hours  |  `36` hours |
| :-------------------------:|:-------------------------:|:-------------------------: |
| ![player_begin_AdobeExpress](https://user-images.githubusercontent.com/100727983/207413233-4415303f-3271-42c4-af07-d9b7d5b9a712.gif)  |  ![player_middle_AdobeExpress](https://user-images.githubusercontent.com/100727983/207415515-d34c404f-16e7-4f90-b753-a5dd04e9d2fb.gif)  |  ![best_player_end_AdobeExpress](https://user-images.githubusercontent.com/100727983/207426190-e87fccd0-d4b1-4658-a46f-fb8f1845e9db.gif) |

#### Demo:
High resolution video demonstration [here](https://drive.google.com/file/d/1MMAwT0LQcNm9O05WWfe1gDebs65NVkSP/view?usp=share_link)

- Best Player - Noisy Cross Entropy Method
- Right (Green) - Trained agent
- Left (Orange) - Hard-coded player

![best_player_end_AdobeExpress](https://user-images.githubusercontent.com/100727983/207426190-e87fccd0-d4b1-4658-a46f-fb8f1845e9db.gif)


### Video
[Link](https://drive.google.com/file/d/1MMAwT0LQcNm9O05WWfe1gDebs65NVkSP/view?usp=share_link)
