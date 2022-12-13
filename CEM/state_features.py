import numpy as np


class pong_features:

    def __init__(self):
        pass


    def bot_position(self, indices):

        bot_idx = np.where(indices[1]==70)[0]
        if len(bot_idx)==0:
            return -1

        bot_position = np.take(indices[0], bot_idx)[0]
        return bot_position
        

    def opp_position(self, indices):
        
        opp_idx = np.where(indices[1]==8)[0]
        if len(opp_idx)==0:
            return -1

        opp_position = np.take(indices[0], opp_idx)[0]
        return opp_position
    

    def ball_position_x(self, indices):
        ball_x_idx = np.where((indices[1]!=8) & (indices[1]!=9) & \
                              (indices[1]!=70) & (indices[1]!=71))[0]
        if len(ball_x_idx)==0:
            return -1
    
        ball_position_x = np.take(indices[0], ball_x_idx)[0]
        return ball_position_x


    def ball_position_y(self, indices):
        ball_y_idx = np.where((indices[1]!=8) & (indices[1]!=9) & \
                              (indices[1]!=70) & (indices[1]!=71))[0]
        if len(ball_y_idx)==0:
            return -1
        
        ball_position_y = np.take(indices[1], ball_y_idx)[0]
        return ball_position_y


    def ball_direction(self, idx_before, idx_after):
        
        pos_before = self.ball_position_y(idx_before)
        pos_after = self.ball_position_y(idx_after)

        if pos_after > pos_before:
            return 1 # towards bot
        
        return -1 # away from bot

    def distance(self, idx_before, idx_after):
        """
        Computes Euclidean disance difference between ball and paddle
        before and after
        """
        bot_b = np.array([self.bot_position(idx_before), 70])
        ball_b = np.array([self.ball_position_x(idx_before), \
                           self.ball_position_y(idx_before)])

        bot_a = np.array([self.bot_position(idx_after), 70])
        ball_a = np.array([self.ball_position_x(idx_after), \
                           self.ball_position_y(idx_after)])

        dist_before = np.linalg.norm(bot_b - ball_b)
        dist_after = np.linalg.norm(bot_a - ball_a)

        return dist_before - dist_after



    def preprocessing(self, image):
        image = image[35:195] # crop
        image = image[::2, ::2, 0] # downsample by factor of 2
        image[image == 144] = 0 # erase background (background type 1)
        image[image != 0] = 1 # everything else(paddles, ball) just set to 1
        
        return image.astype(float).ravel()


    def features(self, obs_before, obs_after):
        
        obs_before = self.preprocessing(obs_before)
        obs_before = obs_before.reshape(80, 80)
        indices_before = np.where(obs_before==1)

        obs_after = self.preprocessing(obs_after)
        obs_after = obs_after.reshape(80, 80)
        indices_after = np.where(obs_after==1)


        state_input = np.array([self.bot_position(indices_before),
                                self.opp_position(indices_before),
                                self.ball_position_x(indices_before),
                                self.ball_position_y(indices_before),
                                self.ball_direction(indices_before, indices_after),
                                self.distance(indices_before, indices_after)])

        return state_input
