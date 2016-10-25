import numpy as np
from scipy import misc

class Object(object):
    pass

class CanvasEnv():
    def __init__(self):
        self.img_dim = 16
        self.num_colors = 1

        # the canvas we'll be drawing on
        self.state = np.zeros([self.img_dim, self.img_dim, self.num_colors])

        self.goal = misc.imread('goal.jpg', flatten=True) / 255.0
        self.goal = np.expand_dims(self.goal, 2)

        self.observation_space = Object()
        self.observation_space.shape = [256]

        self.action_space = Object()
        self.action_space.shape = [2]
        self.action_space.high = 1

    def save(self, name):
        misc.toimage(self.state[:,:,0], cmin=0, cmax=1).save(name+".png")

    def get_state(self):
        return np.reshape(self.state, [self.img_dim * self.img_dim * self.num_colors])

    def step(self, actions):
        # actions = np.array of [center_x, center_y, size]
        center_x = (actions[0] / 2) + 0.5
        center_y = (actions[1] / 2) + 0.5
        size = 3

        scaled_x = int(center_x * self.img_dim)
        min_x = max(0, scaled_x - size)
        max_x = min(15, scaled_x + size)
        scaled_y = int(center_y * self.img_dim)
        min_y = max(0, scaled_y - size)
        max_y = min(15, scaled_y + size)

        self.state[min_x:max_x, min_y:max_y, :] = 1.0;

        reward = -1 * np.sum(np.abs(self.goal - self.state))
        return self.get_state(), reward, False, False

    def reset(self):
        self.state = np.zeros([self.img_dim, self.img_dim, self.num_colors])



        return self.get_state()
