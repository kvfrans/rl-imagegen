import numpy as np
from scipy import misc

class Canvas():
    def __init__(self):
        self.img_dim = 32
        self.num_colors = 1

        # the canvas we'll be drawing on
        self.state = np.zeros([self.img_dim, self.img_dim, self.num_colors])

        self.goal = misc.imread('goal.jpg', flatten=True) / 255.0
        self.goal = np.expand_dims(self.goal, 2)

    def get_state(self):
        # return np.expand_dims(self.state, 0)
        return self.state

    def step(self, actions):
        # actions = np.array of [center_x, center_y, size]
        center_x = actions[0]
        center_y = actions[1]
        size = actions[2] / 3

        x_left = max(0, int(center_x*32.0 - size*32.0))
        x_right = min(32, int(center_x*32.0 + size*32.0))
        y_up = max(0, int(center_y*32.0 - size*32.0))
        y_down = min(32, int(center_y*32.0 + size*32.0))

        self.state[x_left:x_right, y_up:y_down, :] = 1.0;

        # reward = -1 * np.sum(np.square(self.goal - self.state))
        reward = center_x + center_y + size
        return self.get_state(), reward

    def reset(self):
        self.state = np.zeros([self.img_dim, self.img_dim, self.num_colors])
