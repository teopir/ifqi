import gym
import numpy as np
import pygame as pg, random, logging
from gym import spaces
from PIL import Image

logger = logging.getLogger(__name__)


class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, width=12, height=8, cell_size=8, wall=True, wall_random=True):
        # Viewer parameters
        self.width = width
        self.height = height
        self._cell_size = cell_size
        self.wall = wall
        self.wall_random = wall_random
        self.viewer = Viewer(width=self.width, height=self.height, cell_size=self._cell_size, draw_wall=self.wall, wall_random=self.wall_random)

        # MDP parameters
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(self.height * self._cell_size, self.width * self._cell_size, 1)

        # Reset
        self._seed()
        self.reset()

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.viewer.move_agent(action)
        self.state = self.viewer.get_state()
        done = self.viewer.is_on_goal()
        reward = 1 if done else 0
        return self.state, reward, done, {}

    def _reset(self):
        self.viewer.reset_agent()
        self.state = self.viewer.get_state()
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        return self.viewer.render()

    def set_grid_size(self, width, height):
        """
        Change the size of the gridworld and reset the environment
        :param width: the new width (in number of cells) of the gridworld
        :param height: the new height (in number of cells) of the gridworld
        """
        self.width = width
        self.height = height
        self.viewer = Viewer(height=self.height, width=self.width, cell_size=self._cell_size, draw_wall=self.wall, wall_random=self.wall_random)
        self.reset()

    def encode_action(self, action):
        """
        :param action: an action from the environment's action space
        :return: an [x, y] encoding of the direction associated to the action s.t. x,y are in {-1,0,1}
        """
        direction = self.viewer.directions[action]
        out = [direction[0] / self.viewer.cell_size, direction[1] / self.viewer.cell_size]
        return out


class Viewer:
    def __init__(self, width=16, height=9, cell_size=10, draw_wall=True, wall_random=True):
        pg.init()
        # World parameters
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.draw_wall = draw_wall
        self.wall_random = wall_random
        self.screen_size = (self.width * self.cell_size, self.height * self.cell_size)

        # Position data
        self.goal_pos = []
        self.wall_pos = []
        self.char_pos = ()

        # Directions
        self.directions = {
            0: (0, 0),  # Null action
            1: (0, -self.cell_size),  # N
            2: (self.cell_size, -self.cell_size),  # NE
            3: (self.cell_size, 0),  # E
            4: (self.cell_size, self.cell_size),  # SE
            5: (0, self.cell_size),  # S
            6: (-self.cell_size, self.cell_size),  # SW
            7: (-self.cell_size, 0),  # W
            8: (-self.cell_size, -self.cell_size)  # NW
        }

        # Surfaces
        self.surface = pg.Surface(self.screen_size)  # Main game surface
        self.surface.fill((0, 0, 0))
        self.character = pg.Surface((self.cell_size, self.cell_size))  # Agent surface
        self.character.fill((255, 255, 255))
        self.goal = pg.Surface((self.cell_size, self.cell_size))  # Goal surface
        self.goal.fill((255, 255, 255))
        self.wall = pg.Surface((self.cell_size, self.cell_size))  # Wall surface
        self.wall.fill((255, 255, 255))

        # Screen
        self.screen = None

        # Initialize viewer
        self.initialize_env()
        self.reset_agent()
        self.draw()

    def initialize_env(self):
        """
        Initializes the environment by resetting the character position and redrawing the goal and wall surfaces.
        """
        # Reset position data
        self.goal_pos = set()
        self.wall_pos = set()
        self.char_pos = (0, 0)

        # Initialize goal
        for y in range(self.height):
            self.goal_pos.add(((self.width - 1) * self.cell_size, y * self.cell_size))

        # Initialize wall
        if self.draw_wall:
            if self.wall_random:
                x = random.randrange(1, self.width - 1) * self.cell_size  # Wall is randomly placed
            else:
                x = self.width / 2 * self.cell_size  # Wall is in the middle
            for y in range(self.height / 2):
                self.wall_pos.add((x, y * self.cell_size))

    def reset_agent(self):
        """
        Randomly places the agent on any row of the first column
        """
        y = random.randrange(0, self.height) * self.cell_size
        self.char_pos = (0, y)
        self.draw()

    def draw(self):
        """
        Updates the pygame surface to match the current status
        """
        # Clear surface
        self.surface.fill((0, 0, 0))

        # Draw goal
        for p in self.goal_pos:
            self.surface.blit(self.goal, p)

        # Draw wall
        if self.draw_wall:
            for p in self.wall_pos:
                self.surface.blit(self.wall, p)

        # Draw character
        self.surface.blit(self.character, self.char_pos)

    def render(self):
        # Start the game window the first time
        if self.screen is None:
            self.screen = pg.display.set_mode(self.screen_size)
        self.screen.blit(self.surface, (0, 0))
        self.draw()
        pg.display.update()

    def move_agent(self, action):
        self.char_pos = self._get_new_agent_pos(action)
        self.draw()

    def get_state(self):
        """
        :return: grayscale image (PIL Image object) of the current pygame surface.
        """
        img_str = pg.image.tostring(self.surface, 'RGB')
        out = Image.frombytes('RGB', self.screen_size, img_str).convert('L')
        return np.asarray(out)

    def is_on_goal(self):
        """
        :return: True if the character is on the goal of the environment, False otherwise.
        """
        return self.char_pos in self.goal_pos

    def close(self):
        if self.screen is not None:
            pg.display.quit()

    # HELPERS
    def _get_new_agent_pos(self, action):
        addendum = self.directions[action]
        x = self.char_pos[0]
        y = self.char_pos[1]

        # Compute new X coordinate (forbid movements outside of bounds)
        new_x = x + addendum[0]
        if new_x < 0 or new_x >= self.width * self.cell_size:
            new_x = x

        # Compute new Y coordinate (forbid movements outside of bounds)
        new_y = y + addendum[1]
        if new_y < 0 or new_y >= self.height * self.cell_size:
            new_y = y

        new_pos = (new_x, new_y)

        allowed_pos = [(x, y)]
        if new_pos in self.wall_pos:
            if not (new_x, y) in self.wall_pos: # Agent can move horizontally
                allowed_pos.append((new_x, y))
            if not (x, new_y) in self.wall_pos: # Agent can move vertically
                allowed_pos.append((x, new_y))
            # There can be two allowed moves if the agent is on a corner of the wall and tries to move diagonally
            # In that case, choose a random move
            new_pos = random.choice(allowed_pos)

        return new_pos
