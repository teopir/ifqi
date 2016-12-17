import gym, pygame as pg, random, logging
from gym import spaces
from gym.utils import seeding
from PIL import Image

logger = logging.getLogger(__name__)


class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, wall_random=True):
        self.width = 16
        self.height = 9
        self._cell_size = 10

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.height * self._cell_size, self.width * self._cell_size, 1)

        self.wall_random = wall_random
        self.viewer = Viewer(width=self.width, height=self.height, cell_size=self._cell_size, wall_random=self.wall_random)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        self.width = width
        self.height = height
        self.viewer = Viewer(height=self.height, width=self.width, cell_size=self._cell_size, wall_random=self.wall_random)
        self.reset()


class Viewer:
    def __init__(self, width=16, height=9, cell_size=10, wall=True, wall_random=True):
        pg.init()
        # Game parameters
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.wall = wall
        self.wall_random = wall_random
        self.screen_size = (self.width * self.cell_size, self.height * self.cell_size)

        # Position data
        self.goal_pos = []
        self.wall_pos = []
        self.char_pos = ()

        # Directions
        self.directions = {
            0: (0, -self.cell_size), # UP
            1: (0, self.cell_size), # DOWN
            2: (self.cell_size, 0), # RIGHT
            3: (-self.cell_size, 0) # LEFT
        }

        # Surfaces
        self.surface = pg.Surface(self.screen_size)  # Main game surface
        self.surface.fill((0, 0, 0))
        self.character = pg.Surface((self.cell_size, self.cell_size))  # Agent surface
        self.character.fill((255, 255, 255))
        self.goal = pg.Surface((self.cell_size, self.cell_size)) # Goal surface
        self.goal.fill((255, 255, 255))
        self.wall = pg.Surface((self.cell_size, self.cell_size)) # Wall surface
        self.wall.fill((255, 255, 255))

        # Screen
        self.screen = None

        self.initialize_env()
        self.reset_agent()
        self.draw()

    def initialize_env(self):
        # Reset position data
        self.goal_pos = set()
        self.wall_pos = set()
        self.char_pos = (0,0)

        # Init goal
        for y in range(self.height):
            self.goal_pos.add(((self.width - 1) * self.cell_size, y * self.cell_size))

        # Init wall
        if self.wall:
            if self.wall_random:
                x = random.randrange(0, self.width - 1) * self.cell_size # Wall is randomly placed
            else:
                x = self.width / 2 * self.cell_size  # Wall is in the middle
            for y in range(self.height/2):
                self.wall_pos.add((x, y * self.cell_size))

    def reset_agent(self):
        # Randomly place agent on any row of the first column
        y = random.randrange(0, self.height) * self.cell_size
        self.char_pos = (0, y)
        self.draw()

    def draw(self):
        # Clear surface
        self.surface.fill((0, 0, 0))

        # Draw goal
        for p in self.goal_pos:
            self.surface.blit(self.goal, p)

        # Draw wall
        if self.wall:
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
        img_str = pg.image.tostring(self.surface, 'RGB')
        out = Image.frombytes('RGB', self.screen_size, img_str).convert('L')
        return out

    def is_on_goal(self):
        return self.char_pos in self.goal_pos

    def close(self):
        if self.screen is not None:
            pg.display.quit()

    # HELPERS
    def _get_new_agent_pos(self, action):
        addendum = self.directions[action]
        x = self.char_pos[0] + addendum[0]
        y = self.char_pos[1] + addendum[1]
        new_pos = (x, y)
        if new_pos in self.wall_pos or x < 0 or x >= self.width * self.cell_size or y < 0 or y >= self.height * self.cell_size:
            return self.char_pos
        else:
            return new_pos
