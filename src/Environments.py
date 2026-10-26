import bisect 
import os
import time
import numpy as np
from utils import euclidean_distance, euclidean_distance_satisfaction_numba, \
    preprocess_circular_obstacles, preprocess_rectangular_obstacles, \
    preprocess_spherical_obstacles_3d, preprocess_cuboid_obstacles_3d
from enum import Enum

class SquareEnvObsShape(Enum):
    """
    Classes of obstacle 
    """
    RECTANGLE = 1
    CIRCLE = 2
    UNSET = 3
    SPHERE = 4
    CUBOID = 5


class AbstractObstacle:
    """
    Obstacle base class 
    """
    def __init__(self):
        # max time before the goal state of this obstacle
        self.max_time = 0.0
        self.shape = SquareEnvObsShape.UNSET

    def positionAtTime(self, t):
        '''
        Should return a tuple of (x, y, r)
        '''
        raise Exception("This obstacle doesn't have a time component!")
    
    def check_collision(self, agent_state, agent_radius, t=-10., obstacle_buffer=0.):
        '''
        Should return True if collision, False else 
        '''
        raise Exception("Don't instantiate this base class!")


class CircularObstacle2D(AbstractObstacle):
    """
    Circular Obstacle with time component. 

    Even though this obstacle does not move, allowing
    the calling environment to call this obstacle as 
    if it does simplifies things. 
    """
    def __init__(self, x, y, r):
        """Init new circular obstacle with time component. 

        Args:
            x (float): x position of obs center
            y (float): y position of obs center
            r (float): obstacle radius
        """
        super().__init__()
        self.x = x
        self.y = y 
        self.r = r 
        self.pos = np.array([x, y], dtype=np.float64)
        self.shape = SquareEnvObsShape.CIRCLE

    def check_collision(self, agent_state, agent_radius, obstacle_buffer=0.):
        """Returns true if a CIRCULAR agent is in collision with a CIRCULAR obstacle

        Args:
            agent_state (tuple-like): Tuple, list, etc. containing the 'x' position of
                the agent in [0] and the 'y' position in [1]
            agent_radius (float): CIRCULAR agent's radius 
            t (float, optional): Time to check agent position at. NOT USED. Defaults to -10.
            obstacle_buffer (float, optional): Env obstacle buffer. Defaults to 0..

        Returns:
            bool: True if collision detected, false else. 
        """
        # d = euclidean_distance((agent_state[0], agent_state[1]), (self.x, self.y))
        # return d < (agent_radius + self.r + obstacle_buffer)
        flag, d = euclidean_distance_satisfaction_numba(agent_state, self.pos, agent_radius + self.r + obstacle_buffer)
        return flag


class CircularObstacle2DTimed(AbstractObstacle):
    """
    Circular Obstacle with time component. 

    Even though this obstacle does not move, allowing
    the calling environment to call this obstacle as 
    if it does simplifies things. 
    """
    def __init__(self, x, y, r):
        """Init new circular obstacle with time component. 

        Args:
            x (float): x position of obs center
            y (float): y position of obs center
            r (float): obstacle radius
        """
        super().__init__()
        self.x = x
        self.y = y 
        self.r = r 
        self.pos = np.array([x, y], dtype=np.float64)
        self.shape = SquareEnvObsShape.CIRCLE

    def positionAtTime(self, t):
        """Gets the position of this obstacle at time t, which
        as this obstacle does not move will always be its 
        static position

        Args:
            t (float): time at which to find obstacle's position, 
            not used. 

        Returns:
            tuple(float, float, float): obstacle's position (x, y, r)
        """
        return (self.x, self.y, self.r)

    def check_collision(self, agent_state, agent_radius, t=-10., obstacle_buffer=0.):
        """Returns true if a CIRCULAR agent is in collision with a CIRCULAR obstacle

        Args:
            agent_state (tuple-like): Tuple, list, etc. containing the 'x' position of
                the agent in [0] and the 'y' position in [1]
            agent_radius (float): CIRCULAR agent's radius 
            t (float, optional): Time to check agent position at. NOT USED. Defaults to -10.
            obstacle_buffer (float, optional): Env obstacle buffer. Defaults to 0..

        Returns:
            bool: True if collision detected, false else. 
        """
        d = euclidean_distance((agent_state[0], agent_state[1]), (self.x, self.y))
        return d < (agent_radius + self.r + obstacle_buffer)


class RectangleObstacle2D(AbstractObstacle):
    """
    Rectangular Obstacle with time component. 

    Even though this obstacle does not move, allowing
    the calling environment to call this obstacle as 
    if it does simplifies things. 
    """
    def __init__(self, x, y, w, h):
        """Init new rectangle obstacle with time component. 

        Args:
            x (float): x position of obs center
            y (float): y position of obs center
            r (float): obstacle radius
        """
        super().__init__()
        self.x = x
        self.y = y 
        self.w = w
        self.h = h 
        self.shape = SquareEnvObsShape.RECTANGLE

    def positionAtTime(self, t):
        """Gets the position of this obstacle at time t, which
        as this obstacle does not move will always be its
        static position

        Args:
            t (float): time at which to find obstacle's position,
            not used.

        Returns:
            tuple(float, float, float, float): obstacle's position (x, y, w, h)
        """
        return (self.x, self.y, self.w, self.h)

    def check_collision(self, agent_state, agent_radius, t=-10., obstacle_buffer=0.):
        """Returns true if a CIRCULAR agent is in collision with a RECTANGULAR obstacle

        Args:
            agent_state (tuple-like): Tuple, list, etc. containing the 'x' position of
                the agent in [0] and the 'y' position in [1]
            agent_radius (float): CIRCULAR agent's radius 
            t (float, optional): Time to check agent position at. NOT USED. Defaults to -10.
            obstacle_buffer (float, optional): Env obstacle buffer. Defaults to 0..

        Returns:
            bool: True if collision detected, false else. 
        """

        # Calculate half width and half height of the rectangle
        half_width = self.w / 2
        half_height = self.h / 2

        # Find the closest point on the rectangle to the circle's center
        closest_x = agent_state[0]
        closest_y = agent_state[1]

        # Determine the x-coordinate of the closest point
        # if neither of these conditions are true, will be the 
        # x corresponding with the agent's state. Same with 
        # y below. 
        if agent_state[0] < self.x - half_width:
            closest_x = self.x - half_width
        elif agent_state[0] > self.x + half_width:
            closest_x = self.x + half_width

        # Determine the y-coordinate of the closest point
        if agent_state[1] < self.y - half_height:
            closest_y = self.y - half_height
        elif agent_state[1] > self.y + half_height:
            closest_y = self.y + half_height

        d = euclidean_distance((agent_state[0], agent_state[1]), (closest_x, closest_y))
        return d < (agent_radius + obstacle_buffer)


class AgentObstacle(AbstractObstacle):
    def __init__(self, agent, rrt):
        """
        Moving obstacle representing an agent for which a path is already 
        planned. 

        agent: class providing agent dynamics 
        rrt: rrt class, has node information 
        states: list of state ids for valid path to goal
        """
        super().__init__()
        self.state_dict = rrt.get_high_resolution_path()
        self.state_keys = list(self.state_dict.keys())
        self.agent = agent
        self.max_time = self.state_keys[-1]
        # these shouldn't be used, but if they are this should be 
        # outside the env
        self.x = -100
        self.y = -100
        self.r = agent.radius
        self.shape = SquareEnvObsShape.CIRCLE

        if hasattr(self.agent, 'distance_metric_state_size'):
            self.distance_metric_state_size = self.agent.distance_metric_state_size
        else:
            self.distance_metric_state_size = 2

    def positionAtTime(self, t):
        """
        Gets the position of this obstacle at time t

        Args:
            t (float): time to get position of this obs for

        Raises:
            Exception: when t is negative

        Returns:
            tuple(float, float, float): obstacle's position (x, y, r)
        """
        state = None;
        if (t<0): raise Exception("Time cannot be negative")
        idx = bisect.bisect_left(self.state_keys, t)
        if idx > len(self.state_keys) - 1: 
            state = self.state_dict[self.state_keys[-1]]
        elif idx == 0:
            state = self.state_dict[self.state_keys[0]]
        elif t - self.state_keys[idx-1] < self.state_keys[idx] - t:
            state = self.state_dict[self.state_keys[idx-1]]
        else:
            state = self.state_dict[self.state_keys[idx]]
        
        return (state[0], state[1], self.r)
    
    def check_collision(self, agent_state, agent_radius, t=-10., obstacle_buffer=0.):
        """Returns true if a CIRCULAR agent is in collision with a CIRCULAR obstacle

        Args:
            agent_state (tuple-like): Tuple, list, etc. containing the 'x' position of
                the agent in [0] and the 'y' position in [1]
            agent_radius (float): CIRCULAR agent's radius 
            t (float, optional): Time to check agent position at. Defaults to -10.
            obstacle_buffer (float, optional): Env obstacle buffer. Defaults to 0..

        Returns:
            bool: True if collision detected, false else. 
        """
        # avoid some awkwardness in sample_random_point methods
        if t < 0:
            return False
        (x, y, r) = self.positionAtTime(t)
        d = euclidean_distance((agent_state[0], agent_state[1]), (x, y))
        return d < (agent_radius + r + obstacle_buffer)
    
    def to_np(self):
        if self.distance_metric_state_size == 2:
            array = np.zeros((len(self.state_keys), 3), dtype=np.float64)
            for i in range(len(self.state_keys)):
                array[i][0] = self.state_dict[self.state_keys[i]][0]
                array[i][1] = self.state_dict[self.state_keys[i]][1]
                array[i][2] = self.r
            return array
        elif self.distance_metric_state_size == 3:
            array = np.zeros((len(self.state_keys), 4), dtype=np.float64)
            for i in range(len(self.state_keys)):
                array[i][0] = self.state_dict[self.state_keys[i]][0]
                array[i][1] = self.state_dict[self.state_keys[i]][1]
                array[i][2] = self.state_dict[self.state_keys[i]][2]
                array[i][3] = self.r
            return array
        else:
            raise NotImplementedError("AgentObstacle to_np not implemented for position state size ",
                                      self.distance_metric_state_size)


class SquareEnvironment:
    def __init__(self, length, breadth, obs, obs_buffers = True):
        """Get a new square environment with dimensions, obstacles

        Args:
            length (float): environment 'x' distance
            breadth (float): environment 'y' distance
            obs (list(AbstractObstacle)): list of objects to use in the environment 
        """
        self.env_start = np.array([0, 0], dtype=np.float64)
        self.size = np.array([length, breadth], dtype=np.float64)
        self.boundary_buffer = 0.1
        self.obstacles = obs
        self.obstacle_buffer = 0.0125 * (length+breadth)/2
        if not obs_buffers:
            self.obstacle_buffer = 0.0
        self.static_circular_obstacles = None
        self.static_rectangular_obstacles = None
        self.dynamic_obstacles = []
        self.env_dim = 2

        self.set_vectorized_static_obstacles()

    # stub -- only needed for the pybullet env
    def step_environment(self, states, controls, dt, t, num_steps=10):
        return
    
    def add_obstacle(self, obs):
        self.obstacles.append(obs)

    def add_agent(self, agent, start = [0,0], goal=None):
        # do nothing -- this does something more interesting in the pb env
        return
    
    def reset_agent(self, agent):
        # do nothing -- this does something more interesting in the pb env
        return

    def set_vectorized_static_obstacles(self):
        if self.static_circular_obstacles is None:
            self.static_circular_obstacles = preprocess_circular_obstacles(self)

        if self.static_rectangular_obstacles is None:
            self.static_rectangular_obstacles = preprocess_rectangular_obstacles(self)

        return self.static_circular_obstacles, self.static_rectangular_obstacles


# 3D Spherical Obstacle
class SphericalObstacle3D(AbstractObstacle):
    """
    Static spherical obstacle with time component.
    """
    def __init__(self, x, y, z, r):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.pos = np.array([x, y, z], dtype=np.float64)
        self.shape = SquareEnvObsShape.SPHERE

    def check_collision(self, agent_state, agent_radius, t=-10., obstacle_buffer=0.):
        flag, _ = euclidean_distance_satisfaction_numba_with_l(agent_state, self.pos, 3,
                                                agent_radius + self.r + obstacle_buffer)
        return flag

# 3D Cuboid Obstacle
class CuboidObstacle3D(AbstractObstacle):
    """
    Axis-aligned cuboid obstacle with time component.
    """
    def __init__(self, x, y, z, l, w, h):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.w = w
        self.h = h
        self.shape = SquareEnvObsShape.CUBOID

    def check_collision(self, agent_state, agent_radius, t=-10., obstacle_buffer=0.):
        half_l = self.l / 2
        half_w = self.w / 2
        half_h = self.h / 2

        closest_x = min(max(agent_state[0], self.x - half_l), self.x + half_l)
        closest_y = min(max(agent_state[1], self.y - half_w), self.y + half_w)
        closest_z = min(max(agent_state[2], self.z - half_h), self.z + half_h)

        d = euclidean_distance((agent_state[0], agent_state[1], agent_state[2]),
                               (closest_x, closest_y, closest_z))
        return d < (agent_radius + obstacle_buffer)

# 3D Moving Agent Obstacle
class AgentObstacle3D(AbstractObstacle):
    """
    Moving spherical obstacle following a preplanned 3D path.
    """
    def __init__(self, agent, rrt):
        super().__init__()
        self.state_dict = rrt.get_high_resolution_path()
        self.state_keys = list(self.state_dict.keys())
        self.agent = agent
        self.max_time = self.state_keys[-1]
        # these shouldn't be used, but if they are this should be outside the env
        self.x = -100
        self.y = -100
        self.z = -100
        self.r = agent.radius
        self.shape = SquareEnvObsShape.SPHERE

    def positionAtTime(self, t):
        if (t < 0):
            raise Exception("Time cannot be negative")
        idx = bisect.bisect_left(self.state_keys, t)
        if idx > len(self.state_keys) - 1:
            state = self.state_dict[self.state_keys[-1]]
        elif idx == 0:
            state = self.state_dict[self.state_keys[0]]
        elif t - self.state_keys[idx-1] < self.state_keys[idx] - t:
            state = self.state_dict[self.state_keys[idx-1]]
        else:
            state = self.state_dict[self.state_keys[idx]]
        return (state[0], state[1], state[2], self.r)

    def check_collision(self, agent_state, agent_radius, t=-10., obstacle_buffer=0.):
        if t < 0:
            return False
        (x, y, z, r) = self.positionAtTime(t)
        d = euclidean_distance((agent_state[0], agent_state[1], agent_state[2]), (x, y, z))
        return d < (agent_radius + r + obstacle_buffer)

    def to_np(self):
        array = np.zeros((len(self.state_keys), 4), dtype=np.float64)
        for i in range(len(self.state_keys)):
            array[i][0] = self.state_dict[self.state_keys[i]][0]
            array[i][1] = self.state_dict[self.state_keys[i]][1]
            array[i][2] = self.state_dict[self.state_keys[i]][2]
            array[i][3] = self.r
        return array


class CuboidEnvironment:
    def __init__(self, length, breadth, height, obs, obs_buffers=True):
        """3D environment with cuboid bounds and static obstacles.

        Args:
            length (float): environment 'x' distance
            breadth (float): environment 'y' distance
            height (float): environment 'z' distance (used if z_bounds not provided)
            obs (list(AbstractObstacle)): list of obstacles to use in the environment
            obs_buffers (bool, optional): Toggle obstacle buffer usage. Defaults to True.
        """
        self.env_start = np.array([0, 0, 0], dtype=np.float64)
        self.size = np.array([length, breadth, height], dtype=np.float64)
        self.boundary_buffer = 0.1
        self.obstacles = obs
        self.obstacle_buffer = 0.0125 * ((length + breadth + height) / 3)
        if not obs_buffers:
            self.obstacle_buffer = 0.0
        self.static_circular_obstacles = None
        self.static_rectangular_obstacles = None
        self.dynamic_obstacles = []
        self.env_dim = 3

        self.set_vectorized_static_obstacles()

    # stub -- only needed for the pybullet env
    def step_environment(self, states, controls, dt, t, num_steps=10):
        return
    
    def add_obstacle(self, obs):
        self.obstacles.append(obs)

    def add_agent(self, agent, start=[0,0,0], goal=None):
        # do nothing -- this does something more interesting in the pb env
        return
    
    def reset_agent(self, agent):
        # do nothing -- this does something more interesting in the pb env
        return

    def set_vectorized_static_obstacles(self):
        if self.static_circular_obstacles is None:
            self.static_circular_obstacles = preprocess_spherical_obstacles_3d(self)

        if self.static_rectangular_obstacles is None:
            self.static_rectangular_obstacles = preprocess_cuboid_obstacles_3d(self)

        return self.static_circular_obstacles, self.static_rectangular_obstacles


class QuadVisualizer:
    """
    PyBullet visualizer for quadcopter paths
    """
    def __init__(self, env, agent_radius=0.25):
        try:
            import pybullet as p
            import pybullet_data
        except ImportError as e:
            raise RuntimeError("PyBullet is required for QuadVisualizer. Install pybullet to use visualization.") from e

        self.p = p
        self.env = env
        self.agent_radius = agent_radius
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self._load_obstacles()

    def _load_obstacles(self):
        p = self.p
        for obs in getattr(self.env, "obstacles", []):
            if obs.__class__.__name__ == "SphericalObstacle3D":
                visual = p.createVisualShape(p.GEOM_SPHERE, radius=obs.r, rgbaColor=[1.0, 0.0, 0.0, 0.7])
                p.createMultiBody(baseVisualShapeIndex=visual, basePosition=[obs.x, obs.y, obs.z])
            elif obs.__class__.__name__ == "CuboidObstacle3D":
                half_extents = [obs.l / 2, obs.w / 2, obs.h / 2]
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1.0, 0.0, 0.0, 0.5])
                p.createMultiBody(baseVisualShapeIndex=visual, basePosition=[obs.x, obs.y, obs.z])

    def visualize_path(self, path_dict, goal_position=None):
        p = self.p
        if not path_dict:
            print("No path to visualize.")
            return

        times = sorted(path_dict.keys())
        path = [path_dict[t] for t in times]
        # load quadrotor model from visualizations directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "visualizations", "quadrotor.urdf")
        agent_id = p.loadURDF(urdf_path, basePosition=path[0][:3], useFixedBase=True)

        if goal_position is None:
            goal_position = path[-1][:3]
        goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=self.agent_radius * 0.8, rgbaColor=[0.0, 1.0, 0.0, 0.8])
        p.createMultiBody(baseVisualShapeIndex=goal_visual, basePosition=goal_position)

        idx = 0
        while p.isConnected(self.client):
            p.resetBasePositionAndOrientation(agent_id, path[idx][:3], [0, 0, 0, 1])
            p.stepSimulation()
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if k == p.B3G_RIGHT_ARROW and v & p.KEY_WAS_TRIGGERED:
                    idx = min(idx + 1, len(path) - 1)
                elif k == p.B3G_LEFT_ARROW and v & p.KEY_WAS_TRIGGERED:
                    idx = max(idx - 1, 0)
                elif (k == p.B3G_RETURN or k == 27) and v & p.KEY_WAS_TRIGGERED:
                    p.disconnect(self.client)
                    return
            # Small wait to avoid busy looping
            time.sleep(0.01)

    def __del__(self):
        if hasattr(self, "p") and hasattr(self, "client"):
            try:
                self.p.disconnect(self.client)
            except Exception:
                pass
