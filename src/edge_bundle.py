#edge_bundle.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from utils import get_dtype_from_input,find_roundoff_decimal_digits

"""
Vehicle State S = (x,y,theta,v,omega)
Vehicle Control U = (acceleration, steering_angle)

Stored Edge bundle is a dictionary with the following keys
    1) 'actions' : array of actions where every action a_i is a ndarray of values.
    2) 'timesteps': array of time steps where a time step t_i denotes how long the action was executed for
    2) 'final_states' : array of the final state of an agent after the action a_i was executed for time step t_i 
"""

class EdgeBundle:
    def __init__(self,data,*,
                 use_all_edges=False,
                 fix_num_edges=1000,
                 rng_seed=23):

        if use_all_edges:
            fix_num_edges = len(data['timesteps'])
            self.chosen_indices = np.arange(fix_num_edges)
        else:
            self.rng = np.random.default_rng(rng_seed)
            total_edges = len(data['timesteps'])
            self.chosen_indices = self.rng.choice(total_edges, size=fix_num_edges, replace=False)

        self.num_edges = fix_num_edges
        self._fields = list(data.keys())
        for field in self._fields:
            setattr(self, field, data[field][self.chosen_indices])
        self.edge_index = 0
        self.distance_from_random_point = np.zeros(self.num_edges)

        if 'final_states' not in self._fields:
            self.final_states = []
            for i in range(self.num_edges):
                trajectory = self.trajectories[i]
                trajectory_length = self.trajectory_lengths[i]
                final_state = trajectory[trajectory_length - 1]
                self.final_states.append(final_state)
            self.final_states = np.array(self.final_states)
        
    def get_next_edge(self):
        if self.edge_index < self.num_edges:
            action = self.actions[self.edge_index]
            timestep = self.timesteps[self.edge_index]
            final_state = self.final_states[self.edge_index]
            self.edge_index += 1
            return action, timestep, final_state
        else:
            return None, None, None
        
    def get_edge(self, edge_index):
        if edge_index < self.num_edges:
            action = self.actions[edge_index]
            timestep = self.timesteps[edge_index]
            final_state = self.final_states[edge_index]
            return action, timestep, final_state
        else:
            print("Edge index out of bounds")
            return None, None, None
        
    def plot_edges_2d(self, edge_ids=None, show_starts=True):
        """
        Plot edges in 2D using the stored trajectories.

        Parameters
        ----------
        edge_ids : array-like of int, optional
            Indices of edges in this EdgeBundle (0 .. self.num_edges-1).
            If None, all edges in the bundle are plotted.
        show_starts : bool, default True
            Whether to scatter-plot the start states for the selected edges.
        """
        trajs = self.trajectories          # shape: (num_edges, max_len, state_dim)
        lengths = self.trajectory_lengths  # shape: (num_edges,)

        E = self.num_edges

        if edge_ids is None:
            edge_ids = np.arange(E, dtype=int)
        else:
            edge_ids = np.asarray(edge_ids, dtype=int)

        # Filter out-of-range indices
        valid_mask = (edge_ids >= 0) & (edge_ids < E)
        if not np.all(valid_mask):
            print("Warning: some edge_ids are out of range and will be ignored.")
            edge_ids = edge_ids[valid_mask]

        if edge_ids.size == 0:
            print("No valid edge IDs to plot.")
            return

        segments = []
        for i in edge_ids:
            L = int(lengths[i])
            xy = trajs[i, :L, :2]          # take x,y of the trajectory
            if np.isnan(xy).any():
                xy = xy[np.isfinite(xy).all(axis=1)]
            if xy.shape[0] >= 2:
                segments.append(xy)

        if not segments:
            print("No valid segments to plot (all NaN/too short).")
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        lc = LineCollection(segments, linewidths=0.8)
        ax.add_collection(lc)

        all_xy = np.vstack(segments)
        ax.set_xlim(all_xy[:, 0].min(), all_xy[:, 0].max())
        ax.set_ylim(all_xy[:, 1].min(), all_xy[:, 1].max())

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Edge bundle (showing {len(segments)} edges)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Optional: plot start states if present
        if show_starts and hasattr(self, "start_states"):
            starts = np.asarray(self.start_states)
            ax.scatter(starts[edge_ids, 0], starts[edge_ids, 1], s=6, c="red", alpha=0.5)

        plt.show()

    def plot_edges_by_ids(self, edge_ids, show_starts=True):
        """
        Convenience wrapper to plot only the given edge IDs.

        Parameters
        ----------
        edge_ids : array-like of int
            Indices in this EdgeBundle (0 .. self.num_edges-1).
        """
        self.plot_edges_2d(edge_ids=edge_ids, show_starts=show_starts)



class EdgeBundleTraj(EdgeBundle):
    def __init__(self, data, fix_num_edges=1000):
        super().__init__(data, fix_num_edges)
        self.trajectories = data['trajectories'][:fix_num_edges]

    def get_trajectory(self, traj_index):
        if traj_index < self.num_edges:
            return self.trajectories[traj_index]
        else:
            return None

# Load the edge bundle
# edge_bundle_file_location = 'edge_bundles_npz/eb_unicycle_edges_100000.npz' 
# data = np.load(edge_bundle_file_location)
# d = EdgeBundle(data)


"""
Step to run this file or anything in the Python REPL:
    import sys
    sys.path.append('./src')
    from edge_bundle import * 
    # from edge_bundle import EdgeBundle
    # import edge_bundle as eb

    edge_bundle_file_location = 'edge_bundles_npz/eb_unicycle_edges_100000.npz' 
    data = np.load(edge_bundle_file_location)
    d = EdgeBundle(data)
    action, timestep, final_state = d.get_next_edge()
    print(action, timestep, final_state)
"""


class GenerateEdgeBundle:
    def __init__(self,*, 
                env, agent, 
                get_start_state, 
                num_edges=100000,
                minimum_time_step=0.1, 
                max_sample_T=1.5,
                rng_seed=11,
                ):
        self.env = env
        self.agent = agent
        self.num_edges = num_edges
        self.get_start_state_function = get_start_state
        #Smallest time duration between consecutive agent positions
        self.minimum_time_step = minimum_time_step
        #Maximum time duration for which an action can be executed
        self.max_sample_T = max_sample_T
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)
        #Set number of roundoff digits for time sampling
        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        self.edge_bundle = {}

    def get_random_time(self):
        return round(self.rng.uniform(self.minimum_time_step,
                            self.max_sample_T), self.roundoff_digits)

    def generate_edge_bundle(self):
    
        start_states_array = []
        actions_array = []
        timesteps_array = []
        trajectory_array = []
        trajectory_length_array = []

        for _ in range(self.num_edges):
            start_state = self.get_start_state_function(self.env,self.agent,self.rng)
            action = self.agent.get_random_action(self.rng)
            timestep = self.get_random_time()
            num_record_steps = round(timestep/self.minimum_time_step)
            final_state, _path = self.agent.get_next_state(start_state,
                                    action, timestep, num_record_steps)

            start_states_array.append(start_state)
            actions_array.append(action)
            timesteps_array.append(timestep)
            trajectory_array.append(_path)
            trajectory_length_array.append(len(_path))


        np_start_states = np.array(start_states_array)
        np_actions = np.array(actions_array)
        np_timesteps = np.array(timesteps_array)
        np_trajectory_lengths = np.array(trajectory_length_array)

        max_trajectory_length = round(self.max_sample_T/self.minimum_time_step)
        state_length = len(start_states_array[0])
        # np_trajectory_array = np.zeros((self.num_edges, max_trajectory_length, state_length))
        np_trajectory_array = np.full((self.num_edges, max_trajectory_length, state_length), 
                                    np.nan, dtype=np.float64)
        for i in range(self.num_edges):
            np_trajectory_array[i,:np_trajectory_lengths[i],:] = np.array(trajectory_array[i])

        edge_bundle = {
                    'start_states':np.array(np_start_states),
                    'actions':np.array(np_actions),
                    'timesteps':np.array(np_timesteps),
                    'trajectories':np.array(np_trajectory_array),
                    'trajectory_lengths':np.array(np_trajectory_lengths),
                }
        
        self.edge_bundle = edge_bundle
        return edge_bundle

    def plot_length_hist(self, edge_bundle=None):
        if edge_bundle is None:
            edge_bundle = self.edge_bundle

        bins = np.arange(0, round(self.max_sample_T/self.minimum_time_step)+1, 1)-0.5
        lengths = np.asarray(edge_bundle["trajectory_lengths"])
        plt.figure(figsize=(6,4))
        plt.hist(lengths, bins=bins)
        plt.title("Trajectory length distribution")
        plt.xlabel("length (steps)")
        plt.ylabel("count")
        plt.show()

    def plot_edges_2d(self, edge_bundle=None, max_edges=2000, stride=1, show_starts=True):
        if edge_bundle is None:
            edge_bundle = self.edge_bundle

        trajs = edge_bundle["trajectories"]
        lengths = edge_bundle["trajectory_lengths"]
        E = trajs.shape[0]
        idx = np.arange(0, E, stride)[:max_edges]

        segments = []
        for i in idx:
            xy = trajs[i, :int(lengths[i]), :2]
            if np.isnan(xy).any():
                xy = xy[np.isfinite(xy).all(axis=1)]
            if xy.shape[0] >= 2:
                segments.append(xy)

        fig, ax = plt.subplots(figsize=(7, 7))
        lc = LineCollection(segments, linewidths=0.8)
        ax.add_collection(lc)

        if segments:
            all_xy = np.vstack(segments)
            ax.set_xlim(all_xy[:, 0].min(), all_xy[:, 0].max())
            ax.set_ylim(all_xy[:, 1].min(), all_xy[:, 1].max())

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Edge bundle (showing {len(segments)} edges)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show_starts and "start_states" in edge_bundle:
            starts = np.asarray(edge_bundle["start_states"])
            ax.scatter(starts[idx, 0], starts[idx, 1], s=6, c="red", alpha=0.5)

        plt.show()

    def plot_edges_by_ids(self, edge_ids, edge_bundle=None, show_starts=True):
        """
        Plot specific edges from the edge bundle given their indices.

        Parameters
        ----------
        edge_ids : array-like of int
            Indices of edges to plot (0-based).
        edge_bundle : dict, optional
            Edge bundle dict. If None, uses self.edge_bundle.
        show_starts : bool, default True
            Whether to scatter-plot the start states for the selected edges.
        """
        if edge_bundle is None:
            edge_bundle = self.edge_bundle

        trajs = edge_bundle["trajectories"]
        lengths = edge_bundle["trajectory_lengths"]
        E = trajs.shape[0]

        # Convert to NumPy array and filter invalid ids
        edge_ids = np.asarray(edge_ids, dtype=int)
        valid_mask = (edge_ids >= 0) & (edge_ids < E)
        if not np.all(valid_mask):
            print("Warning: some edge_ids are out of range and will be ignored.")
            edge_ids = edge_ids[valid_mask]

        if edge_ids.size == 0:
            print("No valid edge IDs to plot.")
            return

        segments = []
        for i in edge_ids:
            L = int(lengths[i])
            xy = trajs[i, :L, :2]
            if np.isnan(xy).any():
                xy = xy[np.isfinite(xy).all(axis=1)]
            if xy.shape[0] >= 2:
                segments.append(xy)

        if not segments:
            print("No valid segments to plot (all NaN/too short).")
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        lc = LineCollection(segments, linewidths=0.8)
        ax.add_collection(lc)

        all_xy = np.vstack(segments)
        ax.set_xlim(all_xy[:, 0].min(), all_xy[:, 0].max())
        ax.set_ylim(all_xy[:, 1].min(), all_xy[:, 1].max())

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Edge bundle (showing {len(segments)} edges)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show_starts and "start_states" in edge_bundle:
            starts = np.asarray(edge_bundle["start_states"])
            ax.scatter(starts[edge_ids, 0], starts[edge_ids, 1], s=6, c="red", alpha=0.5)

        plt.show()


