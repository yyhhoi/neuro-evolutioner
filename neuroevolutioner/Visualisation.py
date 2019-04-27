import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from .DataProcessing import calculate_firing_rate
from .utils import load_pickle
import skimage as ski
import skvideo.io as skv
import os
from glob import glob
import pandas as pd
import pdb

# 1. Preprocess activity.csv and gene.pickle to generate firing_rate/firing_rate_$TIME.json and weights/weights_$TIME.json
# 2. Call Visualiser.initialise() to get self.weight_range and self.rate_range, for normalisation of color
# 3. Call Visualiser.set_neurons_params() and set_graph_params() to set up necessary parameter
# 4. Call Visualiser.generate_graphs() and store the graphs to graphs/graph_$TIME.png 
# 5. Use skvideo/FFmpeg to combine the graph_$TIME.png as a video

# Work awaits:
# 1. self._find_normalisation_params()

class Visualiser():
    def __init__(self, firing_rate_path, weights_dir, graph_dir, time_step, configs ):
        self.num_neurons = configs["num_neurons"]
        self.anatomy_matrix = configs["anatomy_matrix"]
        self.anatomy_labels = configs["anatomy_labels"]
        self.E_syn = configs["E_syn"]
        self.u_rest = configs["u_rest"]
        self.types_matrix = self._binarise_types(self.E_syn, self.u_rest)
        self.firing_rate_path, self.weights_dir, self.graph_dir = firing_rate_path, weights_dir, graph_dir
        self.time_step = time_step

    def initialise(self):
        """
        Load firing rate to np
        """
        self.weight_max, self.weight_min, self.rate_max, self.rate_min = self._load_and_find_params()
        self.weight_range = self.weight_max - self.weight_min
        self.rate_range = self.rate_max - self.rate_min

    def set_neurons_params(self, neurons_pos, neurons_labels):
        """
        Args:
            neurons_pos: (np.darray) with shape (num_neurons, 2) representing the (x,y) position of each neuron in the graph
            neurons_labels: (list) with length = num_neurons. Labels for all neurons, for example, ["S", "B1", "B2", "A"]...
        """
        self.neuron_pos = neurons_pos
        self.neurons_labels = neurons_labels

    def set_graph_params(self, bgcolor = "white", 
                               graph_labelloc="top", 
                               graph_label_fontsize = 20, 
                               node_color_style = "filled",
                               rate_cmap = "Oranges",
                               node_shape_style = "circle",
                               node_radius = 1.2
                               ):
        self.bgcolor = bgcolor
        self.graph_labelloc = graph_labelloc
        self.graph_label_fontsize = str(graph_label_fontsize)
        self.node_color_style = node_color_style
        self.rate_cmap = plt.get_cmap(rate_cmap)
        self.node_shape_style = node_shape_style
        self.node_radius = str(node_radius)
    
    def visualise_all_moments(self):
        total_len = self.firing_rate_np.shape[0]
        for i in range(total_len):
            # if i != 97:
            #     continue
            print("Visualising %d/%d" % (i, total_len))
            graph_name = "time = %0.4f" % (i*self.time_step)
            file_path = os.path.join(self.graph_dir, "graph_%d" % (i))
            firing_rate = self.firing_rate_np[i, :]
            weights = np.load(os.path.join(self.weights_dir, "weights_%d.npy" % i))
            weights[np.isnan(weights)] = self.weight_min
            weights[np.isinf(weights)] = self.weight_max
            self.visualise_one_moment(graph_name, file_path, firing_rate, weights)



    def visualise_one_moment(self, graph_name, file_path, firing_rate, weights):
        """
        Args:
            graph_name: (str) Name of the graph.
            file_path: (str) Path to save the graph. Extension name should not be included
            firing_rate: (np.darray) with shape (num_neurons,). The filtered firing rate of each neuron
            weights: (np.darray) with shape (num_neurons, num_neurons). Weights of synapse, from presynaptic (row_idx) to post-synaptic (col_idx) neuron

        """

        # Define global graph's/nodes' attributes
        dot = Digraph(engine="neato", format="png")

        dot.attr(bgcolor=self.bgcolor,
                label=graph_name,
                labelloc=self.graph_labelloc,
                fontsize=self.graph_label_fontsize)

        dot.node_attr.update(style=self.node_color_style, 
                             shape=self.node_shape_style, 
                             width=self.node_radius,
                             height=self.node_radius)

        # Draw each node
        for i in range(self.num_neurons):

            # Determine the position of neuron_i
            neuron_pos_str = "%f,%f!" % (self.neuron_pos[i,0], self.neuron_pos[i,1])
            
            if firing_rate[i] == 1:
                color_arr = (np.array(self.rate_cmap(255))*255).astype(int)
            else:
                color_arr = (np.array(self.rate_cmap(0))*255).astype(int)

            # # Determine the color of neuron_i from firing rate
            # firing_rate_norm = self._normalise_rate(firing_rate[i])
            # color_arr = (np.array(self.rate_cmap(firing_rate_norm))*255).astype(int)
            fillcolor = "#%02x%02x%02x%02x" % tuple(color_arr)

            # Add node to the graph
            dot.node(self.neurons_labels[i], 
                     pos = neuron_pos_str, 
                     fillcolor = fillcolor
                     )

        # Find connections that exists
        con_row, con_col = np.where(self.anatomy_matrix == 1)

        # Loop through existing connections and define the edge attribute
        for idx in range(con_row.shape[0]):
            # Retrieve indexes
            row_idx, col_idx = con_row[idx], con_col[idx]
            
            # Define connection attributes
            pre_neuron = self.neurons_labels[row_idx]
            post_neuron = self.neurons_labels[col_idx]
            weight= self._normalise_weight(weights[row_idx, col_idx])
            synapse_type = self.types_matrix[row_idx, col_idx]
            edge_color = "blue" if synapse_type==1 else "red"
            
            # Set edge
            dot.edge(pre_neuron, post_neuron, color = edge_color,
                    penwidth = str(weight*2), arrowsize = str(weight*2), arrowhead="dot")

        dot.render(file_path, cleanup=True, view=False)
    
    def _load_and_find_params(self):
        firing_rate_df = pd.read_csv(self.firing_rate_path)
        self.firing_rate_np = np.array(firing_rate_df.iloc[:, 2:])

        weight_max, weight_min = 20, 0
        rate_max, rate_min  = np.max(self.firing_rate_np),np.min(self.firing_rate_np)
        return weight_max, weight_min, rate_max, rate_min


    def _normalise_weight(self, val):
        return val/self.weight_range
    
    def _normalise_rate(self, val):
        return val/self.rate_range
    @staticmethod
    def _binarise_types(E_syn, u_rest):
        type_matrix = (E_syn > np.repeat(u_rest.reshape(1,-1), u_rest.shape[0], axis = 0)).astype(int)
        return type_matrix


class Visualiser_wrapper():
    def __init__(self, project_name, vis_dir, gen_idx, species_idx, time_step = 0.0001):
        # Paths and names
        self.project_name = project_name
        self.weights_dirname, self.firing_rate_filename, self.configs_filename = "weights", "activity.csv", "gene.pickle"
        self.gen_idx, self.species_idx = gen_idx, species_idx
        self.graph_dir_name, self.vis_dir, self.identifier = "graphs", vis_dir, "gen-{}_species-{}".format(self.gen_idx, self.species_idx)
        self.create_graphs_dir()

        # Configs related
        self.configs = self.load_configs()
        self.anatomy_labels, self.num_neurons = self.configs["anatomy_labels"], self.configs["num_neurons"]
        
        # initialise visualiser
        self.visualiser = Visualiser(self.get_firing_rate_path(),
                                     self.get_weight_dir(), 
                                     self.get_graph_dir(), 
                                     time_step , 
                                     self.configs)
    def initialise(self):
        neurons_pos, neurons_labels = self._setup_pose_labels()
        self.visualiser.initialise()
        self.visualiser.set_neurons_params(neurons_pos, neurons_labels)
        self.visualiser.set_graph_params()
    def generate_graphs(self):
        self.visualiser.visualise_all_moments()

    def combine_graphs_to_video(self, frame_rate = '120/1'):
        all_graphs_paths = glob(os.path.join(self.get_graph_dir(), "*.png"))
        total_graphs_num = len(all_graphs_paths)
        self.vwriter = skv.FFmpegWriter(self.get_video_path(),inputdict={'-r': frame_rate}, outputdict={'-r': frame_rate})
        print("Generate video...")
        for idx in range(total_graphs_num):
            print("\r{}/{}".format(idx,total_graphs_num), end="", flush=True)
            graph_path = self.get_graph_path(idx)
            im = ski.imread(graph_path)
            self.vwriter.writeFrame(im)

        self.vwriter.close()



    def _setup_pose_labels(self):
        num_brain_neurons = self.anatomy_labels["brain"] - self.anatomy_labels["sensory1"]
        neurons_labels = ["S"] +  ["B%d" % (x+1) for x in range(num_brain_neurons)] + ["A"]
        r, r_bias = 3, 2
        neurons_pos = [[-(r + r_bias), -0.5]] + \
                     [[r*np.cos((np.pi/-4)*(i+1)), r*np.sin((np.pi/-4)*(i+1))] for i in range(num_brain_neurons)] + \
                     [[r + r_bias, -0.5]]
        neurons_pos = np.array(neurons_pos)
        return neurons_pos, neurons_labels
    def get_base_dir(self):
        return os.path.join(self.vis_dir, self.project_name, self.identifier)

    def get_weight_dir(self):
        return os.path.join(self.get_base_dir(), self.weights_dirname)
    
    def create_graphs_dir(self):
        os.makedirs(os.path.join(self.get_base_dir(), self.graph_dir_name), exist_ok=True)
        
    def get_video_path(self):
        return os.path.join(self.get_base_dir(), self.identifier + ".mp4")
    def get_graph_dir(self):
        return os.path.join(self.get_base_dir(), self.graph_dir_name)
    def get_graph_path(self, idx):
        return os.path.join(self.get_graph_dir(), "graph_{}.png".format(idx))

    def get_firing_rate_path(self):
        return os.path.join(self.get_base_dir(), self.firing_rate_filename)
    def get_configs_path(self):
        return os.path.join(self.get_base_dir(), self.configs_filename)

    def load_configs(self):
        return load_pickle(self.get_configs_path())

    def __del__(self):
        if self.vwriter is not None:
            self.vwriter.close()