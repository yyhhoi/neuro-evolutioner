import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


# 1. Preprocess activity.csv and gene.pickle to generate firing_rate/firing_rate_$TIME.json and weights/weights_$TIME.json
# 2. Call Visualiser.initialise() to get self.weight_range and self.rate_range, for normalisation of color
# 3. Call Visualiser.set_neurons_params() and set_graph_params() to set up necessary parameter
# 4. Call Visualiser.generate_graphs() and store the graphs to graphs/graph_$TIME.png 
# 5. Use skvideo/FFmpeg to combine the graph_$TIME.png as a video

# Work awaits:
# 1. self._find_normalisation_params()

class Visualiser():
    def __init__(self, firing_rate_dir, weights_dir, configs ):
        self.num_neurons = configs["num_neurons"]
        self.anatomy_matrix = configs["anatomy_matrix"]
        self.anatomy_labels = configs["anatomy_labels"]
        self.types_matrix = configs["types_matrix"]
        self.firing_rate_dir, self.weights_dir = firing_rate_dir, weights_dir
        pass

    def initialise(self):
        self.weight_max, self.weight_min, self.rate_max, self.rate_min = self._find_normalisation_params()
        self.weight_range = self.weight_max - self.weight_min
        self.rate_range = self.rate_max - self.rate_min

    def set_neurons_params(self, neurons_pos, neurons_labels):
        """
        Args:
            neurons_pos: (np.darray) with shape (num_neurons, 2) representing the (x,y) position of each neuron in the graph
            neurons_labels: (np.darray) with shape (num_neurons,). Labels for all neurons, for example, ["S", "B1", "B2", "A"]...
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
            neuron_pos_str = "%f,%f!" % (self.neuron_pos[i][0], self.neuron_pos[i][0])
            
            # Determine the color of neuron_i from firing rate
            firing_rate_norm = self._normalise_rate(firing_rate[i])
            color_arr = (np.array(self.rate_cmap(firing_rate_norm))*255).astype(int)
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
            pre_neuron = neurons_dict[row_idx][0]
            post_neuron = neurons_dict[col_idx][0]
            weight= self._normalise_weight(weights[row_idx, col_idx])
            synapse_type = self._binarise_types(types_matrix[row_idx, col_idx] )
            edge_color = "blue" if synapse_type==1 else "red"
            
            # Set edge
            dot.edge(pre_neuron, post_neuron, color = edge_color,
                    penwidth = str(weight*2), arrowsize = str(weight*2), arrowhead="dot")

        dot.render(file_path, cleanup=True, view=False)
    
    def _find_normalisation_params(self):
        weight_max, weight_min = 0,0
        rate_max, rate_min  = 0,0
        return weight_max, weight_min, rate_max, rate_min


    def _normalise_weight(self, val):
        return val/self.weight_range
    
    def _normalise_rate(self, val):
        return val/self.rate_range
    @staticmethod
    def _binarise_types(val):
        if val > 0.5:
            return 1
        else:
            return 0


        


