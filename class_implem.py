import networkx as nx
import pennylane as qml
import itertools
import numpy as np
from pennylane import qaoa
from scipy import optimize
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

class QAOA:
    def __init__(self, n_qubits, graph, backend = "qiskit.aer", shots = 1000):
        self.n_qubits = n_qubits
        self.graph = graph

        # create quantum device
        self.dev = qml.device(backend, wires = self.n_qubits, shots = shots)
        self.measure_circuit = qml.QNode(self._measure_circuit, self.dev)
        
        # all possible configurations in lexiographic order
        self.bin_strs = np.array([''.join(i) for i in itertools.product('01', repeat = self.n_qubits)])

        # cost of all possible configurations.  
        self.bin_strs_costs = self.bin_strs_costs()
        
        # minimum cost
        self.min_cost = min(self.bin_strs_costs)

        # depth
        self.depth = 0

        # storage 
        self.cost_values = {}
        self.angles = {}
         
    def bin_strs_costs(self):
        """ returns array which is the cost of each binary string in binary_strings """
        bin_strs_costs = np.zeros(len(self.bin_strs))
        for i, bin_str in enumerate(self.bin_strs):
            bin_strs_costs[i] = self.cost(bin_str)
        return bin_strs_costs

    def cost(self, bin_str):
        """ returns cost of a binary string """
        cost = 0.
        for i, j, data in self.graph.edges.data():
            if bin_str[i] != bin_str[j]:
                cost -= data["weight"]
        return cost

    def create_circuit(self, parameters):
        # initial state
        for i in range(self.n_qubits):
            qml.Hadamard(wires = i)
        
        half = len(parameters) // 2
        for i in range(half):
            # problem hamiltonian
            for j, k, data in self.graph.edges.data():
                qml.CNOT(wires = [j, k])
                qml.RZ(-2.0 * data["weight"] * parameters[i], wires = k)
                qml.CNOT(wires = [j, k])

            # mixer hamiltonian
            for j in range(self.n_qubits):
                qml.RX(-2.0 * parameters[half + i], wires = j)

    def _measure_circuit(self, parameters):
        """ 
        creates circuit defined by circuit() and returns probability 
        of each computational basis state in lexiographic order.
        """
        self.create_circuit(parameters)
        return qml.probs(wires=range(self.n_qubits))

    def measurement_expectation(self, parameters):
        s = time.time()
        probs = self.measure_circuit(parameters) 
        expectation_value = np.sum(self.bin_strs_costs * probs)
        print(time.time() - s, expectation_value)
        return expectation_value

    def plot_eigenstates(self, parameters):
        probs = measure_circuit(parameters)
        fig = plt.figure()
        plt.bar(self.bin_strs, probs)
        plt.show()

    def plot_circuit(self, parameters): 
        fig, ax = qml.draw_mpl(measure_circuit)(parameters)
        plt.show()

    def interpolate(self, parameters):
        gammas, betas = np.split(parameters, 2)
        def interp(angles):
            depth = len(angles)
            tmp = np.zeros(len(angles)+2)
            tmp[1:-1] = angles
            w = np.arange(0,depth+1)
            return w / depth * tmp[:-1] + (depth - w) / depth * tmp[1:]
        return np.concatenate((interp(gammas), interp(betas)))

    def increase_depth(self):
        if self.depth == 0:
            # set initial random parameters. to do : implement cost landscape
            self.angles["d1_init"] = np.random.rand(2) * np.pi
        else:
            self.angles[f"d{self.depth + 1}_init"] = self.interpolate(self.angles[f"d{self.depth}_init"])

        result = optimize.minimize(self.measurement_expectation, self.angles[f"d{self.depth + 1}_init"], method = "NELDER-MEAD")
        self.angles[f"d{self.depth + 1}_final"] = result.x
        self.cost_values[f"d{self.depth + 1}"] = result.fun
        self.depth += 1

def relabel_nodes(graph, relabel_type = 0):
    """ rename node labels from strings to integers """

    if relabel_type == 0:
        # direct relabeling such that "0" -> 0, "1" -> 1 ...
        D = {}
        for i in range(len(graph.nodes())):
            D[str(i)] = i
        return nx.relabel_nodes(graph, D)
    else:
        # alternative relabeling, such that graph((0, 2), (2, 1) becomes equivalent to graph((0, 1), (1, 2))
        return nx.convert_node_labels_to_integers(graph, first_label = 0)

#graph = nx.Graph([(0, 1), (1, 2)])
#graph = nx.read_gml("../data/sample_graphs/w_ba_n14_k4_0.gml")
#nx.set_edge_attributes(graph, values = 1.0, name = 'weight')
#graph = relabel_nodes(graph)
#nx.draw(graph, with_labels = True)
#plt.show()

#qaoa_max_cut = QAOA(n_qubits = 14, graph = graph) 
#for i in tqdm(range(4)):
#    qaoa_max_cut.increase_depth()
#print(qaoa_max_cut.angles)
#print(qaoa_max_cut.cost_values)
