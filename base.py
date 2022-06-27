import networkx as nx
import pennylane as qml
import itertools
import numpy as np
from pennylane import qaoa
from scipy import optimize
from matplotlib import pyplot as plt

# create quantum device
dev = qml.device("qiskit.aer", wires = 14, shots = 1000)

def circuit(parameters, n_qubits, graph):
    # initial state
    for i in range(n_qubits):
        qml.Hadamard(wires = i)
    
    half = len(parameters) // 2
    for i in range(half):
        # problem hamiltonian
        for j, k, data in graph.edges.data():
            qml.CNOT(wires = [j, k])
            qml.RZ(-2.0 * data["weight"] * parameters[i], wires = k)
            qml.CNOT(wires = [j, k])

        # mixer hamiltonian
        for j in range(n_qubits):
            qml.RX(-2.0 * parameters[half + i], wires = j)

@qml.qnode(dev)
def measure_circuit(parameters, n_qubits, graph):
    """ 
    creates circuit defined by circuit() and returns probability 
    of each computational basis state in lexiographic order.
    """
    circuit(parameters, n_qubits, graph)
    return qml.probs(wires=range(n_qubits))

def expectation_value(parameters, n_qubits, depth, graph, binary_strings_costs):
    probs = measure_circuit(parameters, n_qubits, graph) 
    expectation_value = 0.
    for i in range(len(probs)):
        expectation_value += binary_strings_costs[i] * probs[i]

    return expectation_value

def cost(binary_string, graph):
    """ 
    returns cost of a binary string
    """
    cost = 0.
    for i, j, data in graph.edges.data():
        if binary_string[i] != binary_string[j]:
            cost -= data["weight"]
    return cost

def get_binary_strings(n):
    """
    returns all possible configurations of n bits in lexiographic order
    """
    return np.array([''.join(i) for i in itertools.product('01', repeat = n)])

def get_binary_strings_costs(binary_strings, graph):
    """
    returns array which is the cost of each binary string in binary_strings
    """
    binary_strings_costs = np.zeros(len(binary_strings))
    for i, binary_string in enumerate(binary_strings):
        binary_strings_costs[i] = cost(binary_string, graph)
    return binary_strings_costs

def min_cost(graph):
    """ 
    calculate lowest possible cost of graph. 
    returns: lowest eigenvalue (float) and eigenvector (binary string)  
    """
    binary_strings = get_binary_strings(len(graph.nodes()), graph)
    binary_strings_costs = get_binary_strings_costs(binary_strings)
    return np.min(binary_strings_costs), binary_strings[np.argmin(binary_strings_costs)]

def plot_eigenstates(parameters, n_qubits, graph):
    probs = measure_circuit(parameters, n_qubits, graph)
    binary_strings = get_binary_strings(n_qubits)
    plt.bar(binary_strings, probs)
    plt.show()

def draw_circuit(n_qubits, graph):
    fig, _ = qml.draw_mpl(measure_circuit)(np.random.rand(n_qubits), n_qubits, graph)
    plt.show()

def interpolate(angles):
    depth = len(angles)
    tmp = np.zeros(len(angles)+2)
    tmp[1:-1] = angles
    w = np.arange(0,depth+1)
    return w / depth * tmp[:-1] + (depth - w) / depth * tmp[1:]

def run(depth, n_qubits, graph):
    # get cost of all binary strings
    binary_strings_costs = get_binary_strings_costs(get_binary_strings(n_qubits), graph)

    # set initial random parameters. to do : implement cost landscape
    parameters_initial = np.random.rand(2) * np.pi

    # initialize return array 
    cost_values = np.zeros(depth)

    # minimize with depth 1
    result = optimize.minimize(expectation_value, parameters_initial, args = (n_qubits, 1, graph, binary_strings_costs), method = "NELDER-MEAD")
    cost_values[0] = result.fun
    gamma_final, beta_final = np.split(result.x, 2)

    for i in range(1, depth):
        print(cost_values)
        # linear interpolation from parameters_final (depth i) to parameters_initial (depth i+1)
        gamma_initial, beta_initial = interpolate(gamma_final), interpolate(beta_final)
        parameters_initial = np.concatenate((gamma_initial, beta_initial))

        # minimize circuit expectation value
        result = optimize.minimize(expectation_value, parameters_initial, args = (n_qubits, 1, graph, binary_strings_costs), method = "NELDER-MEAD")
        cost_values[i] = result.fun
        gamma_final, beta_final = np.split(result.x, 2)

    return cost_values, np.concatenate((gamma_final, beta_final))

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
graph = nx.read_gml("../data/sample_graphs/w_ba_n14_k4_0.gml")
#nx.set_edge_attributes(graph, values = 1.0, name = 'weight')
graph = relabel_nodes(graph)
nx.draw(graph, with_labels = True)
plt.show()

cost_values, angles = run(5, 14, graph)
print(cost_values, angles)
#plot_eigenstates(angles, 14, graph)
