import networkx as nx
import pennylane as qml
from pennylane import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

# create quantum device
dev = qml.device('default.qubit', wires = 3, shots = 1000)

@qml.qnode(dev)
def circuit(parameters, n_qubits, depth, graph):

    # initial state
    for i in range(n_qubits):
        qml.Hadamard(wires = i)

    gamma, beta = np.split(parameters, 2)
    for i in range(depth):

        # problem hamiltonian
        for j, k, data in graph.edges.data():
            qml.CNOT(wires = [j, k])
            qml.RZ(-2.0 * data["weight"] * gamma[i], wires = k)
            qml.CNOT(wires = [j, k])

        # mixer hamiltonian
        for j in range(n_qubits):
            qml.RX(-2.0 * beta[i], wires = j)

    return qml.expval(0.5 * qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliZ(1) @ qml.PauliZ(2) + 0.5 * qml.PauliZ(0) @ qml.PauliZ(2))

def interp(angles):
    depth = len(angles)
    tmp = np.zeros(len(angles)+2)
    tmp[1:-1] = angles
    w = np.arange(0,depth+1)
    return w / depth * tmp[:-1] + (depth - w) / depth * tmp[1:]

def run(depth, n_qubits, graph):

    cost_values = np.zeros(depth)

    # set random initial angles in interval [0, np.pi)
    parameters_initial = np.random.rand(2) * np.pi

    # minimize circuit expectation value
    result = optimize.minimize(circuit, parameters_initial, args = (n_qubits, 1, graph), method = "COBYLA")
    cost_values[0] = result.fun
    gamma_final, beta_final = np.split(result.x, 2)

    for i in range(1, depth):

        # linear interpolation from parameters_final (depth i) to parameters_initial (depth i+1)
        gamma_initial, beta_initial = interp(gamma_final), interp(beta_final)
        parameters_initial = np.concatenate((gamma_initial, beta_initial))

        # minimize circuit expectation value
        result = optimize.minimize(circuit, parameters_initial, args = (n_qubits, i + 1, graph), method = "COBYLA")
        cost_values[i] = result.fun
        gamma_final, beta_final = np.split(result.x, 2)

    return cost_values

graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
nx.set_edge_attributes(graph, values = 1.0, name = 'weight')

test = run(7, 3, graph)
