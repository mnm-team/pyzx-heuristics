import sys; sys.path.append('../..')
from pathlib import Path

sys.path.append("c:\\Users\\wsajk\\Documents\\Arbeit\\MUNIQC-Atoms\\pyzx-heuristics")

import time

import pandas as pd
import re
import logging

from overrides import override
from functools import partial

import pyzx as zx

from pyzx.circuit import Circuit
from pyzx.circuit.gates import CZ, Gate, ZPhase

from pyzx.optimize import Optimizer, toggle_element



logger = logging.getLogger()
logger.setLevel(logging.INFO)
path_to_circuits = 'c:\\Users\\wsajk\\Documents\\Arbeit\\MUNIQC-Atoms\\pyzx-heuristics\\circuits\\qasm\\'
input_data = {"Name": [], "circuit": [], "graph": []}
dataframes = []

gates = []
t_count = []
cliffords = []
cnot = []
other = []
hadamard = []
times = []

for file in Path(path_to_circuits).glob('*.qasm'):
    circuit = zx.Circuit.load(file).to_basic_gates()
    # if circuit.qubits <= 19 and circuit.qubits >= 8 and len(circuit.gates) <= 5000 and len(circuit.gates) >= 100:

    if file.stem == "gf2^5_mult" or file.stem == "gf2^6_mult" or file.stem == "barenco_tof_3" or file.stem == "mod_red_21":
        try:
            circuit = zx.optimize.basic_optimization(circuit)
        except Exception as e:
            pass
        graph = circuit.to_graph()
        graph = graph.copy()

        input_data["Name"].append(file.stem)
        input_data["circuit"].append(circuit)
        input_data["graph"].append(circuit.to_graph())
        logging.info(f"Loaded {file.stem}")
        logging.info(circuit.stats())

        numbers = re.findall(r'\d+', circuit.stats())

        # Assign the numbers to variables
        gates.append(int(numbers[1]))
        t_count.append(int(numbers[2]))
        cliffords.append(int(numbers[3]))
        cnot.append(int(numbers[6]))
        other.append(int(numbers[7]))
        hadamard.append(int(numbers[8]))
        times.append(0)

# Define the column names
columns = input_data["Name"]

# Define the row labels
rows = ["Gates", "T-Count", "Cliffords", "CNOTS", "Other 2 Qubit Gates", "Hadamard", "Time"]

lookahead = list(range(0, 2))

#Define the algorithm
algorithm = ["OR", "TR", "FR"]
algorithm = algorithm + [f"G{la}" for la in lookahead]
algorithm = algorithm + [f"GN{la}" for la in lookahead]

data = [gates, t_count, cliffords, cnot, other, hadamard, times]
dataframes.append(pd.DataFrame(data, columns=columns, index=rows))


def basic_optimization_min_cnots(circuit: Circuit, do_swaps:bool=True, quiet:bool=True) -> Circuit:
    """Optimizes the circuit using a strategy that involves delayed placement of gates
    so that more matches for gate cancellations are found. Specifically tries to minimize
    the number of Hadamard gates to improve the effectiveness 
    of phase-polynomial optimization techniques.

    Args:
        circuit: Circuit to be optimized.
        do_swaps: When set uses some rules transforming CNOT gates into SWAP gates. Generally leads to better results, but messes up architecture-aware placement of 2-qubit gates.
        quiet: Whether to print some progress indicators.
    """
    if not isinstance(circuit, Circuit):
        raise TypeError("Input must be a Circuit")
    o = Optimizer_no_new_cnots(circuit)
    return o.parse_circuit(do_swaps=do_swaps,quiet=quiet)

class Optimizer_no_new_cnots(Optimizer):
    """This class is a subclass of Optimizer that does not allow the creation of new CNOT gates."""

    def __init__(self, circuit: Circuit) -> None:
        super().__init__(circuit)

    @override
    def parse_gate(self, g: Gate) -> None:
        """The main function of the optimization. It records whether a gate needs to be placed at the specified location
        'right now', or whether we can postpone the placement until hopefully it is cancelled against some future gate.
        Only supports ZPhase, HAD, CNOT and CZ gates. """
        g = g.copy()
        # If we have some SWAPs recorded we need to change the target/control of the gate accordingly
        g.target = next(i for i in self.permutation if self.permutation[i] == g.target)
        t = g.target
        if g.name in ('CZ', 'CNOT'):
            g.control = next(i for i in self.permutation if self.permutation[i] == g.control)

        if g.name == 'HAD':
            # If we have recorded a NOT or Z gate at the target location, we push it trough the Hadamard and change the type
            if t in self.nots and t not in self.zs:
                self.nots.remove(t)
                self.zs.append(t)
            elif t in self.zs and t not in self.nots:
                self.zs.remove(t)
                self.nots.append(t)
            # See whether we have a HAD-S-HAD situation
            # And turn it into a S*-HAD-S* situation
            if len(self.gates[t])>1 and self.gates[t][-2].name == 'HAD' and isinstance(self.gates[t][-1], ZPhase):
                    g2 = self.gates[t][-1]
                    if g2.phase.denominator == 2:
                        h = self.gates[t][-2]
                        zp = ZPhase(t, (-g2.phase)%2)
                        zp.index = self.gcount
                        self.gcount += 1
                        g2.phase = zp.phase
                        if g2.name == 'S' and g2.phase.numerator > 1:
                            g2.adjoint = True
                        self.gates[t].insert(-2,zp)
                        return
            toggle_element(self.hadamards, t)
        elif g.name == 'NOT':
            toggle_element(self.nots, t)
        elif isinstance(g, ZPhase):
            if t in self.zs: #Consume a Z gate into the phase gate
                g.phase = (g.phase+1)%2
                self.zs.remove(t)
            if g.phase == 0: return
            if t in self.nots: # Push the phase gate trough a NOT
                g.phase = (-g.phase)%2
            if g.phase == 1: # If the resulting phase is a pi, then we record it as a Z gate
                toggle_element(self.zs, t)
                return
            if g.name == 'S':                           # We might have changed the phase, and therefore
                g.adjoint = g.phase.numerator != 1      # Need to adjust whether the adjoint is true
            if t in self.hadamards: # We can't push a phase gate trough a HAD, so we actually place the HAD down
                self.add_hadamard(t)
            if self.availty[t] == 1 and any(isinstance(g2, ZPhase) for g2 in self.available[t]): # There is an available phase gate
                i = next(i for i,g2 in enumerate(self.available[t]) if isinstance(g2, ZPhase))   # That we can fuse with the new one
                g2 = self.available[t].pop(i)
                self.gates[t].remove(g2)
                phase = (g.phase+g2.phase)%2
                if phase == 1:
                    toggle_element(self.zs, t)
                    return
                if phase != 0:
                    p = ZPhase(t, phase)
                    self.add_gate(t,p)
            else:
                if self.availty[t] == 2: # If previous gate was of X-type
                    self.availty[t] = 1  # We reset the available gates on this qubit
                    self.available[t] = list()
                g = ZPhase(t, g.phase)  # Avoid subclasses of ZPhase with inconsistent phase
                self.add_gate(t, g)
        elif g.name == 'CZ':
            t1, t2 = g.control, g.target
            if t1 > t2: # Normalise so that always g.target<g.control (since CZs are symmetric anyway)
                g.target = t1
                g.control = t2
            # Push NOT gates trough the CZ
            if t1 in self.nots: 
                toggle_element(self.zs, t2)
            if t2 in self.nots:
                toggle_element(self.zs, t1)
            # If there are HADs on both targets, we cannot commute the CZ trough and we place the HADs
            if t1 in self.hadamards and t2 in self.hadamards:
                self.add_hadamard(t1)
                self.add_hadamard(t2)
            if t1 not in self.hadamards and t2 not in self.hadamards:
                self.add_cz(g)
            # Exactly one of t1 and t2 has a hadamard
            # Do not allow the creation of new CNOT gates
            elif t1 in self.hadamards:
                self.add_hadamard(t1)
                self.add_cz(g)
            else:
                self.add_hadamard(t2)
                self.add_cz(g)
            
        elif g.name == 'CNOT':
            c, t = g.control, g.target
            # Commute NOTs and Zs trough the CNOT
            if c in self.nots:
                toggle_element(self.nots, t)
            if t in self.zs:
                toggle_element(self.zs, c)
            # If HADs are on both qubits, we commute the CNOT trough by switching target and control
            if c in self.hadamards and t in self.hadamards:
                g.control = t
                g.target = c
                self.add_cnot(g)
            elif c not in self.hadamards and t not in self.hadamards:
                self.add_cnot(g)
            # If there is a HAD on the target, the CNOT commutes trough to become a CZ
            elif t in self.hadamards:
                cz = CZ(c if c<t else t, c if c>t else t)
                self.add_cz(cz)
            else: # Only the control has a hadamard gate in front of it
                self.add_hadamard(c)
                self.add_cnot(g)
        
        else:
            raise TypeError("Unknown gate {}".format(str(g)))

def run_algorithm(algorithm, input_data, dataframes, algorithm_name, pre_tr:bool = True):
    
    gates = []
    t_count = []
    cliffords = []
    cnot = []
    other = []
    hadamard = []
    times = []

    for name, circuit, graph in zip(input_data["Name"], input_data["circuit"], input_data["graph"]):
        graph_simplified = graph.copy()
        if pre_tr:
            graph_simplified = zx.simplify.teleport_reduce(graph_simplified)
            graph_simplified.track_phases = False
            
        logging.info(f"Running {algorithm} on {name}")

        start = time.perf_counter()
        algorithm(graph_simplified)
        end = time.perf_counter() - start

        with open(f"graphs/{name}_{algorithm_name}.json", 'w') as f:
            f.write(graph_simplified.to_json())

        logging.info(f"Finished execution in {end} seconds")

        qc = zx.extract_circuit(graph_simplified)
        try:
            # qc = basic_optimization_min_cnots(qc.to_basic_gates())
            qc = zx.optimize.basic_optimization(qc.to_basic_gates())
        except Exception as e:
            raise e

        stats = qc.stats()
        logging.info(stats)
        # Extract the numbers
        numbers = re.findall(r'\d+', stats)

        # Assign the numbers to variables
        gates.append(int(numbers[1]))
        t_count.append(int(numbers[2]))
        cliffords.append(int(numbers[3]))
        cnot.append(int(numbers[6]))
        other.append(int(numbers[7]))
        hadamard.append(int(numbers[8]))
        times.append(int(end))

    data = [gates, t_count, cliffords, cnot, other, hadamard, times]
    dataframes.append(pd.DataFrame(data, columns=columns, index=rows))


run_algorithm(zx.simplify.teleport_reduce, input_data, dataframes, algorithm_name="TR", pre_tr=False)
run_algorithm(zx.simplify.full_reduce, input_data, dataframes, algorithm_name="FR", pre_tr=False)

for la in lookahead:
    partial_greedy = partial(zx.simplify.greedy_simp, lookahead=la)
    partial_greedy_neighbors = partial(zx.simplify.greedy_simp_neighbors, lookahead=la)
    run_algorithm(partial_greedy, input_data, dataframes, algorithm_name=f"G{la}", pre_tr=True)
    run_algorithm(partial_greedy_neighbors, input_data, dataframes, algorithm_name=f"GN{la}", pre_tr=True)


df = pd.concat(dataframes, axis=0, keys=algorithm)
df.to_csv('benchmark_greedy.csv')