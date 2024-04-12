import json
import random
import sys

from pathlib import Path
from typing import List
import uuid
import warnings

import numpy as np

project_path = Path(__file__).parent.parent.parent
if project_path not in sys.path:
    sys.path.append(str(project_path))

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
from pyzx.heuristics.simplification import FilterFlowFunc

path_to_circuits = project_path / 'circuits\\qasm'

unique_id = uuid.uuid4().hex
path_to_current_run = project_path / 'demos\\heuristic_demos\\benchmarks' / unique_id
path_to_current_run.mkdir(parents=False, exist_ok=False)

path_to_graphs = path_to_current_run / 'graphs'
path_to_graphs.mkdir(parents=False, exist_ok=True)

logging.basicConfig(filename=path_to_current_run / 'log.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('Greedy_Benchmark')
seed = 1332
random.seed(seed)


def load_circuits(path_to_circuits:Path, circuit_names:str|List[str]|None):

    input_data = {"Name": [], "circuit": [], "graph": []}
    output_data = {"gates": [], "t_count": [], "cliffords": [], "cnot": [], "other": [], "hadamard": [], "verticies": [], "edges": [], "time": []}

    for file in path_to_circuits.glob('*.qasm'):
        circuit = zx.Circuit.load(file).to_basic_gates()
        # if circuit.qubits <= 19 and circuit.qubits >= 8 and len(circuit.gates) <= 5000 and len(circuit.gates) >= 100:
        if circuit_names is None:
            circuits = ["barenco_tof_3", "gf2^6_mult", "tof_10", "mod_red_21", "gf2^5_mult"]
        elif isinstance(circuit_names, list):
            circuits = circuit_names
        else:
            circuits = [circuit_names]

        if file.stem in circuits:
            
            try:
                circuit = zx.optimize.basic_optimization(circuit)
            except Exception as e:
                pass
            graph = circuit.to_graph()
            graph = graph.copy()

            input_data["Name"].append(file.stem)
            input_data["circuit"].append(circuit)
            input_data["graph"].append(circuit.to_graph())
            logger.info(f"Loaded {file.stem}")
            logging.info(circuit.stats())

            numbers = re.findall(r'\d+', circuit.stats())

            # Assign the numbers to variables
            output_data["gates"].append(int(numbers[1]))
            output_data["t_count"].append(int(numbers[2]))
            output_data["cliffords"].append(int(numbers[3]))
            output_data["cnot"].append(int(numbers[6]))
            output_data["other"].append(int(numbers[7]))
            output_data["hadamard"].append(int(numbers[8]))
            output_data["verticies"].append(graph.num_vertices())
            output_data["edges"].append(graph.num_edges())
            output_data["time"].append(0)

    return input_data, output_data

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

def run_algorithm(algorithm, input_data, algorithm_name, pre_tr:bool = True):
    
    output_data = {"gates": [], "t_count": [], "cliffords": [], "cnot": [], "other": [], "hadamard": [], "verticies": [], "edges": [], "time": []}

    for name, graph in zip(input_data["Name"], input_data["graph"]):
        graph_simplified = graph.clone()
        if pre_tr:
            graph_simplified = zx.simplify.teleport_reduce(graph_simplified)
            graph_simplified.track_phases = False
            
        logging.info(f"Running {algorithm} on {name}")

        algorithm_failed = False

        start = time.perf_counter()
        try:
            algorithm(graph_simplified)
            with open(f"{path_to_graphs}/{name}_{algorithm_name}.json", 'w') as f:
                f.write(graph_simplified.to_json())
        except Exception:
            algorithm_failed = True
            warnings.warn(f"Failed to run {algorithm_name} on {name}")
            logging.warning(f"Failed to run {algorithm_name} on {name}")

        end = time.perf_counter() - start

        logging.info(f"Finished execution in {end} seconds")

        if not algorithm_failed:
            try:
                qc = zx.extract_circuit(graph_simplified)
                # qc = basic_optimization_min_cnots(qc.to_basic_gates())
                qc = zx.optimize.basic_optimization(qc.to_basic_gates())

                stats = qc.stats()
                logging.info(f"Stats for {algorithm} on {name}:")
                logging.info(stats+"\n")
                # Extract the numbers
                numbers = re.findall(r'\d+', stats)

                output_data["gates"].append(int(numbers[1]))
                output_data["t_count"].append(int(numbers[2]))
                output_data["cliffords"].append(int(numbers[3]))
                output_data["cnot"].append(int(numbers[6]))
                output_data["other"].append(int(numbers[7]))
                output_data["hadamard"].append(int(numbers[8]))
            except Exception:

                warnings.warn(f"Failed to extract circuit from {name} after {algorithm_name} simplification")
                logging.warning(f"Failed to extract circuit from {name} after {algorithm_name} simplification")
                output_data["gates"].append(np.nan)
                output_data["t_count"].append(np.nan)
                output_data["cliffords"].append(np.nan)
                output_data["cnot"].append(np.nan)
                output_data["other"].append(np.nan)
                output_data["hadamard"].append(np.nan)

            output_data["verticies"].append(graph_simplified.num_vertices())
            output_data["edges"].append(graph_simplified.num_edges())
            output_data["time"].append(int(end))
        else:
            output_data["gates"].append(np.nan)
            output_data["t_count"].append(np.nan)
            output_data["cliffords"].append(np.nan)
            output_data["cnot"].append(np.nan)
            output_data["other"].append(np.nan)
            output_data["hadamard"].append(np.nan)
            output_data["verticies"].append(np.nan)
            output_data["edges"].append(np.nan)
            output_data["time"].append(int(end))

    return output_data


dataframes = []

# circuits = ["barenco_tof_3", "gf2^6_mult", "tof_10", "mod_red_21", "gf2^5_mult"]
circuits = ["barenco_tof_3"]

# Load the circuits and get original data
input_data, output_data_or = load_circuits(path_to_circuits, circuits)

# Define the column names
columns = input_data["Name"]

# Define the row labels
rows = list(output_data_or.keys())

data = list(output_data_or.values())

# Add original data to the dataframe
dataframes.append(pd.DataFrame(data, columns=columns, index=rows))




lookahead = list(range(0, 2))
threshold = 1

# Define the algorithms
algorithms = {
    "OR": None,
    "TR": zx.simplify.teleport_reduce,
    "FR": zx.simplify.full_reduce,
    **{f"G{la}": partial(zx.simplify.greedy_simp, lookahead=la, threshold=threshold, use_yz_phase_gadgets=False, use_xz_phase_gadgets=False) for la in lookahead},
    **{f"GN{la}": partial(zx.simplify.greedy_simp_neighbors, lookahead=la, threshold=threshold, use_yz_phase_gadgets=False, use_xz_phase_gadgets=False) for la in lookahead},
    **{f"GN_PG{la}": partial(zx.simplify.greedy_simp_neighbors, lookahead=la, threshold=threshold, use_yz_phase_gadgets=True, use_xz_phase_gadgets=True) for la in lookahead},
    **{f"G_CFlow{la}": partial(zx.simplify.greedy_simp, lookahead=la, threshold=threshold, use_yz_phase_gadgets=False, use_xz_phase_gadgets=False, flow_function=FilterFlowFunc.C_FLOW_PRESERVING) for la in lookahead},
    **{f"GN_CFlow{la}": partial(zx.simplify.greedy_simp_neighbors, lookahead=la, threshold=threshold, use_yz_phase_gadgets=False, use_xz_phase_gadgets=False, flow_function=FilterFlowFunc.C_FLOW_PRESERVING) for la in lookahead},
    **{f"GN_PG_CFlow{la}": partial(zx.simplify.greedy_simp_neighbors, lookahead=la, threshold=threshold, use_yz_phase_gadgets=True, use_xz_phase_gadgets=True, flow_function=FilterFlowFunc.C_FLOW_PRESERVING) for la in lookahead}
}

#FIXME: There seems to be an error. Some neighbor unfusion algorithms are way faster than other without apperent reason. eg. GN1 and GN_PG1 for gf2^5_mult: ~70s vs ~1500s
for algorithm_name, algorithm in algorithms.items():
    if algorithm is None:
        continue
    if algorithm_name == "TR" or algorithm_name == "FR":
        pre_tr = False
    else:
        pre_tr = True

    # Run the algorithm and get the output data
    output_data = run_algorithm(algorithm, input_data, algorithm_name, pre_tr=pre_tr)
    # Add the output data to the dataframe
    dataframes.append(pd.DataFrame(list(output_data.values()), columns=columns, index=rows))

# Concatenate the dataframes and save them to a csv file
df = pd.concat(dataframes, axis=0, keys=algorithms.keys())
df.to_csv(path_to_current_run / 'benchmark_greedy.csv')

# Save the metadata
run_meta_data = {
    "unique_id": unique_id,
    "seed": seed,
    "lookahead": lookahead,
    "threshold": threshold,
    "algorithms": list(algorithms.keys()),
    "columns": columns,
    "rows": rows,
    "path_to_circuits": str(path_to_circuits),
    "date": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(f"{path_to_current_run}/metadata.json", 'w') as f:
    f.write(json.dumps(run_meta_data, indent=4))