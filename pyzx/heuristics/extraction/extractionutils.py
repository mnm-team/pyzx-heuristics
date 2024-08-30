from pyzx.graph.base import ET, VT, BaseGraph, VertexType
from pyzx.circuit import Circuit, Gate, CNOT, ZPhase
from pyzx.linalg import CNOTMaker
from qiskit import QuantumCircuit
from typing import Dict, Set, List, Optional
import itertools
from pyzx.utils import FractionLike, Fraction
import math

def get_frontier_gadgets(g: BaseGraph, frontier: Dict[int, VT]):
    """Given a graph and a frontier set, returns all phase gadget neighbors of the frontier as a set of tuples (root,top) 
    where root is the (phaseless) root spider, and top the 1-ary spider with phase connected to root"""
    res = set()
    for v in frontier.values():
        for n in g.neighbors(v):
            top = None
            for potential_top in g.neighbors(n):
                if g.vertex_degree(potential_top) == 1 and potential_top not in g.inputs() and potential_top not in g.outputs():
                    res.add((n,potential_top))

    return res

def get_neighbors_of_frontier(g: BaseGraph[VT, ET], frontier_values: List[VT]) -> Set[VT]:
    """Returns the set of neighbors of the frontier."""
    neighbor_set = set()

    for vertex in frontier_values:
        non_start_neighbors = [neighbor for neighbor in g.neighbors(vertex) if g.type(neighbor) == VertexType.Z]
        neighbor_set.update(non_start_neighbors)
    return neighbor_set

def get_exclusive_frontier_gadget_dict(g: BaseGraph[VT,ET], frontier: Dict[int, VT], limit_n: int):
    """returns a dictionary of all gadgets adjacent to only frontier vertices grouped by their degree 
    (i.e. to how many frontier vertices the gadget is connected to)"""
    frontier_gadgets = get_exclusive_frontier_gadgets(g, frontier)
    gadget_dict = dict()
    for root, top in frontier_gadgets:
        neighbors_in_frontier = set(g.neighbors(root)).difference(set([top]))
        if len(neighbors_in_frontier) <= limit_n and len(neighbors_in_frontier) > 1:
            gadget_dict[tuple(neighbors_in_frontier)] = (root,top)

    return gadget_dict

def get_exclusive_frontier_gadgets(g: BaseGraph, frontier: Dict[int, VT]):
    """Given a graph and a frontier set, returns all phase gadget neighbors of the frontier as a set of tuples (root,top) 
    where root is the (phaseless) root spider, and top the 1-ary spider with phase connected to root"""
    res = set()
    for v in frontier.values():
        for n in g.neighbors(v):
            difference_set = set(g.neighbors(n)).union(set(frontier.values())).difference(set(frontier.values())).difference(set(g.outputs()))
            if len(difference_set) == 1: 
                top = difference_set.pop()
                if len(g.neighbors(top)) == 1:
                    #if there are no other neighbors than frontier neighbors+ gadget top
                    res.add((n,top))

    return res

def get_remaining_combinations(combination):
    """returns all combinations necessary to complete a given phase gadget towards cnp structure"""
    remaining_combinations = []
    for degree in range(len(combination)-1,1,-1):
        remaining_combinations += list(itertools.combinations(combination, degree))

    return remaining_combinations

class CNP(Gate):
    name = 'CNP'
    qasm_name = 'cnp'
    print_phase = True
    def __init__(self, phase: FractionLike, qubits: List[int]) -> None:
        self.qubits = qubits
        self.phase = phase

    def to_basic_gates(self):
        gates = []
        odd_phase = self.phase/(2**(len(self.qubits)-1))
        even_phase = -odd_phase
        for degree in range(2,len(self.qubits)+1):
            combinations = list(itertools.combinations(self.qubits, degree))
            for combination in combinations:
                for idx in range(0,len(combination)-1):
                    gates.append(CNOT(combination[idx],combination[idx+1]))
                gates.append(ZPhase(combination[-1], odd_phase if degree % 2 == 1 else even_phase))
                for idx in range(len(combination)-2,-1,-1):
                    gates.append(CNOT(combination[idx],combination[idx+1]))
        for qubit in self.qubits:
            gates.append(ZPhase(qubit, odd_phase))
        return gates                

    def to_graph(self, g, q_mapper, c_mapper):
        for gate in self.to_basic_gates():
            gate.to_graph(g, q_mapper, c_mapper)
    
    def to_qasm(self) -> str:
        phase = "({}*pi)".format(float(self.phase))
        if len(self.qubits) == 2:
            return "cp"+phase+" q["+str(self.qubits[0])+"], q["+str(self.qubits[1])+"];"
        else:
            name = "cnp"+str(len(self.qubits))
            control_string = "".join(["q["+str(control)+"], " for control in self.qubits[:-1]])
            return name+phase+" "+control_string+" q["+str(self.qubits[-1])+"];"
        

def convert_to_qiskit(c: Circuit):
    qc = QuantumCircuit(c.qubits)
    for gate in c.gates:
        if gate.name == "HAD":
            qc.h(gate.target)
        elif gate.name == "ZPhase":
            qc.rz(float(gate.phase)*math.pi, gate.target)
        elif gate.name == "CZ":
            qc.cz(gate.control, gate.target)
        elif gate.name == "CNOT":
            qc.cx(gate.control, gate.target)
        elif gate.name == "CNP":
            qc.mcp(float(gate.phase)*math.pi, gate.qubits[:-1], gate.qubits[-1])
        elif gate.name == "SWAP":
            qc.swap(gate.control, gate.target)
        elif gate.name == "U3":
            qc.u3(float(gate.phases[0])*math.pi, float(gate.phases[1])*math.pi, float(gate.phases[2])*math.pi, gate.target)
        else:
            print("unknown gate",gate)
    return qc

def to_cnots(matrix, optimize: bool = False, use_log_blocksize: bool = False) -> List[CNOT]:
    """
    Copy of pyzx function. Since we here sometimes input a non reversible matrix, we want to return an empty list instead of raising an exception
    Returns a list of CNOTs that implements the matrix as a reversible circuit of qubits."""
    cn: Optional[CNOTMaker]
    if not optimize:
        cn = CNOTMaker()
        blocksize = 5
        if use_log_blocksize:
            blocksize = int(math.log2(matrix.rows()))
        matrix.copy().gauss(full_reduce=True,x=cn, blocksize=blocksize)
    else:
        best = 1000000
        best_cn = None
        for size in range(1,matrix.rows()):
            cn = CNOTMaker()
            matrix.copy().gauss(full_reduce=True,x=cn, blocksize=size)
            if len(cn.cnots) < best:
                best = len(cn.cnots)
                best_cn = cn
        cn = best_cn
    if not cn:
        return []
    return cn.cnots

def convert_mapped_circuit(qasm_circuit, num_qubits, initial_mapping=None):
    c = Circuit(num_qubits)
    if initial_mapping:
        qubit_list = initial_mapping
    else:
        qubit_list = [-1 for _ in range(0,num_qubits)]
    unoccupied = 0
    for gate in qasm_circuit.split(';\n')[3:]:
        components = gate.split(' ')
        gatename = components[0]
        qubits = [int(qubit.translate({ord(c): None for c in 'q[],'})) for qubit in components[1:]]
        for qubit in qubits:
            if not gatename == 'move' and not qubit in qubit_list:
                qubit_list[unoccupied] = qubit
                unoccupied += 1

        if gatename == 'h':
            c.add_gate('HAD',qubit_list.index(qubits[0]))
        elif gatename == 'cz':
            c.add_gate('CZ',qubit_list.index(qubits[0]), qubit_list.index(qubits[1]))
        elif gatename == 'swap':
            c.add_gate('SWAP',qubit_list.index(qubits[0]), qubit_list.index(qubits[1]))
        elif gatename == 'move':
            qubit_list[qubit_list.index(qubits[0])] = qubits[1]
        elif 'rz' in gatename:
            angle = float(gatename.split('(')[1][:-1])
            c.add_gate('ZPhase', qubit_list.index(qubits[0]), round(angle/math.pi,14))

    return c