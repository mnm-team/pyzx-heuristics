from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.circuit import Gate, ZPhase, CZ, Circuit, HAD
from pyzx.routing import Architecture
from pyzx.utils import EdgeType
from pyzx.heuristics.tools import insert_identity

from typing import Dict
import copy
from fractions import Fraction

from .rerouting import calculate_addition_options
from .extractionutils import *


class ExtractionOption:
    logical_circuit: Circuit
    g: BaseGraph[VT,ET]
    frontier: Dict[int,VT]
    architecture: Architecture
    resolved_gadget: bool

    def __init__(self, graph: BaseGraph[VT,ET], frontier: Dict[int,VT], architecture: Architecture, resolved_gadget: bool=False) -> None:
        self.logical_circuit = Circuit(graph.num_inputs())
        self.g = graph
        self.frontier = frontier
        self.architecture = architecture
        self.resolved_gadget = resolved_gadget

    def append_gate(self, gate: Gate):
        # print("append gate",gate)
        self.logical_circuit.add_gate(gate)
    
    def get_logical_circuit(self):
        return self.logical_circuit
    
    def collect_rz_options(self):
        result_option = self.copy()
        for qubit, vertex in result_option.frontier.items():
            if vertex in result_option.g.outputs():
                continue
            phase = result_option.g.phases()[vertex]
            if phase:
                result_option.append_gate(ZPhase(phase=phase, target=qubit))
                result_option.g.set_phase(vertex,0)
        
        return result_option
    
    def collect_cz_options(self):
        result_option = self.copy()
        for qubit, v in result_option.frontier.items():
            for w in set(result_option.g.neighbors(v)).intersection(set(result_option.frontier.values())):
                result_option.append_gate(CZ(qubit, list(result_option.frontier.values()).index(w)))
                result_option.g.remove_edge(result_option.g.edge(v,w))

        return result_option
    
    def collect_had_options(self):
        result_option = self.copy()
        for qubit, v in result_option.frontier.items():
            neighbors = list(result_option.g.neighbors(v))
            if len(neighbors) != 2 or v in result_option.g.outputs() or result_option.g.phase(v) != 0:
                continue
            boundary, z_neighbor = (neighbors[0], neighbors[1]) if neighbors[0] in self.g.inputs() else (neighbors[1], neighbors[0])
            if result_option.g.edge_type(result_option.g.edge(v,z_neighbor)) == EdgeType.HADAMARD:
                result_option.append_gate(HAD(qubit))
            result_option.g.remove_vertex(v)
            result_option.g.add_edge(result_option.g.edge(boundary, z_neighbor), EdgeType.SIMPLE)
            result_option.frontier[list(result_option.frontier.values()).index(v)] = z_neighbor
        
        return result_option
    
    def collect_gadget_options(self):
        result_list = [self.copy()] #do nothing
        gadgets = get_frontier_gadgets(self.g, self.frontier)
        # print("frontier gadgets",gadgets)
        for root, top in gadgets:
            for frontier_neighbor in set(self.g.neighbors(root)).difference(set([top])):
                if frontier_neighbor in self.frontier.values():
                    frontier_qubit = list(self.frontier.values()).index(frontier_neighbor)
                else:
                    continue
                
                current_option = self.copy()
                current_option.resolved_gadget = True
                current_option.append_gate(HAD(frontier_qubit))
                # print("pivot on",root,frontier_neighbor)

                output_neighbor = None
                for neighbor in current_option.g.neighbors(frontier_neighbor):
                    if neighbor in current_option.g.outputs():
                        output_neighbor = neighbor
                        break
                if output_neighbor:
                    #necessary because then frontier_neighbor has two boundary vertices: one input and one output
                    insert_identity(current_option.g,frontier_neighbor,output_neighbor)

                #calculate pivot graph actions
                b = set(current_option.g.neighbors(root)).intersection(current_option.g.neighbors(frontier_neighbor))
                a = set(current_option.g.neighbors(root)).difference(b).difference([frontier_neighbor])
                c = set(current_option.g.neighbors(frontier_neighbor)).difference(b).difference([root])
                input_neighbor = set(current_option.g.inputs()).intersection(current_option.g.neighbors(frontier_neighbor)).pop()
                c.discard(input_neighbor)

                current_option.g.remove_vertex(root)

                #update neighbor set connections
                for v1 in b:
                    for v2 in a.union(c):
                        if current_option.g.connected(v1,v2):
                            current_option.g.remove_edge(current_option.g.edge(v1, v2))
                        else:
                            current_option.g.add_edge(current_option.g.edge(v1, v2), EdgeType.HADAMARD)
                for v1 in a:
                    for v2 in c:
                        if current_option.g.connected(v1,v2):
                            current_option.g.remove_edge(current_option.g.edge(v1, v2))
                        else:
                            current_option.g.add_edge(current_option.g.edge(v1, v2), EdgeType.HADAMARD)
                
                #update frontier neighbor (aka. boundary pivot vertex) connections
                for v1 in c:
                    current_option.g.remove_edge(current_option.g.edge(v1,frontier_neighbor))
                for v1 in a:
                    current_option.g.add_edge(current_option.g.edge(v1,frontier_neighbor), EdgeType.HADAMARD)
                
                #update phases
                for v1 in b:
                    current_option.g.add_to_phase(v1,Fraction(1))

                result_list.append(current_option)
        
        return result_list
    
    def collect_cnot_options(self):
        result_options = [self.copy()] #do nothing
        cnot_addition_options = calculate_addition_options(self.g, self.frontier, self.architecture)
        # print("cnot_addition_options",cnot_addition_options)
        
        for cnot_addition in cnot_addition_options:
            current_option = self.copy()
            for cnot in cnot_addition:
                current_option.append_gate(cnot)
                target_vertex = current_option.frontier[cnot.control]
                control_vertex = current_option.frontier[cnot.target]
                for neighbor in set(current_option.g.neighbors(control_vertex)).difference(set(current_option.g.inputs())):
                    if neighbor in current_option.g.neighbors(target_vertex):
                        current_option.g.remove_edge(current_option.g.edge(target_vertex, neighbor))
                    else:
                        current_option.g.add_edge(current_option.g.edge(target_vertex, neighbor), EdgeType.HADAMARD)

            result_options.append(current_option)

        return result_options
    
    def collect_cnp_options(self, limit_n: int = 5):
        """collects cnps up to length limit_n"""
        """
        Ideen: Jedes Cnp sollte eine einzelne extraction option sein; Kombinationen von cnp gibt es erstmal nicht
        Für jedes frontier phase gadget wird eine Option erzeugt, bei der mit with_insert Strategie auch phase gadgets ergänzt werden
        Beste Struktur für gadget dict: keys = frontier_vertices und als values jeweils das root, top tupel?
        """
        result_options = [self.copy()] #do nothing
        gadget_dict = get_exclusive_frontier_gadget_dict(self.g, self.frontier, limit_n)
        existing_combinations = gadget_dict.keys()

        for current_combination in existing_combinations:
            root,top = gadget_dict[current_combination]
            odd_phase = self.g.phase(top) if len(current_combination) % 2 == 1 else -self.g.phase(top)
            current_option = self.copy()

            for combination in get_remaining_combinations(current_combination):
                odd = len(combination) % 2 == 1
                if combination in existing_combinations:
                    #phase gadget structure exists 
                    phase = self.g.phase(gadget_dict[combination][1])
                    if odd and odd_phase == phase or not odd and odd_phase == -phase:
                        #everything fits; just remove the vertices
                        current_option.g.remove_vertices(gadget_dict[combination])
                    else:
                        #phase gadget stays => update phase
                        current_option.g.add_to_phase(gadget_dict[combination][1], odd_phase if odd else -odd_phase)
                else:
                    #phase gadget structure does not exist; create and set phase
                    insert_root = current_option.g.add_vertex(VertexType.Z)
                    insert_top = current_option.g.add_vertex(VertexType.Z, phase=odd_phase if odd else - odd_phase)
                    current_option.g.add_edge(current_option.g.edge(insert_root, insert_top), EdgeType.HADAMARD)
                    for neighbor in combination:
                        current_option.g.add_edge(current_option.g.edge(insert_root, neighbor), EdgeType.HADAMARD)
            
            current_option.g.remove_vertices([root,top])
            
            #extract gate
            cnp_phase = odd_phase*(2**(len(current_combination)-1)) % Fraction(2,1)
            frontier_qubits = [list(self.frontier.values()).index(v) for v in current_combination]
            current_option.append_gate(CNP(cnp_phase,frontier_qubits))
            # adjust frontier (i.e. unary phase gadgets)
            for neighbor in current_combination:
                phase = current_option.g.phase(neighbor)
                if phase != odd_phase:
                    frontier_qubit = list(self.frontier.values()).index(neighbor)
                    current_option.append_gate(ZPhase(frontier_qubit, phase-odd_phase))
                current_option.g.set_phase(neighbor,0)
            
            result_options.append(current_option)
        
        return result_options


    def copy(self):
        new = ExtractionOption(self.g.clone(), self.frontier.copy(), copy.deepcopy(self.architecture), self.resolved_gadget)
        for gate in self.logical_circuit.gates:
            new.logical_circuit.add_gate(gate)
        return new