
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.utils import EdgeType
from pyzx.circuit import Circuit, HAD
from pyzx.routing import Architecture
from pyzx.extract import bi_adj, extract_circuit, graph_to_swaps

from mqt.qmap import HybridSynthesisMapper

from typing import Dict, List
import numpy as np

from .rerouting import get_neighbors_of_frontier
from .extractionoption import ExtractionOption
from .extractionutils import convert_to_qiskit

#debugging stuff
from pyzx.drawing import draw 
from pyzx.tensor import compare_tensors


class HybridMappingExtractor:
    frontier: Dict[int,VT]
    g: BaseGraph[VT,ET]
    c: Circuit
    mapper: HybridSynthesisMapper
    architecture: Architecture

    def __init__(self, graph: BaseGraph[VT,ET], mapper: HybridSynthesisMapper) -> None:
        self.g = graph.copy()
        if len(graph.inputs()) != len(graph.outputs()):
            raise Exception("Can only extract graphs with same number of inputs and outputs")
        self.mapper = mapper
        self.c = Circuit(len(graph.inputs()))
        self.architecture = Architecture("coupling", coupling_matrix=np.array(mapper.get_circuit_adjacency_matrix()))
        # self.c, self.architecture = get_circuit_from_mapper(self.mapper, get_exact_phases=False)
        self.init_frontier()

    def init_frontier(self) -> Dict[int,VT]:
        """Inits the frontier of a ZX-diagram with the spiders adjacent to the outputs. Extracts Hadamard wires between outputs and frontier"""
        self.frontier: Dict[int,VT] = dict()

        for i, o in enumerate(self.g.inputs()):
            v = list(self.g.neighbors(o))[0]
            if not v in self.g.outputs():
                self.frontier[i] = v
    
    def extract(self, num_it=1) -> Circuit:
        orig_g = self.g.copy() #debugging stuff
        orig_circ = extract_circuit(orig_g.copy()) #debugging stuff

        if not self.frontier:
            self.init_frontier()
        self.clear_initial_hadamards()
        while True:
            extraction_options = self.collect_extraction_options(num_it)
            if not extraction_options:
                print("no valid extraction steps found; this should not happen")
                #But it can happen if we need to resolve multiple phase gadgets before cnot addition works.
                import pdb
                pdb.set_trace()
            for i,option in enumerate(extraction_options):
                print("option",i)
                draw(option.g)
                draw(option.logical_circuit)
            self.apply_best_option(extraction_options)
            
            #debugging stuff
            print("resulting graph:")
            draw(self.g, labels=True, scale=40)
            draw(self.c)
            print(self.frontier)
            print("\n\n\n NEW ITERATION:")
            if not compare_tensors(orig_circ,self.c+extract_circuit(self.g.copy())):
                print("tensors do not match")
                import pdb
                pdb.set_trace()

            if not set(self.frontier.values()).difference(self.g.outputs()):
                #resolve remaining swaps
                swaps = graph_to_swaps(self.g)
                option = ExtractionOption(self.g,self.frontier,self.architecture)
                for swap in swaps:
                    option.logical_circuit.add_gate(swap)
                self.apply_best_option([option])
                return self.c


    
    def collect_extraction_options(self, num_it=1):
        result_options = [ExtractionOption(self.g.clone(), self.frontier.copy(), self.architecture)]
        for i in range(0,num_it):
            result_options = map(lambda option: option.collect_had_options(), result_options)
            result_options = map(lambda option: option.collect_rz_options(), result_options)
            result_options = list(map(lambda option: option.collect_cz_options(), result_options))

            result_options_gadgets: List[ExtractionOption] = []
            for existing_option in result_options:
                for gadget_option in existing_option.collect_gadget_options(): #gadget options should be optionally, i.e. we do not have to apply it if we dont need to
                    result_options_gadgets.append(gadget_option)
            
            for existing_option in result_options:
                for cnp_option in existing_option.collect_cnp_options():
                    result_options_gadgets.append(cnp_option) #add cnp options alongside gadget options
            
            result_options = map(lambda option: option.collect_cz_options(), result_options_gadgets) # in case pivot generated new connections between frontier (impractical for cnot extraction)
            

            result_options_cnots: List[ExtractionOption] = []
            for existing_option in result_options:
                for cnot_option in existing_option.collect_cnot_options(): #TODO: cnot should preferably not(!) be optional, but sometimes we need to resolve multiple phase gadgets before cnot addition works.
                    frontier_vertices_without_outputs = [v for k,v in cnot_option.frontier.items() if not any([n for n in cnot_option.g.neighbors(v) if n in cnot_option.g.outputs()])]
                    frontier_neighbors = list(get_neighbors_of_frontier(cnot_option.g,frontier_vertices_without_outputs))
                    biadj_m = bi_adj(cnot_option.g, frontier_neighbors, frontier_vertices_without_outputs)
                    if any(sum(row) == 1 for row in biadj_m.data) or cnot_option.resolved_gadget or not frontier_neighbors:
                        #kick out options where all biadjacency rows are > 1 and no gadget transformation has happened to prevent non-termination
                        result_options_cnots.append(cnot_option)
            
            result_options = result_options_cnots

        return result_options

    def apply_best_option(self, options: List[ExtractionOption]):
        qiskit_circuits = []
        for option in options:
            qiskit_circuits.append(convert_to_qiskit(option.logical_circuit))
        
        index = self.mapper.evaluate_synthesis_steps(qiskit_circuits, also_map=True)
        print("applied option",index)
        self.g = options[index].g.clone()
        self.c += options[index].logical_circuit.copy()
        self.frontier = options[index].frontier.copy()

        adjacency_matrix = np.array(self.mapper.get_circuit_adjacency_matrix())
        self.architecture = Architecture("new_coupling", coupling_matrix=adjacency_matrix, qubit_map=list(range(self.c.qubits)))

        return index
    
    def clear_initial_hadamards(self):
        option = ExtractionOption(self.g, self.frontier, self.architecture)
        for input in option.g.inputs():
            n = list(option.g.neighbors(input))[0]
            if option.g.edge_type(option.g.edge(input,n)) == EdgeType.HADAMARD:
                option.logical_circuit.add_gate(HAD(list(option.frontier.values()).index(n)))
                option.g.set_edge_type(option.g.edge(input,n), EdgeType.SIMPLE)
        self.apply_best_option([option])
