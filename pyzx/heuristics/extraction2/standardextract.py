
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.utils import EdgeType, VertexType
from pyzx.circuit import Gate, ZPhase, CZ, CNOT, HAD
from pyzx.linalg import Mat2
from pyzx.extract import bi_adj
from .mapper import *
from .rerouting import *
from .extractionutils import *
from mqt.qmap import HybridSynthesisMapper, NeutralAtomHybridArchitecture, HybridMapperParameters, InitialCoordinateMapping, InitialCircuitMapping
from typing import Any, Dict, List
import copy
from pyzx.drawing import draw


def create_na_mapper(config_path: str, num_qubits: int):
    architecture_mapper = NeutralAtomHybridArchitecture(config_path)
    # set mapper parameters (skip to use default values)
    params = HybridMapperParameters()
    # mapper should use SWAP gates or shuttling operations
    params.gate_weight = 0
    params.shuttling_weight = 1
    # look-ahead weights
    params.lookahead_weight_moves = 0.1
    params.lookahead_weight_swaps = 0.1
    # The initial mapping between atoms and hardware
    params.initial_mapping = InitialCoordinateMapping.trivial
    # If mapper should print debug information
    params.verbose = False

    # create mapper
    mapper = HybridSynthesisMapper(arch=architecture_mapper) #, params=params

    mapper.init_mapping(num_qubits)

    return mapper

# def extract_with_mapper(g: BaseGraph[VT, ET], mapper: HybridSynthesisMapper):
#     circuit, architecture = get_circuit_from_mapper(mapper, get_exact_phases=False)
#     frontier = init_frontier(g, circuit)


#ReroutedGate + Veränderung die im Graphen passieren muss. Wie stellt man die Veränderung da?
"""
VertexChange: Create/Delete bzw. phasen update
EdgeChange: Create/Delete bzw. type flip

create_vertex("a")
update_phase("a",5/4)
create_edge("a",42, EdgeType.SIMPLE)
set_edge_type(("a",42),EdgeType.HADAMARD)

"""



# class GraphAction:
#     name: str
#     params: List[str|Fraction|int]

#     def __init__(self, name: str, params: List[str|Fraction|int]) -> None:
#         self.name = name
#         self.params = params

# class GraphActions:
#     actions: List[GraphAction]
#     vertex_variables: Dict[str,VT]

#     def __init__(self, actions: List[GraphAction]|None=None) -> None:
#         self.vertex_variables = []
#         if actions == None:
#             self.actions = []
#         else:
#             for action in actions:
#                 self.add_action(action)
    
#     def add_action(self, action: GraphAction):
#         self.actions.append(action)

#     def get_vertex_from_variable(self, v1: Any):
#         if v1 in self.vertex_variables.keys():
#             return self.vertex_variables[v1]
#         else:
#             return v1

    # def apply_actions_to_graph(self, g: BaseGraph):
    #     for action in self.actions:
    #         if action.name == "create_vertex":
    #             v1 = g.add_vertex(VertexType.Z)
    #             self.vertex_variables[action.params[0]] = v1
    #         elif action.name == "delete_vertex":
    #             v1 = self.get_vertex_from_variable(action.params[0])
    #             g.remove_vertex(v1)
    #         elif action.name == "update_phase":
    #             v1 = self.get_vertex_from_variable(action.params[0])
    #             phase = action.params[1]
    #             g.set_phase(v1,phase)
    #         elif action.name == "create_edge":
    #             v1 = self.get_vertex_from_variable(action.params[0])
    #             v2 = self.get_vertex_from_variable(action.params[1])
    #             edge_type = action.params[2]
    #             g.add_edge(g.edge(v1,v2),edge_type)
    #         elif action.name == "delete_edge":
    #             v1 = self.get_vertex_from_variable(action.params[0])
    #             v2 = self.get_vertex_from_variable(action.params[1])
    #             g.remove_edge(g.edge(v1,v2))
    #         elif action.name == "set_edge_type":
    #             v1 = self.get_vertex_from_variable(action.params[0])
    #             v2 = self.get_vertex_from_variable(action.params[1])
    #             edge_type = action.params[1]
    #             g.set_edge_type(g.edge(v1,v2),edge_type)
    #         else:
    #             raise Exception("Action undefined")

class CircuitOption:
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
            result_option.append_gate(HAD(qubit))
            result_option.g.remove_vertex(v)
            result_option.g.add_edge(result_option.g.edge(boundary, z_neighbor), EdgeType.SIMPLE)
            result_option.frontier[list(result_option.frontier.values()).index(v)] = z_neighbor
        
        return result_option
    
    def collect_gadget_options(self):
        result_list = [self.copy()] #do nothing
        gadgets = get_frontier_gadgets(self.g, self.frontier)
        print("frontier gadgets",gadgets)
        for root, top in gadgets:
            for frontier_neighbor in set(self.g.neighbors(root)).difference(set([top])):
                if frontier_neighbor in self.frontier.values():
                    frontier_qubit = list(self.frontier.values()).index(frontier_neighbor)
                else:
                    continue
                current_option = self.copy()
                current_option.resolved_gadget = True
                current_option.append_gate(HAD(frontier_qubit))
                print("pivot on",root,frontier_neighbor)

                #calculate pivot graph actions
                b = set(current_option.g.neighbors(root)).intersection(current_option.g.neighbors(frontier_neighbor))
                a = set(current_option.g.neighbors(root)).difference(b).difference([frontier_neighbor])
                c = set(current_option.g.neighbors(frontier_neighbor)).difference(b).difference([root])
                input_neighbor = set(current_option.g.inputs()).intersection(current_option.g.neighbors(frontier_neighbor)).pop()
                c.discard(input_neighbor)
                c.add(frontier_neighbor)

                current_option.g.remove_vertex(root)
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
                
                result_list.append(current_option)
        
        return result_list
    
    def collect_cnot_options(self):
        result_options = [self.copy()] #do nothing
        cnot_addition_options = calculate_addition_options(self.g, self.frontier, self.architecture)
        print("cnot_addition_options",cnot_addition_options)
        
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
    
    def copy(self):
        new = CircuitOption(self.g.clone(), self.frontier.copy(), copy.deepcopy(self.architecture), self.resolved_gadget)
        for gate in self.logical_circuit.gates:
            new.logical_circuit.add_gate(gate)
        return new


class HybridMappingExtractor:
    frontier: Dict[int,VT]
    g: BaseGraph[VT,ET]
    c: Circuit
    mapper: HybridSynthesisMapper
    architecture: Architecture

    def __init__(self, graph: BaseGraph[VT,ET], mapper_config: str) -> None:
        self.g = graph.copy()
        if len(graph.inputs()) != len(graph.outputs()):
            raise Exception("Can only extract graphs with same number of inputs and outputs")
        self.mapper = create_na_mapper(mapper_config, len(graph.inputs()))
        self.c, self.architecture = get_circuit_from_mapper(self.mapper, get_exact_phases=False)
        self.init_frontier()

    def init_frontier(self) -> Dict[int,VT]:
        """Inits the frontier of a ZX-diagram with the spiders adjacent to the outputs. Extracts Hadamard wires between outputs and frontier"""
        self.frontier: Dict[int,VT] = dict()

        for i, o in enumerate(self.g.inputs()):
            v = list(self.g.neighbors(o))[0]
            if not v in self.g.outputs():
                self.frontier[i] = v
                if self.g.edge_type(self.g.edge(v,o)) == EdgeType.HADAMARD:
                    self.c.add_gate("HAD", i)
                    self.g.set_edge_type(self.g.edge(v,o),EdgeType.SIMPLE)
    
    def extract(self, num_it=1) -> Circuit:
        if not self.frontier:
            self.init_frontier()
        self.clear_initial_hadamards()
        while True:
            extraction_options = self.collect_extraction_options(num_it)
            if not extraction_options:
                print("this should not happen")
                #But it can happen if we need to resolve multiple phase gadgets before cnot addition works.
                import pdb
                pdb.set_trace()
            self.apply_best_option(extraction_options)
            draw(self.g, labels=True, scale=40)
            draw(self.c)
            print(self.frontier)
            import pdb
            pdb.set_trace()
            if not set(self.frontier.values()).difference(self.g.outputs()):
                break

    
    def collect_extraction_options(self, num_it=1):
        result_options = [CircuitOption(self.g.clone(), self.frontier.copy(), self.architecture)]
        for i in range(0,num_it):
            result_options = map(lambda option: option.collect_had_options(), result_options)
            result_options = map(lambda option: option.collect_rz_options(), result_options)
            result_options = map(lambda option: option.collect_cz_options(), result_options)
            result_options_gadgets: List[CircuitOption] = []
            gadget_options = 0
            for existing_option in result_options:
                for gadget_option in existing_option.collect_gadget_options(): #gadget options should be optionally, i.e. we do not have to apply it if we dont need to
                    print("gadget_option",gadget_option)
                    gadget_options += 1
                    result_options_gadgets.append(gadget_option)
            
            result_options = map(lambda option: option.collect_cz_options(), result_options_gadgets) # in case pivot generated new connections between frontier (impractical for cnot extraction)
            
            result_options_cnots: List[CircuitOption] = []
            for existing_option in result_options:
                for cnot_option in existing_option.collect_cnot_options():
                    draw(cnot_option.g,labels=True, scale=40)
                    print("cnot option",gadget_option)
                    frontier_vertices_without_outputs = [v for k,v in cnot_option.frontier.items() if not any([n for n in cnot_option.g.neighbors(v) if n in cnot_option.g.outputs()])]
                    print(frontier_vertices_without_outputs)
                    frontier_neighbors = list(get_neighbors_of_frontier(cnot_option.g,frontier_vertices_without_outputs))
                    print(frontier_neighbors)
                    biadj_m = bi_adj(cnot_option.g, frontier_neighbors, frontier_vertices_without_outputs)
                    if any(sum(row) == 1 for row in biadj_m.data) or cnot_option.resolved_gadget:
                        #kick out options where all biadjacency rows are > 1 and no gadget transformation has happened to prevent non-termination
                        result_options_cnots.append(cnot_option)
            
            result_options = result_options_cnots
            #Man muss rz cz had evtl. mehrmals wiederholen bevor cnot geht, oder was wenn cnot dann einfach eine unmodifizierte liste zurückgibt?  

        return result_options

    def apply_best_option(self, options: List[CircuitOption]):
        qiskit_circuits = []
        for option in options:
            qiskit_circuits.append(QuantumCircuit().from_qasm_str(option.logical_circuit.to_qasm()))
        
        index = self.mapper.evaluate_synthesis_steps(qiskit_circuits, also_map=True)
        print("applied option",index)
        self.g = options[index].g.clone()
        self.c += options[index].logical_circuit.copy()
        self.frontier = options[index].frontier.copy()

        adjacency_matrix = np.array(self.mapper.get_circuit_adjacency_matrix())
        self.architecture = Architecture("new_coupling", coupling_matrix=adjacency_matrix, qubit_map=list(range(self.c.qubits)))

        return index
    
    def clear_initial_hadamards(self):
        option = CircuitOption(self.g, self.frontier, self.architecture)
        for input in option.g.inputs():
            n = list(option.g.neighbors(input))[0]
            if option.g.edge_type(option.g.edge(input,n)) == EdgeType.HADAMARD:
                option.logical_circuit.add_gate(HAD(list(option.frontier.values()).index(n)))
                option.g.set_edge_type(option.g.edge(input,n), EdgeType.SIMPLE)
        self.apply_best_option([option])








