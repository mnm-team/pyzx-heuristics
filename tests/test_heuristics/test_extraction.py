from pathlib import Path
import sys

project_path = Path(__file__).parent.parent.parent
if project_path not in sys.path:
    sys.path.append(str(project_path))

import matplotlib.pyplot as plt

import random
import pyzx as zx
from pyzx.drawing import draw, draw_matplotlib
from pyzx.routing.architecture import create_line_architecture
from pyzx.heuristics.extraction.extraction import extract_architecture_aware_circuit
from pyzx.heuristics.extraction.extraction_mcp import mcp_aware_extract
from tests.test_heuristics.test_utils import get_circuit_and_fr_graph



class TestExtraction():
 
    def test_architecture_aware_extraction(self):
        
        _, graph = get_circuit_and_fr_graph()

        architecture = create_line_architecture(graph.qubit_count())
        graph_simp = graph.copy()
        new_circuit, new_architecture = extract_architecture_aware_circuit(graph=graph_simp, architecture=architecture, up_to_perm=True, quiet=True)

        for gate in new_circuit.gates:
            if gate.name == "CNOT":
                qubit0 = gate.target
                qubit1 = gate.control
                if not architecture.graph.connected(qubit0, qubit1):
                    assert False

        assert True

    #FIXME: Architecture aware extraction is not allways correct. In this case Circuit 6 can not be extracted correctly
    def test_architecture_aware_extraction_correctness(self):
        
        random.seed(7)
        for i in range(10):
            print(f"Circuit {i}")
            circuit, graph = get_circuit_and_fr_graph()

            architecture = create_line_architecture(graph.qubit_count())
            graph_simp = graph.copy()
            new_circuit, new_architecture = extract_architecture_aware_circuit(graph=graph_simp, architecture=architecture, up_to_perm=False, quiet=True)

            assert zx.compare_tensors(circuit, new_circuit)



    def test_architecture_aware_mapper_extraction_correctness(self):
        
        random.seed(11)
        for i in range(1):
            print(f"Circuit {i}")
            circuit, graph = get_circuit_and_fr_graph(6)

            architecture = create_line_architecture(graph.qubit_count())
            graph_simp = graph.copy()
            new_circuit, new_architecture = extract_architecture_aware_circuit(graph=graph_simp, architecture=architecture, use_gate_mapping=True, up_to_perm=False, quiet=True)

            assert zx.compare_tensors(circuit, new_circuit)


    def test_mcp_architecture_aware_extraction_correctness(self):
        seed = 11
        num_qubits = 6
        depth = 80
        random.seed(seed)

        circuit, graph = get_circuit_and_fr_graph(num_qubits, depth, seed=seed)

        # draw_matplotlib(graph).savefig(f"graph_q{num_qubits}_depth{depth}_s{seed}.png")

        architecture = create_line_architecture(graph.qubit_count())
        graph_simp = graph.copy()
        new_circuit, new_architecture = mcp_aware_extract(graph=graph_simp, architecture=architecture, use_gate_mapping=False)

        assert zx.compare_tensors(circuit, new_circuit)

    
    def test_mcp_architecture_aware_mapper_extraction_correctness(self):
        seed = 11
        num_qubits = 6
        depth = 80
        random.seed(seed)

        circuit, graph = get_circuit_and_fr_graph(num_qubits, depth, seed=seed)

        # draw_matplotlib(graph).savefig(f"graph_q{num_qubits}_depth{depth}_s{seed}.png")

        architecture = create_line_architecture(graph.qubit_count())
        graph_simp = graph.copy()
        new_circuit, new_architecture = mcp_aware_extract(graph=graph_simp, architecture=architecture, use_gate_mapping=True)

        assert zx.compare_tensors(circuit, new_circuit)

        
# if __name__ == '__main__':
#     # pytest.main()