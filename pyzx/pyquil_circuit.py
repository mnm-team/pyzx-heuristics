# PyZX - Python library for quantum circuit rewriting 
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyquil import Program, get_qc
from pyquil.gates import CNOT, S, T, RZ, H, CZ, Z
from pyquil.quil import Pragma
from pyquil.api import WavefunctionSimulator, _quantum_computer
from pyquil.quilbase import Pragma, Gate
from numpy import pi
from qcs_sdk import QCSClient

from pyzx.routing.parity_maps import CNOT_tracker
from pyzx.circuit import Circuit

class PyQuilCircuit(CNOT_tracker):

    def __init__(self, architecture, **kwargs):
        """
        Class to represent a PyQuil program to run on/be compiled for the given architecture

        :param architecture: The Architecture object to adhere to
        """

        super().__init__(n_qubits=architecture.n_qubits, **kwargs)
        client_configuration = QCSClient.load()
        topology = architecture.to_quil_topology()
        self.qc = _quantum_computer._get_qvm_with_topology(
                                        client_configuration=client_configuration,
                                        name=f"{architecture.n_qubits}q-{architecture.name}-pyqvm",
                                        topology=topology,
                                        noisy=False,
                                        qvm_type="pyqvm",
                                        compiler_timeout=30.0,
                                        execution_timeout=30.0,
                                        quilc_client=None,
                                        qvm_client=None,
                                    )
        compiler = WavefunctionSimulator()
        self.qc.compiler = compiler

        self.program = Program()

        self.retries = 0
        self.max_retries = 5
        self.compiled_program = None

    def row_add(self, q0, q1):
        """
        Adds a CNOT gate between the given qubit indices q0 and q1
        :param q0: 
        :param q1: 
        """
        self.program += CNOT(q0, q1)
        super().row_add(q0, q1)

    def col_add(self, q0, q1):
        # TODO prepend the CNOT!
        self.program += CNOT(q1, q0)
        super().col_add(q0, q1)

    def count_cnots(self):
        if self.compiled_program is None:
            return super().count_cnots()
        else:
            return self.compiled_cnot_count()

    def compiled_cnot_count(self):
        if self.compiled_program is None:
            self.compile()
        return len([g for g in self.compiled_program if isinstance(g, Gate) and g.name == "CZ"])

    def to_qasm(self):
        if self.compiled_program is None:
            return super().to_qasm()
        circuit = Circuit(self.n_qubits)
        comments = []
        for g in self.compiled_program:
            if isinstance(g, Pragma):
                wiring = " ".join(["//", g.command, "["+g.freeform_string[2:-1]+"]"])
                comments.append(wiring)
            elif isinstance(g, Gate):
                if g.name == "CZ":
                    circuit.add_gate("CZ", g.qubits[0].index, g.qubits[1].index)
                elif g.name == "RX":
                    circuit.add_gate("XPhase", g.qubits[0].index, g.params[0])
                elif g.name == "RZ":
                    circuit.add_gate("ZPhase", g.qubits[0].index, g.params[0])
                else:
                    print("Unsupported gate found!", g)

        qasm = circuit.to_qasm()
        return '\n'.join(comments+[qasm])


    def update_program(self):
        self.program = Program()
        for gate in self.gates:
            if hasattr(gate, "name"):
                if gate.name == "CNOT":
                    self.program += CNOT(gate.control, gate.target)
                elif gate.name == "CZ":
                    self.program += CZ(gate.control, gate.target)
                elif gate.name == "HAD":
                    self.program += H(gate.target)
                elif gate.name == "S":
                    self.program += S(gate.target)
                elif gate.name == "T":
                    self.program += T(gate.target)
                elif gate.name == "T*":
                    self.program += RZ(3*pi/4, gate.target)
                elif gate.name == "Z":
                    self.program += Z(gate.target)
                else:
                    print(f"Warning: PyquilCircuit does not currently support the gate '{gate}'.")

    @staticmethod
    def from_CNOT_tracker(circuit, architecture):
        new_circuit = PyQuilCircuit(architecture, n_qubits=circuit.qubits, name=circuit.name)
        new_circuit.gates = circuit.gates
        new_circuit.update_matrix()
        new_circuit.update_program()
        return new_circuit

    @staticmethod
    def from_circuit(circuit, architecture):
        new_circuit = PyQuilCircuit(architecture)
        new_circuit.gates = circuit.gates
        new_circuit.update_program()
        return new_circuit

    def compile(self):
        """
        Compiles the circuit/program for created quantum computer
        :return: A string that describes the compiled program in quil
        """
        try:
            ep = self.qc.compile(self.program)
            self.retries = 0
            self.compiled_program = ep.program
            return ep.program
        except KeyError as e:
            print('Oops, retrying to compile.', self.retries)
            if self.retries < self.max_retries:
                self.retries += 1
                return self.compile()
            else:
                raise e
