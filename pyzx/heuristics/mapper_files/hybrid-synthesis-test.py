# %% [markdown]
# # Install instructions
# 
# 1. Clone repository 
# ```bash
# git clone https://github.com/cda-tum/mqt-qmap/tree/na-zx-mapping-synthesis --recursive
# ```
# 2. create venv and inside directory
# ```bash
# pip install .
# ```
# 
# # Use case example
# 

# %%
from qiskit import QuantumCircuit

from mqt.qmap import HybridSynthesisMapper, NeutralAtomHybridArchitecture, HybridMapperParameters, InitialCoordinateMapping, InitialCircuitMapping

# %%
# create two possible synthesis steps
synthesis_step_1 = QuantumCircuit(3)
synthesis_step_1.cx(0, 1)
synthesis_step_1.cx(0, 2)

synthesis_step_2 = QuantumCircuit(3)
synthesis_step_2.cx(0, 2)
synthesis_step_2.cx(1, 2)

synthesis_stesps = [synthesis_step_1, synthesis_step_2]


# %% [markdown]
# ## Create Mapper

# %%
# create a neutral atom hybrid architecture
arch_file = 'rubidium.json'
architecture = NeutralAtomHybridArchitecture(arch_file)

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
params.verbose = True

# create mapper
synthesis_mapper = HybridSynthesisMapper(arch=architecture, params=params)
# alternatively
# synthesis_mapper = HybridSynthesisMapper(arch=architecture)
# synthesis_mapper.set_parameters(params)


# %% [markdown]
# ## Initialize mapping process
# 

# %%
# set the initial circuit mapping and the size of the mapping(number of qubits)
synthesis_mapper.init_mapping(n_qubits=3, initial_mapping=InitialCircuitMapping.identity)

# %% [markdown]
# ## Evaluate different synthesis steps

# %%
# pass the synthesis steps to the mapper
# the returned index is the index of the synthesis step that is the best fit for the architecture
# the parameter indicates if the step should be directly applied or not 
index = synthesis_mapper.evaluate_synthesis_steps(synthesis_stesps, also_map=False)
print(index)

# %% [markdown]
# ## Other methods

# %%
# reset the mapper
synthesis_mapper.init_mapping(n_qubits=3, initial_mapping=InitialCircuitMapping.identity)

# %%
# append a circuit to the mapper by mapping it to the architecture and adding it to the circuit
synthesis_mapper.append_with_mapping(synthesis_step_1)

# %%
# pass a circuit to the mapper without mapping it to the architecture
# the circuit needs to fit the architecture!!!
synthesis_mapper.append_without_mapping(synthesis_step_2)

# %%
# remap the whole circuit again from scratch
synthesis_mapper.complete_remap()

# %% [markdown]
# ## Get resulting circuits

# %%
# the complete synthesized circuit
synthesized_circuit_qasm = synthesis_mapper.get_synthesized_qc()
synthesized_circuit = QuantumCircuit.from_qasm_str(synthesized_circuit_qasm)
synthesized_circuit.draw()

# %%
# the mapped circuit
mapped_circuit_qasm = synthesis_mapper.get_mapped_qc()
# can not be drawn as it is not valid qasm (move operations are not supported by qiskit)
print(mapped_circuit_qasm)

# %%
# convert moves to aod operations
synthesis_mapper.convert_to_aod()
mapped_circuit_qasm = synthesis_mapper.get_mapped_qc_aod()
print(mapped_circuit_qasm)

# %% [markdown]
# ## Get connectivity

# %%
import numpy as np

# %%
# get adjacency matrix of the architecture at the current state of the mapper
adjacency_matrix = np.array(synthesis_mapper.get_circuit_adjacency_matrix())
print(adjacency_matrix)

# %%



