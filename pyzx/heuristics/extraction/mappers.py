from mqt.qmap import HybridSynthesisMapper, NeutralAtomHybridArchitecture, HybridMapperParameters, InitialCoordinateMapping

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