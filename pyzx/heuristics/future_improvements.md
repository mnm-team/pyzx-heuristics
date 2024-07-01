# Future Improvements for PyZX Heuristic Optimization

## Extraction

- **Frontier Dictionary Handling:** Investigate the necessity of filling the frontier dictionary with -1s if a frontier was removed to prevent disallowed CNOTs due to architecture constraints.
- **Basic Extraction Optimization:** Determine if calling `eliminate_yz_spider` is necessary for Basic extraction, as `remove_gadget` might already suffice for removing xz and yz gadgets.
- **Gate Addition Simplification:** Modify the gate class, so `add_gate_to_circuit` can directly call the function from the gate class instead of using a match case over the gate name.
- **Circuit Extraction Flexibility:** Support the choice between using CNOTs or HAD+CZ+HAD gates in circuit extraction through a parameter.
- **MCP Extraction Enhancement:** Enable MCP extraction to support rerouted frontier gates, changing `extract_czs` and `rz_gates` to return a list of `ReroutedGate`s for application with `apply_gates_to_circuit`.
- **YZ Gadgets in MCP Extraction:** Explore the possibility of MCP extraction also considering yz gadgets. Currently the "extract_mcp" only considers xz Gadgets.
- **Multiple MCP Extraction Options:** Extend `extract_mcp` to offer multiple options for MCP extraction, including gadget extraction via pivot, C2P gadgets as MCP gates, and full insertion of missing phase gadgets. Additionally, `extract_mcp` should return a list of `ReroutedGate`s for application with `apply_gates_to_circuit`.
- **Extraction Method Location:** Move `eliminate_yz_spider` to extraction_base.py, replacing `pivot_mcp` and `lcomp_mcp` with standard lcomp and pivot.

## Gate Mapper

- **Mapper Architecture Creation:** Generate the mapper architecture from the coupling matrix derived from the architecture used in extraction, rather than based on the `interactionRadius`.
- **Mapped Circuit Return:** The mapper should return the mapped circuit `get_mapped_qc` but move operations are not yet supported
- **Mapper Object Creation:** Address issues preventing the creation of the mapper object in a separate function, likely rooted in C++ code.
- **Extraction and Mapping Accuracy:** The inaccuracies observed in the circuit post-extraction and gate mapping are likely attributed to the use of `get_synthesized_qc`, which merely collects gates without adapting them to the new architecture. The extraction algorithm, however, still uses the architecture to reroute gates, leading to complications. This issue is shown in the `test_architecture_aware_mapper_extraction_correctness` test case, where, in later iterations, the mapper yields a different set of CNOT operations compared to scenarios without the mapper. For initial iterations, the gates remain consistent despite the unchanged architecture of the mapper throughout the process. This leads to an unexpected outcome where the circuit is ~70% identical to the original, diverging only in the final ~30%. However, it is not yet possible to check if the `get_mapped_qc` is correct, since it can not be transformed to a tensor.

## General

- **Gate Qubit Reversal:** Implement `reverse_gate_qubits` properly within the Gate class.
- **Measurement Types Access:** Make `get_measurement_types` a property of the graph to avoid redundant calculations.
- **Repeated Algorithm Calls:** Investigate the increasing time it takes if the greedy optimization is called multiple times. Currently this is cirumvented by creating child processes for each call (see `benchmark_greedy_heuristic.py`).

## Heuristic and Greedy Optimization

- **Phase Type Handling in Lcomp Heuristic:** Address the hack of returning heuristic_result-1 for phase_type == PhaseType.CLIFFORD in `lcomp_heuristic`.
- **Boundary Conditions in Lcomp Matches:** Reevaluate the restriction on lcomp matches with more than one boundary.
- **Phase Index Tracking:** Consider the utility of tracking the phase index for greedy optimization.
- **Lookup Dictionary Rework:** Assess the effectiveness of the lookup dictionary, given its invalidation upon each match application. This leads to an overhead, since the flow of the graph is calculated for each edge if the dictionary is used, compared to each match, if the dictionary is not used.
- **Lookahead Matches Application:** Debate the merits of applying all lookahead matches versus only the first one.
- **Match Flow Preservation:** Clarify the conditions in `_is_match_flow_preserving` to better identify phase gadgets and their influence. The evaluation should focus on determining whether the current match is a Gadget match or if Gadgets are present in the graph. It might suffice to assess whether the current match would impact any Gadget, rather than verifying the presence of Gadgets.
- **Result Data Structuring:** Propose a more structured approach to incorporating result data, potentially through a dedicated class.
- **Threshold for Search Termination:** Make the `num_matches` threshold for stopping the search in `_apply_and_find_new_matches` a class variable.