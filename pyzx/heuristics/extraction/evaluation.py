import sys
sys.path.append('../../../')
from pyzx.circuit import Circuit
from pyzx.extract import extract_circuit
from pyzx.optimize import basic_optimization
from pyzx.simplify import full_reduce
from pyzx.heuristics.extraction.standardextract import HybridMappingExtractor
from pyzx.heuristics.extraction.mappers import create_na_mapper
from pyzx.heuristics.extraction.extractionutils import convert_to_qiskit
import pandas as pd 
import os
import traceback

def evaluate_circuit(c: Circuit, params):
    data = {}
    c = basic_optimization(c.to_basic_gates().split_phase_gates()).split_phase_gates()
    g = c.to_graph()
    full_reduce(g)

    cpyzx = extract_circuit(g.copy(), up_to_perm=True)
    architecture_name = 'rubidium_4x4.json'
    if len(g.inputs()) > 20:
        architecture_name = 'rubidium_6x6.json'
    elif len(g.inputs()) > 12:
        architecture_name = 'rubidium_5x5.json'
    mapper_pyzx = create_na_mapper(params['config_path']+architecture_name, len(g.inputs()))
    qc = convert_to_qiskit(cpyzx)
    mapper_pyzx.append_with_mapping(qc)
    for k,v in mapper_pyzx.schedule().items():
        data["pyzx_"+k] = v
    
    mapper_standard = create_na_mapper(params['config_path']+architecture_name, len(g.inputs()))
    qc = convert_to_qiskit(c)
    mapper_standard.append_with_mapping(qc)
    for k,v in mapper_standard.schedule().items():
        data["standard_"+k] = v
    
    print(data)
    
    for edge_bias in params['edge_bias']:
        for num_it in params['lookahead_iterations']:
            mapper_hybrid = create_na_mapper(params['config_path']+architecture_name, len(g.inputs()))
            extractor = HybridMappingExtractor(g, mapper_hybrid, edge_bias=edge_bias)
            routed_circuit = extractor.extract(num_it=num_it, up_to_perm=True)
            for k,v in extractor.mapper.schedule().items():
                data["hybrid_"+str(edge_bias)+"_"+str(num_it)+"_"+k] = v
            
            print(data)


    return pd.DataFrame(data, index=[0])
    

def evaluate_benchmark_circuits(benchmark='feyn'):
    """main evaluation routine: generates benchmark circuits from qasm files 
    and evaluates different synthesis strategies"""
    if benchmark == 'feyn':
        circuits = generate_feynman_circuits()
    elif benchmark == 'mqt':
        circuits = generate_mqt_circuits()
    else:
        circuits = generate_qasm_bench_circuits(benchmark)    
    
    params = {'edge_bias': [0,0.005,0.05], 'lookahead_iterations': [1,2], 'config_path': 'architectures/'}
    data = pd.DataFrame()
    for name, circuit in circuits:
        filename = name.split('/')[-1]
        if filename in ['adder_n64.qasm','multiplier_n45.qasm','swap_test_n115.qasm','wstate_n76.qasm','multiplier_n75.qasm','adder_n433.qasm','swap_test_n361.qasm','gf2^8_mult.qasm','gf2^9_mult.qasm','gf2^10_mult.qasm','adder_n118.qasm']:
            #skip some very large circuits
            continue
        print("eval",filename)
        try:
            circ_data = evaluate_circuit(circuit, params)
            circ_data['name'] = filename
        except:
            traceback.print_exc()
            print('\n')
            # import pdb
            # pdb.set_trace()
            continue

        data = pd.concat([data,circ_data],ignore_index=True)
    
        print(data) 

    data.to_csv(benchmark+'.csv')


def generate_feynman_circuits():
    circuits = []
    directory = '../../../../../neutral-atom-gate-decomposer/feyn_bench/'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.getsize(f) > 10000:
            print("file too large, skip",f)
            continue
        try:
            circuits.append((f,Circuit.from_qasm_file(f)))
        except:
            print("conversion error",f)
    
    return circuits

def generate_mqt_circuits():
    circuits = []
    directory = '../../../../../neutral-atom-gate-decomposer/mqt_bench/'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        try:
            circ = Circuit.from_qasm_file(f)
            circ.remove_final_measurements()
            circuits.append((f,circ))
        except:
            print("conversion error",f) 
    return circuits

def generate_qasm_bench_circuits(folder):
    circuits = []
    directory = '../../../../../neutral-atom-gate-decomposer/QASMBench/'+folder
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename+'/'+filename+'.qasm')
        try:
            if os.path.getsize(f) > 100000:
                print("file too large, skip",f)
                continue
            circuits.append((f,Circuit.from_qasm_file(f)))
        except:
            print("conversion error",f)
    
    return circuits    

if __name__ == "__main__":
    """
    evaluates benchmark circuits and saves results in csv file.
    possible benchmark parameters are:
    feyn (arithmetic)
    small (qasm bench small)
    medium (qasm bench medium)
    large (qasm bench large)
    mqt (mqt bench circuits)
    """
    evaluate_benchmark_circuits(benchmark='small')