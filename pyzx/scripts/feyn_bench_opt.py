# Parameter: 

# input_file output_file <strategie='greedy'> <cap=1> <maxv=False>

# Output: 
# (name, )

import os
import sys; sys.path.append('../..')
import pyzx as zx
import pickle

import datetime
import time

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def evaluate_strategy(circuit, strategy, params = {}):
    g = circuit.to_graph()
    g = zx.simplify.teleport_reduce(g)
    g.track_phases = False
    t_s = time.time()

    if strategy == "greedy_simp":
        zx.simplify.greedy_simp(g, cap=params['cap'], max_v=params['max_v'])
    elif strategy == "random_simp":
        zx.simplify.random_simp(g, cap=params['cap'], max_v=params['max_v'])
    elif strategy == "random_simp_neighbors":
        zx.simplify.random_simp_neighbors(g, params['cap'], params['max_v'])
    elif strategy == "greedy_simp_neighbors":
        zx.simplify.greedy_simp_neighbors(g, params['cap'], params['max_v'])

    else:
        if not strategy == "pyzx":
            print("Wrong name: ",strategy)
            return (None, None)
    t_d = time.time() - t_s
    if strategy == "pyzx":
        zx.simplify.clifford_simp(g)
        # c_ext = zx.Circuit.from_graph(g)
    # else:
    c_ext = zx.extract_circuit(g.copy())
    c_ext = zx.optimize.basic_optimization(c_ext.to_basic_gates()).to_basic_gates().split_phase_gates()
    return (c_ext, t_d)

def generate_stat_entry(circuit, strategy, time = -1, params = {}):
    return {'strategy': strategy, '1-qubit': len(circuit.gates), '2-qubit': circuit.twoqubitcount(), 'T-gates': circuit.tcount(), 'time': time, 'params': str(params)}

def evaluate_circuit(path, name, todd=False):
    stats = []
    circuit = zx.Circuit.load(os.path.join(path,name)).to_basic_gates()
    stats.append(generate_stat_entry(circuit, 'original'))

    if todd:
        c_basic_opt = zx.optimize.full_optimize(circuit)
    else:
        c_basic_opt = zx.optimize.basic_optimization(circuit)
        
    stats.append(generate_stat_entry(c_basic_opt, 'basic_opt'))

    c_opt, time_opt = evaluate_strategy(c_basic_opt, 'pyzx')
    stats.append(generate_stat_entry(c_opt, 'pyzx', time_opt))
    best_circuit = c_opt

    for strategy in ['greedy_simp', 'random_simp', 'greedy_simp_neighbors', 'random_simp_neighbors']:
        for cap in [1, -5, -20]:
            print("evaluate strategy ",strategy)
            params = {'cap': cap, 'max_v': False if cap == 1 else True}
            c_opt, time_opt = evaluate_strategy(c_basic_opt, strategy, params)
            stats.append(generate_stat_entry(c_opt, strategy, time_opt, params))

            if c_opt.twoqubitcount() < best_circuit.twoqubitcount():
                best_circuit = c_opt
    
    return (best_circuit, stats)

def evaluate_folder(folder):
    source_folder = folder + 'before/'
    dest_folder = folder + 'after/'
    stats_folder = folder + 'stats/'
    list_of_files = filter( lambda x: os.path.isfile(os.path.join(source_folder, x)),os.listdir(source_folder) )
    list_of_files = sorted( list_of_files, key =  lambda x: os.stat(os.path.join(source_folder,x)).st_size)
    for file in list_of_files:
        print("evaluate circuit ",file, " at ",datetime.datetime.now())
        for todd in [False]:
            best_circuit, stats = evaluate_circuit(source_folder, file, todd)

            filename = file + '_t' if todd else file
            save_obj(stats, os.path.join(stats_folder,filename))
            f = open(os.path.join(dest_folder,filename), 'w')
            f.write(best_circuit.to_qasm())
            f.close()

if __name__ == "__main__":
    evaluate_folder(sys.argv[1])
    # evaluate_circuit(name=sys.argv[1])