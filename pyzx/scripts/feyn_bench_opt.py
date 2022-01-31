# Parameter: 

# input_file output_file <strategie='greedy'> <cap=1> <maxv=False>

# Output: 
# (name, )

import sys; sys.path.append('../..')
import pyzx as zx
import pickle

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
    # elif strategy == "random_simp_neighbors":
    #     zx.simplify.random_simp_neighbors(g, params['boundaries'], params['gadgets'], params['cap'], params['max_v'])
    elif strategy == "greedy_simp_neighbors":
        zx.simplify.greedy_simp_neighbors(g, params['cap'], params['max_v'])

    else:
        if not strategy == "pyzx":
            print("Wrong name: ",strategy)
            return (None, None)
    t_d = time.time() - t_s
    if strategy == "pyzx":
        c_ext = zx.Circuit.from_graph(g)
    else:
        c_ext = zx.extract_circuit(g.copy())
    c_ext = zx.optimize.basic_optimization(c_ext.to_basic_gates()).to_basic_gates().split_phase_gates()
    return (c_ext, t_d)

def generate_stat_entry(circuit, strategy, time = -1, params = {}):
    return {'strategy': strategy, '1-qubit': len(circuit.gates), '2-qubit': circuit.twoqubitcount(), 'T-gates': circuit.tcount(), 'time': time, 'params': str(params)}

def evaluate_circuit(name):
    stats = []
    circuit = zx.Circuit.load(name).to_basic_gates()
    stats.append(generate_stat_entry(circuit, 'original'))
    c_basic_opt = zx.optimize.basic_optimization(circuit)
    stats.append(generate_stat_entry(c_basic_opt, 'basic_opt'))
    best_circuit = c_basic_opt
    for strategy in ['pyzx', 'greedy_simp', 'random_simp', 'greedy_simp_neighbors' ]: #,random_simp_neighbors
        params = {'cap': 1, 'max_v': False}
        c_opt, time_opt = evaluate_strategy(c_basic_opt, strategy, params)
        stats.append(generate_stat_entry(c_opt, strategy, time_opt, params))

        if c_opt.twoqubitcount() < best_circuit.twoqubitcount():
            best_circuit = c_opt
    
    save_obj(stats, name+'_stats')
    f = open(name+'_after', 'w')
    f.write(best_circuit.to_qasm())
    f.close()

if __name__ == "__main__":
    evaluate_circuit(name=sys.argv[1])