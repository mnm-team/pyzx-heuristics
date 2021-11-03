import sys; sys.path.append('..')
import random
import pyzx as zx
import os
import pickle
import time

def save_obj(obj, name):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def evaluate_strategy(circuit, strategy, teleport=False, params = {}):
    g = circuit.to_graph()
    if teleport:
        g = zx.simplify.teleport_reduce(g)
    g.track_phases = False
    print("apply strategy: ",strategy)
    t_s = time.time()
    if strategy == "full_reduce":
        zx.simplify.full_reduce(g)
    elif strategy == "clifford_simp":
        zx.simplify.clifford_simp(g)
    elif strategy == "greedy_simp":
        zx.simplify.greedy_simp(g, params['boundaries'], params['gadgets'], params['cap'], params['max_v'])
    elif strategy == "random_simp":
        zx.simplify.random_simp(g, params['boundaries'], params['gadgets'], params['cap'], params['max_v'])
    elif strategy == "simulated_annealing_simp":
        zx.simplify.simulated_annealing_simp(g, params['iterations'], params['alpha'], params['cap'])
    elif strategy == "greedy_simp_neighbors":
        zx.simplify.greedy_simp_neighbors(g, params['cap'], params['max_v'])
    elif strategy == "sim_annealing_post":
        zx.simplify.full_reduce(g)
        g, _ = zx.anneal(g, iters=params['iterations'], full_reduce_prob=1)

    else:
        print("Wrong name: ",strategy)
        return (None, None)
    t_d = time.time() - t_s
    c_ext = zx.extract_circuit(g.copy())
    return (len(c_ext.gates),c_ext.twoqubitcount(),t_d)

def file_parser(root_path, param_list):
    statistics = []
    for filename in os.listdir(root_path):#["dnn_n2.qasm"]:
        print("parse file ",filename)
        try: 
            c = zx.Circuit.load(os.path.join(root_path, filename)).to_basic_gates()
        except Exception as e:
            print("could not parse ",filename, "error ",e)
            continue
        print("circuit size ",len(c.gates))
        if len(c.gates) > 10000:
            print("no evaluation of circuit ",filename, " because of size ",len(c.gates))
            continue
        stat_obj = {'name': filename, 'qubits': c.qubits, 'gatecount': len(c.gates), 'gatecount2': c.twoqubitcount()}
        c_opt = zx.optimize.basic_optimization(c.split_phase_gates())
        stat_obj['basic_opt'] = len(c_opt.gates)
        stat_obj['basic_opt2'] = c_opt.twoqubitcount()
        for strategy in ["full_reduce", "clifford_simp", "sim_annealing_post", "greedy_simp", "random_simp", "simulated_annealing_simp", "greedy_simp_neighbours"]:
            stat_obj[strategy] = evaluate_strategy(c_opt, strategy, teleport=True, params=param_list)
        statistics.append(stat_obj)
        print(stat_obj)
    return statistics

def generate_filename(qasm_folder, params):
    name = qasm_folder + "_"
    if params['boundaries']:
        name += "b_"
    if params['gadgets']:
        name += "g_"
    if params['max_v']:
        name+= "m_"
    name += "c" + str(params['cap'])
    name += "a" + str(params['alpha'])
    name += "i" + str(params['iterations'])
    return name

def evalutate_feyn():
    qasm_folder = "feyn_bench"
    for bg in [False]:
        for cap in [1, -10]:
            param_list = {
                'boundaries': bg,
                'gadgets': bg,
                'cap': cap,
                'max_v': False if cap == 1 else True,
                'iterations': 100,
                'alpha': 0.99
            }
            f_name = generate_filename(qasm_folder, param_list)
            print("evaluating ",f_name)
            stats = file_parser("../circuits/feyn_bench/qasm/", param_list)
            save_obj(stats, f_name)

if __name__ == "__main__":
    evalutate_feyn()