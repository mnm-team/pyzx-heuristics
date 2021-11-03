#!/usr/bin/env python
import sys; sys.path.append('..')
import random
import pyzx as zx
import os
import pickle
import time

def save_obj(obj, name):
    with open('data/qasmbench/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('data/qasmbench/' + name + '.pkl', 'rb') as f:
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


def evaluate(root_folder):
    root_path = "../circuits/QASMBench/"+ root_folder + "/"
    complete_stats = {}
    num_shots = 10
    for filename in os.listdir(root_path):
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
        stat_obj = {'gatecount': (len(c.gates),c.twoqubitcount(),0),}
        t_s = time.time()
        c_opt = zx.optimize.basic_optimization(c.split_phase_gates())
        stat_obj['basic_opt'] = (len(c_opt.gates),c_opt.twoqubitcount(),time.time()-t_s)
        iterations = len(c_opt.gates)
        stat_obj['full_reduce'] = evaluate_strategy(c_opt, 'full_reduce', teleport=True, params={})
        stat_obj['clifford_simp'] = evaluate_strategy(c_opt, 'clifford_simp', teleport=True, params={})
        stat_obj['sim_annealing_post'] = evaluate_strategy(c_opt, 'sim_annealing_post', teleport=True, params={'iterations': iterations})
        stat_obj['simulated_annealing_simp'] = evaluate_strategy(c_opt, 'simulated_annealing_simp', teleport=True, params={'iterations': iterations, 'alpha': 0.99, 'cap': -10000})
        for cap in [1,-10]:
            stat_obj['greedy_simp_neighbors_c'+str(cap)] = evaluate_strategy(c_opt, 'greedy_simp_neighbors', teleport=True, params={'cap': cap, 'max_v': False if cap == 1 else True})
            for bg in [True, False]:
                stat_obj['greedy_simp_c'+str(cap)+'_b_'+str(bg)] = evaluate_strategy(c_opt, 'greedy_simp', teleport=True, params={'cap': cap, 'max_v': False if cap == 1 else True, 'boundaries':bg, 'gadgets': bg})
                stat_obj['random_simp_c'+str(cap)+'_b_'+str(bg)] = evaluate_strategy(c_opt, 'random_simp', teleport=True, params={'cap': cap, 'max_v': False if cap == 1 else True, 'boundaries':bg, 'gadgets': bg})
        print("name ",filename, "stats ", stat_obj)
        complete_stats[filename] = stat_obj
    save_obj(complete_stats, root_folder)

if __name__ == "__main__":
    evaluate('small')
