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


import os
import sys; sys.path.append('../..')
# import pyzx

from pyzx.circuit import Circuit, determine_file_type
from pyzx import simplify
from pyzx import extract
from pyzx import optimize

description="""End-to-end circuit optimizer

For simple optimisation of a circuit run as
    python -m pyzx opt circuit.extension

This puts an optimised version of the circuit in the same directory and of the same file type.

If we want to specify the output location and type we can run
    python -m pyzx opt -d outputfile.qc -t qc inputfile.qasm
"""

import argparse
parser = argparse.ArgumentParser(prog="pyzx opt", description=description, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('source',type=str,help='source circuit')
parser.add_argument('-d',type=str,help='destination for output file', dest='dest',default='')
parser.add_argument('-o',type=str,default='match', dest='outformat',
    help='Specify the output format (qasm, qc, quipper). By default matches the input')
parser.add_argument('-v',default=False, action='store_true', dest='verbose',
    help='Output verbose information')
parser.add_argument('-s',type=str,default='greedy', dest='simp', 
    help='ZX-simplifier to use. Options are greedy (default), greedyn (greedy + neighbor unfusion), random or randomn (random + neighbor unfusion)')
parser.add_argument('-p',default=True, action='store_true', dest='phasepoly',
    help='Whether to also run the phase-polynomial optimizer (default is true)')
parser.add_argument('-c',type=int, default=1, dest='cap',
    help='Cap for heuristics; default is 1, i.e. only rules which decrease H-wires are applied')
parser.add_argument('-maxv',default=False, action='store_true', dest='maxv',
    help='Maxv for heuristics, has to be true if cap <= 0, otherwise procedure may not terminate')

def main(args):
    print("Starting program")
    options = parser.parse_args(args)
    if not os.path.exists(options.source):
        print("File {} does not exist".format(options.source))
        return
    ctype = determine_file_type(options.source)
    if options.outformat == 'match':
        dtype = ctype
    elif options.outformat not in ('qasm', 'qc', 'quipper'):
        print("Unsupported circuit type {}. Please use qasm, qc or quipper".format(options.outformat))
        return
    else:
        dtype = options.outformat
    if not options.dest:
        base = os.path.splitext(options.source)[0]
        dest = base + "." + dtype
    else:
        dest = options.dest

    c = Circuit.load(options.source).to_basic_gates().split_phase_gates()
    if options.verbose:
        print("Starting circuit:")
        print(c.stats())
    c1 = optimize.basic_optimization(c)
    print("c1 ",c1.stats())
    g = c1.to_graph()
    if options.phasepoly:
        if options.verbose: print("Running phase teleportation algorithm")
        g = simplify.teleport_reduce(g,quiet=(not options.verbose))
        g.track_phases = False

    print("Applying simplification strategy ",options.simp)
    if options.simp == 'greedy':
        simplify.greedy_simp(g, False, False, quiet=(not options.verbose), threshold=options.cap, max_v=options.maxv)
    if options.simp == 'greedyn':
        simplify.greedy_simp_neighbors(g,quiet=(not options.verbose), threshold=options.cap, max_v=options.maxv)
    if options.simp == 'random':
        simplify.random_simp(g,False, False, quiet=(not options.verbose), threshold=options.cap, max_v=options.maxv)
    if options.simp == 'randomn':
        simplify.random_simp_neighbors(g, quiet=(not options.verbose), threshold=options.cap, max_v=options.maxv)
    if options.verbose: print("Extracting circuit...")

    c4 = extract.extract_circuit(g.copy())
    c5 = optimize.basic_optimization(c4.to_basic_gates()).to_basic_gates().split_phase_gates()

    if options.verbose: print(c5.stats())
    print("Writing output to {}".format(os.path.abspath(dest)))
    if dtype == 'qc': output = c5.to_qc()
    if dtype == 'qasm': output = c5.to_qasm()
    if dtype == 'quipper': output = c5.to_quipper()
    f = open(dest, 'w')
    f.write(output)
    f.close()

if __name__ == "__main__":
    main(args=sys.argv[1:])