// Initial wiring: [0, 1, 2, 3, 4]
// Resulting wiring: [0, 1, 2, 3, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cx q[0], q[1];
cx q[1], q[3];
