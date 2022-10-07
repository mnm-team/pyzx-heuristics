OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];
h q[0];
h q[1];

cz q[0], q[1];


rz(pi/6) q[0];
h q[0];
rz(0) q[0];
h q[0];


rz(pi/4) q[1];
h q[1];
rz(0) q[1];
h q[1];


rz(0) q[0];
h q[0];

rz(0) q[1];
h q[1];

rz(0) q[0];
h q[0];
rz(pi) q[0];
h q[0];

rz(0) q[1];
h q[1];
rz(pi) q[1];
h q[1];

cz q[0], q[1];

rz(0) q[0];
h q[0];
rz(pi) q[0];
h q[0];

rz(0) q[1];
h q[1];
rz(pi) q[1];
h q[1];

rz(0) q[0];
h q[0];

rz(0) q[1];
h q[1];
