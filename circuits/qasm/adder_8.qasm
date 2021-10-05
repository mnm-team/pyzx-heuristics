OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[24];
cx qubits[3],qubits[2];
cx qubits[8],qubits[7];
cx qubits[14],qubits[13];
cx qubits[21],qubits[20];
cx qubits[3],qubits[4];
cx qubits[8],qubits[9];
cx qubits[14],qubits[15];
cx qubits[21],qubits[22];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[21];
h qubits[3];
ccx qubits[0],qubits[1],qubits[3];
h qubits[3];
h qubits[8];
ccx qubits[5],qubits[6],qubits[8];
h qubits[8];
h qubits[14];
ccx qubits[11],qubits[12],qubits[14];
h qubits[14];
h qubits[21];
ccx qubits[18],qubits[19],qubits[21];
h qubits[21];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[21];
h qubits[4];
h qubits[9];
h qubits[15];
h qubits[22];
h qubits[10];
h qubits[16];
h qubits[23];
h qubits[4];
ccx qubits[2],qubits[3],qubits[4];
h qubits[4];
h qubits[9];
ccx qubits[7],qubits[8],qubits[9];
h qubits[9];
h qubits[10];
ccx qubits[7],qubits[8],qubits[10];
h qubits[10];
h qubits[15];
ccx qubits[13],qubits[14],qubits[15];
h qubits[15];
h qubits[16];
ccx qubits[13],qubits[14],qubits[16];
h qubits[16];
h qubits[22];
ccx qubits[20],qubits[21],qubits[22];
h qubits[22];
h qubits[23];
ccx qubits[20],qubits[21],qubits[23];
h qubits[23];
cx qubits[6],qubits[5];
cx qubits[12],qubits[11];
cx qubits[19],qubits[18];
cx qubits[5],qubits[8];
cx qubits[11],qubits[14];
cx qubits[18],qubits[21];
h qubits[10];
ccx qubits[7],qubits[8],qubits[10];
h qubits[10];
h qubits[16];
ccx qubits[13],qubits[14],qubits[16];
h qubits[16];
h qubits[23];
ccx qubits[20],qubits[21],qubits[23];
h qubits[23];
h qubits[4];
h qubits[10];
h qubits[15];
h qubits[16];
h qubits[23];
h qubits[17];
h qubits[17];
ccx qubits[16],qubits[23],qubits[17];
h qubits[17];
h qubits[22];
ccx qubits[15],qubits[23],qubits[22];
h qubits[22];
h qubits[9];
ccx qubits[4],qubits[10],qubits[9];
h qubits[9];
h qubits[17];
h qubits[9];
h qubits[15];
h qubits[22];
ccx qubits[9],qubits[17],qubits[22];
h qubits[22];
h qubits[15];
ccx qubits[9],qubits[16],qubits[15];
h qubits[15];
h qubits[15];
h qubits[22];
h qubits[17];
h qubits[17];
ccx qubits[16],qubits[23],qubits[17];
h qubits[17];
h qubits[17];
h qubits[10];
h qubits[16];
h qubits[23];
h qubits[10];
ccx qubits[7],qubits[8],qubits[10];
h qubits[10];
h qubits[16];
ccx qubits[13],qubits[14],qubits[16];
h qubits[16];
h qubits[23];
ccx qubits[20],qubits[21],qubits[23];
h qubits[23];
cx qubits[5],qubits[8];
cx qubits[11],qubits[14];
cx qubits[18],qubits[21];
cx qubits[6],qubits[5];
cx qubits[12],qubits[11];
cx qubits[19],qubits[18];
h qubits[10];
ccx qubits[7],qubits[8],qubits[10];
h qubits[10];
h qubits[16];
ccx qubits[13],qubits[14],qubits[16];
h qubits[16];
h qubits[23];
ccx qubits[20],qubits[21],qubits[23];
h qubits[23];
h qubits[23];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[21];
h qubits[3];
ccx qubits[0],qubits[1],qubits[3];
h qubits[3];
h qubits[8];
ccx qubits[5],qubits[6],qubits[8];
h qubits[8];
h qubits[14];
ccx qubits[11],qubits[12],qubits[14];
h qubits[14];
h qubits[21];
ccx qubits[18],qubits[19],qubits[21];
h qubits[21];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[21];
cx qubits[3],qubits[2];
cx qubits[8],qubits[7];
cx qubits[14],qubits[13];
cx qubits[21],qubits[20];
cx qubits[6],qubits[5];
cx qubits[12],qubits[11];
cx qubits[19],qubits[18];
cx qubits[6],qubits[8];
cx qubits[12],qubits[14];
cx qubits[19],qubits[21];
cx qubits[4],qubits[6];
cx qubits[9],qubits[12];
cx qubits[15],qubits[19];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[21];
h qubits[3];
ccx qubits[0],qubits[1],qubits[3];
h qubits[3];
h qubits[8];
ccx qubits[5],qubits[6],qubits[8];
h qubits[8];
h qubits[14];
ccx qubits[11],qubits[12],qubits[14];
h qubits[14];
h qubits[21];
ccx qubits[18],qubits[19],qubits[21];
h qubits[21];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[21];
cx qubits[3],qubits[2];
cx qubits[8],qubits[7];
cx qubits[14],qubits[13];
cx qubits[21],qubits[20];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[21];
h qubits[3];
ccx qubits[0],qubits[1],qubits[3];
h qubits[3];
h qubits[8];
ccx qubits[5],qubits[6],qubits[8];
h qubits[8];
h qubits[14];
ccx qubits[11],qubits[12],qubits[14];
h qubits[14];
h qubits[21];
ccx qubits[18],qubits[19],qubits[21];
h qubits[21];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[21];
cx qubits[6],qubits[5];
cx qubits[12],qubits[11];
cx qubits[19],qubits[18];
cx qubits[4],qubits[6];
cx qubits[9],qubits[12];
cx qubits[15],qubits[19];
cx qubits[6],qubits[8];
cx qubits[12],qubits[14];
cx qubits[19],qubits[21];
cx qubits[1],qubits[0];
cx qubits[6],qubits[5];
cx qubits[12],qubits[11];
cx qubits[19],qubits[18];
x qubits[0];
x qubits[2];
x qubits[5];
x qubits[7];
x qubits[11];
x qubits[13];
cx qubits[3],qubits[2];
cx qubits[8],qubits[7];
cx qubits[14],qubits[13];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[3];
ccx qubits[0],qubits[1],qubits[3];
h qubits[3];
h qubits[8];
ccx qubits[5],qubits[6],qubits[8];
h qubits[8];
h qubits[14];
ccx qubits[11],qubits[12],qubits[14];
h qubits[14];
h qubits[3];
h qubits[8];
h qubits[14];
cx qubits[6],qubits[5];
cx qubits[12],qubits[11];
h qubits[10];
ccx qubits[7],qubits[8],qubits[10];
h qubits[10];
h qubits[16];
ccx qubits[13],qubits[14],qubits[16];
h qubits[16];
cx qubits[5],qubits[8];
cx qubits[11],qubits[14];
h qubits[10];
ccx qubits[7],qubits[8],qubits[10];
h qubits[10];
h qubits[16];
ccx qubits[13],qubits[14],qubits[16];
h qubits[16];
h qubits[10];
h qubits[16];
h qubits[15];
h qubits[15];
ccx qubits[9],qubits[16],qubits[15];
h qubits[15];
h qubits[9];
h qubits[9];
ccx qubits[4],qubits[10],qubits[9];
h qubits[9];
h qubits[4];
h qubits[10];
h qubits[16];
h qubits[10];
ccx qubits[7],qubits[8],qubits[10];
h qubits[10];
h qubits[16];
ccx qubits[13],qubits[14],qubits[16];
h qubits[16];
cx qubits[5],qubits[8];
cx qubits[11],qubits[14];
h qubits[10];
ccx qubits[7],qubits[8],qubits[10];
h qubits[10];
h qubits[16];
ccx qubits[13],qubits[14],qubits[16];
h qubits[16];
cx qubits[6],qubits[5];
cx qubits[12],qubits[11];
h qubits[9];
ccx qubits[7],qubits[8],qubits[9];
h qubits[9];
h qubits[15];
ccx qubits[13],qubits[14],qubits[15];
h qubits[15];
h qubits[4];
ccx qubits[2],qubits[3],qubits[4];
h qubits[4];
h qubits[4];
h qubits[9];
h qubits[10];
h qubits[15];
h qubits[16];
h qubits[3];
h qubits[8];
h qubits[14];
h qubits[3];
ccx qubits[0],qubits[1],qubits[3];
h qubits[3];
h qubits[8];
ccx qubits[5],qubits[6],qubits[8];
h qubits[8];
h qubits[14];
ccx qubits[11],qubits[12],qubits[14];
h qubits[14];
h qubits[3];
h qubits[8];
h qubits[14];
cx qubits[3],qubits[4];
cx qubits[8],qubits[9];
cx qubits[14],qubits[15];
cx qubits[3],qubits[2];
cx qubits[8],qubits[7];
cx qubits[14],qubits[13];
x qubits[0];
x qubits[2];
x qubits[5];
x qubits[7];
x qubits[11];
x qubits[13];
