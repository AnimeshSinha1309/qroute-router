OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
cx q[15],q[7];
cx q[9],q[3];