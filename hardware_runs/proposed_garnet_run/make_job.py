"""Write job.qasm for candidate A from the angles in candidates.json (run prepare.py first)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prepare  # noqa: E402

params = json.load(open(os.path.join(HERE, "candidates.json")))["A: plain QAOA, fixed cost, 1 layer"]["params"]
qc = prepare.build("plain", params)

lines = ["OPENQASM 3.0;", 'include "stdgates.inc";',
         "gate rzz(p0) _gate_q_0, _gate_q_1 {", "  cx _gate_q_0, _gate_q_1;",
         "  rz(p0) _gate_q_1;", "  cx _gate_q_0, _gate_q_1;", "}",
         f"bit[{prepare.N}] c;", f"qubit[{prepare.N}] q;"]
for inst in qc.data:
    name = inst.operation.name
    qs = [qc.find_bit(q).index for q in inst.qubits]
    if name == "h":
        lines.append(f"h q[{qs[0]}];")
    elif name in ("rz", "rx"):
        lines.append(f"{name}({float(inst.operation.params[0])!r}) q[{qs[0]}];")
    elif name == "rzz":
        lines.append(f"rzz({float(inst.operation.params[0])!r}) q[{qs[0]}], q[{qs[1]}];")
    elif name == "measure":
        lines.append(f"c[{qs[0]}] = measure q[{qs[0]}];")
    else:
        raise ValueError(f"unexpected gate {name}")
open(os.path.join(HERE, "job.qasm"), "w").write("\n".join(lines) + "\n")
print(f"wrote job.qasm: {len(lines)} lines, {qc.count_ops()}")
