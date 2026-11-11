"""
Submit job.qasm to IQM Garnet through qBraid. THIS SPENDS CREDITS.

    pip install qbraid
    export QBRAID_API_KEY=...        # your key; never commit it
    python hardware_runs/proposed_garnet_run/submit.py            # shows status and cost, asks before submitting
    python hardware_runs/proposed_garnet_run/submit.py --fetch    # collect the result of the job already submitted
    python hardware_runs/proposed_garnet_run/submit.py --as-circuit   # only if the service rejects the QASM text:
                                                                      # sends the same circuit as a Qiskit object,
                                                                      # which is how the June job was sent

The job ID is written to submitted_job.json as soon as the job is accepted, so an
interrupted run can be picked up with --fetch without paying twice.
"""
import json
import os
import sys

DEVICE = "aws:iqm:qpu:garnet"
SHOTS = 1000
# Observed on the job of 1 June 2026: 320 credits for 2,000 shots = 30 per job + 0.145 per shot.
PER_JOB, PER_SHOT = 30.0, 0.145

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "submitted_job.json")
RESULT = os.path.join(HERE, "result.json")


def qasm_to_qiskit(text):
    """The gates in job.qasm (h, rzz, rz, rx, measure) as a Qiskit circuit."""
    import re
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(8, 8)
    for line in text.splitlines():
        m = re.match(r"(h|rzz|rz|rx)(?:\(([-\d.e]+)\))? q\[(\d)\](?:, q\[(\d)\])?;", line.strip())
        if m:
            gate, theta, a, b = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            if gate == "h":
                qc.h(a)
            elif gate == "rzz":
                qc.rzz(float(theta), a, int(b))
            else:
                getattr(qc, gate)(float(theta), a)
        m = re.match(r"c\[(\d)\] = measure q\[(\d)\];", line.strip())
        if m:
            qc.measure(int(m.group(2)), int(m.group(1)))
    return qc


def save_result(job):
    job.wait_for_final_state()
    status = job.status()
    print("final status:", status)
    counts = job.result().data.get_counts()
    json.dump({"job_id": job.id, "device": DEVICE, "shots": sum(counts.values()), "counts": counts},
              open(RESULT, "w"), indent=1)
    print(f"saved {RESULT}\nnext: python hardware_runs/proposed_garnet_run/score.py {RESULT}")


def main():
    from qbraid.runtime import QbraidProvider

    key = os.environ.get("QBRAID_API_KEY")
    if not key:
        sys.exit("Set QBRAID_API_KEY first.")
    device = QbraidProvider(api_key=key).get_device(DEVICE)

    if "--fetch" in sys.argv:
        if not os.path.exists(RECORD):
            sys.exit("No submitted_job.json: nothing has been submitted from here.")
        job_id = json.load(open(RECORD))["job_id"]
        from qbraid.runtime import QbraidJob
        return save_result(QbraidJob(job_id, device=device))

    if os.path.exists(RECORD):
        sys.exit(f"A job was already submitted ({json.load(open(RECORD))['job_id']}). "
                 "Use --fetch, or delete submitted_job.json if you really mean to pay for another.")

    status = device.status()
    print("device:", DEVICE, "| status:", status)
    if "ONLINE" not in str(status).upper() and "--force" not in sys.argv:
        sys.exit("The device is not online, so nothing was submitted. Try again later "
                 "(or pass --force to queue the job anyway).")
    try:
        print("jobs queued:", device.queue_depth())
    except Exception as e:                                   # not every device reports this
        print("queue depth unavailable:", e)
    try:
        pricing = device.metadata().get("pricing")
        print("pricing reported by qBraid:", pricing)
    except Exception as e:
        print("pricing unavailable:", e)
    print(f"estimate from the June job's price: {PER_JOB + PER_SHOT * SHOTS:.0f} credits for {SHOTS} shots. "
          "Check this against your balance and the pricing above.")

    if input(f"Type 'submit' to send {SHOTS} shots to {DEVICE}: ").strip() != "submit":
        sys.exit("Nothing submitted.")

    program = open(os.path.join(HERE, "job.qasm")).read()
    if "--as-circuit" in sys.argv:
        program = qasm_to_qiskit(program)
    try:
        job = device.run(program, shots=SHOTS)
    except Exception as e:
        sys.exit(f"The service rejected the submission: {e}\n"
                 "Check the qBraid jobs page to confirm nothing was created before trying again. "
                 "If the error is about the program format, rerun with --as-circuit.")
    if isinstance(job, list):
        job = job[0]
    print("submitted:", job.id)
    json.dump({"job_id": job.id, "device": DEVICE, "shots": SHOTS}, open(RECORD, "w"))
    save_result(job)


if __name__ == "__main__":
    main()
