from qbraid.runtime import QbraidProvider

provider = QbraidProvider()
job = provider.get_device("aws:iqm:qpu:emerald").get_job(
    "aws:iqm:qpu:emerald-eecd-qjob-6a1cfc993d86f02c1354cb85")
print(job.status())
if job.status().name == "COMPLETED":
    print(job.result().data.get_counts())
