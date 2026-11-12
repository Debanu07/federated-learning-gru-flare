import sys, os
sys.path.insert(0, os.getcwd())

from nvflare.job_config.api import FedJob
from nvflare.app_common.workflows.fedavg import FedAvg

def main():
    job = FedJob(name="mnist_flare")

    controller = FedAvg(
        num_clients=2,
        num_rounds=3,
    )

    job.to_server(controller)

    try:
        from nvflare.job_config.script_runner import ScriptRunner
    except ImportError:
        from nvflare.app_common.executors.script_runner import ScriptRunner

    runner = ScriptRunner(script="flare_train.py")
    job.to_clients(runner)

    job.simulator_run(
        workspace=os.path.expanduser("~/flare_workspace"),  # ✅ Linux path
        n_clients=2,
        threads=2,
    )

if __name__ == "__main__":
    main()