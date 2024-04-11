"""
This takes an MCLI run config and attempts to manage the run as follows:
- If a run with the same name on the specified cluster is already running or queued, it does nothing.
- If there's enough nodes available to run the job, it submits and monitors a light-weight test run on each
  available node to determine which nodes are working properly.
- If there's enough working nodes it will launch a new run on a subset of the working nodes.
"""

import argparse
from concurrent.futures import as_completed
from typing import List, Optional, Set

import mcli
import mcli.api.runs
import yaml
from mcli.api.model.cluster_details import Instance
from mcli.api.runs import Run, RunConfig, RunStatus
from rich import print
from rich.progress import track
from rich.prompt import Confirm

_SKIP_CONFIRMATION = False


def get_test_config(
    *, cluster_name: str, image_name: str, node_name: str, instance_name: Optional[str] = None
) -> RunConfig:
    """
    Get a run config for testing if a node is working properly.
    """
    run_config = RunConfig(
        name="test-run",
        image=image_name,
        compute=dict(cluster=cluster_name, nodes=1, node_names=[node_name]),  # type: ignore
        command='''python -c "import torch; torch.rand(2, 3).cuda() @ torch.rand(3, 2).cuda(); print('All good!')"''',
    )
    if instance_name is not None:
        run_config.compute["instance"] = instance_name
    return run_config


def submit_runs(run_configs: List[RunConfig], timeout: int = 30) -> List[Run]:
    """
    Submit a list of runs.
    """
    futures = []
    for run_config in run_configs:
        futures.append(mcli.api.runs.create_run(run_config, future=True))

    runs = []
    for future in track(
        as_completed(futures, timeout=timeout), total=len(futures), description="Submitting runs..."
    ):
        runs.append(future.result())

    return runs


def wait_on_runs(runs: List[Run], timeout: int = 360) -> List[Run]:
    """
    Wait on a list of runs to reach 'COMPLETED' status (or a failure of some kind).
    """
    futures = []
    for run in runs:
        futures.append(mcli.api.runs.wait_for_run_status(run, RunStatus.COMPLETED, future=True))

    results = []
    for future in track(
        as_completed(futures, timeout=timeout), total=len(futures), description="Waiting on runs..."
    ):
        results.append(future.result())

    return results


def identify_bad_nodes(
    *,
    available_nodes: Set[str],
    cluster_name: str,
    image_name: str,
    instance_name: Optional[str] = None,
    timeout: int = 360,
) -> Set[str]:
    """
    Identify faulty nodes from a set of nodes on a cluster.
    """
    bad_nodes = set()
    test_runs = submit_runs(
        [
            get_test_config(
                cluster_name=cluster_name, image_name=image_name, node_name=node_name, instance_name=instance_name
            )
            for node_name in available_nodes
        ]
    )

    try:
        test_runs = wait_on_runs(test_runs, timeout=timeout)
    except BaseException:
        print("Stopping test runs due to error...")
        mcli.api.runs.stop_runs(test_runs)
        raise

    for run in test_runs:
        assert len(run.nodes) == 1
        node_name = run.nodes[0].name
        if run.status in {RunStatus.FAILED, RunStatus.UNKNOWN, RunStatus.STOPPED}:
            bad_nodes.add(node_name)
            print(f"  ✖️ '{node_name}' {run.status} (run '{run.name}')")
        elif run.status in {RunStatus.COMPLETED}:
            print(f"  ✔️ '{node_name}' {run.status} (run '{run.name}')")
        else:
            print(f"  ❔ '{node_name}' {run.status} (run '{run.name}')")

    return bad_nodes


def confirm_continue(prompt: str) -> bool:
    if _SKIP_CONFIRMATION:
        print(prompt)
        return True
    else:
        return Confirm.ask(f"{prompt} Continue?")


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    cluster_name = config["compute"]["cluster"]
    instance_name = config["compute"].get("instance")
    image_name = config["image"]
    run_prefix = config["name"]
    gpus_required = config["compute"]["gpus"]

    cluster = mcli.get_cluster(cluster_name)
    assert cluster.utilization is not None

    # Check if config is already running or queued.
    for run in cluster.utilization.active_runs_by_user:
        if run.name.startswith(run_prefix):
            print(f"Run '{run.name}' is already active")
            return
    for run in cluster.utilization.queued_runs_by_user:
        if run.name.startswith(run_prefix):
            print(f"Run '{run.name}' is already queued")
            return

    # Collect instance data.
    instance: Optional[Instance] = None
    for instance_util in cluster.utilization.cluster_instance_utils:
        if instance_name is None or instance_util.instance.name == instance_name:
            instance = instance_util.instance
            break
    assert instance is not None
    assert gpus_required % instance.gpus == 0
    nodes_required = gpus_required // instance.gpus

    # Gather all nodes.
    all_nodes = set()
    for node in instance.node_details:
        all_nodes.add(node.name)

    print(f"There are {len(all_nodes)} total nodes")

    if nodes_required > len(all_nodes):
        raise RuntimeError(f"Not enough nodes to meet requirement of {nodes_required} ({gpus_required} GPUs)")

    # Filter out nodes that already have a job.
    available_nodes = all_nodes.copy()
    for run in cluster.utilization.active_runs_by_user:
        run = mcli.get_run(run.name)
        for node in run.nodes:
            if node.name in available_nodes:
                available_nodes.remove(node.name)

    print(f"There are {len(available_nodes)} available nodes")

    if nodes_required > len(available_nodes):
        raise RuntimeError(
            f"Not enough nodes available to meet requirement of {nodes_required} ({gpus_required} GPUs)"
        )

    if not confirm_continue("Submitting test runs to determine working nodes..."):
        return
    bad_nodes = identify_bad_nodes(
        available_nodes=available_nodes,
        cluster_name=cluster_name,
        image_name=image_name,
        instance_name=instance_name,
    )
    print(f"Identified {len(bad_nodes)} bad nodes")

    # Gather all working nodes.
    working_nodes = set()
    for node in available_nodes:
        if node not in bad_nodes:
            working_nodes.add(node)

    print(f"There are {len(working_nodes)} working available nodes")

    if nodes_required > len(working_nodes):
        raise RuntimeError(
            f"Not enough working nodes available to meet requirement of {nodes_required} ({gpus_required} GPUs)"
        )

    # Initialize run config to submit.
    run_config = RunConfig(**config)
    run_config.compute["node_names"] = list(working_nodes)[:nodes_required]

    # Submit job.
    if not confirm_continue("Launching new run..."):
        return
    run = mcli.create_run(run_config, timeout=30)
    print(f"Launched new run '{run.name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mcli-run-manager")
    parser.add_argument("run_config")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    args = parser.parse_args()
    if args.yes:
        _SKIP_CONFIRMATION = True

    main(args.run_config)
