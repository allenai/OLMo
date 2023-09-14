import platform
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union, cast

import requests
import rich
from beaker import (
    Beaker,
    Dataset,
    DatasetConflict,
    DatasetNotFound,
    Digest,
    ExperimentSpec,
    Priority,
    SecretNotFound,
    TaskResources,
    TaskSpec,
    WorkspaceNotSet,
)
from rich import print, prompt
from rich.console import Console

from . import constants
from .aliases import PathOrStr
from .constants import GITHUB_TOKEN_SECRET
from .exceptions import *
from .version import VERSION

if TYPE_CHECKING:
    from datetime import timedelta


def unique_name() -> str:
    import uuid

    import petname

    return cast(str, petname.generate()) + "-" + str(uuid.uuid4())[:7]


def stderr_console() -> Console:
    return Console(stderr=True)


def print_stderr(*args, **kwargs):
    stderr_console().print(*args, **kwargs)


def print_exception(*args, **kwargs):
    stderr_console().print_exception(*args, **kwargs)


def parse_git_remote_url(url: str) -> Tuple[str, str]:
    """
    Parse a git remote URL into a GitHub (account, repo) pair.

    :raises InvalidRemoteError: If the URL can't be parsed correctly.
    """
    try:
        account, repo = (
            url.split("https://github.com/")[-1].split("git@github.com:")[-1].split(".git")[0].split("/")
        )
    except ValueError:
        raise InvalidRemoteError(f"Failed to parse GitHub repo path from remote '{url}'")
    return account, repo


def display_logs(logs: Iterable[bytes], ignore_timestamp: Optional[str] = None) -> Optional[str]:
    console = rich.get_console()
    latest_timestamp: Optional[str] = None

    def print_line(line: str):
        if not line:
            return
        nonlocal latest_timestamp
        # Remove timestamp
        try:
            timestamp, line = line.split("Z ", maxsplit=1)
            latest_timestamp = f"{timestamp}Z"
            if ignore_timestamp is not None and latest_timestamp == ignore_timestamp:
                return
        except ValueError:
            pass
        console.print(line, highlight=False, markup=False)

    line_buffer = ""
    for bytes_chunk in logs:
        chunk = line_buffer + bytes_chunk.decode(errors="ignore")
        chunk = chunk.replace("\r", "\n")
        lines = chunk.split("\n")
        if chunk.endswith("\n"):
            line_buffer = ""
        else:
            # Last line chunk is probably incomplete.
            lines, line_buffer = lines[:-1], lines[-1]
        for line in lines:
            print_line(line)

    print_line(line_buffer)
    return latest_timestamp


def ensure_repo(allow_dirty: bool = False) -> Tuple[str, str, str, bool]:
    from git.repo import Repo

    repo = Repo(".")
    if repo.is_dirty() and not allow_dirty:
        raise DirtyRepoError("You have uncommitted changes! Use --allow-dirty to force.")
    git_ref = str(repo.commit())
    account, repo = parse_git_remote_url(repo.remote().url)
    response = requests.get(f"https://github.com/{account}/{repo}")
    if response.status_code not in {200, 404}:
        response.raise_for_status()
    is_public = response.status_code == 200
    return account, repo, git_ref, is_public


def ensure_entrypoint_dataset(beaker: Beaker) -> Dataset:
    import hashlib
    from importlib.resources import read_binary

    import eb_gantry

    workspace_id = beaker.workspace.get().id

    # Get hash of the local entrypoint source file.
    sha256_hash = hashlib.sha256()
    contents = read_binary(eb_gantry, constants.ENTRYPOINT)
    tag_start = contents.find(b"${{")
    while tag_start != -1:
        tag_end = contents.find(b"}}") + 2
        tag = contents[tag_start:tag_end]
        constant_name = tag.split(b" ")[1].decode()
        contents = contents.replace(tag, getattr(constants, constant_name).encode())  # type: ignore
        tag_start = contents.find(b"${{", tag_end)
    assert b"${{" not in contents
    sha256_hash.update(contents)

    entrypoint_dataset_name = f"gantry-v{VERSION}-{workspace_id}-{sha256_hash.hexdigest()[:6]}"

    # Ensure gantry entrypoint dataset exists.
    gantry_entrypoint_dataset: Dataset
    try:
        gantry_entrypoint_dataset = beaker.dataset.get(entrypoint_dataset_name)
    except DatasetNotFound:
        # Create it.
        print(f"Creating entrypoint dataset '{entrypoint_dataset_name}'")
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdir = Path(tmpdirname)
                entrypoint_path = tmpdir / constants.ENTRYPOINT
                with open(entrypoint_path, "wb") as entrypoint_file:
                    entrypoint_file.write(contents)
                gantry_entrypoint_dataset = beaker.dataset.create(entrypoint_dataset_name, entrypoint_path)
        except DatasetConflict:  # could be in a race with another `gantry` process.
            time.sleep(1.0)
            gantry_entrypoint_dataset = beaker.dataset.get(entrypoint_dataset_name)

    # Verify contents.
    err_msg = (
        f"Checksum failed for entrypoint dataset {beaker.dataset.url(gantry_entrypoint_dataset)}\n"
        f"This could be a bug, or it could mean someone has tampered with the dataset.\n"
        f"If you're sure no one has tampered with it, you can delete the dataset from "
        f"the Beaker dashboard and try again."
    )
    ds_files = list(beaker.dataset.ls(gantry_entrypoint_dataset))
    if len(ds_files) != 1:
        raise EntrypointChecksumError(err_msg)
    if ds_files[0].digest != Digest.from_decoded(sha256_hash.digest(), "SHA256"):
        raise EntrypointChecksumError(err_msg)

    return gantry_entrypoint_dataset


def ensure_github_token_secret(beaker: Beaker, secret_name: str = constants.GITHUB_TOKEN_SECRET) -> str:
    try:
        beaker.secret.get(secret_name)
    except SecretNotFound:
        raise GitHubTokenSecretNotFound(
            f"GitHub token secret '{secret_name}' not found in Beaker workspace!\n"
            f"You can create a suitable GitHub token by going to https://github.com/settings/tokens/new "
            f"and generating a token with '\N{ballot box with check} repo' scope.\n"
            f"Then upload your token as a Beaker secret using the Beaker CLI or Python client."
        )
    return secret_name


def format_timedelta(td: "timedelta") -> str:
    def format_value_and_unit(value: int, unit: str) -> str:
        if value == 1:
            return f"{value} {unit}"
        else:
            return f"{value} {unit}s"

    parts = []
    seconds = int(td.total_seconds())
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        parts.append(format_value_and_unit(days, "day"))
    if hours:
        parts.append(format_value_and_unit(hours, "hour"))
    if minutes:
        parts.append(format_value_and_unit(minutes, "minute"))
    if seconds:
        parts.append(format_value_and_unit(seconds, "second"))
    return ", ".join(parts)


def ensure_datasets(beaker: Beaker, *datasets: str) -> List[Tuple[str, str]]:
    out = []
    for dataset_str in datasets:
        try:
            dataset_name, path = dataset_str.split(":")
        except ValueError:
            raise ValueError(
                f"Bad '--dataset' specification: '{dataset_str}'\n"
                f"Datasets should be in the form of 'dataset-name:/mount/location'."
            )
        dataset_id = beaker.dataset.get(dataset_name).id
        out.append((dataset_id, path))
    return out


def build_experiment_spec(
    task_name: str,
    clusters: List[str],
    task_resources: TaskResources,
    arguments: List[str],
    entrypoint_dataset: str,
    github_account: str,
    github_repo: str,
    git_ref: str,
    description: Optional[str] = None,
    beaker_image: Optional[str] = None,
    docker_image: Optional[str] = None,
    gh_token_secret: Optional[str] = constants.GITHUB_TOKEN_SECRET,
    conda: Optional[PathOrStr] = None,
    pip: Optional[PathOrStr] = None,
    venv: Optional[str] = None,
    nfs: Optional[bool] = None,
    datasets: Optional[List[Tuple[str, str]]] = None,
    env: Optional[List[Tuple[str, str]]] = None,
    env_secrets: Optional[List[Tuple[str, str]]] = None,
    priority: Optional[Union[str, Priority]] = None,
    install: Optional[str] = None,
    replicas: Optional[int] = None,
    leader_selection: bool = False,
    host_networking: bool = False,
    mounts: Optional[List[Tuple[str, str]]] = None,
):
    task_spec = (
        TaskSpec.new(
            task_name,
            beaker_image=beaker_image,
            docker_image=docker_image,
            result_path=constants.RESULTS_DIR,
            command=["bash", "/gantry/entrypoint.sh"],
            arguments=arguments,
            resources=task_resources,
            priority=priority,
            replicas=replicas,
            leader_selection=leader_selection,
            host_networking=host_networking,
        )
        .with_constraint(cluster=clusters)
        .with_env_var(name="GANTRY_VERSION", value=VERSION)
        .with_env_var(name="GITHUB_REPO", value=f"{github_account}/{github_repo}")
        .with_env_var(name="GIT_REF", value=git_ref)
        .with_dataset("/gantry", beaker=entrypoint_dataset)
    )

    if gh_token_secret is not None:
        task_spec = task_spec.with_env_var(name="GITHUB_TOKEN", secret=gh_token_secret)

    for name, val in env or []:
        task_spec = task_spec.with_env_var(name=name, value=val)

    for name, secret in env_secrets or []:
        task_spec = task_spec.with_env_var(name=name, secret=secret)

    if conda is not None:
        task_spec = task_spec.with_env_var(
            name="CONDA_ENV_FILE",
            value=str(conda),
        )
    elif Path(constants.CONDA_ENV_FILE).is_file():
        task_spec = task_spec.with_env_var(
            name="CONDA_ENV_FILE",
            value=constants.CONDA_ENV_FILE,
        )
    elif Path(constants.CONDA_ENV_FILE_ALTERNATE).is_file():
        task_spec = task_spec.with_env_var(
            name="CONDA_ENV_FILE",
            value=constants.CONDA_ENV_FILE_ALTERNATE,
        )
    else:
        task_spec = task_spec.with_env_var(
            name="PYTHON_VERSION", value=".".join(platform.python_version_tuple()[:-1])
        )

    if pip is not None:
        task_spec = task_spec.with_env_var(
            name="PIP_REQUIREMENTS_FILE",
            value=str(pip),
        )

    if venv is not None:
        task_spec = task_spec.with_env_var(
            name="VENV_NAME",
            value=venv,
        )

    if install is not None:
        task_spec = task_spec.with_env_var(name="INSTALL_CMD", value=install)

    if nfs is None and clusters and all(["cirrascale" in cluster for cluster in clusters]):
        nfs = True

    if nfs:
        task_spec = task_spec.with_dataset(constants.NFS_MOUNT, host_path=constants.NFS_MOUNT)

    if datasets:
        for dataset_id, path in datasets:
            task_spec = task_spec.with_dataset(path, beaker=dataset_id)
    if mounts:
        for source, target in mounts:
            task_spec = task_spec.with_dataset(target, host_path=source)

    return ExperimentSpec(description=description, tasks=[task_spec])


def check_for_upgrades():
    import packaging.version
    import requests

    try:
        response = requests.get("https://api.github.com/repos/allenai/beaker-gantry/releases/latest", timeout=1)
        if response.ok:
            latest_version = packaging.version.parse(response.json()["tag_name"])
            if latest_version > packaging.version.parse(VERSION):
                print_stderr(
                    f":warning: [yellow]You're using [b]gantry v{VERSION}[/], "
                    f"but a newer version ([b]v{latest_version}[/]) is available: "
                    f"https://github.com/allenai/beaker-gantry/releases/tag/v{latest_version}[/]\n"
                    f"[yellow i]You can upgrade by running:[/] pip install --upgrade beaker-gantry\n",
                )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        pass


def ensure_workspace(
    workspace: Optional[str] = None,
    yes: bool = False,
    gh_token_secret: str = GITHUB_TOKEN_SECRET,
    public_repo: bool = False,
) -> Beaker:
    beaker = (
        Beaker.from_env(session=True)
        if workspace is None
        else Beaker.from_env(session=True, default_workspace=workspace)
    )
    try:
        permissions = beaker.workspace.get_permissions()
        if not public_repo and len(permissions.authorizations) > 1:
            print_stderr(
                f"[yellow]Your workspace [b]{beaker.workspace.url()}[/] has multiple contributors! "
                f"Every contributor can view your GitHub personal access token secret ('{gh_token_secret}').[/]"
            )
            if not yes and not prompt.Confirm.ask("[yellow][i]Are you sure you want to use this workspace?[/][/]"):
                raise KeyboardInterrupt
        elif workspace is None:
            default_workspace = beaker.workspace.get()
            if not yes and not prompt.Confirm.ask(
                f"Using default workspace [b cyan]{default_workspace.full_name}[/]. [i]Is that correct?[/]"
            ):
                raise KeyboardInterrupt
    except WorkspaceNotSet:
        raise ConfigurationError("'--workspace' option is required since you don't have a default workspace set")
    return beaker
