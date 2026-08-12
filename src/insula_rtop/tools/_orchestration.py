"""Shared submitit dispatch helpers used by every per-subject CLI in this package.

Adapted from ``spherical_integral_gnn.tools._submitit_orchestration``.

``submitit``'s per-job ``executor.submit()`` calls are independent SLURM jobs;
unlike ``executor.map_array``, ``array_parallelism`` does *not* throttle them
(it only limits array-job task concurrency). Submitting hundreds of individual
jobs "unthrottled" onto a shared cluster can starve other users and cause
widespread timeouts under contention. :func:`submit_with_concurrency_cap`
provides a real sliding-window cap via polling; :func:`run_subject_jobs` wraps
it together with the resume-filtering / local-vs-SLURM-executor boilerplate.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SlurmOptions:
    """How per-subject work is dispatched.

    These eight settings always travel together -- from the CLI or Hydra config,
    through each step's ``run_*`` wrapper, into :func:`run_subject_jobs` -- and
    none of them mean anything on their own. Passing them as one object keeps
    the wrappers' signatures about the *step*, not about SLURM.
    """

    slurm: bool = False
    n_jobs: int = 1
    partition: str | None = None
    account: str | None = None
    time: int = 60
    cpus_per_task: int = 1
    mem: str | None = None
    max_jobs: int | None = None

    @classmethod
    def from_args(cls, args) -> SlurmOptions:
        """Build from the flags :func:`add_common_orchestration_args` adds."""
        return cls(
            slurm=args.slurm and not args.local,
            n_jobs=args.n_jobs,
            partition=args.slurm_partition,
            account=args.slurm_account,
            time=args.slurm_time,
            cpus_per_task=args.slurm_cpus_per_task,
            mem=args.slurm_mem,
            max_jobs=args.slurm_max_jobs,
        )


def skip_completed(
    complete: bool,
    subject_id: str,
    output: Path,
    *,
    force: bool,
    skip_existing: bool,
) -> bool:
    """The force / skip_existing / refuse contract, shared by every step.

    Returns True when the caller should return early. Refusing by default --
    rather than silently overwriting or silently skipping -- is deliberate: a
    re-run that quietly did nothing is indistinguishable from one that worked.
    """
    if not complete:
        return False
    if skip_existing:
        print(f"[{subject_id}] output exists, skipping.")
        return True
    if force:
        return False
    raise FileExistsError(
        f"[{subject_id}] {output} already exists. Use force=True to overwrite "
        "or skip_existing=True to skip."
    )


def submit_with_concurrency_cap(
    executor,
    fn,
    kwargs_list: list[dict],
    max_concurrent: int | None = None,
    poll_interval: float = 5.0,
) -> list[tuple[dict, Exception | None]]:
    """Submit ``fn(**kwargs)`` for every ``kwargs`` in *kwargs_list*.

    If *max_concurrent* is set, at most that many jobs are in flight at once
    (a new job is submitted only once an earlier one finishes). If None, every
    job is submitted immediately (submitit's own default behavior).

    Returns
    -------
    list of (kwargs, exception)
        In completion order; ``exception`` is None on success.
    """
    if not max_concurrent:
        jobs = [(kw, executor.submit(fn, **kw)) for kw in kwargs_list]
        results = []
        for kw, job in jobs:
            try:
                job.result()
                results.append((kw, None))
            except Exception as e:
                results.append((kw, e))
        return results

    pending = list(kwargs_list)
    in_flight: list[tuple[dict, object]] = []
    results = []
    while pending or in_flight:
        while pending and len(in_flight) < max_concurrent:
            kw = pending.pop(0)
            in_flight.append((kw, executor.submit(fn, **kw)))

        done_now = [item for item in in_flight if item[1].done()]
        if not done_now:
            time.sleep(poll_interval)
            continue

        for item in done_now:
            in_flight.remove(item)
            kw, job = item
            try:
                job.result()
                results.append((kw, None))
            except Exception as e:
                results.append((kw, e))
    return results


def run_subject_jobs(
    subject_kwargs: list[dict],
    fn,
    log_dir: Path,
    *,
    resume: bool = False,
    is_complete: Callable[[dict], bool] | None = None,
    options: SlurmOptions = SlurmOptions(),
) -> tuple[int, int]:
    """Dispatch ``fn(**kw)`` for every ``kw`` in *subject_kwargs*, then report.

    Filters already-complete subjects when *resume* is set (via *is_complete*),
    runs a plain Python loop when neither SLURM nor local multiprocessing is
    requested -- which is what makes local and CI runs debuggable -- and
    otherwise builds a submitit executor and dispatches through
    :func:`submit_with_concurrency_cap`.

    Parameters
    ----------
    subject_kwargs : list of dict
        Keyword arguments for one call to *fn* per subject; each dict must
        include a ``"subject_id"`` key (used for progress/error messages).
    fn : callable
        Per-subject function, called as ``fn(**kw)``.
    log_dir : Path
        submitit log folder (only used when dispatching via an executor).
    is_complete : callable, optional
        ``is_complete(kw) -> bool``, used to filter already-done subjects when
        *resume* is True. Required if *resume* is True.
    options : SlurmOptions
        Dispatch settings; the default runs everything in-process.

    Returns
    -------
    tuple of (n_succeeded, n_failed)
    """
    use_slurm = options.slurm or bool(os.environ.get("SLURM_JOB_ID"))

    if resume:
        if is_complete is None:
            raise ValueError("resume=True requires is_complete")
        n_before = len(subject_kwargs)
        subject_kwargs = [kw for kw in subject_kwargs if not is_complete(kw)]
        n_skipped = n_before - len(subject_kwargs)
        print(
            f"{n_skipped} subject(s) already complete, "
            f"{len(subject_kwargs)} subject(s) to process."
        )
    else:
        print(f"Found {len(subject_kwargs)} subject(s) to process.")

    if not subject_kwargs:
        print("No subjects found. Nothing to do.")
        return 0, 0

    n_succeeded = 0
    n_failed = 0

    if not use_slurm and options.n_jobs <= 1:
        for kw in subject_kwargs:
            try:
                fn(**kw)
                n_succeeded += 1
            except Exception as e:
                n_failed += 1
                print(f"[{kw['subject_id']}] FAILED: {e}")
        print(f"Done. {n_succeeded} subject(s) processed, {n_failed} failed.")
        return n_succeeded, n_failed

    import submitit

    log_dir.mkdir(parents=True, exist_ok=True)
    if use_slurm:
        executor = submitit.SlurmExecutor(folder=str(log_dir))
        slurm_params: dict = {}
        if options.partition is not None:
            slurm_params["partition"] = options.partition
        if options.account is not None:
            slurm_params["account"] = options.account
        if options.mem is not None:
            slurm_params["mem"] = options.mem
        slurm_params["time"] = options.time
        slurm_params["cpus_per_task"] = options.cpus_per_task
        # Without this, submitit emits no --ntasks-per-node and SLURM starts one
        # task per allocated CPU, running the per-subject function once per CPU
        # with every copy writing the same output file concurrently. These
        # functions are single-process and not write-safe against themselves, so
        # the request must be pinned. `scontrol show job` reports the *requested*
        # NumTasks=1 and hides this -- the submitit log ranks are what reveal it.
        slurm_params["ntasks_per_node"] = 1
        executor.update_parameters(**slurm_params)
        max_concurrent = options.max_jobs
        print(
            f"Submitting {len(subject_kwargs)} job(s) to SLURM "
            f"(max {max_concurrent or 'unbounded'} concurrent)..."
        )
    else:
        executor = submitit.AutoExecutor(folder=str(log_dir), cluster="local")
        executor.update_parameters(
            cpus_per_task=options.n_jobs, timeout_min=options.time
        )
        max_concurrent = None

    results = submit_with_concurrency_cap(
        executor, fn, subject_kwargs, max_concurrent=max_concurrent
    )
    for kw, err in results:
        if err is None:
            n_succeeded += 1
            if use_slurm:
                print(f"  Subject {kw['subject_id']} done.")
        else:
            n_failed += 1
            print(f"  Subject {kw['subject_id']} FAILED: {err}")

    print(f"Done. {n_succeeded} subject(s) processed, {n_failed} failed.")
    return n_succeeded, n_failed


def add_common_orchestration_args(
    parser, default_slurm_time: int = 60, default_cpus: int = 1
) -> None:
    """Add the CLI flags shared by every per-subject entry point."""
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--slurm-partition", type=str, default=None)
    parser.add_argument("--slurm-account", type=str, default=None)
    parser.add_argument("--slurm-time", type=int, default=default_slurm_time)
    parser.add_argument("--slurm-cpus-per-task", type=int, default=default_cpus)
    parser.add_argument("--slurm-mem", type=str, default=None)
    parser.add_argument("--slurm-max-jobs", type=int, default=None)
