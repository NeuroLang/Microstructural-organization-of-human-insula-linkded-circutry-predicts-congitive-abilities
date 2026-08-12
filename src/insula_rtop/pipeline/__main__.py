"""Pipeline entry point: runs the requested steps in order.

    uv run python -m insula_rtop.pipeline site=mysite

Dispatch is one literal table, :data:`STEP_RUNNERS`, so the set of valid steps
and the set of dispatched steps cannot drift apart. Each ``_run_<step>`` wrapper
spells out every argument it passes rather than splatting the config, which is
what ``tests/test_pipeline.py`` pins: a key missing from the YAML is otherwise
invisible until the step is actually launched on the cluster, and by then Hydra
has created the output directory and the run looks like it started.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from insula_rtop.analysis.run import run_analysis
from insula_rtop.atlases.run import run_atlases
from insula_rtop.figures.run import run_figures
from insula_rtop.hcp.cohort import run_cohort
from insula_rtop.hcp.to_bids import run_bidsify
from insula_rtop.rtop.run import run_rtop
from insula_rtop.surface.run import run_surface
from insula_rtop.tools._orchestration import SlurmOptions

logger = logging.getLogger("insula_rtop.pipeline")


def _subjects(cfg: DictConfig) -> list[str] | None:
    if cfg.subjects is None:
        return None
    # OmegaConf parses unquoted numeric CLI list entries (subjects=[100206]) as
    # int, which breaks every path built from them.
    return [str(s) for s in cfg.subjects]


def _orchestration(cfg: DictConfig, step: DictConfig) -> SlurmOptions:
    """Dispatch settings: cluster identity from the site, sizing from the step."""
    return SlurmOptions(
        slurm=bool(cfg.slurm.use),
        n_jobs=int(cfg.slurm.n_jobs),
        partition=cfg.slurm.partition,
        account=cfg.slurm.account,
        time=int(step.slurm_time),
        cpus_per_task=int(step.slurm_cpus_per_task),
        mem=step.slurm_mem,
        max_jobs=cfg.slurm.max_jobs,
    )


def _run_cohort(cfg: DictConfig) -> None:
    releases = cfg.cohort.releases
    run_cohort(
        Path(cfg.site.hcp_data_root),
        Path(cfg.cohort.behavioral_csv),
        Path(cfg.bids_root),
        restricted_csv=(
            Path(cfg.cohort.restricted_csv) if cfg.cohort.restricted_csv else None
        ),
        releases=tuple(releases) if releases is not None else None,
        require_behavior=bool(cfg.cohort.require_behavior),
        subjects=_subjects(cfg),
    )


def _run_hcp2bids(cfg: DictConfig) -> None:
    run_bidsify(
        Path(cfg.site.hcp_data_root),
        Path(cfg.bids_root),
        subjects=_subjects(cfg),
        force=bool(cfg.hcp2bids.force),
        skip_existing=bool(cfg.hcp2bids.skip_existing),
        resume=bool(cfg.hcp2bids.resume),
        options=_orchestration(cfg, cfg.hcp2bids),
    )


def _run_rtop_volume(cfg: DictConfig) -> None:
    run_rtop(
        Path(cfg.bids_root),
        Path(cfg.deriv_root),
        subjects=_subjects(cfg),
        fit_mask=str(cfg.rtop_volume.fit_mask),
        ribbon_dilation=int(cfg.rtop_volume.ribbon_dilation),
        ventricle_erosion=int(cfg.rtop_volume.ventricle_erosion),
        radial_order=int(cfg.rtop_volume.radial_order),
        fit_jobs=int(cfg.rtop_volume.fit_jobs),
        force=bool(cfg.rtop_volume.force),
        skip_existing=bool(cfg.rtop_volume.skip_existing),
        resume=bool(cfg.rtop_volume.resume),
        options=_orchestration(cfg, cfg.rtop_volume),
    )


def _run_rtop_surface(cfg: DictConfig) -> None:
    run_surface(
        Path(cfg.bids_root),
        Path(cfg.deriv_root),
        subjects=_subjects(cfg),
        force=bool(cfg.rtop_surface.force),
        skip_existing=bool(cfg.rtop_surface.skip_existing),
        resume=bool(cfg.rtop_surface.resume),
        options=_orchestration(cfg, cfg.rtop_surface),
    )


def _run_atlas_labels(cfg: DictConfig) -> None:
    acc_labels = cfg.atlas_labels.acc_labels
    run_atlases(
        Path(cfg.deriv_root),
        cache_dir=(
            Path(cfg.atlas_labels.cache_dir) if cfg.atlas_labels.cache_dir else None
        ),
        acc_labels=(
            {hemi: Path(path) for hemi, path in acc_labels.items()}
            if acc_labels
            else None
        ),
        insula_labels=(
            OmegaConf.to_container(cfg.atlas_labels.insula_labels, resolve=True)
            if cfg.atlas_labels.insula_labels
            else None
        ),
        force=bool(cfg.atlas_labels.force),
    )


def _run_analysis(cfg: DictConfig) -> None:
    run_analysis(
        Path(cfg.bids_root),
        Path(cfg.deriv_root),
        subjects=_subjects(cfg),
        rebuild_table=bool(cfg.analysis.rebuild_table),
        outlier_statistic=str(cfg.analysis.outlier_statistic),
        n_resamples=int(cfg.analysis.n_resamples),
        covariate_columns=tuple(cfg.analysis.covariates),
        group_column=cfg.analysis.group_column,
        gradient_smoothing=int(cfg.analysis.gradient_smoothing),
        gradient_frame=str(cfg.analysis.gradient_frame),
    )


def _run_figures(cfg: DictConfig) -> None:
    run_figures(
        Path(cfg.deriv_root),
        formats=tuple(cfg.figures.formats),
        dpi=int(cfg.figures.dpi),
        only=list(cfg.figures.only) if cfg.figures.only else None,
    )


#: Every step the pipeline knows how to run, in canonical order. One literal
#: table means ``steps:`` cannot name something that is never dispatched, and a
#: new step cannot be added without appearing here.
STEP_RUNNERS = {
    "cohort": _run_cohort,
    "hcp2bids": _run_hcp2bids,
    "rtop_volume": _run_rtop_volume,
    "rtop_surface": _run_rtop_surface,
    "atlas_labels": _run_atlas_labels,
    "analysis": _run_analysis,
    "figures": _run_figures,
}

STEPS = tuple(STEP_RUNNERS)


@hydra.main(version_base=None, config_path=".", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    deriv_root = Path(cfg.deriv_root)
    deriv_root.mkdir(parents=True, exist_ok=True)
    # The resolved config is the run's provenance record.
    OmegaConf.save(cfg, deriv_root / "config.yaml")

    unknown = [step for step in cfg.steps if step not in STEP_RUNNERS]
    if unknown:
        raise ValueError(
            f"Unknown pipeline step(s): {unknown}. Known steps: {', '.join(STEPS)}"
        )

    for step in cfg.steps:
        logger.info("Running step: %s", step)
        STEP_RUNNERS[step](cfg)
    logger.info("Pipeline complete. Derivatives: %s", deriv_root)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
