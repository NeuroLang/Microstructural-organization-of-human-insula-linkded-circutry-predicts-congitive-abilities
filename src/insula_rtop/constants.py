"""Constants fixed by the paper and by the HCP acquisition protocol.

Every value here is quoted from Menon et al. (2020) eLife 9:e53470, Appendix 1
("HCP diffusion MRI data acquisition and preprocessing"):

    TE/TR 89 ms/5.5 s; 1.25 mm isotropic voxels; b = 0 (6 images), and 271
    directions equally distributed across 3 b-shells of 1, 2, and 3 ms um^-2;
    diffusion times D = 43.1 ms and d = 10.6 ms.
"""

from __future__ import annotations

#: Gradient pulse separation Delta, in seconds.
BIG_DELTA_S = 43.1e-3

#: Gradient pulse duration delta, in seconds.
SMALL_DELTA_S = 10.6e-3

#: Effective diffusion time t = Delta - delta/3, in seconds (Appendix 1).
#: This is the ``t`` in both P_t and the normalisation R_t = P_t (4 pi D t)^(3/2).
DIFFUSION_TIME_S = BIG_DELTA_S - SMALL_DELTA_S / 3.0

#: b-values at or below this are treated as b0 (s/mm^2).
B0_THRESHOLD = 50.0

#: FreeSurfer ``aparc+aseg`` labels making up the ventricular CSF compartment
#: used to estimate D_vent: left/right lateral, 3rd and 4th ventricle.
VENTRICLE_LABELS = (4, 43, 14, 15)

#: Historical HCP releases making up the "Q1-Q6 Data Release" of the paper. In
#: the S1200 behavioural table the Q4-Q6 releases are folded into ``S500``, so
#: this set -- not a literal ``Q1..Q6`` match -- is the reproducible spelling of
#: the paper's cohort. See README, "Cohort".
Q1_Q6_RELEASES = ("Q1", "Q2", "Q3", "S500")

#: The 11 behavioural measures entering the CCA (Results, "Insula microstructure
#: and relation to cognitive control ability"): six in-scanner task measures and
#: five out-of-scanner NIH Toolbox measures.
CCA_BEHAVIORAL_COLUMNS = (
    "WM_Task_Acc",
    "WM_Task_Median_RT",
    "Relational_Task_Acc",
    "Relational_Task_Median_RT",
    "Gambling_Task_Perc_Larger",
    "Gambling_Task_Median_RT_Larger",
    "ListSort_Unadj",
    "Flanker_Unadj",
    "CardSort_Unadj",
    "PicSeq_Unadj",
    "ProcSpeed_Unadj",
)

#: The three insular subdivisions, in the order the paper reports them.
INSULA_SUBDIVISIONS = ("vAI", "dAI", "PI")

#: The three ACC subdivisions, named after the insular subdivision each is
#: preferentially connected to (Figure 7).
ACC_SUBDIVISIONS = ("ACC-vAI", "ACC-dAI", "ACC-PI")

HEMISPHERES = ("L", "R")
