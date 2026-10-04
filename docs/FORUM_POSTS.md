# Forum post drafts

Post on the competition's Discussion tab. Review the wording before posting; record replies in `docs/COMPETITION_RULES.md`.

## #1 — External data announcement (week 1)

**Title:** External data: ETS photogranulometry dataset (CC BY 4.0)

> Hi all, in line with rule 2.6 (external data must be public and freely accessible to everyone), I'd like to announce that I plan to use the following public dataset for pretraining:
>
> Plante St-Cyr et al., *Dataset of soil images with corresponding particle size distributions for photogranulometry* — 321 soil samples, 12,714 images, sieve PSD 80 µm–80 mm. FRDR, DOI 10.20383/103.01316, licence CC BY 4.0 (paper: arXiv:2506.17469).
>
> It is free to download via Globus. None of its samples come from the competition sites. @lukasleiboldboku, please let me know if you see any problem with this.

## #2 — Unlabelled test photos for self-supervised pretraining (week 1)

**Title:** Clarification: may unlabelled test photos be used for self-supervised pretraining?

> Hi @lukasleiboldboku, a rules question. The training photos come from Motorola/Samsung phones while the test photos come from iPhones. Is it allowed to use the test *images* (never any labels, no hand labelling) for self-supervised / unsupervised pretraining of an image encoder, e.g. contrastive learning on image crops? I'd like to make sure this is within the rules before relying on it. Thank you!

## #3 — Optional, after week 3: share a finding (only if our own ablation confirms it)

**Title:** Physical scale matters: resampling all phones to the same px/mm

> Short note for anyone starting out: `ppm_updated.csv` ranges from 11.5 px/mm (Motorola Edge) to 26.3 px/mm (Samsung A52), and the test iPhones are 13.9 / 19.5. In our grouped CV, resampling every photo to a common px/mm before cropping changed EMD from X to Y. Also note that samples H366–H374 are near-identical, so random K-fold leaks — group them.

Fill in X/Y from the registry before posting. Any code shared must be shared publicly on Kaggle, never privately (Foundational Rules 5.d / 6.a–b).
