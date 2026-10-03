# Kaggle Workflow

The dataset remains inside Kaggle. GitHub stores code, configs, documentation, experiment metadata, and lightweight summaries only.

## Recommended Kaggle session

```bash
!git clone https://github.com/rexxar280903/PhyGrainNet.git
%cd PhyGrainNet
!pip install -e .
```

The competition data should appear under a Kaggle input mount. Do not copy the dataset into the repository.

## First executable task

```bash
python scripts/audit_dataset.py \
  --data-root /kaggle/input/soil-grain-size-from-photos
```

Save the resulting JSON as a Kaggle output artifact, inspect it, then transfer only the compact audit findings into the research docs.

## Research discipline

Never begin architecture comparison before the sample grouping key and exact metric have been verified.
