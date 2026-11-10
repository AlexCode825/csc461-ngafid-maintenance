# Predicting maintenance needs from Cessna 172 flight data

**Is the model seeing wear, or just context?**

Course project for CSC 461 (Machine Learning), University of Rhode Island, Fall 2026.
Individual project by Alejandro Lacosta. Work in progress.

## The question

Published models on the NGAFID aviation maintenance dataset reach about 76% accuracy at
telling flights recorded shortly *before* an unplanned repair from flights recorded *after*
it. The dataset authors themselves warn that pilot actions, weather and flight plans explain
much of the variation in the sensors, not only the condition of the aircraft.

So this project asks whether that accuracy measures engine wear or flight context:

- **RQ1 — wear or context?** Does the model use signs of engine problems, or information
  about the day and the flight?
- **RQ2 — honest testing.** Do results drop when flights from the same repair event cannot
  appear in both training and test data, or when a whole type of problem is hidden from
  training?
- **RQ3 — which sensors?** Which sensor groups does the model need, and does that make
  sense physically?
- **RQ4 — pilot or aircraft?** How much of the variation in the sensors is explained by how
  the flight was flown, and is the leftover part what lets a model separate before from after?

The full write-up of the plan is in [`docs/proposal.pdf`](docs/proposal.pdf).

## Data

This repository contains **no data**. The dataset is the NGAFID Aviation Maintenance Dataset
(Yang and Desell, 2022): 28,935 real Cessna 172 flights from one flight school, 23 sensors
recorded every second, linked to 2,111 unplanned maintenance events.

- Benchmark subset (`2days`, about 1.1 GB): downloaded automatically by
  `src/ngafid_pilot.py --download`, from the same source the dataset authors' own code uses.
- Full dataset (4.3 GB): [Zenodo](https://doi.org/10.5281/zenodo.6624956) or
  [Kaggle](https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid).

## Reproducing

Most of the work runs in Google Colab, because the dataset authors trained their neural
networks on a TPU and a laptop is not enough.

```bash
pip install -r requirements.txt
python src/ngafid_pilot.py --download --data-dir ./ngafid_data
```

This downloads the benchmark subset, prints what is actually in it (columns, labels, folds,
missing values, and whether a repair-event identifier exists), builds summary features per
flight, and runs logistic regression and a random forest over the authors' 5 folds. It writes
`results/pilot_report.txt`.

To reproduce the published deep-learning baselines, run the dataset authors' own notebooks
from [their repository](https://github.com/hyang0129/NGAFIDDATASET).

## Layout

| Path | What it holds |
| --- | --- |
| `src/` | Scripts |
| `notebooks/` | Colab notebooks, one per research question |
| `results/` | Figures, tables and the pilot report |
| `docs/` | The project proposal |

## Status

- [x] Proposal submitted
- [ ] Benchmark reproduced
- [ ] RQ1 — wear or context
- [ ] RQ2 — honest testing
- [ ] RQ3 — which sensors
- [ ] RQ4 — pilot or aircraft
- [ ] Final report

## References

- Yang, H., Desell, T. *A Large-Scale Annotated Multivariate Time Series Aviation Maintenance
  Dataset from the NGAFID.* [arXiv:2210.07317](https://arxiv.org/abs/2210.07317) (2022).
- Yang, H., LaBella, A., Desell, T. *Predictive maintenance for general aviation using
  convolutional transformers.* [arXiv:2110.03757](https://arxiv.org/abs/2110.03757) (2021).
- Ismail Fawaz, H., et al. *InceptionTime: Finding AlexNet for time series classification.*
  Data Mining and Knowledge Discovery 34(6), 2020.
- Kapoor, S., Narayanan, A. *Leakage and the reproducibility crisis in machine-learning-based
  science.* [Patterns 4(9)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856), 2023.

## License

GPL-3.0, to match the dataset authors' repository, parts of which this project builds on.
