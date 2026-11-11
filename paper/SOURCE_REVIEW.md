# Primary-source review — 11 November 2026

The bibliography uses sources inspected during this drafting session. This is a targeted verification, not a systematic review or proof of novelty. No claim is made about journal quartile or acceptance likelihood.

| Source | Primary page checked | Use and limits |
| --- | --- | --- |
| McMahan et al., 2017 | https://proceedings.mlr.press/v54/mcmahan17a.html | Original federated model averaging; no general privacy guarantee attributed |
| Li et al., 2020 | https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html | FedProx and heterogeneity; no attack protection implied |
| Haarnoja et al., 2018 | https://proceedings.mlr.press/v80/haarnoja18b.html | Off-policy maximum entropy actor-critic |
| Haarnoja et al., applications | https://arxiv.org/abs/1812.05905 | SAC updates and applications; author manuscript |
| Sievers et al., 2025 | https://doi.org/10.1016/j.egyai.2025.100521 and https://publikationen.bibliothek.kit.edu/1000181773 | Building batteries, federation, cost/emissions and transfer; no claim that this paper originated federated batteries |
| Blanchard et al., 2017 | https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf | Krum and its original assumptions |
| Yin et al., 2018 | https://proceedings.mlr.press/v80/yin18a.html | Median and trimmed-mean distributed learning |
| Cao et al., 2021 | https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/ | Reference data, trust bootstrapping |
| Nguyen et al., 2022 | https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen | Clustering, clipping and noise for backdoors; adapted task is disclosed |
| Fang et al., 2025 | https://arxiv.org/abs/2502.08123 and official proceedings https://archives.iw3c2.org/www2025/www2025-proceedings.pdf | Normalized attack and ensemble FRL; fixed-grid adaptation disclosed |
| Sepehrzad et al., 2026 | https://www.sciencedirect.com/science/article/pii/S0306261925020124 | Trust-aware microgrid market framework; comparator remains a reconstruction |
| Ratnam et al., 2017 | https://researchportalplus.anu.edu.au/en/publications/residential-load-and-rooftop-pv-generation-an-australian-distribu/ | Author institution verifies dataset description and publication metadata |
| Sturmberg and Shaw, 2025 | https://zenodo.org/records/14885589 | Dataset creators, version and description |
| CSIRO taxonomy | https://github.com/csiro-energy-systems/RepresentativeLVNetworks | Official code/data release and recommended citation |
| Energex dynamic connections | Official URL in bibliography | Export cap and ordinary floor; emergency exception acknowledged |
| QCA 2025–26 feed-in tariff | Official PDF in bibliography | 8.66 c/kWh historical input |

## Still requiring verification

- Retail import tariff verification is now complete against the retailer's previous-rates table; preserving an archival snapshot remains advisable.
- AEMO threshold fact sheet table and downloaded demand-series provenance. The draft treats the threshold as a scenario input and avoids claiming contemporaneous operational threshold equivalence.
- Exact NASA POWER climatology query parameters and irradiance inputs.
- Full-paper component mapping for the TAFDRL reconstruction and broad literature coverage. Publication metadata and broad scope were checked; no exhaustive equation-level revalidation was performed in this session.
- Data licence terms, code-release permission, authorship, affiliations, funding and disclosures.

## Evidence interpretation

Every plotted energy or attack value comes from local logs, not a published comparison copied from another study. Paired final-minus-original analyses are newly computed **exploratory** summaries of those logs and use matched seeds. No new control experiment was run. Source hashes identify the archived inputs.

## Energy and AI preparation audit additions

- Ergon historical import rates and service fee verified: https://www.ergon.com.au/retail/residential/tariffs-and-prices/residential-tariffs (previous rates from 1 July 2025). All four implemented inputs match.
- Exact signed-rank procedure and its tie/zero assumptions checked against official SciPy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html. The installed computational version is retained in the environment record.
- Holm procedure checked in the original article (author-paper PDF hosted by the University of Granada): https://sci2s.ugr.es/keel/pdf/specific/articulo/0052_001.pdf. Metadata: Scandinavian Journal of Statistics 6(2), 65–70, 1979; https://www.jstor.org/stable/4615733.
- Elsevier LaTeX instructions, highlights and AI policies checked; journal-specific Energy and AI guide returned 403 and its exact requirements remain unverified. See SUBMISSION_READINESS.md.
- Local NASA climatology CSV headers specify coordinates, irradiance parameter and the 2001–2025 climatology period. That local metadata narrows the remaining provenance work but does not replace an archived request and source verification.
