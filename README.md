# Jangkar & Kompas Riset Ilmiah: Pustaka Metasains & Tolok Ukur Integritas Kesarjanaan (Scholarly Core)

[![CI](https://github.com/zaditprodakwah/jangkar-kompas-riset-ilmiah/actions/workflows/ci.yml/badge.svg)](https://github.com/zaditprodakwah/jangkar-kompas-riset-ilmiah/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![RO-Crate 1.3](https://img.shields.io/badge/RO--Crate-1.3-orange.svg)](ro-crate-metadata.json)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/muhzadit/scholarly-metascience-benchmarks)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/code/muhzadit/psychometric-validity-and-metascience-benchmark)

Repositori ini memuat artefak komputasi pendamping resmi untuk monograf kesarjanaan:
> **Jangkar & Kompas Riset Ilmiah: Epistemologi, Metodologi, dan Rekayasa Riset Kesarjanaan Modern**  
> *Penulis:* **Muhammad Khoiruzzadittaqwa** (256 Halaman, Standar PO-PAK 40 KUM, UGM Press)

---

## 🌐 English Executive Summary

**Scholarly Core (`scholarly-core`)** is a deterministic Python library and empirical benchmark suite designed to enforce methodological rigor, evidential integrity, and open reproducibility across scholarly disciplines. Developed as the computational companion to the 18-chapter monograph *Jangkar & Kompas Riset Ilmiah*, this package provides five rigorous metascience modules:
1. **Statistical Reporting Forensics (`forensics`):** Recalculation of APA test statistics ($t, F, r, \chi^2$) and decision error detection via the Nuijten et al. (2016) psychology corpus, Ioannidis (2005) Positive Predictive Value (PPV), Gelman & Carlin (2014) Type S/M errors, and discrete granularity checks (GRIM, GRIMMER, DEBIT).
2. **Causal Diagnostics (`causal`):** Difference-in-Differences (2x2), Hake (1998) Normalized Gain ($g$) with headroom ceiling effect calibration (Nissen et al., 2018), and standardized effect sizes.
3. **Bibliometrics & Knowledge Topology (`bibliometrics`):** Funk & Owen-Smith (2019) Consolidation-Disruption Index (CD-Index), Uzzi et al. (2013) citation combination novelty, and Rao-Stirling interdisciplinary diversity.
4. **Qualitative Reliability (`qualitative`):** Krippendorff's Alpha for arbitrary raters and missing data, Fleiss' Kappa, and Mixed Methods Appraisal Tool (MMAT 2018) scoring.
5. **Exact Metrology & Validity (`validity`):** Discrete polynomial convolution for exact right-tail $p$-values of Aiken's $V$, Gwet's $AC_1$ inter-rater agreement, and 10,000-iteration bootstrap percentile confidence intervals.

---

## 🏛️ Triad Ekosistem Komputasi Terbuka

Untuk menegakkan Deklarasi Barcelona tentang Informasi Riset Terbuka (*Barcelona Declaration on Open Research Information*, 2024), seluruh metodologi monograf terhubung secara simetris ke dalam tiga pilar:

1. **Pustaka Python Mandiri (`scholarly-core`):** Pustaka modular tanpa bloat, dirancang deterministik untuk audit integritas riset.
2. **Suite Tolok Ukur Kaggle (Multi-Cluster Benchmark):** 4 korpus data empiris di `benchmarks/data/` dan buku catatan eksekusi master di `benchmarks/notebooks/`.
3. **Ilmify Evidence Lab:** Antarmuka kalkulator interaktif peramban di [https://ilmify.id/lab](https://ilmify.id/lab).

---

## 📦 Pemasangan

```bash
git clone https://github.com/zaditprodakwah/jangkar-kompas-riset-ilmiah.git
cd jangkar-kompas-riset-ilmiah
pip install -e .
```

---

## 🚀 Contoh Penggunaan Cepat

### 1. Forensik Replikasi: Uji Statcheck & Granularitas GRIM (Bab 1 & 11)
```python
from scholarly_core import recalculate_statcheck, run_grim_test

# Audit laporan statistik: t(28) = 2.20, reported p = .03
audit = recalculate_statcheck(test_type="t", test_stat=2.20, df1=28, p_reported=0.03)
print(f"P-Value Dihitung : {audit['p_computed_exact']:.4f}")  # 0.0362
print(f"Status Konsistensi: {audit['status']}")

# Uji kemungkinan aritmetika rerata skor Likert (Brown & Heathers, 2017)
# Rerata dilaporkan 3.42 dengan N = 24 pada skala butir tunggal
grim = run_grim_test(mean=3.42, sample_size=24, items=1)
print(f"Konsistensi GRIM : {grim['consistent']}")  # False (Angka mustahil secara matematis)
```

### 2. Inferensi Kausal: Kalibrasi Headroom Hake Gain (Bab 5 & 6)
```python
from scholarly_core import calculate_hake_gain

# Mengatasi distorsi efek batas atas (ceiling effect) pada evaluasi intervensi
pre_scores = [70.0, 75.0, 80.0]
post_scores = [85.0, 90.0, 95.0]
gain_res = calculate_hake_gain(pre_scores, post_scores, max_score=100.0)
print(f"Hake's Normalized Gain (<g>): {gain_res['average_normalized_gain']:.3f}")
print(f"Kategori Gain: {gain_res['hake_category']}")
```

### 3. Topologi Pengetahuan: CD-Index Disrupsi Riset (Bab 15 & 16)
```python
from scholarly_core import calculate_cd_index

# Funk & Owen-Smith (2019): Mengukur apakah karya ilmiah bersifat mendisrupsi atau mengonsolidasi
# n_i = sitasi hanya ke karya fokus, n_j = sitasi ke karya fokus dan pendahulu, n_k = sitasi hanya ke pendahulu
cd = calculate_cd_index(citations_only_focal=120, citations_both=15, citations_only_prior=40)
print(f"CD-Index: {cd['cd_index']:.3f}")  # Positif mendekati +1 = Sangat Mendisrupsi
```

### 4. Metrologi Validitas: Exact P-Value Konvolusi Polinomial (Lampiran C)
```python
from scholarly_core import calculate_aiken_v, calculate_aiken_exact_p_value, bootstrap_aiken_ci

ratings = [5, 5, 5, 4, 4, 4, 4]  # 7 rater ahli skala 5
v = calculate_aiken_v(ratings, lo=1, hi=5)
p_exact = calculate_aiken_exact_p_value(ratings, lo=1, hi=5)
ci_lo, ci_hi = bootstrap_aiken_ci(ratings, lo=1, hi=5, n_resamples=10000)

print(f"Aiken V        : {v:.4f}")
print(f"Exact P-Value  : {p_exact:.6f}")
print(f"95% Bootstrap  : [{ci_lo:.4f}, {ci_hi:.4f}]")
```

---

## 📊 Suite Tolok Ukur Empiris (4 Kluster)

Dataset tolok ukur tersimpan di direktori `benchmarks/data/`:
1. `01_statcheck_psychology_corpus.csv`: 150 kasus audit inferensial psikologi (Nuijten et al., 2016).
2. `02_causal_concept_inventory_gains.csv`: 50 kohort evaluasi intervensi kausal (Nissen et al., 2018).
3. `03_discrete_forensics_lattice.csv`: 100 matriks kalibrasi granularitas diskrit (GRIM, GRIMMER, Aiken Exact).
4. `04_knowledge_topology_disruption.csv`: 50 simpul jejaring sitasi historis CD-Index (Funk & Owen-Smith, 2019).

Buku catatan eksekusi interaktif tersedia di:
```bash
jupyter notebook benchmarks/notebooks/01_scholarly_metascience_integrity_benchmark.ipynb
```

---

## 🧪 Verifikasi Mutu Deterministik

Jalankan seluruh rangkaian uji unit (pytest):
```bash
pytest tests/ -v
```

---

## 📜 Sitasi Resmi

Jika Anda menggunakan pustaka ini atau dataset tolok ukur dalam penelitian Anda, mohon merujuk monograf babon:

```bibtex
@book{khoiruzzadittaqwa2026jangkar,
  author    = {Muhammad Khoiruzzadittaqwa},
  title     = {Jangkar & Kompas Riset Ilmiah: Epistemologi, Metodologi, dan Rekayasa Riset Kesarjanaan Modern},
  publisher = {UGM Press},
  year      = {2026},
  pages     = {256},
  isbn      = {978-623-00-XXXX-X},
  address   = {Yogyakarta, Indonesia}
}

@dataset{khoiruzzadittaqwa2026metascience_benchmarks,
  author    = {Muhammad Khoiruzzadittaqwa},
  title     = {Scholarly Metascience & Research Integrity Benchmarks: Multi-Cluster Empirical Suite},
  year      = {2026},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/muhzadit/scholarly-metascience-benchmarks}
}
```

---

## ⚖️ Lisensi

- Perangkat Lunak (`scholarly-core`): **MIT License**.
- Data Tolok Ukur & Dokumentasi: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.
