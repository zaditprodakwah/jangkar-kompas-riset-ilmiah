# Jangkar & Kompas Riset Ilmiah: Artefak Komputasi & Tolok Ukur Terbuka

[![CI](https://github.com/zaditprodakwah/jangkar-kompas-riset-ilmiah/actions/workflows/ci.yml/badge.svg)](https://github.com/zaditprodakwah/jangkar-kompas-riset-ilmiah/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![RO-Crate 1.3](https://img.shields.io/badge/RO--Crate-1.3-orange.svg)](ro-crate-metadata.json)

Repositori ini memuat artefak komputasi pendamping resmi untuk monograf kesarjanaan:
> **Jangkar & Kompas Riset Ilmiah: Memandu Penyelidikan Kesarjanaan dari Keraguan Epistemik Menuju Kebenaran Teruji**
> *Bab 11: Presisi Tabel & Visualisasi &bull; Lampiran C: Validitas Isi Aiken & Bootstrap Non-Parametrik*

---

## 🏛️ Triad Ekosistem Komputasi Terbuka

Untuk menjamin prinsip keterulangan sains (_open science reproducibility_), metodologi di buku terhubung dengan tiga pilar komputasi terbuka:

1. **Pustaka Python Mandiri (`scholarly-core`):** Modul komputasi murni tanpa dependensi eksternal berat untuk menghitung _exact p-value_ Aiken V via konvolusi polinomial diskrit, Gwet's AC1, dan selang kepercayaan bootstrap.
2. **Kaggle Benchmark Suite:** Kumpulan data tolok ukur 1.000 butir psikometri dan buku catatan eksekusi interaktif (_interactive executable notebook_).
3. **Ilmify Evidence Lab:** Antarmuka kalkulator interaktif peramban di [https://ilmify.id/lab#aiken](https://ilmify.id/lab#aiken) yang didukung oleh Cloudflare Pages Functions.

---

## 📦 Pemasangan

```bash
git clone https://github.com/zaditprodakwah/jangkar-kompas-riset-ilmiah.git
cd jangkar-kompas-riset-ilmiah
pip install -e .
```

---

## 🚀 Contoh Penggunaan Cepat

```python
from scholarly_core import (
    calculate_aiken_v,
    calculate_aiken_exact_p_value,
    get_aiken_critical_value,
    calculate_gwet_ac1,
    bootstrap_aiken_ci
)

# Skor penilaian 7 rater ahli pada butir kuesioner skala 5
ratings = [5, 5, 5, 4, 4, 4, 4]

# 1. Koefisien Validitas Aiken V
v = calculate_aiken_v(ratings, lo=1, hi=5)
print(f"Koefisien Aiken V : {v:.4f}")  # 0.7857

# 2. Exact P-Value via Konvolusi Polinomial Diskrit
p_val = calculate_aiken_exact_p_value(ratings, lo=1, hi=5)
print(f"Exact P-Value     : {p_val:.6f}")  # Terhitung analitis

# 3. Kesepakatan Antar-Rater Gwet's AC1
ac1 = calculate_gwet_ac1(ratings, lo=1, hi=5)
print(f"Gwet's AC1        : {ac1:.4f}")

# 4. Selang Kepercayaan 95% Bootstrap (10.000 Iterasi)
ci_lower, ci_upper = bootstrap_aiken_ci(ratings, lo=1, hi=5, n_resamples=10000)
print(f"95% Bootstrap CI  : [{ci_lower:.4f}, {ci_upper:.4f}]")
```

---

## 📊 Dataset Tolok Ukur (1.000 Butir)

Dataset tolok ukur terletak di `benchmarks/data/psychometric_validation_benchmark.csv`, mencakup empat arketipe perilaku rater:
- **High Consensus:** Kesepakatan tinggi rater pada butir berkualitas (Valid).
- **Moderate Agreement:** Kesepakatan moderat di ambang batas signifikansi.
- **Polarization Dilemma:** Polarisasi tajam (bimodal split) untuk membuktikan superioritas Gwet's AC1 atas indeks klasik.
- **Low Validity:** Butir cacat dengan ketidaksepakatan tinggi (Gugur / Wajib Revisi).

Jalankan pengujian buku catatan melalui:
```bash
jupyter notebook benchmarks/notebooks/01_psychometric_validity_benchmark.ipynb
```

---

## 🧪 Verifikasi Otomatis

Jalankan pengujian unit deterministik:
```bash
pytest tests/ -v
```

---

## 📜 Sitasi & Lisensi

Jika Anda menggunakan pustaka ini atau dataset tolok ukur dalam penelitian Anda, mohon mengutip monograf rujukan:

```bibtex
@book{jangkar_kompas_2026,
  title     = {Jangkar & Kompas Riset Ilmiah: Memandu Penyelidikan Kesarjanaan dari Keraguan Epistemik Menuju Kebenaran Teruji},
  author    = {Tim Arsitektur Kesarjanaan},
  year      = {2026},
  publisher = {Penerbit Akademik Terakreditasi},
  address   = {Yogyakarta},
  isbn      = {978-623-000-000-0}
}
```

Kode sumber dilisensikan di bawah **MIT License**. Data dan dokumentasi dilisensikan di bawah **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
