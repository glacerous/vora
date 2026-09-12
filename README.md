# vora

## Setup Instructions

Clone the dependency repository:
```bash
git clone https://github.com/NVlabs/InstantSplat.git instantsplat_repo
```

---

## Metodologi Kalkulasi Karbon

Estimasi karbon mengikuti: foto → rekonstruksi 3D (MASt3R / Gaussian Splatting via
Modal) → ekstraksi DBH & tinggi dari point cloud → kalkulasi allometrik.

**Rumus Allometrik** (`carbon/allometric.py`)
- AGB (Chave et al. 2005):
  - DBH-only: `ρ·exp(a + b·lnD + 0.207·(lnD)² − 0.0281·(lnD)³)` per forest type
    (dry: a=-0.667,b=1.784; moist: a=-1.499,b=2.148; wet: a=-1.239,b=1.980).
  - Dengan tinggi: `exp(a + b·ln(ρD²H))` (dry: a=-2.187,b=1.016; moist: a=-1.499,b=0.976;
    wet: a=-1.239,b=0.947). **Hanya dipakai bila tinggi terbukti = tinggi total pohon.**
- BGB = AGB × R:S, dengan `R:S` per forest type dari **IPCC 2006 Tier 1 (Vol.4 Ch.4
  Table 4.4)**: moist/wet = 0.37, dry = 0.28 (bukan konstanta 0.24 global).
  Rasio yang dipakai dicantumkan di response (`root_to_shoot_ratio`).
- Carbon = Total biomass × 0.47 (IPCC default). CO₂e = Carbon × 3.67 (44/12).

**Akurasi & kualitas output**
1. **Scale calibration wajib & eksplisit.** `_load_scale_factor_for_scan()` tidak lagi
   diam-diam memakai 1.0; mengembalikan `(scale_factor, is_calibrated, source)`.
   Tanpa kalibrasi apa pun, fallback-nya `(1.0, False, "uncalibrated_default")` —
   hasil TIDAK pernah ditandai `calibrated`. Response menandai `scale_status` =
   `calibrated` / `uncalibrated`. Jika tidak dikalibrasi, hasil **tetap ditampilkan
   untuk preview tetapi ditandai jelas** (`confidence` + badge merah di frontend)
   dan `calibrated=false`.
2. **Auto-kalibrasi pose.** Jika orang terdeteksi di frame (MediaPipe) dengan confidence
   cukup dan terlokalisasi di point cloud, `scale_factor` diturunkan otomatis
   (`calibration_source = "auto_pose"`). Kalibrasi manual (`calibration.json`) selalu
   menang atas auto-pose.
3. **Validasi tinggi total.** `is_full_tree_height()` mengecek keberadaan titik dekat
   pangkal (ground) dan sinyal tajuk (percabangan) di ujung atas. Jika tinggi yang
   terekam hanyalah segmen batang, sistem **memaksa fallback DBH-only**
   (`height_used = "dbh_only_fallback"` dengan alasan di `height_fallback_reason`),
   dan menyimpan tinggi segmen mentah terpisah (`segment_height_m`).
   Validasi ini dipakai **di semua jalur** lewat satu helper terpusat
   `carbon/dbh_extractor.resolve_height_usage()` — termasuk jalur otomatis
   (`run_carbon_analysis`) dan **semua endpoint manual** (manual override 3D transform,
   recalculate 2D clicks, adjust-geometry). Setiap endpoint manual mengembalikan field
   standar yang sama: `height_used`, `total_height_used_m`, `segment_height_m`,
   `height_fallback_reason`, `height_validated`, `height_validation_reason`.
   - Tinggi **dari sistem** (extractor) yang gagal validasi → dipaksa fallback DBH-only
     (`height_validated: false`).
   - Tinggi **diinput manual oleh user** (mis. transform controls) → dihormati (tidak
     dipaksa fallback), tetapi ditandai `height_validated: false` dengan alasan eksplisit.

4. **Quality gate rekonstruksi.** `quality_status` = `ok` / `low_points` /
   `high_fit_error` berdasarkan jumlah titik slice dan residual fit lingkaran.
5. **Interval ketidakpastian.** Output mengembalikan rentang CO₂e
   (`co2e_low_kg`..`co2e_high_kg`, ±`co2e_uncertainty_pct`%) karena beberapa sumber
   error (skala, tinggi, spesies/densitas) menumpuk.
6. **Threshold spesies (Pl@ntNet).** Densitas kayu spesifik hanya dipakai bila
   confidence top-1 ≥ 30%; di bawah itu fallback ke default 0.6 dengan flag
   `"generic-default (spesies tidak pasti)"`.

**Migrasi basis data:** `db/migration_006.sql` dan `db/migration_007.sql` menambahkan
kolom metadata akurasi (`scale_status`, `height_used`, `height_validated`,
`root_to_shoot_ratio`, `co2e_low_kg`, dst.) ke tabel `tree_scans`. Terapkan migrasi
001–007 sebelum menjalankan versi baru.

---

## Repository Structure

```
vora/
├── app/                  # Core backend entrypoints (FastAPI server, Modal GPU app)
│   ├── server.py
│   └── modal_app.py
├── carbon/               # Core MRV allometrics, geometry extraction, and reconstruction engines
│   ├── allometric.py
│   ├── dbh_extractor.py
│   ├── reconstruction_engine.py
│   ├── commercial_pipeline.py
│   └── scale_calibrator.py
├── web/                  # Web frontend assets and viewers
│   ├── index.html
│   ├── viewer.html
│   └── gaussian-splats-3d.umd.js
├── storage/              # Cloud storage clients (Cloudflare R2, D1, auth utils)
├── scripts/              # Operational & CLI utility scripts
│   ├── calibrate_scale.py
│   ├── cleanup_ply.py
│   ├── keep_alive.py
│   ├── seed_demo_account.py
│   ├── COLMAP.bat
│   └── RUN_TESTS.bat
├── tests/                # Automated test suites and regression evaluations
│   ├── test_accuracy.py
│   ├── test_concurrency_and_gates.py
│   ├── test_carbon.py
│   ├── test_storage.py
│   ├── test_bandwidth_verification.py
│   ├── run_test.py
│   ├── benchmark_accuracy.py
│   └── scratch_eval/     # Standalone historical evaluation & audit scripts
├── db/                   # Database schemas and migrations (D1 SQLite)
├── bin/                  # Prebuilt binaries and DLLs (COLMAP, ONNX Runtime CUDA, Qt plugins)
├── docs/                 # Architectural specifications, licensing docs, and benchmark reports
├── output/               # Gitignored output directory for job reconstructions
└── scratch/              # Gitignored live working directory for ephemeral pipeline processing
```

## Running the Application

- **Start Local API Server**:
  ```bash
  uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
  # or
  python app/server.py
  ```

- **Deploy Modal GPU App**:
  ```bash
  modal deploy app/modal_app.py
  ```

- **Run Automated Test Suite**:
  ```bash
  python tests/test_accuracy.py
  python tests/test_concurrency_and_gates.py
  ```


