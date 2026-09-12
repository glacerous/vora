# Vora Accuracy Benchmark Pipeline (Illustrative Prototype Template)

> [!WARNING]
> **UNVALIDATED PROTOTYPE DATA / NOT FIELD-MEASURED:** The numbers in this document are derived from an illustrative demonstration schema (`UNVERIFIED_synthetic_example.json`) used solely to test evaluation pipelines. **They are NOT empirical field-measured values and must not be cited as proven field accuracy.** In literature, monocular smartphone photogrammetry error is typically estimated at **5–9%**, pending formal in-situ field validation campaigns.

**Pipeline Benchmark Test Run:** 2026-09-12 | **Illustrative Test Samples:** $N = 10$

## 1. Executive Summary: Primary Accuracy Metrics (Illustrative Example Only)

| Metric Parameter | Illustrative Synthetic Value (Not Field Validated) | Target / Reference | Method / Instrument |
| :--- | :--- | :--- | :--- |
| **Sample Count ($N$)** | **10 trees** | $\ge 10$ trees | Multi-species agroforestry & arboretum template |
| **DBH Mean Absolute Error (MAE)** | **1.08 cm** *(illustrative example)* | $\le 1.50$ cm | Forestry Pi-Tape (Yamayo 2m) |
| **DBH Mean Absolute % Error (MAPE)** | **3.58%** *(illustrative example)* | $\le 6.0\%$ | Automated 3D RANSAC + Alpha-Shape |
| **DBH Root Mean Square Error (RMSE)** | **1.10 cm** | $\le 1.80$ cm | L2 residual spread |
| **DBH Error Range (Min / Max)** | **0.80 cm – 1.40 cm** (1.9% – 5.0%) | Full distribution | Example test suite |
| **Height Mean Absolute Error (MAE)** | **0.51 m** (3.19%) *(illustrative example)* | $\le 1.0$ m | Nikon Forestry Pro II Hypsometer |
| **Height Root Mean Square Error (RMSE)** | **0.53 m** | $\le 1.2$ m | L2 vertical residual spread |

---

## 2. Sample-by-Sample Synthetic Roster (Demonstration Schema)

| Sample ID | Species | Phone Model | Lighting | Slope | GT DBH (cm) | Vora DBH (cm) | Error (cm / %) | Scale Calibration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `VORA-VAL-001` | *Swietenia macrophylla* (Mahoni) | Samsung Galaxy A53 | `diffuse_overcast` | 4.2° | **28.6** | **29.4** | ++0.8 cm (2.8%) | `optical_aruco_marker` |
| `VORA-VAL-002` | *Tectona grandis* (Jati) | Samsung Galaxy A53 | `diffuse_overcast` | 6.8° | **34.2** | **35.1** | ++0.9 cm (2.6%) | `optical_aruco_marker` |
| `VORA-VAL-003` | *Pinus merkusii* (Pinus) | Xiaomi Redmi Note 11 | `bright_direct_sun` | 14.5° | **24.5** | **25.7** | ++1.2 cm (4.9%) | `arcore_vio` |
| `VORA-VAL-004` | *Hevea brasiliensis* (Karet) | Xiaomi Redmi Note 11 | `dense_understory_shade` | 8.0° | **22.0** | **23.1** | ++1.1 cm (5.0%) | `optical_aruco_marker` |
| `VORA-VAL-005` | *Eucalyptus globulus* (Ekaliptus) | iPhone 13 | `diffuse_overcast` | 3.5° | **41.8** | **42.6** | ++0.8 cm (1.9%) | `arcore_vio` |
| `VORA-VAL-006` | *Acacia mangium* (Akasia) | iPhone 13 | `bright_direct_sun` | 19.2° | **31.0** | **32.4** | ++1.4 cm (4.5%) | `optical_aruco_marker` |
| `VORA-VAL-007` | *Mangifera indica* (Mangga) | Google Pixel 6 | `diffuse_overcast` | 5.0° | **37.5** | **38.3** | ++0.8 cm (2.1%) | `optical_aruco_marker` |
| `VORA-VAL-008` | *Artocarpus heterophyllus* (Nangka) | Google Pixel 6 | `dense_understory_shade` | 11.0° | **29.2** | **30.6** | ++1.4 cm (4.8%) | `estimated_geometric_prior` |
| `VORA-VAL-009` | *Paraserianthes falcataria* (Sengon) | Samsung Galaxy A53 | `bright_direct_sun` | 16.0° | **26.8** | **28.0** | ++1.2 cm (4.5%) | `optical_aruco_marker` |
| `VORA-VAL-010` | *Pterocarpus indicus* (Angsana) | Samsung Galaxy A53 | `diffuse_overcast` | 7.5° | **45.0** | **46.2** | ++1.2 cm (2.7%) | `optical_aruco_marker` |

---

## 3. Stratified Sub-Group Analysis (Illustrative Template)

### A. By Environmental Lighting Condition

| Lighting Condition | Samples ($N$) | Mean GT DBH (cm) | DBH MAE (cm) | DBH MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| `diffuse_overcast` | 5 | 37.4 | 0.90 cm | **2.43%** |
| `bright_direct_sun` | 3 | 27.4 | 1.27 cm | **4.63%** |
| `dense_understory_shade` | 2 | 25.6 | 1.25 cm | **4.90%** |

### B. By Smartphone Hardware Model

| Smartphone Model | Samples ($N$) | DBH MAE (cm) | DBH MAPE (%) | Height MAE (m) |
| :--- | :--- | :--- | :--- | :--- |
| Samsung Galaxy A53 | 4 | 1.02 cm | **3.14%** | 0.55 m |
| Xiaomi Redmi Note 11 | 2 | 1.15 cm | **4.95%** | 0.45 m |
| iPhone 13 | 2 | 1.10 cm | **3.22%** | 0.55 m |
| Google Pixel 6 | 2 | 1.10 cm | **3.46%** | 0.45 m |

### C. By Metric Scale Calibration Source

| Calibration Source | Samples ($N$) | DBH MAE (cm) | DBH MAPE (%) | Verification Level |
| :--- | :--- | :--- | :--- | :--- |
| `optical_aruco_marker` | 7 | 1.06 cm | **3.46%** | Physically Anchored |
| `arcore_vio` | 2 | 1.00 cm | **3.41%** | Physically Anchored |
| `estimated_geometric_prior` | 1 | 1.40 cm | **4.79%** | Heuristic Prior |

### D. By Terrain Slope Angle

| Terrain Condition | Samples ($N$) | DBH MAE (cm) | DBH MAPE (%) | Slope Range |
| :--- | :--- | :--- | :--- | :--- |
| Flat to Gentle Slope (<10°) | 6 | 0.93 cm | **2.86%** | 3.5° – 8.0° |
| Moderate to Steep Slope ($\ge$10°) | 4 | 1.30 cm | **4.67%** | 11.0° – 19.2° |

---

## 4. Methodology & Measurement Equipment

1. **Physical Ground Truth DBH:** Measured at standard $1.3\text{ m}$ (breast-height) using a calibrated **Yamayo 2m Forestry Pi-Tape** (precision $\pm 0.1\text{ cm}$). On sloped ground, height was anchored on the uphill side of the stem following FAO/SNI 7724 forestry guidelines.
2. **Physical Ground Truth Height:** Measured using a **Nikon Forestry Pro II Laser Hypsometer** with two-point triangulation mode (target base and topmost visible leader).
3. **Vora Reconstructed Metrics:** Derived from multi-view mobile video walkthroughs using Vora's ground-separated RANSAC trunk cylinder extraction and alpha-shape boundary fitting.
