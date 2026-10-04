# Jadwal Kerja PhyGrainNet (4 Okt – 30 Nov 2026)

**Tujuan:** naik leaderboard (private) dengan model yang dilatih sendiri dari nol (DEC-001), boleh memakai data tambahan publik (DEC-004).
**Deadline Kaggle:** Senin **30 Nov 2026, 18:00 WIB**. Target final dikunci **Jumat 27 Nov**, cadangan waktu 3 hari.

Legenda: 🧪 eksperimen di Kaggle · 💬 diskusi/forum · 🐙 GitHub · 📝 dokumen/paper · ✅ kriteria selesai

---

## Rutinitas tetap

| Kapan | Apa |
|---|---|
| Setiap hari kerja | Maksimal **2 submission/hari** (jatah 5), masing-masing tercatat di `docs/LEADERBOARD_LOG.md` dengan experiment ID + commit. Public LB **tidak** dipakai untuk memilih model (DEC-006). |
| Senin & Kamis | 💬 Cek forum lomba (tab Discussion, urutkan *Recent*). Catat jawaban host yang mengubah aturan/data ke `docs/COMPETITION_RULES.md`. |
| Minggu malam | 🐙 Review mingguan: salin baris baru `experiments/registry.csv` dari Kaggle → commit `exp(<ID>): ...`; update `project_status.json` → `python scripts/update_readiness.py`; push. |
| Setiap selesai eksperimen | Simpan `metrics.json` (berisi `cv_emd`, `worst_samples`, `per_point`). Kirim ke Claude untuk dibahas bila hasilnya aneh. |

Format commit: `feat(data): ...`, `exp(C100): ...`, `docs(schedule): ...`, `fix(metric): ...`.

---

## Minggu 0 — Min 4 Okt – Sen 5 Okt: Persiapan

- 🐙 Review & **merge PR** `feat/competition-pipeline` ke `main`.
- Baca `docs/DECISIONS.md` (12 keputusan) dan `docs/SCHEDULE.md` ini.
- 🧪 Di Kaggle: buat notebook baru dari `notebooks/kaggle_runner.ipynb`, Internet ON, attach data lomba.
- 📝 Daftar akun **Globus** (globus.org, login pakai Google/ORCID) untuk unduh data ETS.
- ✅ Notebook bisa `git clone` dan `pip install -e .` tanpa error.

## Minggu 1 — Sen 5 – Min 11 Okt: Audit + submission pertama (P2–P5)

- 🧪 Sel 1 `audit_dataset.py` → pastikan `problems` kosong. Kalau ada file yang tidak terbaca sample_id/kameranya, kirim daftar ke Claude.
- 🧪 Sel 2 `baseline_prior.py` → **submit A000** (median training). Ini jaring pengaman.
- 🧪 Sel 3 A001 (fitur fisik + ridge/PLS/kNN, CPU) → submit model dengan CV terbaik.
- 💬 **Posting forum #1** (draft di `docs/FORUM_POSTS.md`): umumkan pemakaian dataset ETS (CC BY 4.0) sebagai external data.
- 💬 **Posting forum #2**: tanya host apakah foto test *tanpa label* boleh dipakai untuk self-supervised pretraining (DEC-010).
- 🧪 Globus: transfer **subset kecil dulu** (±20 sampel) untuk mengecek pola nama file & isi Excel PSD.
- 🐙 Commit hasil audit ke `docs/DATASET.md` + registry A000/A001.
- ✅ Ada submission valid; grouped-CV A001 < 60 (target awal; median ≈ 90–97).

## Minggu 2 — Sen 12 – Min 18 Okt: CNN dari nol + data ETS (P6)

- 🧪 C100 uji cepat: `train.py --config configs/phygrainnet/mv_scratch.yaml training.epochs=2 cv.max_folds=1` → cek waktu per epoch. Lalu jalankan penuh (±1–2 jam GPU).
- 🧪 C000 dan B000 (ablation sederhana) di sesi GPU yang sama bila kuota cukup (kuota GPU Kaggle ±30 jam/minggu).
- 📝 ETS: `convert_ets_labels.py --inspect` pada Excel → pastikan 14 ayakan terdeteksi; catat apakah ada data hidrometer.
- 🧪 ETS: transfer penuh subset terpilih (≤6 foto/sampel), `prepare_ets.py`, upload sebagai Kaggle Dataset privat `ets-photogranulometry-10ppm`.
- 💬 Baca balasan forum #1/#2. Jika host melarang foto test untuk SSL → set `ssl.include_test_photos: false`.
- ✅ C100 punya OOF CV; dataset ETS sudah ada di Kaggle.

## Minggu 3 — Sen 19 – Min 25 Okt: Self-supervised pretraining (P7)

- 🧪 P000 SimCLR (`configs/pretrain/ssl.yaml`, nyalakan `ssl.use_ets: true` bila dataset ETS siap).
- 🧪 D100 fine-tune dari encoder P000. Bandingkan dengan C100 pada fold yang sama.
- 📝 Analisis error: 5 sampel terburuk dari `metrics.json` (kemungkinan H374, H616, H037, F827). Lihat foto-fotonya; tulis temuan di `docs/LEADERBOARD_LOG.md`.
- 💬 **Sesi diskusi dengan Claude**: kirim `metrics.json` C100/D100/A001 → putuskan arah minggu 4.
- ✅ Jawaban RQ: apakah SSL menurunkan CV EMD ≥ 2 poin?

## Minggu 4 — Sen 26 Okt – Min 1 Nov: Pretraining ETS (P7–P8)

- 🧪 P100 supervised ETS (`configs/pretrain/ets.yaml`) → D200 fine-tune (`mv_ets.yaml`).
- 🧪 Uji `external.moisture`: [dry] vs [humid] vs semua — pilih yang CV-nya terbaik.
- 🧪 Ablation inti untuk paper (masing-masing 1 run): `texture_stream` on/off, `fields_mm` 1 vs 2 skala, `mix_prob` 0 vs 0.3.
- 🐙 Commit tabel ablation ke `experiments/registry.csv`.
- ✅ D200 vs C100 vs D100 tercatat; keputusan pipeline final.

## Minggu 5 — Sen 2 – Min 8 Nov: Rantai penuh + ensembel (P9–P11)

- 🧪 D300 (`configs/phygrainnet/full.yaml`): SSL → ETS → fine-tune, `full_fit: true`, **3 seed** (`seed=42/43/44`).
- 🧪 `ensemble.py` → H001 (median per titik, dipilih greedy dari OOF). Submit H001.
- 📝 Mulai tulis deskripsi metode (wajib bagi pemenang: arsitektur, preprocessing, loss, hyper-parameter, link repo).
- ✅ Ensembel valid dengan OOF ≤ model tunggal terbaik.

## Minggu 6 — Sen 9 – Min 15 Nov: Ketahanan terhadap kamera baru (P10)

- 🧪 **Camera-holdout CV**: latih tanpa Samsung, validasi pada Samsung (meniru iPhone yang tidak pernah dilihat). Bandingkan `color_mode: grayworld` vs `lab` vs `none`.
- 🧪 Uji `inference.tta` 1/4/8 dan `max_tiles_per_image`.
- 🧪 Uji shrinkage ke prior (`postprocess.shrink_to_prior`, α 0–0.3) pada OOF untuk kandidat Final 2 (DEC-012).
- 💬 Cek apakah host merilis data tambahan (pernah disebut di forum). Kalau ya, ulangi audit + CV.
- ✅ Pengaturan warna & TTA dikunci berdasar CV.

## Minggu 7 — Sen 16 – Min 22 Nov: Freeze (P11–P12)

- 🧪 Jalankan ulang kandidat final di **sesi Kaggle bersih** dari commit yang di-tag → hasil harus identik (reproducibility).
- 🐙 Tag rilis `v0.9-final-candidate`.
- 📝 Lengkapi `docs/PAPER_PLAN.md` dengan tabel baseline/ablation; error analysis.
- ✅ Tidak ada perubahan kode setelah Minggu 7 kecuali bug fix.

## Minggu 8 — Sen 23 – Sen 30 Nov: Final

- Senin–Rabu: submission terakhir kandidat Final 1 (CV terbaik) dan Final 2 (konservatif).
- **Jumat 27 Nov**: pilih 2 final submission di tab *Submissions* (centang "Select").
- Sabtu–Minggu: cadangan bila ada error; jangan eksperimen baru.
- **Senin 30 Nov 18:00 WIB**: lomba ditutup.
- 🐙 Tag `v1.0-competition`, push semua log.

## Setelah lomba (Desember)

- Bila masuk 3 besar: kirim kode + dokumentasi ke sponsor dalam 1 minggu setelah notifikasi.
- 📝 Draft paper (bukti sudah terkumpul di registry & ablation).

---

## Prioritas bila waktu/GPU kurang

1. Submission valid (A000) → 2. A001 → 3. C100 → 4. D200 (ETS) → 5. ensembel → 6. SSL/ablation.
Bila ETS gagal diunduh sampai **25 Okt**, lanjut dengan SSL di foto lomba saja (P000 tanpa ETS) dan ensembel A001 + D100.
