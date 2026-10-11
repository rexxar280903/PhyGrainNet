# Rilis pengembangan lokal v0.3 — 10 Oktober 2026

Pengembangan rilis ini mencakup perangkat yang dapat diselesaikan dan diverifikasi di workspace. Dataset kompetisi tidak terpasang dan PyTorch lokal 2.13.0+cpu tidak menyediakan CUDA. Karena itu kemajuan riset dan kesiapan kompetisi tetap 10%; tidak ada skor CV, training GPU, atau leaderboard baru.

## Yang ditambahkan

- `docs/dashboard.html`: HTML mandiri dengan alur klik, simulasi head/CDF/EMD, konversi kamera, kalkulator waktu/biaya, matriks ablation, gate, import OOF, dan kamus istilah. Hover, keyboard focus dan ketuk membuka komentar. Semua simulasi berlabel.
- `docs/paper.html`: paper metode dalam Bahasa Indonesia, 17 persamaan, 10 tabel, 3 figur, hyperparameter dari YAML, kompleksitas forward/backward/inference/classical/SSL, censoring quantile, sumber, batas klaim, dan stylesheet print.
- `web/`: sumber HTML/CSS/JS tanpa CDN/font eksternal, serta `scripts/build_portal.py` untuk sinkronisasi status/config/profile dan generasi HTML mandiri.
- `scripts/profile_model.py` / `src/phygrainnet/profiling.py`: parameter dan MAC Conv/Linear, informasi GPU/CUDA/cuDNN, benchmark training sintetis tersinkronisasi dengan optimizer/EMA, peak allocated/reserved, dan estimasi per-fold dengan batas cakupan eksplisit.
- `scripts/research_report.py` / `reporting.py`: complete-OOF validation, zero group overlap, cakupan sekali per sampel, bootstrap kelompok, fraction MAE, diameter D10/D30/D50/D60 tanpa extrapolation, paired comparison pada fold identik, JSON untuk portal.
- Tujuh konfigurasi ablation F101–F107. F103 mengurangi tile, bukan ablation view murni. F106 sengaja mempertahankan piksel native; fields_mm kehilangan arti fisik konsisten. F107 menggantikan statistik attention dengan mean pada dimensi output sama.
- Logging environment GPU, learning rate dan waktu per step termasuk data loader. Peak allocated dicatat tiap epoch. Notebook Kaggle mengukur profile sebelum training dan mengekspor report sesudah CV penuh. Classical notebook memakai fold yang sama dengan C100.
- Pooling satu tile mempunyai floor varians positif untuk menghindari gradien sqrt tak hingga; mask kosong ditolak. Checkpoint lama tetap dapat dibaca karena perubahan ini tidak mengubah keys bobot.
- Arsip hasil kini mencakup sumber portal dan dokumentasi, selain kode/config/checkpoint.

## Memakai portal

Klik dua kali `docs/dashboard.html` atau `docs/paper.html`. Keduanya memuat CSS, JS dan snapshot di dalam HTML sehingga berjalan tanpa server. Kedua file sebaiknya disimpan bersama untuk navigasi. Untuk mengedit, ubah file di `web/`, lalu jalankan:

```text
python scripts/update_readiness.py
python scripts/build_portal.py
python scripts/build_portal.py --check
node tests/test_portal.cjs
```

Untuk server pengembangan, dari root proyek gunakan `python -m http.server 8879 --bind 127.0.0.1`; buka `http://127.0.0.1:8879/web/`. Root proyek dibutuhkan agar tautan relatif ke docs/config/README berfungsi.

## Run Kaggle yang masih diperlukan

Kode baru di workspace belum otomatis tersedia pada branch GitHub `main`. Untuk membawa versi ini tanpa push, jalankan `python scripts/export_source.py`, unggah `outputs/phygrainnet_source.zip` sebagai Kaggle Dataset kode privat, attach pada notebook, dan isi `SOURCE_ARCHIVE` dengan path zip mount yang nyata. Notebook memverifikasi semua checksum dan menolak mencampur arsip dengan checkout yang sudah ada. Alternatifnya, gunakan branch/commit remote yang sudah berisi rilis ini.

Paket tidak berisi raw dataset, venv, .git, atau kredensial. Saat memakai arsip tanpa .git, provenance memakai SHA-256 source_manifest.json dan pemeriksaan kecocokan source, sementara git_commit dilaporkan `unknown`. Ini berbeda dari mengklaim commit remote sudah berisi perubahan lokal.

1. Attach `soil-grain-size-from-photos`, verifikasi runtime/GPU dan audit file nyata.
2. Baseline A000/A001/B000 + smoke C100. Smoke tidak menjadi evidence seleksi.
3. C100 full grouped CV, lalu F101–F107 dengan folds/seed/schedule identik. Laporkan perubahan parameter/compute.
4. Finalis pada 42/123/2026; kelompok terburuk, per-diameter, bootstrap, dan analisis kamera/view khusus.
5. Pretraining ETS opsional setelah akses/lisensi/aturan diumumkan. SSL yang memakai foto validasi/test dilabeli transductive; fold-specific inductive SSL belum otomatis tersedia.
6. Ensemble setelah CV lengkap; clean-session restore, submission tervalidasi, arsip dan commit provenance.
7. Impor report untuk membaca hasil, lalu isi tabel paper hasil dan audit gate G0–G5 sebelum mengubah kemajuan riset.

## Batas verifikasi

Tes sintetis memverifikasi pemetaan ID, loss/head, CV, checkpoint, validitas submission, pipeline, reporting dan perhitungan. Mereka tidak mengukur akurasi data kompetisi. Hitungan MAC mengecualikan functional texture/norm/aktivasi/pooling/backward/IO. Benchmark sintetis mengecualikan loader, augmentation, validation, test dan IO; nama GPU dan konfigurasi harus menyertai estimasi. VM/ketersediaan/kuota Kaggle tidak dapat disimpulkan dari perangkat lokal.

Dashboard/paper bukan model inference yang dilatih. Tabel hasil tetap kosong sebelum bukti nyata. Importer melakukan pemeriksaan schema, kurva dan skor di browser, tetapi tidak dapat mengautentikasi bahwa file berasal dari dataset kompetisi. Exporter Python melakukan pemeriksaan provenance fold; source dataset tetap perlu diaudit.
