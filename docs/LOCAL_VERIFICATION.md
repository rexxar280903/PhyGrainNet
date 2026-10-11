# Verifikasi lokal rilis 0.3 — 10 Oktober 2026 (WIB)

## Bukti yang dihasilkan

- Suite Python lengkap: **38 passed**, 23,09 detik, menggunakan data sintetis. Suite sebelumnya lulus 32 dan 37 tes; final mencakup pengujian arsip sumber/checksum.
- Numerical JavaScript checks lulus: softmax stabil untuk logit ekstrem, total massa, monotonicity, bobot log EMD berjumlah 5, batas 500, perhitungan waktu multi-fold/seed, dan penolakan input invalid.
- Profil C100: **1.518.836 parameter**, **9.717.806.080 MAC/batch** pada B=2,T=6,S=2,256×256. Counter mencakup Conv2d/Linear saja.
- Benchmark **CPU sintetis**, 1 warm-up dan 2 step: median **1,76275 detik/step** pada Torch 2.13.0+cpu, Python 3.13.15, dua thread. Ini kalibrasi lokal singkat, bukan throughput Kaggle atau waktu full training. JSON tersedia di `docs/local_benchmark.json`.
- Browser: dashboard dan paper tampil tanpa error/warning console pada pemeriksaan. Alur klik berpindah ke tahap output; edit logit mengubah EMD; tiga seed menghasilkan 30.000 update / 5,42 jam pada asumsi 0,5 detik dan 30% overhead.
- Tooltip GSD tampil setelah klik; penjelasan dapat dibaca di kamus. Layout dashboard diperiksa pada lebar 360 px tanpa horizontal overflow; kontrol kurva tetap bekerja.
- Paper memuat 13 section, 17 baris hyperparameter dari YAML, dan delapan baris diagram tensor.
- Impor laporan fixture `SYNTHETIC_UI_TEST_NOT_COMPETITION` diterima, kurva/skor konsisten, tabel dan grafik terisi. Schema laporan invalid ditolak. Fixture ini bukan hasil model pada dataset kompetisi.

## Batas

Tidak ada dataset Kaggle asli di workspace, training CUDA, score leaderboard, atau bukti akurasi geoteknik baru. Tests dan benchmark CPU tidak mengubah gate riset G1–G5 menjadi passed. Runtime/input/licensing ETS dan kuota GPU tetap diperiksa pada Kaggle. File hasil yang dipilih di importer dibaca lokal dan tidak dikirim ke server.

Pytest memerlukan eksekusi di luar sandbox terbatas Windows karena folder privat yang dibuat pytest ditolak oleh sandbox; run tetap menggunakan folder sementara di `outputs/` proyek. Tidak ada perubahan pengaturan sistem.
