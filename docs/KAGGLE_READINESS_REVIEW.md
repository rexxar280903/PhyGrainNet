# Pemeriksaan kesiapan Kaggle — 10 Oktober 2026

**Keputusan: siap memulai audit, baseline dan uji training singkat di Kaggle. Belum ada bukti bahwa full training pada data asli, penyimpanan versi Kaggle, atau evaluasi final sudah berhasil.**

Pemeriksaan ini dilakukan dari repository lokal. Tidak ada sesi Kaggle yang dijalankan atau submission yang dikirim. `project_status.json` dan gate riset tetap mencerminkan belum adanya hasil data asli.

## Hasil pemeriksaan

| Bagian | Kesiapan kode | Bukti yang masih diperlukan |
|---|---|---|
| Setup notebook | Perintah berhenti bila gagal; commit tercetak; checkout tidak dihapus; tahap mahal/ETS opsional | Input kompetisi, instalasi dan GPU pada sesi Kaggle sebenarnya |
| Audit data | Label, sample ID, PPM, foto, grup dan duplikasi byte diperiksa; kegagalan fatal menghentikan notebook | Hasil audit seluruh data asli, pemeriksaan skala/resize, near-duplicate dan validitas grup berdasarkan domain |
| Training | Profil GPU awal lebih kecil; uji singkat memakai ID terpisah; augmentasi berubah per epoch; NaN/Inf ditolak | Memori GPU, runtime, loss dan seluruh fold pada data asli |
| Penyimpanan | Prediksi, target, grup, konfigurasi, environment, log dan bobot tersimpan; ZIP dengan checksum dan snapshot source | Saved Version berhasil dan arsip muncul di Output serta dapat diunduh |
| Evaluasi | EMD, OOF, per-sample/per-group/per-point, MAE dan keanggotaan fold tersimpan; baseline/fold scratch diselaraskan | Skor baseline nyata, full CV, perbandingan model, ablation dan beberapa seed |
| Pemakaian ulang | `scripts/predict.py` memuat bobot tersimpan dan menghasilkan submission tanpa training | Sesi Kaggle baru dengan input output versi lama menghasilkan CSV yang sama |

## Perbaikan dari pemeriksaan ini

- Runner bawaan menjalankan audit → A000 → `C100_smoke` → arsip; full CV, SSL, ETS dan ensemble perlu diaktifkan sesuai tahap.
- Kegagalan pemasangan dependensi/perintah tidak lagi tersembunyi oleh shell notebook.
- A000 menyimpan evaluasi lengkap dan registry, sehingga hasil pertama dapat dievaluasi ulang dari target dan OOF.
- A000/A001/C100 menggunakan lima grouped folds dan seed 42. LOGO tersedia untuk sensitivitas terpisah.
- CV parsial diberi status `smoke_only`; ensemble menolak OOF parsial atau sample ID yang tidak lengkap.
- Pekerja pemuat data dibuat ulang tiap epoch sehingga seed augmentasi mengikuti epoch; cache gambar memasukkan asal file dan PPM.
- NaN/Inf tidak lagi diubah menjadi kurva yang tampak valid saat postprocessing.
- `scripts/export_results.py` membuat arsip yang mencakup hasil, bobot, audit, registry, source/config dan manifest checksum; `scripts/predict.py` menyediakan inference dari checkpoint.

## Batas yang perlu diketahui

Checkpoint disimpan setelah fold selesai atau pretraining selesai. Bobot ini dapat dipakai untuk inference/inisialisasi, tetapi **belum menyimpan optimizer/scheduler untuk melanjutkan training dari epoch yang terputus**. Memilih T4 x2 tetap menggunakan satu GPU pada implementasi saat ini.

Group lokasi masih perkiraan dari pola ID. SSL saat ini melihat foto validation dan test tanpa label; hasilnya perlu dilaporkan sebagai evaluasi transduktif. Metrik geoteknik tambahan, uncertainty multi-seed, ablation dan plot analisis belum lengkap. Tidak ada bukti peningkatan model sebelum hasil data asli tersedia.

Perubahan pemeriksaan ini berada di working tree lokal; notebook yang mengambil `main` dari GitHub hanya mendapat perubahan tersebut setelah disimpan ke commit/branch yang tersedia di GitHub. Pilih branch/commit yang sudah ditinjau atau unggah snapshot repository.

## Urutan menjalankan

1. Gunakan runner terbaru, pasang input kompetisi, aktifkan Internet dan pilih GPU. Untuk baseline CPU saja, set `RUN_SMOKE=False`.
2. Jalankan mode bawaan; periksa audit, skor A000, loss/waktu uji singkat, `fold0.pt`, registry dan ZIP.
3. Simpan versi Kaggle dengan tahap yang diinginkan, tunggu berhasil, periksa Output dan unduh arsip. Panduan: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks), [CLI resmi untuk output](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#kaggle-kernels-output).
4. Setelah audit dan smoke lolos, aktifkan A001/full C100. Evaluasi dengan seluruh grouped folds; abaikan skor `smoke_only` untuk memilih model.
5. Pada sesi baru, tambahkan output versi tersimpan sebagai input, muat checkpoint menggunakan `scripts/predict.py`, lalu bandingkan submission hasil pemuatan ulang dengan hasil semula.

Detail perintah dan file: [Kaggle Workflow](KAGGLE_WORKFLOW.md).

## Verifikasi lokal

**32 pengujian lulus (36,80 detik, Python 3.13 / PyTorch CPU).** Pemeriksaan konsistensi status README dan pemeriksaan whitespace perubahan juga lulus. Prediksi hasil pemuatan ulang cocok dengan prediksi tersimpan pada toleransi pembulatan CSV 0,000051, dan seluruh checksum file arsip cocok.

Pengujian memakai dataset sintetis dengan pola nama file kompetisi. Cakupan mencakup pipeline classical/CNN/SSL, EMD, label ETS, kurva/submission, baseline tersimpan, audit gagal pada label/duplikasi bermasalah, arsip dan checksum, restore checkpoint, penolakan ensemble parsial, cache serta sintaks sel notebook. Hasil ini memverifikasi integrasi lokal; tidak menggantikan pengujian Kaggle dan data asli.
