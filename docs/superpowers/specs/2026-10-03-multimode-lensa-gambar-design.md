# Desain: Mode Multi FILTERS + LENSA + GAMBAR

Tanggal: 2026-10-03
Status: menunggu review
Baseline: repo `~/Filters` (branch `main`)

## 1. Latar & Tujuan

Proyek Filters saat ini: portal 2 tangan dengan 8 filter, ganti filter via
gerakan maju-mundur (histeresis), sumber video fleksibel
(`--source`, launcher `filters`, thread baca frame anti-delay).

Script terpisah `~/Downloads/main.py` (RETROLENS) berisi fitur yang ingin
ditambahkan: quad dari gestur "L" dua tangan + warp perspektif + 7 lensa
retro + auto-foto + mode gambar (kanvas + pena gesture).

Tujuan: **satu aplikasi multi-mode** — tidak ada fitur Filters lama yang
hilang, tidak ada fitur RETROLENS yang hilang.

## 2. Cakupan

Masuk:
- Mode LENSA (quad-L, smoothing, warp perspektif, 7 lensa retro, grade global)
- Mode GAMBAR (kanvas, pena, gesture, warna pena)
- Auto-foto 3 detik + flash + simpan manual ke `foto/`
- State machine 3 mode dengan transisi gestur hold 2 detik + keyboard
- HUD retro (REC, timestamp, FPS, mode, jumlah foto)

Tidak masuk (non-goal):
- Perubahan render portal mode FILTERS (mask poligon tetap)
- Penggantian API tracking (tetap `mp.solutions.hands`, tanpa model
  `hand_landmarker.task`)
- Perubahan launcher `filters`, `frame_source`, `--source`, `filters.py`

## 3. Arsitektur

### 3.1 Pipeline (main.py, sekali per frame)

```
FrameSource → cv2.flip → hand_tracking.detect(frame) → mode.update(...)
→ HUD → cv2.imshow → waitKey → event keyboard global
```

### 3.2 Interface mode (modes.py)

```python
class Mode:
    def on_enter(self): ...        # reset state (quad/kanvas/indeks filter)
    def update(self, frame, hands, now, key) -> (frame, pindah: str | None)
```

- `hands`: objek hasil `hand_tracking.detect()` — berisi tangan kiri/kanan
  (landmark + label), daftar semua tangan dengan flag jari terbuka.
- `pindah`: nama mode tujuan bila gestur hold terpicu, selain itu `None`.
- `key`: satu byte tombol yang sudah dialokasikan untuk mode itu
  (global ditangani main).

State machine: `mode ∈ {FILTERS, LENSA, GAMBAR}` + `mode_sebelumnya`
(untuk kembali dari GAMBAR). Keyboard `f`/`l`/`g` langsung pindah;
`m` cycle; `on_enter` selalu dipanggil saat masuk.

### 3.3 Pembagian file

| File | Tanggung jawab |
|---|---|
| `main.py` | loop, pipeline, state machine, keyboard global, HUD |
| `modes.py` | `FiltersMode` (logika lama dipindah utuh), `HoldTransition`, classifier gestur, pemetaan transisi |
| `lensa.py` | quad-L (`urutkan_quad`, `cocokkan`, smoothing), `warp_efek`, komposit, 7 lensa, `grade_retro` |
| `draw.py` | kanvas, pena, histeresis mulai/berhenti, gesture hapus/pindah |
| `capture.py` | simpan foto (auto/manual), flash, konter, path absolut `foto/` |
| `hand_tracking.py` | diperluas: `detect(frame)` → label kiri/kanan + jari terbuka + landmark |

Tidak berubah: `frame_source.py`, `launcher.py`, `filters.py`,
`geometry.py`, `test_*` yang ada.

## 4. Gestur & Transisi

Classifier (dari `get_extended_fingers`):
- `tunjuk_ketat`: telunjuk saja (jempol + sisa turun)
- `telapak`: kelima jari terbuka
- `kepal`: semua jari turun
- Gesti L (LENSA): jempol+telunjuk terbuka, sisanya turun (2 tangan)

`HoldTransition(hold_s=2.0, reset_grace_s=0.3)`: gestur harus bertahan
2 detik → trigger sekali; hilang >0,3 detik → reset. Grace 0,5 detik saat
`on_enter` supaya sisa pose lama tidak langsung memicu.

Aturan jumlah tangan: semua hold dihitung dari **≥1 tangan yang cocok**,
kecuali **FILTERS → LENSA (kepal) mensyaratkan kedua tangan** — karena di
FILTERS kedua tangan selalu terlihat, syarat dua tangan menekan pemicu
tidak sengaja.

Pemetaan (hold 2 detik):

| Dari | Gestur | Ke | Alasan bebas konflik |
|---|---|---|---|
| FILTERS | kepal (2 tangan) | LENSA | kepal tak dipakai di FILTERS |
| FILTERS | tunjuk_ketat | GAMBAR | jempol turun, tak bentrok pose portal |
| LENSA | tunjuk_ketat | GAMBAR | konsisten RETROLENS |
| LENSA | kepal | FILTERS | kepal tak dipakai saat quad-L |
| GAMBAR | telapak | mode_sebelumnya | kepal tetap = hapus kanvas |

Catatan: di FILTERS, deteksi tangan untuk portal tetap seperti sekarang
(tidak memeriksa ekstensi jari); detektor hold dijalankan paralel dengan
grace supaya tidak ada pemicu tak sengaja saat portal baru terbuka.

Keyboard global (cadangan): `f` FILTERS, `l` LENSA, `g` GAMBAR, `m` cycle,
`q` keluar, `d` debug. Keyboard mode: `spasi` ganti filter (FILTERS),
`p` warna pena, `s` simpan manual, `c` hapus kanvas.

## 5. Mode FILTERS (regresi)

Logika lama pindah ke `modes.py::FiltersMode` tanpa perubahan perilaku:
portal 4 titik (telunjuk+jempol × 2 tangan) → `render_portal` dengan
`FILTROS[i]` → `ClosingGestureDetector` ganti filter saat tangan
didekatkan. `geometry.py` dan `filters.py` tidak disentuh.

## 6. Mode LENSA

Port RETROLENS dengan perilaku identik:
- 4 titik = (jempol, telunjuk) × 2 tangan berlabel L → `urutkan_quad`
  (urut sudut) → `cocokkan` (roll vs frame sebelumnya) → smoothing 0,4
- `hilang` counter; hilang > 6 frame → quad dilepas; >20 → reset penuh
- `diag < 120px` → status GENGGAM (lingkaran + "TARIK UNTUK MEMBUKA")
- Buka → `warp_efek` (perspektif ke rect) → lensa `(roi, t)` → komposit
  mask ke tampilan
- Lensa: `MONO, KONTRAS, FILM, GARIS, AMBANG, DITHER, NEGATIF`
  (registry `LENSA_LIST`, terpisah dari `FILTROS`)
- Seluruh frame kena `grade_retro` (LUT + vignette + scanline); quad
  di-composit dari sumber tanpa grade (sama asli)
- Overlay: kurung quad, progress bar tahan foto, angka hitung mundur,
  ROLL/PITCH/YAW saat quad stabil

## 7. Mode GAMBAR

- Kanvas `uint8` hitam, ukuran frame saat `on_enter` (cek ulang bila
  ukuran berubah); bertahan antar pindah mode sampai `c`/hapus
- Telunjuk: pena — smoothing EMA 0,5, tolak lompatan > 160px, tebal 6
- Histeresis frame: mulai gambar setelah 2 frame berturut tunjuk,
  berhenti setelah 4 frame tak cocok (angka sama seperti RETROLENS)
- Kebral: hapus seluruh kanvas + reset state pena
- Peace: indikator "PINDAH" (tanpa aksi, sesuai asli)
- Latar: `grade_retro × 0.5` + glow Gaussian kanvas + kanvas overlay
- Warna pena: `[PUTIH, CYAN, MAGENTA, AMBER]`, ganti `p`

## 8. Capture

- Path: `~/Filters/foto/` (path absolut tetap — foto selalu di rumah, dari mana pun kode dijalankan)
- Auto (LENSA): quad aktif + geser ≤ 9px → tahan 3 detik → simpan **hasil
  lensa (rect warp)** sebagai `LENS_YYYYmmdd_HHMMSS.png` → flash putih
  0,3s → reset quad (hilang=99, quad=None) → konter +1
- Manual: `s` simpan frame tampilan penuh `MANUAL_*.png`
- Gagal tulis: cetak pesan, aplikasi tetap jalan

## 9. HUD

Port elemen RETROLENS: titik REC berkedip, timestamp, baris bawah
`MODE | LENS | FPS | FOTO n`, teks status gerakan, ROLL/PITCH/YAW (LENSA),
`GORESAN n` (GAMBAR), indikator sumber aktif dari `--source`.

## 10. Strategi Testing

TDD, unittest (sudah ada 28 test — wajib tetap hijau):

Unit baru:
- `HoldTransition`: trigger 2s, reset grace, latch sekali, grace on_enter
- Classifier jari: seluruh kombinasi penting (L, tunjuk_ketat, telapak,
  kepal, peace)
- `urutkan_quad`/`cocokkan`/`sisi`/`orientasi`: invarian rotasi/refleksi
- `warp_efek`: tolak quad < 16px, meta bbox dalam batas frame
- `capture`: penamaan unik, direktori dibuat, tidak menimpa
- `draw`: kepal→kanvas kosong, stroke tergambar saat tunjuk stabil,
  lompatan > 160px ditolak, kanvas bertahan pindah mode
- `FiltersMode`: regresi identik — portal terbuit saat 2 tangan,
  ClosingGestureDetector pindah filter
- Pemetaan transisi: semua sel pada tabel §4

E2E manual (checklist): `filters` → 3 mode via gestur + keyboard,
foto tersimpan di `foto/`, flash muncul, stream delay tetap rendah,
`--source` kamera/WiFi/USB tetap jalan.

## 11. Risiko & Mitigasi

- Port logika merusak FILTERS → mitigasi: test regresi + pindah tanpa
  mengubah fungsi asli
- Konflik gestur hold vs pose portal → mitigasi: pemetaan bebas konflik §4
  + grace 0,5s
- Kamera 1080p30 + proses berat → `FrameSource` sudah membuang frame basi;
  ukuran kanvas mengikuti frame, warp dibatasi bbox
- Dua registry filter (FILTROS vs LENSA_LIST) tertukar → test eksplisit
