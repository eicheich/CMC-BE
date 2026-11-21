# Check My Citation API

Backend FastAPI dengan endpoint proof of concept untuk memeriksa satu referensi akademik.

Jalankan perintah berikut di PowerShell dari folder `backend`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger UI tersedia di <http://127.0.0.1:8000/docs>. Endpoint kesehatan tersedia di <http://127.0.0.1:8000/health>.

Gunakan `POST /api/v1/verifications/preview` di Swagger UI dengan contoh berikut:

```json
{
  "citation": "Vaswani, A., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems."
}
```

Input dibatasi 2.000 karakter. Backend mencari metadata di Crossref dan OpenAlex. Jika DOI ada, pencarian DOI didahulukan; pencarian teks dipakai jika tidak ada hasil DOI. Masing-masing provider dibatasi lima kandidat. `CROSSREF_CONTACT` dan `OPENALEX_API_KEY` opsional; jika digunakan, atur sebagai environment variable PowerShell sebelum menjalankan server, misalnya `$env:CROSSREF_CONTACT="nama@example.com"`. File `.env` tidak dimuat otomatis.

Status hasil: `verified_match` berarti judul dan setidaknya satu atribut pendukung cocok kuat; `possible_match` berarti kandidat relevan tetapi buktinya kurang; `metadata_mismatch` berarti atribut penting bertentangan; `not_found` berarti pencarian berhasil tetapi tidak menemukan kandidat cukup relevan; `verification_unavailable` berarti kedua sumber gagal. `not_found` bukan bukti referensi palsu. Skor kecocokan adalah skor heuristik, bukan probabilitas.

Parser hanya mengenali pola APA sederhana secara best effort. Judul yang mengandung titik, nama penulis tidak umum, dan metadata provider yang tidak lengkap dapat menimbulkan false positive atau false negative. Periksa `comparison`, `reasons`, dan `source_url` sebelum mengambil keputusan.

Untuk menjalankan test:

```powershell
python -m pytest
```

