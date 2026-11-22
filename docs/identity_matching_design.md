# Phase 4A — Identity Matching dan Metadata Checking

Dokumen ini merancang perubahan berikutnya. Tidak ada perubahan kode produksi, kontrak API, model, atau threshold pada Phase 4A.

## Arsitektur saat ini

`verify()` menerima teks sitasi dan style, lalu memanggil parser untuk menghasilkan `Citation`. `fetch_candidates()` mencoba DOI di Crossref dan OpenAlex terlebih dahulu. Jika salah satu menghasilkan `Work`, fungsi langsung mengembalikan kandidat tersebut. Jika tidak, fungsi mencari berdasarkan judul (atau teks sitasi mentah), paling banyak lima hasil per sumber. `Work` menyimpan judul, penulis, tahun, DOI, `source`, dan `source_url`; OpenAlex ID belum menjadi field tersendiri.

Untuk setiap kandidat, `compare()` menghasilkan `Comparison` dan `match_score`. Skor memakai bobot judul 0,6, penulis 0,1, tahun 0,1, dan DOI 0,2, lalu dinormalisasi atas atribut yang tersedia. Kandidat dipilih dengan urutan `(relevant, score, not issues)`. `consistency_issues()` baru memeriksa konflik bila judul cukup relevan atau DOI cocok. `classify()` memberi prioritas pada issue, lalu mempertimbangkan kelengkapan, bukti identitas, dan skor untuk `verified_match`; sisanya menjadi `possible_match` atau `not_found`.

Respons memakai `PreviewResponse` dengan `status`, `match_score`, `input`, `best_match`, `comparison`, dan `reasons`. `verification_unavailable` dipakai jika tidak ada kandidat dan kedua sumber metadata tidak berhasil. ML v4.1 hanya diamati lewat `observe_shadow()` untuk kandidat terpilih; probabilitasnya tidak memengaruhi ranking, skor, status, atau respons.

Keterbatasan utama: skor yang sama dipakai untuk mengurutkan kandidat identitas dan merangkum konsistensi metadata. Tahun, DOI, atau penulis salah dapat menurunkan skor pasangan yang sebenarnya merujuk publikasi sama. DOI yang berhasil di-resolve juga menghentikan pencarian judul, sehingga publikasi yang dimaksud tidak ditemukan jika DOI sitasi menunjuk karya lain. Audit training v4.1 menunjukkan label negatif sering berarti metadata sitasi salah, sehingga model tersebut belum layak dijadikan penentu identitas.

## Arsitektur yang diusulkan

```text
Citation
  → DOI lookup + title search (bila judul tersedia)
  → normalisasi dan deduplikasi kandidat
  → identity assessment per kandidat
  → pilih kandidat dengan bukti identitas terbaik
  → metadata checking per field untuk kandidat terpilih
  → pemetaan ke status publik dan PreviewResponse
  → observasi ML v4.1 tetap shadow
```

### A. Identity Matching

Identity Matching menjawab apakah **pasangan** `Citation`–`Work` merujuk karya yang sama. Ia menghasilkan penilaian internal `same`, `possible`, atau `different`, dengan alasan dan atribut yang mendukungnya. Jika tidak ada kandidat yang dapat dinilai, hasil keseluruhan adalah `unresolved`. Penilaian `different` pada satu kandidat tidak membuktikan bahwa referensi input palsu.

- DOI cocok dan judul selaras: bukti identitas kuat, walaupun tahun atau penulis sitasi salah. DOI cocok dengan judul yang bertentangan tidak boleh otomatis dianggap karya sama; DOI mungkin menunjuk publikasi lain.
- DOI sitasi berbeda, tetapi judul sangat selaras dan penulis atau atribut independen lain mendukung: kandidat dapat dinilai `same` setelah hasil pencarian judul dibandingkan dengan target DOI. Perbedaan DOI dicatat di tahap metadata.
- Judul mirip saja, terutama judul umum: `possible`. Judul mirip dengan DOI, penulis, atau konteks bibliografi yang bertentangan dapat menghasilkan `different` untuk kandidat itu. Metadata yang hilang tetap `unknown`, bukan `match` maupun `mismatch`.
- Kecocokan DOI/OpenAlex ID dari Crossref dan OpenAlex dapat menggabungkan rekaman duplikat kandidat. DOI berbeda tidak cukup untuk menyatakan dua rekaman karya berbeda; periksa judul, penulis, versi/preprint, dan provenance sebelum memutuskan.

Pemilihan kandidat mendahulukan kekuatan bukti identitas, bukan `match_score` lama atau jumlah metadata yang kebetulan terisi. Jika tidak ada kandidat `same`, kandidat `possible` boleh ditampilkan dengan ketidakpastian. Kandidat yang jelas berbeda tidak boleh ditampilkan sebagai publikasi yang telah terverifikasi.

### B. Metadata Checking

Setelah kandidat dipilih, periksa judul, penulis, tahun, dan DOI secara terpisah. Setiap field mendapat keadaan internal `match`, `mismatch`, atau `unknown`. `unknown` dipakai saat parsing atau metadata sumber tidak cukup; ia tidak menambah bukti positif maupun konflik. Pemeriksaan penulis mempertahankan toleransi untuk inisial, `et al.`, nama keluarga multikata, dan variasi urutan nama. Issue yang dapat dipastikan menghasilkan alasan spesifik, misalnya tahun sitasi dan tahun metadata, atau DOI sitasi dan DOI karya yang ditemukan. Ringkasan internal metadata adalah `consistent`, `mismatch`, atau `insufficient`.

`match_score` tetap memakai arti dan format saat ini selama migrasi, agar klien lama tidak rusak. Ia **bukan** probabilitas identitas dan tidak menjadi gerbang tunggal status. Jangan mengubah bobot atau threshold produksi hanya berdasarkan 50 kasus evaluasi.

## Alur keputusan dan status

| Keadaan | Status publik | Penilaian identitas | Pemeriksaan metadata |
| --- | --- | --- | --- |
| Kandidat karya sama, field yang dapat diperiksa selaras dan bukti cukup | `verified_match` | `same` | `consistent` |
| Kandidat karya sama, tahun salah | `metadata_mismatch` | `same` | `mismatch` |
| Kandidat karya sama, DOI sitasi salah | `metadata_mismatch` | `same` | `mismatch` |
| Kandidat karya sama, penulis sitasi salah | `metadata_mismatch` | `same` | `mismatch` |
| Judul mirip tetapi bukti identitas belum cukup | `possible_match` | `possible` | `insufficient` atau `mismatch` yang belum dapat dikaitkan dengan karya sama |
| Semua kandidat yang diperiksa berbeda dan karya yang dimaksud tidak ditemukan | `not_found` | `unresolved` | `insufficient` |
| Sumber kurang lengkap, tanpa konflik yang dapat dipastikan | `possible_match` | `same` atau `possible` | `insufficient` |
| Seluruh sumber metadata gagal dan tidak ada kandidat | `verification_unavailable` | `unresolved` | `insufficient` |

Untuk kompatibilitas, jika DOI input berhasil di-resolve ke **publikasi yang berbeda** dan pencarian judul tidak menemukan karya yang dimaksud, `metadata_mismatch` masih dapat dipakai dengan alasan eksplisit “DOI resolves to a different publication”. Dalam kasus itu penilaian identitas kandidat adalah `different`, bukan `same`; `best_match` adalah target DOI, bukan karya yang telah diverifikasi. Perilaku ini perlu didokumentasikan jelas di UI. `not_found` berarti tidak ditemukan pada sumber yang tersedia, bukan bukti referensi palsu. Jika satu API gagal tetapi API lain memberi kandidat, respons tetap mengikuti bukti yang ada dan menyebut keterbatasan sumber bila relevan.

## Perubahan schema dan API yang minimal

Lima nama `status` tetap cukup sebagai hasil ringkas, tetapi **tidak cukup** untuk membedakan “karya sama dengan metadata salah” dari “DOI menunjuk kandidat lain”. Usulan perubahan aditif pada `PreviewResponse`:

```python
identity_assessment: Literal["same", "possible", "different", "unresolved"] | None = None
metadata_assessment: Literal["consistent", "mismatch", "insufficient"] | None = None
```

Field opsional menjaga request lama (`style` default `apa`) dan pembaca respons lama tetap berfungsi. `Citation`, `Work`, `Comparison`, `match_score`, `reasons`, dan endpoint tetap ada. `Comparison` tetap memakai `None` untuk atribut yang tidak dapat dibandingkan. Dalam implementasi, hasil detail per field dan provenance kandidat dapat disimpan secara internal; tidak perlu membuat field respons baru untuk setiap atribut. Frontend Vue 3 dapat mengadopsi dua field baru bertahap, dengan fallback ke `status` selama transisi.

## Risiko dan edge cases

- DOI salah yang valid dapat menutup title search pada implementasi saat ini. Pencarian tambahan perlu dibatasi jumlah request dan timeout agar tetap layak pada hosting cPanel.
- Judul umum, judul versi preprint, transliterasi, serta metadata Crossref/OpenAlex yang berbeda dapat mengaburkan identitas. Gunakan `possible` saat bukti belum cukup.
- Penulis dengan inisial, `et al.`, nama keluarga multikata atau ber-hyphen tidak boleh dianggap konflik hanya karena bentuk teksnya berbeda.
- Tahun online dan tahun terbit final dapat berbeda. Tunjukkan perbedaan yang teramati tanpa menyimpulkan identitas berbeda dari tahun saja.
- Metadata sumber yang kosong tidak boleh menaikkan keyakinan karena skor ternormalisasi. Kegagalan satu sumber berbeda dari kegagalan seluruh sumber.
- Deduplikasi Crossref/OpenAlex berdasarkan DOI atau OpenAlex ID tidak boleh menghapus provenance atau menyamakan dua versi hanya karena judul sama.
- Model v4.1 tetap shadow karena label training mencampur ketepatan metadata dengan identitas publikasi. Probabilitasnya tidak dipakai sebagai penentu.

## Rencana implementasi bertahap

1. Tambahkan hasil identity assessment dan metadata checking **internal** serta unit test, tanpa mengubah respons. Catat keputusan untuk diagnosis, tanpa teks sitasi mentah dalam log.
2. Perbaiki retrieval agar DOI lookup dan title search dapat dibandingkan ketika DOI dan judul saling bertentangan. Batasi hasil, timeout, dan jumlah request; gunakan `httpx.AsyncClient` yang sudah ada. Tidak perlu database, Redis, worker, atau GPU di cPanel.
3. Pilih kandidat berdasarkan bukti identitas; jalankan metadata checking pada kandidat itu; petakan hasil ke lima status lama. Pertahankan `match_score` lama sebagai informasi kompatibilitas.
4. Setelah perilaku dan kontrak diuji, tambahkan dua field opsional pada `PreviewResponse` dan koordinasikan pemakaian bertahap di frontend. Model tetap shadow sampai ada model dengan label identitas yang benar dan evaluasi independen.

## Strategi testing

Gunakan respons Crossref/OpenAlex yang di-mock; tidak ada network dalam unit test. Uji pasangan karya sama yang konsisten, tahun/DOI/penulis salah, DOI yang menunjuk publikasi lain sementara title search menemukan karya yang dimaksud, judul mirip pada karya berbeda, metadata penulis/DOI/tahun kosong, sumber gagal sebagian/seluruhnya, dan versi preprint. Pastikan atribut tak tersedia tetap `unknown`, `not_found` tidak diberi alasan “palsu”, serta ranking dan status tidak berubah karena probabilitas ML shadow. Tambahkan regression test untuk bentuk JSON lama dan field opsional baru. Evaluasi offline berikutnya harus memakai label identitas publikasi; pemeriksaan metadata dinilai terpisah.
