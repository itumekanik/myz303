# İncelemeler

Projenin incelemeleri ve her bulgunun ne olduğu bu klasörde. Bir incelemenin konusunun kapanıp
kapanmadığı, neyin sonraya bırakıldığı buradan izlenir.

## Dosya adları

`<sıra>_<tarih>_<konu>.md`. Örnek: `5_2026-11-26_pedagojik-inceleme.md`.

- **Sıra:** 1, 2, 3 … İncelemeler yapıldıkları sırayla numaralanır.
- **Tarih:** incelemenin yapıldığı gün (YYYY-AA-GG).
- **Konu:** kısa, küçük harfli, tireli.

## Her dosyada

1. **Kapsam:** neye ve nasıl bakıldı.
2. **Bulgular:** her biri bir kimlikle (örneğin `P3`, `K2`). Bulgunun yeri, sorunu ve durumu bir
   tabloda yazılır.
3. **Doğrulanamayanlar:** bakılamayan ya da denenemeyen şeyler.
4. **Takip:** sonradan ne yapıldı. Bir bulgu sonraki bir işte kapanırsa, durumu bu dosyada
   güncellenir ve hangi işte kapandığı yazılır.

## Durumlar

| Durum | Anlamı |
| --- | --- |
| **Çözüldü** | Düzeltildi ve doğrulandı. Hangi commit'te olduğu `git log -- reviews/<dosya>` ile ya da yazılan commit numarasından bulunur. |
| **Açık: eğitmen kararı** | Bir öğretim tercihi. Karar verilince yapılır ya da bırakılır. |
| **Açık: büyük iş** | Yapılması kararlaştırılırsa ayrı bir iş olarak ele alınır. |
| **Açık** | Yapılacak, henüz yapılmadı. |
| **Ertelendi** | Bilerek sonraya bırakıldı, nedeni yanında yazılı. |
| **Bırakıldı** | Bilerek yapılmayacak, nedeni yanında yazılı. |

## Liste

| Dosya | Tarih | Konu | Açık kalanlar |
| --- | --- | --- | --- |
| [1_2026-11-14_arayuz-ve-duyarlilik.md](1_2026-11-14_arayuz-ve-duyarlilik.md) | 2026-11-14 | Arayüz, 320 px'den geniş ekrana | yok (3 madde bilerek bırakıldı) |
| [2_2026-11-19_pedagojik-inceleme.md](2_2026-11-19_pedagojik-inceleme.md) | 2026-11-19 | İlk üç bölümün pedagojik incelemesi | 1 (CNN Teori sırası, 5. incelemede C3) |
| [3_2026-11-22_genel-gozden-gecirme.md](3_2026-11-22_genel-gozden-gecirme.md) | 2026-11-22 | Site ve Python panelinin genel gözden geçirmesi (1.0.1) | 2 |
| [4_2026-11-26_genel-kontrol.md](4_2026-11-26_genel-kontrol.md) | 2026-11-26 | Genel kontrol, ana sayfa bağlantısı, kod taraması | 3 |
| [5_2026-11-26_pedagojik-inceleme.md](5_2026-11-26_pedagojik-inceleme.md) | 2026-11-26 | İlk üç bölümün ikinci pedagojik incelemesi | 22 (çoğu eğitmen kararı) |

İlk üç dosya, o günlerdeki konuşmalardan ve commit'lerden sonradan yazıldı. Onlarda yalnızca
bulguların özeti ve bugünkü durumları var.
