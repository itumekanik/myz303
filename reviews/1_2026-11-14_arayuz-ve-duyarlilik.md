# 1 · Arayüz ve duyarlılık incelemesi

- **Tarih:** 2026-11-14
- **Commit:** `1ed5f98` (Make the dashboard responsive from 320px to wide screens)
- **Not:** Bu kayıt 2026-11-26'da, o günkü rapordan sonradan yazıldı. İnceleme, sitenin `web/`
  klasörüne taşınmasından önce, Python panelinin `static/` sayfalarında yapıldı. `static/` 1.0.0'da
  emekliye ayrıldı ve düzeltmeler `web/`'e taşındı.

## Kapsam

O sırada tek sayfa olan CNN panelinin beş sekmesi, 340–1920 piksel arası 8 genişlikte ölçüldü:
taşma, üst çubuk, dokunmatik kullanım ve yazı kesilmesi.

## Bulgular

| No | Bulgu | Durum |
| --- | --- | --- |
| A1 | Üst çubuk dar ekranda satır satır uzuyordu; telefonda ekranın yaklaşık %27'sini kaplıyordu, 768 px'de başlık sekmelerin altında kesiliyordu | Çözüldü (`1ed5f98`): dar ekranda kısa rozetler ve ikonlar, telefonda aşağı kaydırınca gizlenen çubuk |
| A2 | Dokunmatik ekranda küçük dokunma alanları (bilgi düğmeleri, dil/tema, kopyalama, filtre okları, onay kutuları) | Çözüldü (`1ed5f98`) |
| A3 | KaTeX'in ekran okuyucu kopyası telefonda sayfayı yana taşırıyordu (340 px'de Matematik sekmesi) | Çözüldü (`1ed5f98`) |
| A4 | `optimizer.zero_grad()` gibi kod adları kelimenin ortasından bölünüyordu | Çözüldü (`1ed5f98`) |
| A5 | Çizim alanı yalnızca "Fareyle" diyordu | Çözüldü (`1ed5f98`) |
| A6 | Metin içindeki kavram bağlantılarının dokunma alanı küçük | Bırakıldı: satır içi bağlantılar için kabul gören istisna |
| A7 | Matris hücrelerindeki 9,5 piksellik sayılar | Bırakıldı: küçük ızgaralara sığmaları gerekiyor |
| A8 | Teori'deki mimari şemasında işlemlerin üzerindeki küçük bağlantılar | Bırakıldı: telefonda yana kaydırılan bir diyagram |

## Takip

- 2026-11-22 (1.0.2): Firefox, WebKit ve öykünülmüş telefonlarda da kontrol edildi.
- 2026-11-26: üst çubuğun geniş ekranlarda iki satıra düşmesi ayrıca ele alındı
  ([4. inceleme](4_2026-11-26_genel-kontrol.md), K2).
