# 2 · İlk üç bölümün pedagojik incelemesi

- **Tarih:** 2026-11-19
- **Commit:** `135901d` (Review the first three chapters for teaching and fix what was found)
- **Not:** Bu kayıt 2026-11-26'da, o günkü rapordan sonradan yazıldı.

## Kapsam

Perceptron, MLP ve CNN bölümlerinin bütün sekmeleri. Sayısal iddialar kod çalıştırılarak (Node,
NumPy, PyTorch) doğrulandı.

## Genel sonuç

İçerik matematiksel olarak sağlamdı. Zayıflıklar üç yerde toplanıyordu: bölümler arası süreklilik
(CNN, 1. ve 2. bölümden önce yazıldığı için onları anmıyordu), etkin öğrenme (hedef, özet,
"tahmin et ve gözle" yönergeleri yoktu) ve yanlış kavrayışa yol açan bazı ifadeler.

## Bulgular

| No | Bulgu | Durum |
|---|---|---|
| B1 | CNN bölümü nöronu, aktivasyonu, çapraz entropiyi ve geri yayılımı 1–2. bölümlere bağlanmadan yeniden anlatıyordu | Çözüldü (`135901d`): giriş MLP'nin görüntüdeki sorunundan başlıyor, tekrarlar bağlantıyla hatırlatılıyor |
| B2 | MLP bölümü perceptron kuralını "yalnızca hata sayan" bir yöntem gibi anlatıyordu (1. bölümle çelişki) | Çözüldü (`135901d`) |
| B3 | Öğrenme hedefi, bölüm sonu özeti, tahmin yönergeleri ve sıralı, ipuçlu alıştırmalar yoktu | Çözüldü (`135901d`) |
| B4 | XOR demosunda "yerel minimum" denmesi (aslında doymuş, düz bölge) | Çözüldü (`135901d`); literatür notu 5. incelemede düzeltildi (M3) |
| B5 | "PyTorch'un başlatması değerleri ne patlatır ne söndürür" | Çözüldü (`135901d`) |
| B6 | Dropout'un "birbirini tamamlayan öznitelikler" öğrettiği | Çözüldü (`135901d`) |
| B7 | Havuzlamanın kaymaya "tamamen" dayanıklı olduğu | Çözüldü (`135901d`) |
| B8 | Eğitim kaybının doğrulama kaybından yüksek olması yalnızca Dropout'a bağlanıyordu | Çözüldü (`135901d`) |
| B9 | Perceptron'da "×2" düğmesinin sınırı oynatması | Çözüldü (`135901d`) |
| B10 | Bozuk alıştırmalar: perceptron öğrenme oranı alıştırması fark göstermiyordu, MLP ReLU alıştırmasında doğru kod gradyan kontrolünden geçmiyordu, "1 piksel kayınca girdilerin çoğu değişir" yanlıştı | Çözüldü (`135901d`) |
| B11 | Terim birliği: "çapraz entropi", z = ön-aktivasyon; **bölüm** = chapter, **kısım** = sekme içindeki numaralı başlık | Çözüldü (`135901d`); bir kalıntı 5. incelemede düzeltildi (P9) |
| B12 | CNN sayfasının adı "SimpleCNN Paneli" idi | Çözüldü: eğitmen "Evrişimli ağ (CNN)" adını seçti |
| B13 | CNN Teori sırası: en ağır kısım BatchNorm, ReLU hatırlatmasından önce geliyor | **Açık: eğitmen kararı.** Yalnızca "ilk okumada atlayabilirsiniz" notu eklendi; 5. incelemede C3 olarak yeniden ele alındı |
| B14 | Gösterim: MLP Matematik'te L hem kayıp hem katman sayısı; CNN Matematik'te K hem katman hem sınıf sayısı | Not eklendi. 5. incelemede CNN'deki notun aslında eksik olduğu görüldü (C5) |
| B15 | Doğrulanması gereken tarihler: Mark I (1960) ve IBM 704 gösterimi, Werbos 1974, Fukushima 1980, LeCun 1989, Ioffe ve Szegedy 2015, Santurkar 2018, Hamey 1998 | Çözüldü (2026-11-26, 5. inceleme): hepsi kaynakla doğrulandı. Hamey 1998'in iddiası tartışmalı çıktı, metin buna göre düzeltildi (M3) |

## Takip

- 2026-11-26: [5. inceleme](5_2026-11-26_pedagojik-inceleme.md) aynı üç bölüme, bu arada eklenen
  Alıştırma sekmesiyle birlikte yeniden baktı.
