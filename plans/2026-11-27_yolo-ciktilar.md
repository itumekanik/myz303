# YOLO bölümüne yeni sekme: "Çıktılar" (eğitimin ürettiği dosyaları okumak)

- **Tarih:** 2026-11-27
- **Durum:** uygulandı (2026-11-27): A (veri: `yolo-outputs.js`), B (sekme `outputs`), C (10 kısım ve demolar), D (Alıştırma'da 10 soru, sözlükte 8 terim), E (doğrulama).
- **Öğretim üyesinin isteği:** YOLO eğitimi sırasında ve sonrasında oluşan çıktıların anlatımı için ayrı bir
  bölüm. Özellikle karışıklık matrisi ve TP, FP, FN… kavramları, pedagojik örneklerle.

## Kararlar (öğretim üyesi, 2026-11-27)

1. **Yer:** YOLO bölümünde yeni bir sekme, "Eğitim"den hemen sonra: **Çıktılar** (`#ciktilar`, sekme kimliği
   `outputs`). Bölüm numaraları değişmez.
2. **Örnekler:** dördü birden:
   - bizim modelin gerçek karışıklık matrisi (131 doğrulama karesi), hücreye tıklayınca örnekler;
   - oyuncak örnek (öğrencinin eşiği değiştirdiği küçük bir kare);
   - günlük hayattan / mühendislikten benzetme (deprem sonrası hasar taraması: FP ve FN'nin bedeli farklı);
   - Alıştırma sekmesine karışıklık matrisi soruları.

## Ultralytics'ten doğrulanan olgular (ultralytics 8.4.219 kodu)

- `confusion_matrix.png` eğitimin sonunda **best.pt** ile yapılan doğrulamada çizilir. Doğrulamanın güven
  eşiği varsayılan olarak **0,001**'dir (`validator.py`: `conf=None → 0.001`) ve matris bu eşikle doldurulur
  (`detect/val.py`: `process_batch(..., conf=self.args.conf)`). Bizim matristeki 24.474 "bina ↔ arka plan"
  yanlış pozitifi bundandır; 0,25'te çok daha azdır. Öğrencinin en çok yanlış okuyacağı nokta.
- Segmentasyon modelinde de matris **kutularla** kurulur (kutu IoU > 0,5), sınıftan bağımsız ve açgözlü
  eşleştirmeyle (skora göre sırayla, her gerçek nesne en fazla bir tespitle). Yanlış sınıflı eşleşme hem FP
  hem FN'dir ve köşegen dışına düşer.
- Satırlar **tahmin**, sütunlar **gerçek**; son satır/sütun "background". Arka plan × arka plan hücresi
  boştur: tespitte TN sayılamaz (sayılamayacak kadar çok "boş yer" vardır).
- Normalize edilmiş matris **sütunlara** göre bölünür: her sütun (gerçek sınıf) 1'e toplanır; köşegen
  duyarlılıktır. 0,005'in altı yazılmaz.
- Anlamsal modelde matris **piksel** sayar (6 × 6, arka plan ayrı bir sınıf): burada arka planın doğru
  bulunduğu pikseller vardır. IoU = TP / (TP + FP + FN), iou_bar_chart.png bunu çizer.
- `best.pt`: uygunluk (fitness) = mAP50-95(B) + mAP50-95(M) en yüksek olan epoch; `last.pt` son epoch.
- `close_mosaic: 10`: son 10 epoch mozaik yok; `train_batch11880–11882.jpg` epoch 91'in ilk üç yığını
  (132 yığın × 90 = 11880).
- Konsol: eğitim satırı `Epoch GPU_mem box_loss seg_loss cls_loss l1_loss sem_loss Instances Size`, doğrulama
  tablosu `Class Images Instances Box(P R mAP50 mAP50-95) Mask(...)`; bizim eğitimin son tablosu günlükte
  (`yolo-spike/train_seg.log`): all 131 / 2501, bina 110 / 2329, saha 65 / 172.

## Sekmenin akışı

1. **Klasör:** `runs/yolo/seg-pretrained/` ağacı; her dosya ne zaman yazılır (eğitim başında, her epoch'ta,
   sonunda) ve hangi soruyu cevaplar. Dosyaya tıklayınca ilgili kart.
2. **Eğitim sürerken ekran:** konsol satırları, sütun sütun (gerçek günlükten).
3. **results.csv ve results.png:** eğitim ve doğrulama kayıpları, (B) ve (M), dalgalanma, ezberleme işaretleri,
   best.pt / last.pt ve uygunluk.
4. **labels.jpg:** sınıf sayıları, konumlar, boyutlar → dengesiz veri (bina 2329, saha 172 doğrulamada).
5. **train_batch ve val_batch görüntüleri:** mozaik ve veri artırma; etiket ile tahmini yan yana koymak.
6. **Karışıklık matrisi I: benzetme.** Deprem sonrası hasar taraması, 2 × 2 tablo, TP/FP/FN/TN, doğruluk,
   kesinlik, duyarlılık, özgüllük, F1; dengesiz veride "doğruluk tuzağı". Eşik kaydırıcı.
7. **Karışıklık matrisi II: oyuncak kare.** 10 bina + 2 spor sahası, tahmin kutuları; eşik kaydırıcı, 3 × 3
   matris canlı; hücreye tıklayınca kutular vurgulanır. TN neden yok, arka plan satırı/sütunu, yanlış sınıf.
8. **Karışıklık matrisi III: bizim model.** Gerçek matris, güven eşiği kaydırıcısıyla (0,001 → Ultralytics'in
   resmi; 0,25 → Tespit sekmesi); sayı / normalize; sınıf başına P, R, F1; hücre örnekleri (kırpılmış
   doğrulama görüntüleri).
9. **Karışıklık matrisi IV: arazi örtüsü.** 6 × 6 piksel matrisi; hangi sınıf hangisiyle karışıyor
   (su → yeşil, otopark → arka plan) ve neden (etiket kalitesi); IoU'nun matristen hesabı.
10. **P, R, F1 ve PR eğrileri:** eşiğe göre; en iyi F1 eşiği; Box ve Mask farkı.
11. **Özet:** eğitim bitince bakılacaklar listesi.

## Aşamalar

- **A. Veri:** `export_yolo_assets.py` → `public/data/yolo-outputs.js`: matrisler (eşik başına), hücre
  örnekleri, P/R/F1/PR eğrileri, anlamsal matris, etiket istatistikleri, küçültülmüş görüntüler, konsol
  satırları, dosya listesi. Denetim: 0,001'deki matris `confusion_matrix.png` ile aynı olmalı.
- **B. İskelet:** sekme kimliği `outputs` (şema, chapters.mjs, core.js, i18n, build), chapter.json,
  panels.html, boş sekme, e2e.
- **C. İçerik:** kartlar ve demolar, iki dil.
- **D. Alıştırma ve sözlük:** karışıklık matrisi soruları; yeni terimler (karışıklık matrisi, TP, FP, FN, TN,
  F1, doğruluk, özgüllük).
- **E. Doğrulama ve inceleme:** lint, typecheck, birim, Playwright (8 profil), axe; commit (yerel).

## Ölçülenler (2026-11-27, kendi doğrulamamız, fp32)

- Matris 0,001'de: bina [2069, 18, 24461], saha [5, 103, 5522], arka plan [255, 51, –]. Eğitim sonundaki PNG
  (fp16): 2070 / 18 / 24474 … birkaç tespit farklı. Ultralytics tablosu birebir aynı (all 131 / 2501, P 0,64 …).
- Matris 0,25'te: bina [1520, 8, 687], saha [2, 57, 53], arka plan [807, 107, –].
- En iyi ortalama F1: kutu 0,26 (0,550), maske 0,33 (0,517). best.pt = epoch 88 (uygunluk 0,542; son 0,514).
- Arazi örtüsü: piksel doğruluğu 0,706, mIoU 0,464; otopark → arka plan 0,41, yol → arka plan 0,34, su → yeşil 0,23.
- Etiketler: 10.974 bina, 529 saha; binaların %56'sının uzun kenarı karenin onda birinden kısa.

Not: Y6/Y7 (yeniden eğitim) hâlâ açık. Yeniden eğitilince bu sekmenin verisi de aynı araçla yeniden üretilir.
