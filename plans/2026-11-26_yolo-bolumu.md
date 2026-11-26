# 4. bölüm planı: "Nesne tespiti ve segmentasyon: YOLO"

- **Tarih:** 2026-11-26
- **Durum:** onaylandı (2026-11-26); ön denemenin ikinci turu sürüyor. Kararlar aşağıda.
- **Öğretim üyesinin isteği (2026-11-26):** YOLO'nun son sürümüyle, inşaat mühendisliğine uygun bir
  kuş bakışı görüntü örneği. YOLO'nun nasıl çalıştığı ve nasıl eğitildiği, tam bir ders gibi ve
  pedagojik olarak.
- **Dayanak:** `C:\Users\murat\Desktop\MYZ303E_CODE\yolo-spike\` klasöründeki ön deneme (2026-11-26).

## Kararlar (öğretim üyesi, 2026-11-26)

1. **Veri:** indirme 2 GB'a kadar olabilir; aşırı büyük indirme olmasın.
2. **Sınıflar:** mümkün olduğunca çok. Uygulama: binalar SpaceNet'ten, öteki sınıflar OpenStreetMap'ten
   (ODbL), etiketler bölümün kendi betiğiyle üretilir. Sayılabilen nesneler (bina, spor sahası) örnek
   segmentasyonuyla, yüzeyler (yol, yeşil alan, su, otopark) anlamsal segmentasyonla; iki model.
   Yüzme havuzu ve ibadet yeri bu dört şehrin OSM verisinde çok seyrek (300 karede 15 ve 10 nesne),
   bu yüzden yok. Son sınıf listesi, seçilen karelerdeki örnek sayısına göre kesinleşir.
3. **Python dosyaları:** içeriğe en uygun düzen (öneri: veri hazırlama ve eğitim iki ayrı dosya).
4. **Lisanslar:** README'ye her parçanın lisansı yazılır (SpaceNet CC BY-SA 4.0, OpenStreetMap ODbL,
   Ultralytics AGPL-3.0, repo MIT).
5. **Yer:** ders haftasına bağlı değil; konunun yeri önemli. CNN'den sonra 4. bölüm.

## Özet: öğretim üyesine

1. **Yer:** CNN'den hemen sonra, 4. bölüm. Planlanan RNN, Dikkat ve Transformer bölümleri birer sıra
   kayar (5, 6, 7). Henüz içerikleri olmadığı için yalnızca `chapter.json` dosyaları değişir.
2. **Omurga dört soru:** *Görüntünün neresinde ne var?* (1–2) → *Ağ bunu tek bakışta nasıl yapar?*
   (3–5) → *Nasıl eğitilir, ne kadar iyi?* (6–7) → *Mühendis sonucu nasıl okur?* (8).
3. **Ana örnek:** SpaceNet'in 0,3 m çözünürlüklü uydu görüntüleri (Las Vegas, Paris, Şanghay,
   Hartum). İki model: YOLO26n-seg (bina, spor sahası) ve YOLO26n-sem (yol, yeşil alan, su, otopark),
   bölümün kendi verisiyle eğitilmiş. İnşaat bağlantısı:
   maskeden m² cinsinden bina taban alanı ve karedeki yapılaşma oranı (TAKS'a benzer bir oran).
4. **Her şey tarayıcıda:** eğitilmiş model sayfada, çevrimdışı ZIP dahil çalışır. Ön denemede görüntü
   başına yaklaşık 0,5 saniye sürdü, GPU gerekmedi. Eğitim tarayıcıda yapılmaz: Kodlama sekmesi Colab'da
   ya da öğrencinin bilgisayarında eğitmeyi anlatır; Eğitim sekmesi bizim gerçek eğitimimizi gösterir.
5. **Sekmeler:** Teori, Tespit (canlı model), Eğitim (kaydedilmiş eğitim), Matematik, Kodlama,
   Alıştırma. CNN bölümüyle aynı yapı.
6. **İki oturuma bölünebilir:** 1–5. kısımlar tespitin ve maskenin nasıl üretildiğini, 6–8. kısımlar
   eğitimi, ölçmeyi ve mühendislik okumasını anlatır. CNN bölümündeki yeni sırayla aynı fikir.
7. **Dürüst mesaj:** model 240 görüntüyle eğitildi; doğrulamada maske mAP50 0,62. Siyah çatıları
   kaçırıyor, Hartum'da bitişik evleri tek blok sayıyor. Bu hatalar bölümde saklanmaz, öğretilir.
8. **Ölçü:** Teori 8 kısım ve 8 demo, Matematik 8 kısım, Kodlama 14 adım, Alıştırma yaklaşık 25 soru.
9. **Açık sorular:** §9'daki sorular yukarıdaki kararlarla cevaplandı.

---

## Etiketler

- **✓:** ön denemede ölçüldü ya da hesaplandı.
- **⟨ölçülecek⟩:** bölüm yazılırken bölümün kendi koduyla ölçülecek; o zamana kadar metne girmez.
- **(doğrulanacak):** tarih ya da iddia birincil kaynaktan doğrulanmadı; doğrulanmazsa metne girmez.

## 1. Öğrenme hedefleri

Bölümün sonunda öğrenci şunları yapabilir:

1. Sınıflandırma, nesne tespiti, anlamsal segmentasyon ve örnek segmentasyonu arasındaki farkı aynı
   görüntü üzerinde gösterir.
2. İki kutunun IoU'sunu elle hesaplar ve 0,5 eşiğinin ne anlama geldiğini açıklar.
3. YOLO'nun 640×640 bir görüntü için neden 8400 aday ürettiğini (80² + 40² + 20²) ve her adayın neden
   37 sayı taşıdığını (4 kutu + 1 sınıf + 32 maske katsayısı) gösterir. ✓
4. Güven eşiğinin ve NMS'in aday kutuları nasıl azalttığını adım adım izler.
5. Bir maskenin 32 prototip haritanın doğrusal birleşimiyle nasıl üretildiğini açıklar.
6. Eğitimin üç kayıp terimini (kutu, sınıf, maske) ve önceden eğitilmiş ağırlıkların neden işe
   yaradığını açıklar.
7. Kesinlik, duyarlılık, AP, mAP50 ve mAP50-95'i bir örnek üzerinde hesaplar.
8. Bir maskeden m² cinsinden alan hesaplar ve bu alanın neden hatalı olabileceğini (çatı ile taban
   farkı, maske çözünürlüğü, gölge, eğik çekim) tartışır.
9. Kendi verisini YOLO biçimine çevirir, Colab'da bir model eğitir, sonucunu ölçer ve dışa aktarır.

**Ön koşullar:** Bölüm 3'teki evrişim, stride, havuzlama, öznitelik haritası ve algı alanı; Bölüm 2'deki
eğitim, doğrulama ve test ayrımı ile aşırı uyum; ikili çapraz entropi.

## 2. Veri ve model (ön denemeden)

| Konu | Değer |
| --- | --- |
| Veri | SpaceNet 2 binalar, dört şehirden 75'er kare: 240 eğitim, 60 doğrulama ✓ |
| Lisans | CC BY-SA 4.0, kaynak gösterilerek ([spacenet.ai](https://spacenet.ai)) ✓ |
| Görüntü | 650×650 piksel, 0,3 m/piksel (bir kare ≈ 195 m × 195 m); 16 bit RGB, %2–98 germeyle 8 bit JPG ✓ |
| Etiketler | 5.498 bina çokgeni, GeoJSON'dan YOLO biçimine benim çevirimle ✓ |
| Model | YOLO26n-seg (Ultralytics, Ocak 2026), 3.082.751 parametre, COCO'da önceden eğitilmiş ✓ |
| Eğitim | 50 epoch, imgsz 640, batch 8, RTX 2060: 1007 s, 3,9 GB GPU belleği ✓ |
| Sonuç | Doğrulamada kutu mAP50 0,626, maske mAP50 0,618, maske mAP50-95 0,330; 50. epoch'ta hâlâ artıyor ✓ |
| ONNX | 11,1 MB; çıktılar (1, 37, 8400) ve prototipler (1, 32, 160, 160) ✓ |
| Tarayıcı | ONNX Runtime Web 1.31.1 (WASM, tek iş parçacığı): görüntü başına 0,46–0,57 s; Chromium, Firefox, WebKit, `file://` ✓ |
| Boyut | WASM 3,8 MB, model 11,1 MB (gzip+base64 olarak 5,1 + 13,1 MB) ✓ |
| Tutarlılık | PyTorch ve ONNX aynı sayıda bina buluyor; tarayıcı 4 görüntünün 2'sinde ±1 (yeniden boyutlandırma farkı) ✓ |

**Bölüm yazılmadan önce ön denemeye eklenecekler:**

- Veri: şehir başına 150 kare (indirme ≈ 1,5 GB, TIF'ler silinince ≈ 120 MB). 1 GB sınırı indirmeyi mi,
  kalan veriyi mi kapsıyor? (§9, soru 3.)
- 100 epoch ve "sıfırdan eğitim" karşılaştırması: önceden eğitilmiş ağırlığın faydası bir sayıyla
  gösterilsin. ⟨ölçülecek⟩
- Eğitim sekmesi için 1., 5., 10., 25. ve son epoch'taki ağırlıklarla aynı doğrulama karesinin tahmini.
- YOLO26'nın NMS'siz (uçtan uca) çıktısıyla dışa aktarma; böylece sayfada iki yol da gösterilebilir.
- Modeli float16'ya çevirmek (yaklaşık 5,6 MB): doğruluk kaybı ölçülecek. ⟨ölçülecek⟩
- Tarayıcı ile Python'un aynı yeniden boyutlandırmayı kullanması (tutarlılık farkını kapatmak için).

## 3. Sekmeler ve sayfa yapısı

| Sekme | İçerik |
| --- | --- |
| **Teori** | 8 kısım, her birinde gerçek modelle canlı bir demo (§4). |
| **Tespit** | Doğrulama karelerinden birini seç ya da kendi görüntünü yükle. Güven eşiği ve NMS kaydırıcıları; kutular, maskeler, bina sayısı, toplam alan (m²) ve yapılaşma oranı. Doğru etiketlerle karşılaştırma: doğru tespitler yeşil, yanlışlar kırmızı, kaçanlar turuncu. |
| **Eğitim** | Bizim eğitimimizin kaydı: kayıp eğrileri (kutu, sınıf, maske), mAP eğrisi, aynı karenin 1., 5., 10., 25. ve son epoch'taki tahmini. Tarayıcıda eğitim yok; nedeni yazıyor. |
| **Matematik** | 8 kısım (§5). |
| **Kodlama** | 14 adım, "Bu dosyayı nasıl çalıştırırım?" kutusuyla (Colab T4 önerilir) (§6). |
| **Alıştırma** | Yaklaşık 25 soru; cevaplar bölümün kendi koduyla hesaplanır (§7). |

Model ve WASM yalnızca gerektiğinde yüklenir (Tespit sekmesi ya da modele ihtiyaç duyan bir demo
açılınca); metin ve modelsiz demolar hemen açılır.

## 4. Teori (8 kısım)

### Bölüm A: tespit ve maske (ilk oturum)

**1. Ne ve nerede? Dört görev.**
Aynı kare dört kez: sınıflandırma ("bu karede bina var"), tespit (kutular), anlamsal segmentasyon (her
piksel bina mı değil mi), örnek segmentasyonu (her bina ayrı bir maske). Mühendislik bağlamı: kentsel
envanter, ruhsatsız yapı tespiti, afet sonrası hasar tespiti, yapılaşma yoğunluğu. Neden YOLO: tek
bakışta (You Only Look Once), gerçek zamanlı. Kısa tarih: Redmon ve ark. 2016; Ultralytics YOLOv8
(2023, çapasız); YOLOv10 (2024, NMS'siz); YOLO26 (Ocak 2026) (doğrulanacak).
*Demo:* dört görev arasında geçiş yapan tek bir kare.

**2. Kutu ve IoU.**
Kutunun iki yazılışı: (x1, y1, x2, y2) ve (cx, cy, w, h). IoU = kesişim / birleşim. Neden 0,5 eşiği,
neden IoU mesafeden iyi. *Demo:* iki kutuyu sürükle, IoU canlı; doğru etiket ile modelin kutusu üst üste.

**3. Tek bakışta: üç ölçekte ızgara.**
CNN bölümünden köprü: evrişim katmanları görüntüyü 8, 16 ve 32 kat küçültür; 640 piksellik girişte
80×80, 40×40 ve 20×20'lik haritalar oluşur. Her hücre bir adaydır: 6400 + 1600 + 400 = 8400 ✓. Her aday
hücresinden dört kenara uzaklık (çapasız), bir güven skoru ve 32 maske katsayısı tahmin eder
(37 sayı ✓). Gövde, boyun ve baş: gövde Bölüm 3'teki gibi bir CNN; boyun ölçekleri birleştirir; baş
tahmin eder. Küçük ev ince ızgarada, büyük depo kaba ızgarada bulunur. Yeni YOLO'lardaki küçük dikkat
blokları yalnızca adıyla anılır, Dikkat bölümüne bağlanır. *Demo:* bir ölçek seç, hücrenin üzerine
gel: hücrenin adayı ve algı alanı görüntüde.

**4. Çok fazla aday: eşik ve NMS.**
8400 adayın çoğu boş. Güven eşiği boşları atar; aynı binayı bulan komşu adayları NMS (en iyisini tut,
ona çok benzeyenleri sil) temizler. YOLO26 eğitimde her binaya tek bir aday atayarak NMS'e gerek
bırakmıyor (bire bir atama) (doğrulanacak). *Demo:* gerçek modelin bir karedeki ham adayları; eşik ve
IoU kaydırıcıları; NMS'i adım adım oynat.

**5. Maskeler: prototipler ve katsayılar.**
Ağ her görüntü için 160×160'lık 32 "prototip" harita üretir; her bina için 32 katsayı. Maske =
σ(Σ katsayı × prototip), kutuyla kırpılır (YOLACT fikri, Bolya ve ark. 2019 (doğrulanacak)). 160 = 640/4;
bir maske pikseli yaklaşık 1,22 m ✓. Anlamsal segmentasyonun farkı (yol, yeşil alan gibi sayılamayan
yüzeyler). *Demo:* bir binaya tıkla; 32 prototipi ve katsayılarını gör, katsayıları değiştirince
maskenin nasıl değiştiğini izle.

### Bölüm B: eğitim, ölçme ve okuma (ikinci oturum)

**6. Nasıl eğitilir?**
Etiket ataması: hangi aday hangi binadan sorumlu. Üç kayıp: kutu (IoU tabanlı), sınıf (ikili çapraz
entropi), maske (piksel başına ikili çapraz entropi). Önceden eğitilmiş ağırlıklar: COCO'da "bina" sınıfı
yok, ama kenar ve doku bilgisi işe yarıyor (sıfırdan eğitimle karşılaştırma ⟨ölçülecek⟩). Veri artırma:
kuş bakışı görüntünün "yukarısı" yok, döndürmek ve çevirmek serbest; mozaik. Eğitim ve doğrulama ayrımı
Bölüm 2'deki gibi. *Demo:* Eğitim sekmesinin küçük bir kopyası: aynı karenin epoch epoch tahmini.

**7. Ne kadar iyi? Kesinlik, duyarlılık, mAP.**
Doğru tespit (IoU ≥ 0,5), yanlış tespit, kaçırılan bina. Kesinlik ve duyarlılık; eşiği değiştirince
ikisinin birbirine karşı değişmesi; PR eğrisi ve altındaki alan (AP); mAP50 ve mAP50-95. Bizim modelin
sayıları ✓ ve ne anlama geldikleri. *Demo:* eşik kaydırıcısı PR eğrisinde bir noktayı hareket ettirir,
karede yeşil, kırmızı ve turuncu kutular değişir.

**8. Mühendis sonucu nasıl okur?**
Maskeden alana: piksel × 0,09 m². Karedeki yapılaşma oranı (bina alanı / kare alanı) ve TAKS'tan farkı
(kare parsel değildir; uydu çatıyı görür, tabanı değil). Hata kaynakları: maske çözünürlüğü, gölge, eğik
çekim, siyah çatılar, bitişik nizam. Alan değişimi: başka bir şehre (örneğin İstanbul'a) taşınınca model
neden kötüleşebilir. Görüntü lisansları: neden Google ekran görüntüsü kullanılmadı. *Demo:* bir binanın
alanı, maske sınırı bir piksel büyütülüp küçültülünce alanın nasıl değiştiği.

**Özet ve kendinizi sınayın:** CNN bölümündeki gibi dört soru ve katlanmış cevaplar.

## 5. Matematik (8 kısım)

1. **Gösterim ve boyutlar:** 640 → 80/40/20, 8400 aday, 37 kanal, 160×160×32 prototip; YOLO26n-seg'in
   parametre sayısı ✓ ve SimpleCNN'le karşılaştırma (582.218).
2. **Kutunun kodlanması:** hücre merkezi ve dört uzaklık (l, t, r, b) × stride → (x1, y1, x2, y2).
   Sayılarla örnek.
3. **IoU ve IoU kayıpları:** IoU'nun türevi; kutular hiç kesişmeyince gradyanın sıfır olması ve GIoU'nun
   bunu nasıl çözdüğü; CIoU'ya kısa bakış (YOLO26'nın kullandığı terim (doğrulanacak)).
4. **Sınıf skoru:** neden softmax değil sigmoid (bir aday birden fazla sınıfa ait olabilir); ikili
   çapraz entropi ve gradyanı (Bölüm 2'deki ŷ − y).
5. **Maske:** m = σ(Σₖ cₖ Pₖ); kaybın katsayılara göre türevi ∂L/∂cₖ = Σ (m − y) Pₖ; kutuyla kırpma.
6. **NMS:** algoritma, karmaşıklığı, eşiğin etkisi; bire bir atamayla NMS'siz tahminin fikri.
7. **Kesinlik, duyarlılık, AP:** küçük bir örnek üzerinde elle PR eğrisi; COCO'nun 101 noktalı
   enterpolasyonu; mAP50-95'in ortalaması.
8. **Alan ve hata:** GSD, piksel alanı, m²; sınır her yandan bir maske pikseli (160'lık maskede
   650/640 × 0,3 × 4 ≈ 1,22 m) kayarsa 10 m × 10 m'lik bir binanın alanı %43 küçülür ya da %55 büyür ✓.
   Maske tam çözünürlüğe büyütülünce hata ne kadar azalıyor ⟨ölçülecek⟩.

## 6. Kodlama (14 adım)

Kök klasöre bölümün Python dosyası gelir (öneri: `YOLO_SpaceNet.py`; §9, soru 4). Adımlar:

1. **Ne yapacağız?** İş akışı: veri → etiket → kontrol → eğitim → ölçme → tahmin → dışa aktarma.
2. **Ortamı hazırlamak:** "Bu dosyayı nasıl çalıştırırım?" kutusu; Colab'da T4 GPU; `pip install ultralytics rasterio shapely`.
3. **Veriyi indirmek:** SpaceNet'in herkese açık AWS deposundan küçük bir örnek; lisans ve kaynak gösterme.
4. **Görüntüyü 8 bite çevirmek:** 16 bit uydu görüntüsü, yüzdelik germe.
5. **Etiketleri YOLO biçimine çevirmek:** enlem/boylam → piksel (rasterio), çokgeni kırpmak, normalize
   etmek, satır biçimi. Boş karelerdeki yer tutucu nokta.
6. **Etiketleri gözle kontrol etmek:** çokgenleri görüntünün üzerine çizmek (en sık hata burada yakalanır).
7. **Eğitim ve doğrulama ayrımı, `data.yaml`.**
8. **Eğitmek:** `model.train(...)` parametreleri tek tek; kuş bakışına uygun veri artırma; Windows'ta
   `if __name__ == "__main__":` zorunluluğu ✓.
9. **Eğitim eğrilerini okumak:** `results.csv`, aşırı uyum belirtileri.
10. **Ölçmek:** `model.val()`, mAP'leri okumak, karışıklık matrisi.
11. **Tahmin ve alan:** yeni bir karede binalar, maskeden m² ve yapılaşma oranı.
12. **Dışa aktarmak:** ONNX; bu sitenin modeli nasıl kullandığı.
13. **Tam dosya.**
14. **Alıştırmalar:** daha çok epoch; sıfırdan eğitim; yalnızca Vegas'la eğitip Hartum'da sınamak (alan
    değişimi); imgsz'i küçültmek; SpaceNet 3 yollarıyla ikinci sınıf eklemek (anlamsal segmentasyon).

## 7. Alıştırma (yaklaşık 25 soru)

Kısımlar Teori'nin sırasında, sonda "Hepsi bir arada" ve "Eğitilmiş modelle deneyler". Önceki bölümlerdeki
gibi sayısal sorularda tipik yanlışlara özel geri bildirim. Örnekler:

- İki kutunun IoU'su (tipik yanlış: birleşim yerine iki alanın toplamına bölmek).
- 512 piksellik girişte kaç aday? (64² + 32² + 16² = 5376; tipik yanlış: yalnızca en ince ölçek.)
- Aday başına kanal sayısı 3 sınıflı bir modelde (4 + 3 + 32 = 39).
- Verilen beş kutuya NMS uygulanınca kaç kutu kalır?
- TP, FP, FN sayılarından kesinlik ve duyarlılık.
- Maske piksel sayısından m² (0,3 m GSD; tipik yanlış: 0,3 ile çarpmak, karesini almamak).
- Tahmin: güven eşiği 0,25'ten 0,6'ya çıkınca bu karede bina sayısı ne olur? (Gerçek modelle.)
- Transfer: siyah çatılı binaların neden kaçırıldığını en iyi açıklayan şık; İstanbul'a taşınınca ne
  beklenir; çatı alanı ile taban alanı.

Bütün cevaplar birim testinde bölümün kodu ve gerçek modelle yeniden hesaplanır.

## 8. İş sırası ve tahmini büyüklük

| Aşama | İş | Çıktı |
| --- | --- | --- |
| 0 | Ön denemeyi tamamlamak (§2'deki eklemeler) | ölçümler, son model, örnek kareler |
| 1 | `engines/yolo/`: ONNX oturumu, ön işleme, NMS, maske, metrikler (TypeScript) ve Python'la tutarlılık testleri | motor + birim testleri |
| 2 | Bölüm iskeleti, Tespit ve Eğitim sekmeleri, bölüm numaralarının kayması | çalışan sayfa |
| 3 | Teori: 8 kısım ve demolar | |
| 4 | Matematik | |
| 5 | Kodlama ve `YOLO_SpaceNet.py` | |
| 6 | Alıştırma ve testleri | |
| 7 | İnceleme: 8 tarayıcı profili, telefon, iki dil, erişilebilirlik; `reviews/`'a bir inceleme | |

Her aşama ayrı bir commit; her aşamanın sonunda size gösteririm. Büyüklük CNN bölümüne yakın.

## 9. Açık sorular

1. **Yer:** 4. bölüm, CNN'den hemen sonra mı? (Öneri: evet.)
2. **Sınıflar:** yalnızca bina mı, yoksa SpaceNet 3'ten yollar da mı (anlamsal segmentasyon, ikinci
   model)? Öneri: bölüm binayla; yollar Kodlama'da ★★★ alıştırma. İstanbul'daki cami, stadyum, yeşil alan
   gibi sınıflar ancak İBB'den izin alınırsa ve etiketlenirse.
3. **Veri sınırı:** 1 GB indirilen veri için mi, diskte kalan veri için mi? Şehir başına 150 kare (≈ 1,5 GB
   indirme, ≈ 120 MB kalan) modeli belirgin iyileştirir.
4. **Python dosyası:** tek dosya `YOLO_SpaceNet.py` mı, yoksa `yolo_data.py` ve `yolo_train.py` diye iki
   dosya mı? Öneri: tek dosya, adımlar fonksiyon fonksiyon.
5. **Eğitim sekmesi:** salt okunur bir Eğitim sekmesi olsun mu, yoksa eğitim yalnızca Teori 6'da ve
   Kodlama'da mı anlatılsın? Öneri: olsun; CNN bölümüyle tutarlı.
6. **Kendi görüntüsünü yükleme:** Tespit sekmesinde olsun mu? Öneri: olsun, yanında "model 0,3 m
   çözünürlüklü uydu görüntüsüyle eğitildi; farklı ölçek ve kaynakta sonuç kötüleşir" uyarısıyla.
7. **Lisans notu:** SpaceNet verisi CC BY-SA 4.0 (sitede kaynak gösterilecek; türetilen kareler ve etiketler
   de aynı lisansla). Ultralytics AGPL-3.0; onunla eğitilen modelin de bu lisansa tabi olduğunu söylüyor.
   Repo MIT lisanslı. README'ye her parçanın lisansını yazan bir not öneriyorum; kesin yorum için İTÜ'nün
   hukuk birimine danışılabilir.
8. **Ders haftası:** MYZ303E'nin 9. haftası (Tespit ve segmentasyon) RDD2022 ve çatlak segmentasyonu
   planlıyor. Bu bölüm o haftanın etkileşimli sayfası mı olsun, yoksa ayrı mı dursun?
