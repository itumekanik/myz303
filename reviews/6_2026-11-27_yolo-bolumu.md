# 6 · YOLO bölümünün incelemesi

- **Tarih:** 2026-11-27
- **Sürüm:** 1.3.0 yayında. YOLO bölümü (4. bölüm) yalnızca yerelde commit edildi; sürüm numarası
  değişmedi, push yapılmadı.
- **Plan:** [plans/2026-11-26_yolo-bolumu.md](../plans/2026-11-26_yolo-bolumu.md), 7. aşama.

## Kapsam

1. **Otomatik denetim:** altı sekme (Teori, Tespit, Eğitim, Matematik, Kodlama, Alıştırma) iki dilde
   ve üç genişlikte (1280, 375, 320 px): sayfa hataları, KaTeX hataları, yatay taşma, İngilizce
   sayfada kalan Türkçe metin. axe ile erişilebilirlik, açık ve koyu temada, her sekmede.
2. **Kod taraması:** bölümün kodu (motor, altı sekme, iki Python dosyası, dışa aktarma aracı ve
   paylaşılan `lesson.js` ile `runbox.js`'teki değişiklikler) yalnızca hata açısından. Taramayı
   bölümü yazmamış ayrı bir Claude ajanı yaptı; her bulgu kod okunarak ya da çalıştırılarak
   doğrulandı, sonra düzeltildi.
3. **İçerik:** metinlerdeki sayıların ölçümlerle ve birbiriyle tutarlılığı; YOLO26, YOLOv10,
   YOLACT ve Ultralytics hakkındaki iddiaların kaynağı.
4. **Bölümler arası:** öğrenme yolunda CNN'den YOLO'ya geçiş.
5. **Testler:** lint, tip denetimi, birim testleri ve sekiz tarayıcı profilinde uçtan uca testler,
   temiz bir kopyada.

## Bulgular

| No | Bulgu | Durum |
| --- | --- | --- |
| Y1 | axe, boş tablo başlığı (küçük): Teori'deki karşılaştırma tablosunda, Tespit'teki sonuç ve arazi örtüsü tablolarında ilk sütunun başlığı boştu; Matematik 6'daki NMS tablosunun başlığı yalnızca bir formüldü, Matematik 7'deki PR tablosunda TP/FP sütununun başlığı yoktu | Çözüldü: başlıklar yazıldı ("Özellik", "ölçü", "sınıf", "IoU eşiği", "TP / FP") |
| Y2 | axe, odaklanamayan kaydırma alanı (ciddi): Matematik 2'deki kutu formülü 1280 px'de bile sığmayıp yatay kayıyordu; klavyeyle kaydırılamıyordu | Çözüldü: formül iki satıra bölündü, artık sığıyor |
| Y3 | axe, aynı adlı iki bölge (orta): Kodlama'daki iki "Nasıl çalıştırırım?" kutusunun ekran okuyucu adı aynıydı | Çözüldü: ad dosyayı söylüyor ("YOLO_SpaceNet.py dosyasını çalıştırmak"); öteki bölümlerdeki kutular da bundan yararlanıyor |
| Y4 | Tespit sekmesi "bir kare yaklaşık 195 m × 195 m" diyordu. Kareler derece cinsinden olduğu için doğu-batı boyu enleme göre kısalır: Paris'te 128 m, Hartum'da 188 m | Çözüldü: "kuzey-güneyde 195 m, doğu-batıda şehre göre 128–188 m" |
| Y5 | CNN bölümünün sonunda, öteki bölümlerdeki gibi "Sıradaki bölüm" kartı yoktu (RNN henüz yazılmadığı için konmamıştı); YOLO gelince yol CNN'de kopuyordu | Çözüldü: CNN'in özet kartının altında YOLO'ya giden kart, iki dilde. Testi eklendi |
| Y6 | Kod taraması: arazi örtüsü maskelerinde, aynı sınıftan çokgenler tek bir `cv2.fillPoly` çağrısıyla dolduruluyordu. OpenCV bu durumda örtüşen yerleri boş bırakır: yol kavşaklarında yol genişliği kadar kareler ve park içindeki spor sahaları "arka plan" olarak etiketlenmiş. OpenCV ile denenerek doğrulandı | Kod çözüldü: `YOLO_SpaceNet_data.py` her çokgeni ayrı boyuyor. **Açık: eğitmen kararı.** Sitedeki arazi örtüsü modeli bu etiketlerle eğitildi; örnek karelerin "doğru harita"ları (Tespit, Arazi örtüsü) da bu delikleri taşıyor. Öğretim üyesinin kararı (2026-11-27): şimdilik yeniden eğitilmeyecek, ileride ele alınacak |
| Y7 | Kod taraması: birden çok parçadan oluşan OSM alanları (çok parçalı ilişkiler) yalnızca ilk parçalarıyla okunuyor, açık bir çizgi kaldığı için atlanıyordu: Las Vegas'taki 22 spor alanı ilişkisinin 18'i, Paris'te 39 alan ilişkisi. Kapalı yollar (döner kavşak, servis halkaları; yaklaşık 80 tane) alan sayılıp yol etiketlerinden düşüyordu | Kod çözüldü: dış parçalar birleştirilip alana çevriliyor, yollar her zaman çizgi. **Açık: eğitmen kararı** (Y6 ile aynı karar): bina/saha modelinin etiketlerinde bu spor alanları yok. İki model yeniden eğitilince bölümdeki sayılar (mAP, alıştırma deneyleri, Teori'deki örnekler) güncellenmeli; birim testleri değişen cevapları gösterir |
| Y8 | Kod taraması: OSM önbelleği yalnızca şehre bağlıydı. Önce `--per-city 20` ile, sonra bütün veriyle çalıştırınca yeni karelere hiç OSM etiketi gelmiyordu. Ayrıca daha önce indirilmiş bütün kareler veri setine giriyordu | Çözüldü: önbellek hangi kareler için alındığını biliyor ve kareler eklenince yeniden soruyor; veri seti yalnızca bu çalıştırmanın kareleriyle kuruluyor |
| Y9 | Kod taraması: Tespit sekmesinde bir kare çalışırken kendi görüntünüzü yüklerseniz, ilk çalışmanın sonu yenisinin "Model çalışıyor" durumunu siliyor, hatası varsa yeni görüntünün üstüne yazıyordu | Çözüldü: bir çalışmanın sonucu, hatası ve bitişi yalnızca o görüntü hâlâ ekrandaysa uygulanıyor |
| Y10 | Kod taraması: çok büyük bir fotoğraf (örneğin 100 megapiksel) tam boyutta okunuyordu (~700 MB); Safari'nin tuval sınırını aşınca yanıltıcı bir "görüntü açılamadı" hatası veriyordu | Çözüldü: yüklenen görüntü en fazla 4000 pikselle okunuyor; alan hesabı için piksel başına metre de buna göre ölçekleniyor |
| Y11 | Kod taraması: Eğitim sekmesinde, 7'den az epoch'u kaydedilmiş bir eğitimde (örneğin `--epochs 20`) "aynı kare" görünümü hiç çizilmiyordu; hızlı tıklamada geç açılan bir arazi örtüsü haritası yeni seçimin üzerine çiziliyordu; dışa aktarma aracı eğitim ayarlarını sabit yazıyordu | Çözüldü: seçim kaydedilenlerle sınırlanıyor, yalnızca son tıklama çiziyor, ayarlar eğitimin `args.yaml`'ından okunuyor |
| Y12 | Kod taraması, erişilebilirlik: ızgara, maske ve alan demolarında seçim yalnızca fareyle yapılabiliyordu; prototip düğmelerinin adı katsayıydı ("0,12"); aynı şehrin iki küçük resminin adı aynıydı | Çözüldü: ızgarada ok tuşları, maske ve alan demolarında "Sonraki nesne" düğmesi; düğme adları "Prototip 3, katsayı 0,12" ve "Las Vegas · img2660" |
| Y13 | Kod taraması, küçük görsel hatalar: IoU demosunda kutular kesişmeyince boyut tutamacı kayboluyordu; İngilizcede binlik virgülü formülde "6, 000" gibi dizgileniyordu; alan demosunda büyütülen maske görüntünün dışına taşıp kenar piksellerini öteki satıra sarabiliyordu; yüklenen kare olmayan görüntülerde maskenin gri kenara taşan kısmı alana sayılıyordu | Çözüldü |
| Y14 | Kod taraması: sayfa modelin "yaklaşık 7 MB" olduğunu söylüyordu; Teori ve Alıştırma'da ONNX Runtime ile birlikte yaklaşık 12 MB, Tespit'te 16 MB yükleniyor | Çözüldü: metinler 12 ve 16 MB diyor |

## İçerik

- Metindeki sayılar bölümün kendi ölçümleridir ve sekmeler arasında tutarlıdır: maske mAP50 0,50
  (bina 0,69, spor sahası 0,30), mAP50-95 0,25; sıfırdan eğitimde 0,32; arazi örtüsü mIoU 0,46;
  528 eğitim, 131 doğrulama karesi; 78 dakika; 3.083.302 parametre; Şanghay karesinde 107 aday,
  15 nesne. Alıştırma sekmesindeki her cevap ve beş model deneyi birim testinde yeniden hesaplanır.
- Kaynaktan doğrulananlar: YOLO26 ilk kez Ultralytics 8.4.0 ile (PyPI, 2026-01-14); YOLO27
  Eylül 2026'da duyuruldu, henüz yayımlanmadı; YOLOv10 (Tsinghua, NeurIPS 2024) ve YOLACT (Bolya
  ve ark., ICCV 2019). Kayıplar, atama ve sınıf hedefleri kurulu Ultralytics kodundan okundu.
- Bir iddia ölçüme göre yumuşatıldı (Alıştırma, eşik deneyi): "skorların çoğu 0,3–0,6 arasında"
  yerine "bu karedeki binaların çoğunun skoru 0,6'nın altında".

## Boyut

| Ne zaman | Dosyalar | Boyut |
| --- | --- | --- |
| Teori açılınca | örnek kareler, eğitim kaydı | 2,2 MB |
| Bir demo modeli isteyince | ONNX Runtime'ın WASM'ı, bina modeli | 11,9 MB |
| Tespit sekmesinde ayrıca | arazi örtüsü modeli | 3,9 MB |

Çevrimdışı ZIP 31,6 MB oldu (yaklaşık 14 MB'ı bu bölümün). Sayfanın kendi betiği 0,78 MB.

## Doğrulanamayanlar

- Gerçek telefonlarda dokunmayla sürükleme (Teori 2'deki kutu): yalnızca öykünülmüş Android ve
  iPhone'da denendi.
- Bölümün Python dosyalarının Colab'da baştan sona çalışması: dosyalar yerelde (Windows, RTX 2060)
  her komutla çalıştırıldı; bütün veriyle 100 epoch'luk eğitim ön denemede yapıldı, Colab'da değil.
- Yavaş bilgisayarlarda modelin çalışma süresi: yalnızca bu bilgisayarda ölçüldü (0,4–0,8 s).

Kod taramasının iyi bulduğu yerler: `lesson.js` ve `runbox.js`'teki değişiklikler öteki bölümleri
bozmuyor; Kodlama sekmesinin 18 alıntısının hepsi Python dosyalarında bulunuyor; letterbox ve
koordinat dönüşümleri, sıfıra bölme korumaları, Alıştırma'daki 90° döndürme, Matematik 7'deki AP
örneği (0,69) ve 746 karelik liste doğru.

## Testler

Bu incelemenin commit'inden önce, bütün düzeltmelerle temiz bir kopyada:

- Lint ve tip denetimi temiz; 205 birim testi geçti.
- Uçtan uca testler sekiz profilde de geçti (Chrome, Firefox, WebKit'te http ve `file://`, öykünülmüş
  Android ve iPhone). Yeni test: CNN'den YOLO'ya geçiş kartı.
- YOLO'nun altı sekmesi iki dilde ve 1280, 375, 320 px'de: sayfa hatası, KaTeX hatası, taşma ve
  İngilizce sayfada Türkçe metin yok. axe: düzeltmelerden sonra açık ve koyu temada ihlal yok.

## Takip

- Y6 ve Y7 için: veri `YOLO_SpaceNet_data.py` ile baştan üretilip üç model yeniden eğitilecek
  (yaklaşık 3,5 saat), sonra `python web/tools/export_yolo_assets.py <klasör>` ile sitenin dosyaları
  ve metinlerdeki sayılar güncellenecek.
