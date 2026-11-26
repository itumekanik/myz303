# 4 · Genel kontrol

- **Tarih:** 2026-11-26
- **Sürüm:** 1.3.0 yayında (derleme `44e1670`). Bu incelemedeki düzeltmeler yalnızca yerelde commit
  edildi, sürüm numarası değişmedi.

## Kapsam

1. **Canlı site ve yayın:** iki depoya push, GitHub iş akışının ilk çalışması, canlı sitede üç
   bölüm ve alıştırmalar.
2. **İstenen yenilik:** her bölümden ana sayfaya dönüş bağlantısı.
3. **Arayüz:** üst çubuk 320–1700 px arasında, her 10 pikselde, iki dilde ve üç bölümde ölçüldü.
   Ölçüm, son commit'teki sürümle karşılaştırmalı yapıldı.
4. **Kod taraması:** 1.0.2'den bu yana değişen kod (41 dosya, yaklaşık 3.400 satır: Alıştırma
   sekmesi, yeni bölümler, iş akışı) yalnızca hata açısından okundu. Bulgular mümkün olduğunda kod
   çalıştırılarak doğrulandı.
5. **Testler:** lint, tip denetimi, birim testleri ve sekiz tarayıcı profilinde uçtan uca testler,
   temiz bir kopyada.

## Bulgular

| No | Bulgu | Durum |
| --- | --- | --- |
| K1 | Bölüm sayfalarından ana sayfaya doğrudan bir bağlantı yoktu; başlangıç sayfasına yalnızca adres çubuğundan dönülebiliyordu | Çözüldü: üst çubuğun en solunda ev ikonu (telefonda logonun yerini alır, 360 px'in altında gizlenir), bölüm menüsünün başında "← Ana sayfa". Testi eklendi |
| K2 | Geniş ekranda üst çubuk sığmayınca rozetler ikinci satıra tek başına düşüyordu. CNN sayfasında bu, son commit'te de 1050–1630 px arasında 16–30 genişlikte vardı; ev ikonu bunu biraz artırdı | Çözüldü: çubuk sığmadığında önce alt başlık ve uzun rozetler, sonra model rozeti gizleniyor, en son sekmeler kendi satırına iniyor (dar ekranlardaki düzenin aynısı). Rozetler artık hiçbir genişlikte tek başına kalmıyor |
| K3 | Kod taraması #1: 1000 ve üzeri sayısal cevaplar. "1.024" (Türkçe yazım) yanlış sayılıyordu; "Cevabı göster" de "1.024" gösterip aynı yazımı reddediyordu (CNN `pl-flat`, cevap 1024) | Çözüldü: binlik ayırıcılı yazım da kabul ediliyor, gösterilen cevapta ayırıcı yok. Birim ve tarayıcı testleri eklendi |
| K4 | Kod taraması #2: tarayıcıda saklanan bozuk bir model sonucu (boş değer listesi) Alıştırma sekmesini yarıda kesiyordu; sonraki kısımlar, içindekiler işaretleri ve sıfırlama çalışmıyordu | Çözüldü: bozuk sonuç gösterilmiyor, sekme çalışıyor. Tarayıcı testi eklendi |
| K5 | Kod taraması #3: arka planda biten bir model sorusu, okuyucu başka yere kaydırmış olsa bile sayfayı o soruya geri çekiyordu; açık "Tüm cevaplar silinsin mi?" sorusunu da kapatıyordu | Çözüldü: yalnızca soru görünürdeyse odak oraya gidiyor; açık onay sorusu ve özetteki odak korunuyor |
| K6 | Kod taraması #5: soru bankası değişince başlangıç sayfası eski toplamı gösteriyordu (örneğin 15 soruluk bölüm 17 olunca "15/15") | Çözüldü: bölüm açılınca kayıt güncelleniyor. Birim testi eklendi |
| K7 | Kod taraması #4: aynı bölüm iki sekmede açıkken birinin ilerlemesi ötekinin üzerine yazılıyor | **Açık** (düşük): kayıt yazılırken saklanan kayıtla birleştirilmeli ya da `storage` olayı dinlenmeli |
| K8 | Kod taraması #6: "Sıfırlama çalışan bir deneyi iptal eder" testi, korunan kod silinse de geçebiliyor | **Açık** (test): deneyi sıfırlamadan önce "Model çalışıyor" durumunda tutmanın güvenilir bir yolu gerekiyor |
| K9 | Kod taraması #7: telefonda "altı sekme de görünüyor" denetimi, sekmeler alt satıra kaydığı için neredeyse başarısız olamıyor | **Açık** (test): son sekmenin görünür ve tıklanabilir olduğu denetlenmeli |
| K10 | Python ile düzenlenen bir dosyada satır sonları CRLF olmuştu | Çözüldü (commit'ten önce LF'ye döndürüldü) |

Kod taramasının iyi bulduğu yerler: iş akışının koşulları (yayın yalnızca `itumekanik/myz303`
main'ine gelen push'ta; tarayıcı testleri yalnızca sürüm etiketinde), Türkçe ondalık virgülü,
eksi işaretleri, toleranslar, ekran okuyucu etiketleri, iki dil, `file://` ile açma ve gizli
modda depolamanın hata vermesi.

## Canlı site

- 2026-11-23'te push edilen `44e1670` için GitHub iş akışı başarılı: Python, kontroller ve yayın
  geçti; tarayıcı testleri planlandığı gibi atlandı (yalnızca sürüm etiketinde çalışıyorlar).
- Canlı sitede altı bölüm var, son üçü "yakında". Üç hazır bölüm ve alıştırmaları (15, 19, 20
  soru) açılıyor, CNN model deneyi çalışıyor, sayfa hatası yok.

## Testler

Sonuçlar bu incelemenin commit'inden hemen önce, bütün düzeltmelerle (5. incelemedekiler de dahil)
temiz bir kopyada alındı:

- Lint ve tip denetimi temiz. Birim testleri 136/136 geçti (11 yeni test).
- Uçtan uca testler sekiz profilde de geçti: Chrome, Firefox ve WebKit'te http ve `file://`,
  öykünülmüş Android ve iPhone. Her profilde 55 test var; http profillerinde 1, `file://`
  profillerinde 6 test o ortama uymadığı için atlanıyor. Yeni testler: ev ikonu, menüdeki ana sayfa bağlantısı, "1.024"
  cevabı, bozuk kayıtla açılan Alıştırma sekmesi.
- Telefonda parmakla çizim ve dokunma, 320 px'de İngilizce sayfalar, depolaması kapalı tarayıcı,
  üç tarayıcıda aynı CNN tahminleri, Alıştırma sekmesinde axe (açık ve koyu temada ihlal yok),
  `file://`'da ilerlemenin başlangıç sayfasına geçmesi: hepsi beklendiği gibi.
- Düzeltilen metinler iki dilde ve iki genişlikte (1280 ve 375 px) sayfada görünüyor. Hiçbir
  sekmede KaTeX hatası, taşma ya da sayfa hatası yok.
- Üst çubuk 320–1700 px arasında her 10 pikselde: taşma yok, rozetler hiçbir genişlikte tek
  başına ikinci satırda değil, telefonda çubuk eskisinden yüksek değil.

## Doğrulanamayanlar

- Gerçek iPhone ve Android cihazlar, gerçek ekran okuyucu.
- Tarayıcı testlerinin GitHub'daki ilk çalışması (bir sonraki sürüm etiketinde olacak).
- Özel depo `origin`'de GitHub Actions'ın açık olup olmadığı (iş akışı orada zaten hiçbir şey
  çalıştırmıyor).
