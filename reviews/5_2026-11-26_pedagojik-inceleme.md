# 5 · İlk üç bölümün ikinci pedagojik incelemesi

- **Tarih:** 2026-11-26
- **Önceki inceleme:** [2. inceleme](2_2026-11-19_pedagojik-inceleme.md) (2026-11-19). O incelemede
  düzeltilenler yeniden sayılmadı.
- **Yeni olan:** üç bölümün Alıştırma sekmesi (15 + 19 + 20 soru) ilk kez incelendi.

## Kapsam ve yöntem

Her bölüm, deneyimli bir eğitmen gözüyle ayrı ayrı okundu: Teori, Deneme alanı, Matematik, Kodlama
ve Alıştırma sekmeleri, sözlük maddeleri ve Kodlama'da gösterilen Python dosyaları. Ölçütler
şunlardı:

1. Doğruluk.
2. Kavramların sırası ve ön koşulları.
3. Yanlış kavrayış riski.
4. Alıştırma sorularının niteliği.
5. Türkçe ve terim birliği.
6. Tarihî bilgiler.

Sayısal iddiaların ve 54 alıştırma cevabının hepsi kod çalıştırılarak doğrulandı (Node, NumPy,
PyTorch; CNN eğitimi RTX 2060'ta yeniden üretildi). Tarihler web kaynaklarıyla karşılaştırıldı.

## Genel değerlendirme

**Güçlü yanlar.** Sayılar şaşırtıcı ölçüde doğru: metinlerdeki her sayı, her program çıktısı ve her
alıştırma cevabı yeniden hesaplandığında tuttu. Alıştırma cevaplarının, metinden değil modelin kendi
kodundan hesaplanması (ve bunun testle korunması), içerik ile demo arasındaki kaymayı önleyen çok iyi
bir tasarım. Matematik sekmeleri doğru düzeyde ve eksiksiz. Demolar dürüst: başarısızlığı da
gösteriyorlar (XOR'da takılma, H = 1 ile kapasite). Türkçe akıcı, TR/EN eşliği iyi.

**Zayıf yanlar.** Bulguların ağırlığı artık hesapta değil, öğretim tasarımında:

1. **Alıştırmalar çoğunlukla hatırlama ve tek adımlık hesap.** 54 sorunun büyük kısmı formüle sayı
   koymak ya da tanımı tanımak. Bilgiyi yeni bir duruma taşıyan (transfer) soru az. Bazı çeldiriciler
   kimseyi yanıltmayacak kadar zayıf. Sayısal sorularda tipik yanlış cevaplara (bias'ı unutmak, log
   tabanı, +1'i unutmak) özel geri bildirim yok. (P11, P12, M10, C10)
2. **Aşırı uyum ve doğrulama kavramı, CNN'den önce hiç işlenmiyor.** MLP bölümünde yalnızca bir
   Python alıştırmasında geçiyor; Deneme alanında ölçülemiyor. CNN ise doğrulama ayrımını hazır
   kullanıyor. "%99 ne demek, ne demek değil" sorusu hiçbir yerde tartışılmıyor. (M9, C6, C7)
3. **CNN bölümü tek oturum için çok büyük ve Eğitim sekmesiyle açılıyor.** Öğrenci CNN'in ne
   olduğunu öğrenmeden hiperparametre formuyla karşılaşıyor. Bölüm 1 ve 2 ise Teori ile açılıyor. En
   ağır kısım (BatchNorm) hâlâ ikinci sırada. (C1, C2, C3)
4. **Kodlama sekmeleri Python kurulumunu varsayıyor.** Hedef kitlenin çoğunda Python yok. Kurulum
   ancak 3. bölümde anlatılıyor, Perceptron.py'yi siteden indirmenin yolu yok. (P2)
5. **Bölümler arası köprüler yer yer yalnızca Matematik sekmesinde.** Örneğin perceptron kuralının
   bir gradyan inişi olduğu Teori'de anlatılmıyor, ama MLP Teori'si bunu bilinen kabul ediyor. (P5)
6. **Aynı kavram birçok sekmede yeniden anlatılıyor**, terimler yer yer ikili: "aşırı uyum /
   aşırı öğrenme / ezberleme", "adım" sözcüğünün dört anlamı, "ileri hesap / ileri yayılım". (M12,
   C14, C15, C19, P16)

**Önerim (sırasıyla):**

1. CNN bölümünün Teori ile açılması (C1). Tek satırlık bir değişiklik, en büyük etki. *(Yapıldı,
   2026-11-26.)*
2. MLP bölümüne kısa bir "eğitim, doğrulama ve test" kısmı ile Deneme alanına test doğruluğu
   (M9). CNN'e geçmeden önceki en önemli eksik bu.
3. Her bölüme 4–6 transfer sorusu ve tipik yanlışlara özel geri bildirim (M10, C10, P6).
4. Kodlama sekmelerine "Python'unuz yoksa" kutusu: dosyayı indirme bağlantısı, kurulum ya da
   Colab (P2).
5. CNN bölümünün bölünmesi ya da sırasının değişmesi (C2, C3). Büyük bir iş; ders takvimine göre
   karar verilmeli.

## Bölüm 1 · Perceptron

| No | Önem | Yer | Bulgu | Durum |
| --- | --- | --- | --- | --- |
| P1 | yüksek | `Perceptron.py`, Kodlama | Yakınsamayan her kapı için "no straight line separates XOR" yazılıyordu. Alıştırmanın önerdiği seed 3'te AND 20 epoch'a sığmıyor (22 epoch gerekiyor), öğrenci AND'in ayrılamadığı sonucunu çıkarabiliyordu | Çözüldü: mesaj her veri için doğru ("…no straight line separates the data, or it needs more epochs"), Kodlama'daki çıktı güncellendi, seed alıştırmasına not eklendi |
| P2 | yüksek | Kodlama | Python ve `.venv` kurulu varsayılıyor; kurulum ancak 3. bölümde, dosyayı indirme bağlantısı yok, komut yalnızca Windows için | **Açık: eğitmen kararı / büyük iş** |
| P3 | orta | Alıştırma `c-eta` | "Kendiniz deneyin" bağlantısı, cevapla çelişen bir demoya götürüyordu (o demo sıfırdan başlamıyor, orada η epoch sayısını değiştiriyor) | Çözüldü: bağlantı, iddianın ispatlandığı Matematik kısmına gidiyor |
| P4 | orta | Teori, tarih | "Bu döneme ilk yapay zekâ kışı dendi": ilk yapay zekâ kışı 1974–1980 arası ve bütün alanı kapsıyor | Çözüldü |
| P5 | orta | Teori ↔ MLP Teori | Kayıp ve gradyan inişi 1. bölümde yalnızca Matematik'te; MLP Teori'si bunları 1. bölümden biliniyor sayıyor | **Açık: eğitmen kararı** (p-rule'a 2–3 cümle ve bir soru önerildi) |
| P6 | orta | Alıştırma | MLP'ye geçişin ana fikirleri sorulmuyor: gizli gösterim, gizli ağırlıkların neden perceptron kuralıyla öğrenilemediği, kayıp | **Açık: eğitmen kararı** |
| P7 | orta | Matematik | "z = 0'da adım atmak da atmamak da geçerli" ifadesi: sıfırdan başlayan perceptronu yürüten şey, z = 0'ı y = 1 için hata saymak | Çözüldü: açıklayıcı cümle eklendi |
| P8 | orta-düşük | Teori, sözlük | (R/γ)² sınırı yalnızca sıfırdan başlayınca geçerli; R, bias için eklenen 1 bileşeniyle ölçülüyor | Çözüldü (Teori ve sözlük) |
| P9 | düşük-orta | Teori | "Bölümler sırayla okunmak üzere yazıldı": kısım olmalı | Çözüldü |
| P10 | düşük | Teori, tarih | Mark I aslında üç katmanlı: 400 fotosel, rastgele ve sabit bağlı 512 ilişki birimi, 8 çıktı. Bu, 2. bölümün konusuna doğal bir köprü | **Açık: eğitmen kararı** |
| P11 | düşük | Alıştırma `c-flip` ↔ `x-why` | Bir sorunun açıklaması sonrakinin cevabını veriyor; doğru şık belirgin biçimde uzun | **Açık: eğitmen kararı** |
| P12 | düşük | Alıştırma `c-bound` | Formüle sayı koymaktan ibaret ("marj yarıya inerse sınır kaç katına çıkar?" önerildi) | **Açık: eğitmen kararı** |
| P13 | düşük | Kodlama | "Perceptron kuralı gradyan istemez" ifadesi Matematik'teki "kural bir gradyan inişidir" ile çelişiyordu | Çözüldü: "gradyanı elle yazılacak kadar basit, autograd gerekmez" |
| P14 | düşük | Teori, sözlük | Basamak fonksiyonunun eğimi "her yerde 0" değil, z = 0 dışında 0 | Çözüldü (Teori ve sözlük) |
| P15 | düşük | Kodlama, Deneme alanı | Eksik iki fikir: x₃ = x₁x₂ ile sınır "özelliklerde doğrusal, girdilerde eğri"; ayrılamayan gürültülü veride perceptron hiç durulmuyor | **Açık: eğitmen kararı** |
| P16 | düşük | panels, sözlük | "Perceptron'u" / "perceptronu" yazımı karışık; "bias (yanlılık)" istatistikteki yanlılıkla karışabilir | **Açık: eğitmen kararı** |
| P17 | düşük | Kodlama (PyTorch) | w 16 basamakla, b 2 basamakla basılıyor | Açık |

## Bölüm 2 · Çok katmanlı ağ (MLP)

| No | Önem | Yer | Bulgu | Durum |
| --- | --- | --- | --- | --- |
| M1 | yüksek | Alıştırma | Açıklama "geri hesabın maliyeti ağırlık sayısıyla artmaz" diyordu; yanlış (ileri hesap gibi doğrusal artar; ileri hesaba oranı sabittir) ve Teori ile çelişiyordu | Çözüldü |
| M2 | yüksek | Alıştırma `bp-sym` | "Aynı ağırlıklarla başlatıp" belirsiz: yalnızca giren ağırlıklar eşitse nöronlar ayrışıyor | Çözüldü: giren ağırlıklar, biaslar ve çıkış ağırlıkları |
| M3 | yüksek | Teori, sözlük | "Bu küçük XOR ağında gerçek bir yerel minimum olmadığı gösterilmiştir" iddiası tartışmalı (Hamey 1998'e karşı Sprinkhuizen-Kuyper ve Boers 1999); sözlük de Teori ile çelişiyordu | Çözüldü: tarafsız ifade, sözlük uyumlu |
| M4 | yüksek | Teori, sözlük | "MLP" Teori'de hiç açılmıyor; YSA/ANN ve "çok katmanlı algılayıcı (ÇKA)" ile ilişkisi söylenmiyor. Derste "YSA/ANN" denince öğrenci bunu kendisi eşlemek zorunda | Çözüldü: girişe bir cümle, sözlüğe eş adlar (YSA, ANN, ÇKA) |
| M5 | orta | Matematik | Evrensel yaklaşım teoremindeki "tepelerle yaklaşma" yapısı uç noktada yanlış (x = 0'da hata f(0)/2'ye gidiyor, sıfıra inmiyor); n boyutta "kapalı aralık" yerine "kompakt küme" olmalı | Çözüldü |
| M6 | orta | Teori (kapasite) | Demo kare hata ve Adam kullanıyor, metin söylemiyor; regresyon ile sınıflandırma ayrımı yapılmıyor | Çözüldü: bir cümle |
| M7 | orta | Teori (görüntüler) | "Evrişimin yalnızca 52.096 parametresi var" karşılaştırması bütün CNN'in daha az parametreli olduğunu düşündürüyor (SimpleCNN 582.218, tek katmanlı MLP 401.920) | Çözüldü |
| M8 | orta | Teori (geri yayılım) | ∂L/∂w gösterimi Teori'de tanımlanmadan beş çarpanlı zincire geçiliyor; çarpanların demodaki adımlarla eşleşmesi gösterilmiyor | Çözüldü |
| M9 | orta | Teori, Deneme alanı | Aşırı uyum ve doğrulama CNN'den önce gereken en önemli eksik; Deneme alanında test doğruluğu yok | **Açık: büyük iş** (önerimin 2. maddesi) |
| M10 | orta | Alıştırma | Zayıf çeldiriciler; 19 sorunun neredeyse hepsi tek adımlık. Mini-batch, η'nın etkisi, kaybolan gradyan, aşırı uyum sorulmuyor | **Açık: eğitmen kararı** |
| M11 | orta | Teori (XOR) | Takılmanın asıl mekanizması gösterilmiyor (iki farklı etiketli nokta gizli uzayda üst üste düşüyor); gizli nöronların 1. bölümdeki OR/AND'e benzemesi köprüsü kaçıyor | Çözüldü |
| M12 | orta | Teori, sözlük | Terimler: "ileri hesap" ile "ileri yayılım"; "hata" üç anlamda (δ, kayıp, MSE); sözlükte geri yayılıma "geri besleme" eş adı (RNN'deki feedback ile karışır) | Kısmen çözüldü: "geri besleme" sözlükten çıktı. Gerisi **Açık: eğitmen kararı** |
| M13 | düşük-orta | Teori (kayıp demosu) | Bu veride kayıp hiçbir η'da ıraksamıyor; büyük η daha hızlı varıyor, öğrenci "büyük η hep iyi" sonucuna varabilir | **Açık: eğitmen kararı** |
| M14 | düşük | Teori, Alıştırma, sözlük | "σ′ ≤ 0,25 olduğu için sinyal küçülür" ancak ağırlıklar büyük değilse doğru | Çözüldü |
| M15 | düşük | Matematik | "Bu bölümdeki bütün kodlar bu başlatma kuralını kullanır": PyTorch kodu kendi varsayılanını kullanıyor | Çözüldü |
| M16 | düşük | Kodlama (alıştırma ipucu) | "Yanlış gradyan kabaca doğru yönü gösterir, fark etmek zor" ipucu: öğrencinin gördüğü sonuç çelişiyor (XOR açıkça başarısız) | Çözüldü |
| M17 | düşük | Teori (kapasite) | Basamak hedefinde H = 3, H = 2'den kötü çıkıyor; açıklanmıyor (iyi bir öğretme fırsatı) | **Açık: eğitmen kararı** |
| M18 | düşük | Alıştırma `h-xor` | Kesin yargı istiyor ama 2-2-1 ağı bazı başlangıçlarda takılıyor | Çözüldü |
| M19 | düşük | Teori (geri yayılım demosu), Matematik EN | Hep aynı örnekte "Uygula"ya basınca ağ yalnızca onu öğreniyor; İngilizce metinde eksik boşluk | Çözüldü |

## Bölüm 3 · Evrişimli ağ (CNN)

| No | Önem | Yer | Bulgu | Durum |
| --- | --- | --- | --- | --- |
| C1 | yüksek | `chapter.json` | Bölüm Eğitim sekmesiyle açılıyor; öğrenci CNN'i öğrenmeden hiperparametre formuyla karşılaşıyor. Bölüm 1 ve 2 Teori ile açılıyor | Çözüldü (2026-11-26, eğitmen kararıyla): bölüm Teori ile açılıyor, sekme sırası Teori, Çıkarım, Eğitim, Matematik, Kodlama, Alıştırma |
| C2 | yüksek | bütün bölüm | Tek oturum için çok büyük (Teori 7 kısım ve 7 demo, Matematik 10 kısım, Kodlama 17 adım, 20 soru); optimizasyon kısmı CNN'e özgü değil | **Açık: büyük iş** (3a "Evrişimli ağ" ve 3b "Derin ağı eğitmek" diye bölmek önerildi) |
| C3 | yüksek | Teori, Matematik, Alıştırma | BatchNorm hâlâ 2. sırada (2. incelemede B13). Ek kanıt: BatchNorm'suz model aynı seed ile %99,30, BatchNorm'lu %99,29 | **Açık: eğitmen kararı** |
| C4 | orta-yüksek | Teori, Alıştırma `sm-equal` | "−log 0,1 ≈ 2,30": lisede log 10 tabanlı; öğrenci 1 bulur. ln olduğu yalnızca Matematik'te yazıyor | Çözüldü: ilk geçtiği yerde ve soruda "log = doğal logaritma (ln)" |
| C5 | orta | Matematik | K hem sınıf hem katman sayısı; 2. incelemede eklendiği yazılan not dosyada yoktu | Çözüldü: not eklendi |
| C6 | orta | Çıkarım, Kodlama | "%99" sonucunun anlamı tartışılmıyor: 71 hatanın 20'sinde model %90'dan emin, ters renkli rakamlarda %50, 4 piksel kaymada %82 | **Açık: büyük iş** ("Sonucu okumak" kartı önerildi) |
| C7 | orta | Teori | Eksik konular: derinliğin neden işe yaradığı, veri artırma, eşdeğişkenlik/değişmezlik, tümevarımsal yanlılık, erken durdurma | **Açık: eğitmen kararı** |
| C8 | orta | Alıştırma `ac-why`, Teori | "Aktivasyon yoksa evrişimler tek doğrusal katmana eşit": SimpleCNN'de aradaki max pooling de doğrusal değil | Çözüldü |
| C9 | orta | Alıştırma `e-shift` | "Küçük kayma" 1–2 piksel demek; 4 pikselde CNN de %82'ye düşüyor | Çözüldü: açıklamaya eklendi |
| C10 | orta | Alıştırma | 20 sorunun 17'si tek adımlık; üçü Teori'deki öz-sınamayla örtüşüyor; zayıf çeldiriciler; algı alanı, kayıp eğrisi okuma, max-pool gradyanı sorulmuyor | **Açık: eğitmen kararı** |
| C11 | orta | Teori, sözlük | "Sonraki çalışmalar gösterdi": Santurkar ve ark. (2018) öne sürdü, konu tartışmalı; sözlük eski açıklamayı tekrarlıyordu | Çözüldü (Teori ve sözlük) |
| C12 | orta | Kodlama (alıştırma) | "BatchNorm'suz büyük fark arayın" ipucu, gerçek sonuçla çelişiyor (fark neredeyse yok) | Çözüldü |
| C13 | düşük-orta | Teori (BatchNorm) | "γ = σ, β = μ ile normalleştirme tamamen geri alınır": yalnızca yaklaşık | Çözüldü |
| C14 | düşük-orta | sözlük, Kodlama, Teori | "aşırı öğrenme", "aşırı uyum", "ezberleme" karışık | **Açık: eğitmen kararı** (öneri: "aşırı uyum") |
| C15 | düşük-orta | Teori | "adım" dört anlamda (stride, eğitim adımı, türetme adımı, Kodlama adımı); başlıkların bir kısmı çevrilmemiş | **Açık: eğitmen kararı** |
| C16 | düşük | Kodlama | "Softmax ve cross-entropy" bağlantı adı, kısmın adıyla uyuşmuyordu | Çözüldü |
| C17 | düşük | Alıştırma | Aynı formülde ayraç olarak hem virgül hem noktalı virgül | Çözüldü |
| C18 | düşük | sözlük | Hareketli ortalama 0,95/0,05 ile tanımlanıyor; BatchNorm 0,9/0,1 kullanıyor | Çözüldü |
| C19 | düşük | sekmeler arası | Aynı sayılar ve açıklamalar (parametre sayıları, eğitim > doğrulama kaybı, dropout ölçeklemesi) 3–6 yerde tekrar ediliyor | **Açık: eğitmen kararı** |
| C20 | düşük | Teori, tarih | LeNet-5 ve MNIST (1998), AlexNet (2012) anılmıyor; Dropout ve Adam'ın kaynağı yok | Çözüldü |

## Tarihî bilgiler

| Bilgi | Sonuç | Kaynak |
| --- | --- | --- |
| Rosenblatt 1958 (Psychological Review 65(6)); IBM 704 gösterimi 7 Temmuz 1958; Mark I'ın ilk halka açık gösterimi 23 Haziran 1960; 400 fotosel | Doğru | Wikipedia "Perceptron", Cornell Chronicle |
| Novikoff 1962, Minsky–Papert 1969 | Doğru | Wikipedia "Perceptron" |
| İlk "yapay zekâ kışı" | Düzeltildi (P4) | Wikipedia "AI winter" |
| Linnainmaa 1970, Werbos 1974, Rumelhart–Hinton–Williams 1986 | Doğru | Schmidhuber (2015, arXiv:1404.7828) |
| Hamey 1998: 2-2-1 XOR ağında yerel minimum yok | Tartışmalı, düzeltildi (M3) | Sprinkhuizen-Kuyper ve Boers 1999 (Ann. Math. AI 25) |
| Fukushima 1980, LeCun 1989, LeNet-5 1998 | Doğru | yayın künyeleri |
| Ioffe ve Szegedy 2015, Santurkar ve ark. 2018 | Doğru | arXiv:1502.03167, arXiv:1805.11604 |
| Srivastava ve ark. 2014, Kingma ve Ba 2015 | Doğru | JMLR 15, arXiv:1412.6980 |

## Doğrulanamayanlar

- Sayfalar bu inceleme için tarayıcıda gözle okunmadı; demoların görsel davranışı koddan
  değerlendirildi. Düzeltilen metinlerin görünümü otomatik testlerle kontrol edildi.
- Tarayıcıdaki eğitim süresi iddiaları ("35–40 s/epoch") yeniden ölçülmedi.
- Hamey ile Sprinkhuizen-Kuyper ve Boers makalelerinin tam metni okunamadı.

## Takip

- 2026-11-26: "Çözüldü" diye işaretlenen maddeler bu incelemenin commit'inde (`f97cdad`)
  düzeltildi.
- 2026-11-26: eğitmen, önerinin 1. maddesini onayladı; CNN bölümü artık Teori ile açılıyor (C1).
  Eğitim sekmesini doğrudan açan testler `#egitim` adresine geçti. Bölümün Teori ile açıldığını ve
  Eğitim sekmesine geçince grafiklerin çizildiğini denetleyen bir test eklendi.
