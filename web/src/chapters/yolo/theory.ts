/**
 * The YOLO Theory tab: eight sections in two sessions (1–5: how the network turns an image into boxes
 * and masks; 6–8: how it is trained, measured and read), each next to a live demo (theory-demos.ts).
 *
 * Texts are { tr, en } pairs of HTML; \( TeX \) is typeset by math.js and [[key|text]] links a
 * glossary term. Every number in them was measured on this chapter's models or data (see
 * plans/2026-11-26_yolo-bolumu.md); dates and claims about other papers were checked at the source.
 */
import { $, h } from "../../shared/core.js";
import { KIT } from "../../shared/kit.js";
import type { Text } from "../../shared/chapters.ts";
import { T } from "./page.ts";
import { areaDemo, epochDemo, gridDemo, iouDemo, masksDemo, metricsDemo, nmsDemo, resetDemos, tasksDemo } from "./theory-demos.ts";

const R = String.raw;
const { html, formula, sectionCard, nextChapterCard } = KIT as {
  html: (tag: string, attrs: object, content: Text) => HTMLElement;
  formula: (tex: string) => HTMLElement;
  sectionCard: (id: string, num: number, title: Text, paragraphs: (Node | null)[], demo: Node, mathSection?: string | null) => HTMLElement;
  nextChapterCard: () => HTMLElement | null;
};

interface Section {
  h: Text;
  p: Text[];
  model: Text;
}

const INTRO = {
  h: { tr: "Nesne tespiti ve segmentasyon: YOLO nasıl çalışır?", en: "Object detection and segmentation: how does YOLO work?" },
  p: [
    {
      tr: R`CNN bölümündeki ağ bir görüntüye tek bir cevap veriyordu: "bu bir 7". Kuş bakışı bir şehir karesinde soru başkadır: karede kaç bina var, nerede, sınırları nereden geçiyor, ne kadar alan kaplıyor? Bu bölüm, bu soruları görüntüye tek bir kez bakarak cevaplayan YOLO'yu (<i>You Only Look Once</i>) anlatıyor.`,
      en: R`The network of the CNN chapter gave an image one answer: "this is a 7". On an aerial view of a city the question is different: how many buildings are in the tile, where are they, where do their outlines run, how much area do they cover? This chapter is about YOLO (<i>You Only Look Once</i>), which answers these questions by looking at the image once.`,
    },
    {
      tr: R`Örnek gerçektir: bu bölüm için SpaceNet'in yaklaşık 0,3 m çözünürlüklü uydu görüntüleri (Las Vegas, Paris, Şanghay, Hartum) üzerinde iki YOLO26 modeli eğittik. Biri binaları ve spor sahalarını tek tek bulur; öteki her pikseli bina, yol, yeşil alan, su ya da otopark olarak boyar. İkisi de bu sayfada, tarayıcınızda çalışır; demolardaki her sayı gerçek modelden gelir.`,
      en: R`The example is real: for this chapter we trained two YOLO26 models on SpaceNet satellite images of about 0.3 m resolution (Las Vegas, Paris, Shanghai, Khartoum). One finds buildings and sports fields one by one; the other paints every pixel as building, road, green, water or parking. Both run on this page, in your browser; every number in the demos comes from the real model.`,
    },
    {
      tr: R`Bölüm iki oturuma bölünebilir. 1–5. kısımlar ağın bir görüntüden kutuları ve maskeleri nasıl ürettiğini, 6–8. kısımlar nasıl eğitildiğini, ne kadar iyi olduğunun nasıl ölçüldüğünü ve bir mühendisin sonucu nasıl okuması gerektiğini anlatır. Bölüm 3'teki [[convolution|evrişim]], [[stride|stride]] ve [[feature-map|öznitelik haritası]] burada temel kavramlardır.`,
      en: R`The chapter can be split into two sessions. Sections 1–5 show how the network turns an image into boxes and masks; sections 6–8 how it is trained, how good it is and how an engineer should read the result. The [[convolution|convolution]], [[stride|stride]] and [[feature-map|feature map]] of chapter 3 are the basics here.`,
    },
  ],
  compare: {
    title: { tr: "CNN bölümünden YOLO'ya", en: "From the CNN chapter to YOLO" },
    head: [
      { tr: "", en: "" },
      { tr: "SimpleCNN (Bölüm 3)", en: "SimpleCNN (chapter 3)" },
      { tr: "YOLO26n-seg (bu bölüm)", en: "YOLO26n-seg (this chapter)" },
    ],
    rows: [
      [
        { tr: "Girdi", en: "Input" },
        { tr: "28 × 28 gri rakam", en: "28 × 28 grey digit" },
        { tr: "640 × 640 renkli uydu karesi", en: "640 × 640 colour satellite tile" },
      ],
      [
        { tr: "Çıktı", en: "Output" },
        { tr: "10 olasılık", en: "10 probabilities" },
        { tr: "8400 aday: kutu, sınıf skoru, maske", en: "8400 candidates: box, class score, mask" },
      ],
      [
        { tr: "Parametre", en: "Parameters" },
        { tr: "582.218", en: "582,218" },
        { tr: "yaklaşık 3 milyon", en: "about 3 million" },
      ],
      [
        { tr: "Eğitim", en: "Training" },
        { tr: "tarayıcıda, dakikalar", en: "in the browser, minutes" },
        { tr: "ekran kartında, 78 dakika", en: "on a graphics card, 78 minutes" },
      ],
    ] as Text[][],
  },
};

const S: Record<string, Section> = {
  tasks: {
    h: { tr: "Ne ve nerede? Dört görev", en: "What and where? Four tasks" },
    p: [
      {
        tr: R`Aynı kareye dört farklı soru sorulabilir. <b>Sınıflandırma</b> "karede bina var mı?" diye sorar ve tek cevap verir; CNN bölümündeki ağ bunu yapıyordu. [[object-detection|<b>Nesne tespiti</b>]] her nesneyi bir [[bounding-box|kutu]] ve bir sınıfla bulur. [[semantic-segmentation|<b>Anlamsal segmentasyon</b>]] her pikseli bir sınıfa atar; [[instance-segmentation|<b>örnek segmentasyonu</b>]] ise her nesneye kendi maskesini verir, böylece bitişik iki bina ayrı sayılır.`,
        en: R`The same tile can be asked four different questions. <b>Classification</b> asks "is there a building in the tile?" and gives one answer; the network of the CNN chapter did this. [[object-detection|<b>Object detection</b>]] finds every object with a [[bounding-box|box]] and a class. [[semantic-segmentation|<b>Semantic segmentation</b>]] gives every pixel a class; [[instance-segmentation|<b>instance segmentation</b>]] gives every object its own mask, so two buildings that touch are counted apart.`,
      },
      {
        tr: R`İnşaat mühendisliğinde karşılıkları: kentsel yapı envanteri (kaç bina, ne büyüklükte), yeni ya da ruhsatsız yapıların tespiti, afet sonrası hasar taraması, yapılaşma yoğunluğu ve yeşil alan oranı. Sayılabilen nesneler (bina, saha) için örnek segmentasyonu, sayılamayan yüzeyler (yol, su, yeşil alan) için anlamsal segmentasyon uygundur; bu yüzden bu bölümde iki model var.`,
        en: R`In civil engineering they stand for: an urban building inventory (how many buildings, how large), finding new or unpermitted buildings, damage surveys after a disaster, building density and the share of green space. Instance segmentation suits objects you can count (buildings, fields), semantic segmentation surfaces you cannot (roads, water, grass); that is why this chapter has two models.`,
      },
      {
        tr: R`YOLO'yu 2015'te Joseph Redmon ve arkadaşları tanıttı (makalesi 2016'da CVPR'da yayımlandı). Fikri: görüntüyü ağdan bir kez geçirip bütün kutuları birlikte tahmin etmek; önceki yöntemler aday bölgeleri tek tek inceliyordu. Sonraki sürümlerden bu bölüm için önemli olanlar: YOLOv8 (Ultralytics, Ocak 2023) önceden tanımlı "çapa" kutularını bıraktı; YOLOv10 (Tsinghua Üniversitesi, 2024) NMS adımını gereksiz kılan bir eğitim yöntemi getirdi; YOLO26 (Ultralytics, Ocak 2026) ikisini birleştirir. Bu bölümün modelleri YOLO26'dır; Ultralytics'in Eylül 2026'da duyurduğu YOLO27 henüz yayımlanmadı.`,
        en: R`YOLO was introduced in 2015 by Joseph Redmon and colleagues (the paper appeared at CVPR in 2016). The idea: pass the image through the network once and predict all boxes together; earlier methods examined candidate regions one by one. Of the later versions, these matter for this chapter: YOLOv8 (Ultralytics, January 2023) dropped predefined "anchor" boxes; YOLOv10 (Tsinghua University, 2024) brought a training method that makes the NMS step unnecessary; YOLO26 (Ultralytics, January 2026) combines both. The models of this chapter are YOLO26; YOLO27, which Ultralytics announced in September 2026, is not released yet.`,
      },
    ],
    model: {
      tr: R`Bu bölümün modelleri: YOLO26n-seg (n: nano, en küçük boy) iki sınıfla (bina, spor sahası) ve YOLO26n-sem altı sınıfla (arka plan, bina, yol, yeşil alan, su, otopark). Demodaki etiketler modelin değil, doğru etiketlerdir: binalar SpaceNet'te elle çizildi, öteki sınıflar OpenStreetMap'ten üretildi.`,
      en: R`The models of this chapter: YOLO26n-seg (n: nano, the smallest size) with two classes (building, sports field), and YOLO26n-sem with six (background, building, road, green, water, parking). The labels in the demo are the true labels, not the model's: buildings were drawn by hand in SpaceNet, the other classes made from OpenStreetMap.`,
    },
  },
  iou: {
    h: { tr: "Kutu ve IoU", en: "Boxes and IoU" },
    p: [
      {
        tr: R`Bir [[bounding-box|kutu]] dört sayıdır. İki yazılışı var: köşelerle \((x_1, y_1, x_2, y_2)\) ya da merkez ve boyutla \((c_x, c_y, w, h)\). Biri ötekine kolayca çevrilir. Görüntü koordinatlarında \(y\) aşağı doğru artar.`,
        en: R`A [[bounding-box|box]] is four numbers. It has two forms: by its corners \((x_1, y_1, x_2, y_2)\), or by its centre and size \((c_x, c_y, w, h)\). One converts easily into the other. In image coordinates \(y\) grows downwards.`,
      },
      {
        tr: R`Tahmin edilen bir kutu ne kadar doğru? Ölçüt [[iou|IoU]]'dur (<i>intersection over union</i>, kesişim bölü birleşim): iki kutunun ortak alanı, kapladıkları toplam alana bölünür. Tam örtüşmede 1, hiç örtüşmeyince 0'dır.`,
        en: R`How right is a predicted box? The measure is [[iou|IoU]] (intersection over union): the area the two boxes share, divided by the area they cover together. It is 1 when they match exactly, 0 when they do not overlap at all.`,
      },
      {
        tr: R`Neden merkezler arasındaki mesafe değil? Mesafe boyutu görmez: 5 m kayma küçük bir ev için büyük bir hatadır, büyük bir depo için önemsizdir. IoU orana baktığı için nesnenin büyüklüğünden bağımsızdır. Bir tahmin genellikle IoU ≥ 0,5 ise doğru sayılır; 7. kısımdaki ölçümler bu eşiğe dayanır. Segmentasyonda aynı formül kutular yerine maskelerle hesaplanır: <b>maske IoU</b>.`,
        en: R`Why not the distance between the centres? Distance does not see size: a 5 m shift is a big mistake for a small house and nothing for a large warehouse. IoU looks at a ratio, so it does not depend on the size of the object. A prediction usually counts as correct when IoU ≥ 0.5; the scores of section 7 rest on this threshold. In segmentation the same formula is computed with masks instead of boxes: <b>mask IoU</b>.`,
      },
    ],
    model: {
      tr: R`Tespit sekmesindeki "doğru etiketlerle karşılaştır" seçeneği maske IoU ≥ 0,5 kuralını kullanır: yeşil tespitler bu eşiği geçenlerdir.`,
      en: R`The "compare with the true labels" option of the Detection tab uses the rule mask IoU ≥ 0.5: the green detections are the ones that pass it.`,
    },
  },
  grid: {
    h: { tr: "Tek bakışta: üç ölçekte ızgara", en: "One look: a grid at three scales" },
    p: [
      {
        tr: R`YOLO'nun gövdesi Bölüm 3'teki gibi bir [[cnn|CNN]]'dir: [[stride|stride]] 2 olan [[convolution|evrişim]] katmanları görüntüyü küçültürken kanal sayısını artırır. 640 × 640 bir girişten 8, 16 ve 32 kat küçültülmüş üç [[feature-map|öznitelik haritası]] alınır: 80 × 80, 40 × 40 ve 20 × 20. "Boyun" bu üç ölçeği birbirine karıştırır, "baş" her hücre için tahmin yapar.`,
        en: R`The backbone of YOLO is a [[cnn|CNN]] as in chapter 3: [[convolution|convolution]] layers with [[stride|stride]] 2 shrink the image while adding channels. From a 640 × 640 input it takes three [[feature-map|feature maps]], 8, 16 and 32 times smaller: 80 × 80, 40 × 40 and 20 × 20. The "neck" mixes the three scales; the "head" predicts for every cell.`,
      },
      {
        tr: R`Her hücre bir aday demektir. Her aday 38 sayı verir: hücre merkezinden kutunun dört kenarına uzaklık (4 sayı; YOLO26'da önceden tanımlı çapa kutusu yoktur), iki sınıfın skoru (bina, spor sahası) ve maskesi için 32 katsayı (5. kısım).`,
        en: R`Every cell is one candidate. Every candidate gives 38 numbers: the distances from the cell's centre to the four sides of its box (4 numbers; YOLO26 has no predefined anchor boxes), the scores of the two classes (building, sports field) and 32 coefficients for its mask (section 5).`,
      },
      {
        tr: R`Neden üç ölçek? İnce ızgaranın bir hücresi 8 model pikselidir (bu karelerde 1,6–2,4 m) ve küçük bir evi yakalar; kaba ızgaranın hücresi 32 pikseldir (6,5–10 m) ve daha geniş bir [[receptive-field|algı alanıyla]] bir futbol sahasını bütün olarak görür. Yeni YOLO'lar gövdeye küçük dikkat blokları da ekler; dikkat mekanizması Bölüm 6'nın konusu.`,
        en: R`Why three scales? A cell of the fine grid is 8 model pixels (1.6–2.4 m on these tiles) and catches a small house; a cell of the coarse grid is 32 pixels (6.5–10 m) and, with a wider [[receptive-field|receptive field]], sees a football field as a whole. Newer YOLOs also add small attention blocks to the backbone; attention is the subject of chapter 6.`,
      },
    ],
    model: {
      tr: R`YOLO26n-seg'in yaklaşık 3 milyon parametresi var, SimpleCNN'in (582.218) beş katı kadar. 640 × 640 bir kare için bir geçiş bu tarayıcıda yarım saniye civarı sürer.`,
      en: R`YOLO26n-seg has about 3 million parameters, some five times SimpleCNN's 582,218. One pass over a 640 × 640 tile takes about half a second in this browser.`,
    },
  },
  nms: {
    h: { tr: "Çok fazla aday: güven eşiği ve NMS", en: "Too many candidates: the confidence threshold and NMS" },
    p: [
      {
        tr: R`8400 adayın çoğu boş zemine bakar ve skorları sıfıra yakındır. [[confidence-score|Güven eşiği]] (genellikle 0,25) bunları atar. Kalanlar arasında aynı binayı bulan birkaç komşu aday vardır; her biri biraz farklı bir kutu önerir.`,
        en: R`Most of the 8400 candidates look at empty ground and score close to zero. The [[confidence-score|confidence threshold]] (usually 0.25) drops them. Among those left, several neighbouring candidates find the same building, each proposing a slightly different box.`,
      },
      {
        tr: R`[[nms|NMS]] (<i>non-maximum suppression</i>, en büyük olmayanları bastırma) bu kopyaları temizler: adaylar skora göre sıralanır; en yüksek skorlu aday tutulur; aynı sınıftan olup onunla IoU'su eşiği (burada 0,7) aşan adaylar silinir; kalanlarla aynı adım tekrarlanır. Demoda adımları tek tek oynatabilirsiniz.`,
        en: R`[[nms|NMS]] (non-maximum suppression) cleans up these copies: the candidates are sorted by score; the highest one is kept; those of the same class that overlap it by more than the IoU threshold (here 0.7) are removed; the same step repeats with what is left. You can play the steps one by one in the demo.`,
      },
      {
        tr: R`NMS'in bedeli var: ayrı bir adımdır, aday sayısıyla yavaşlar ve bitişik iki gerçek nesneyi tek nesneye indirebilir. YOLOv10 ve YOLO26 eğitimde ikinci bir "bire bir" baş kullanır; bu baş her nesne için tek aday seçmeyi öğrenir ve NMS gerekmez. Bizim modelde ölçtük: NMS'siz baş maske mAP50'de 0,47, NMS'li çıktı 0,50 verdi (mAP 7. kısımda). Bu sayfada NMS'li çıktı çalışıyor.`,
        en: R`NMS has a price: it is a separate step, it slows down with the number of candidates, and it can merge two real objects that touch into one. YOLOv10 and YOLO26 train a second, "one-to-one" head that learns to pick a single candidate per object, so NMS is not needed. We measured it on our model: the head without NMS gave a mask mAP50 of 0.47, the output with NMS 0.50 (mAP is section 7). This page runs the output with NMS.`,
      },
    ],
    model: {
      tr: R`Şanghay karesinde: skoru 0,25'i geçen 107 adaydan NMS sonunda 15 nesne kalır (11 bina, 4 spor sahası). Python'daki ONNX Runtime da aynı sayıları verir.`,
      en: R`On the Shanghai tile: of the 107 candidates scoring above 0.25, 15 objects are left after NMS (11 buildings, 4 sports fields). ONNX Runtime in Python gives the same numbers.`,
    },
  },
  masks: {
    h: { tr: "Maskeler: prototipler ve katsayılar", en: "Masks: prototypes and coefficients" },
    p: [
      {
        tr: R`Her nesne için 640 × 640'lık ayrı bir maske tahmin etmek pahalı olurdu. YOLO bunun yerine YOLACT'ın (Bolya ve arkadaşları, 2019) fikrini kullanır: ağ her görüntü için 32 tane 160 × 160'lık [[mask-prototype|prototip]] harita üretir; her aday da bu prototipler için 32 katsayı verir (3. kısımdaki 38 sayının 32'si).`,
        en: R`Predicting a separate 640 × 640 mask for every object would be expensive. Instead YOLO uses the idea of YOLACT (Bolya and colleagues, 2019): for every image the network makes 32 [[mask-prototype|prototype]] maps of 160 × 160; every candidate gives 32 coefficients for them (32 of the 38 numbers of section 3).`,
      },
      {
        tr: R`Bir nesnenin maskesi, prototiplerin katsayılarla ağırlıklı toplamıdır. [[sigmoid|Sigmoid]]'den sonra 0,5'i geçen ve kutunun içinde kalan pikseller maskeyi oluşturur. Prototipler bütün nesneler için aynıdır; biri kenarları, biri çatıların içini, biri bir yönü vurgulayabilir. Katsayılar bunlardan hangisinin ne kadar kullanılacağını söyler. Demoda bir katsayıyı değiştirip maskenin nasıl değiştiğini görebilirsiniz.`,
        en: R`An object's mask is the prototypes' sum weighted by the coefficients. The pixels inside the box that pass 0.5 after the [[sigmoid|sigmoid]] form the mask. The prototypes are the same for every object; one may stress edges, one the inside of roofs, one a direction. The coefficients say how much of each to use. In the demo you can change a coefficient and watch the mask change.`,
      },
      {
        tr: R`160 = 640 / 4: bir prototip pikseli 4 model pikseline, bu karelerde 0,8–1,2 m'ye karşılık gelir (bir pikselin doğu-batı boyu şehre göre değişir). Maske büyütülürken bu ayrıntı sınırı kalır; 8. kısımda alana etkisini göreceğiz. Arazi örtüsü modeli (anlamsal segmentasyon) bu düzeneği kullanmaz: her piksel için altı sınıfın skorunu doğrudan verir ve en yükseğini seçer; nesneleri birbirinden ayırmaz.`,
        en: R`160 = 640 / 4: one prototype pixel is 4 model pixels, 0.8–1.2 m on these tiles (the east-west size of a pixel depends on the city). That limit to the detail stays when the mask is enlarged; section 8 shows what it does to the area. The land-cover model (semantic segmentation) does not use this mechanism: it gives the scores of the six classes for every pixel directly and picks the highest; it does not tell objects apart.`,
      },
    ],
    model: {
      tr: R`Prototipler 32 × 160 × 160 sayıdır ve bir görüntü için bir kez hesaplanır; bir nesnenin maskesi için yalnızca 32 katsayısı gerekir. Tespit sekmesinde bir eşiği değiştirince ağın yeniden çalışmaması bundandır.`,
      en: R`The prototypes are 32 × 160 × 160 numbers, computed once per image; an object's mask needs only its 32 coefficients. That is why the network does not run again when you move a threshold in the Detection tab.`,
    },
  },
  train: {
    h: { tr: "Nasıl eğitilir?", en: "How is it trained?" },
    p: [
      {
        tr: R`Eğitim verisi, her karede doğru nesnelerin çokgenleridir. İlk soru, 8400 adaydan hangisinin hangi binadan sorumlu olacağıdır: [[label-assignment|etiket ataması]]. YOLO26 bir adayın hem sınıf skoruna hem de kutusunun gerçek kutuyla IoU'suna bakar (skor × IoU⁶) ve her nesneye en uygun adayları atar: bire-çok başta en fazla 10, bire bir başta tek aday. Eğitim ilerledikçe ağırlık bire-çok baştan bire bir başa kayar.`,
        en: R`The training data are the outlines of the true objects in every tile. The first question is which of the 8400 candidates is responsible for which building: [[label-assignment|label assignment]]. YOLO26 looks at both a candidate's class score and its box's IoU with the true box (score × IoU⁶) and gives each object its best candidates: up to 10 in the one-to-many head, a single one in the one-to-one head. As training goes on, the weight shifts from the one-to-many head to the one-to-one head.`,
      },
      {
        tr: R`Atanan adaylar için üç [[loss|kayıp]] hesaplanır: <b>kutu</b> (IoU tabanlı CIoU ve dört uzaklık üzerinde L1), <b>sınıf</b> (her sınıf için ayrı bir evet/hayır: [[cross-entropy|ikili çapraz entropi]]; softmax değil, çünkü bir aday hiçbir sınıfa ait olmayabilir) ve <b>maske</b> (her piksel için ikili çapraz entropi). Toplam kayıp bunların ağırlıklı toplamıdır; [[optimizer|optimizasyon]] Bölüm 3'teki gibi ilerler. Ultralytics bizim eğitimde AdamW'yi seçti.`,
        en: R`For the assigned candidates three [[loss|losses]] are computed: <b>box</b> (the IoU-based CIoU, plus L1 on the four distances), <b>class</b> (a separate yes/no for every class: [[cross-entropy|binary cross-entropy]]; not softmax, because a candidate may belong to no class at all) and <b>mask</b> (binary cross-entropy for every pixel). The total is their weighted sum; [[optimizer|optimization]] works as in chapter 3. In our training Ultralytics chose AdamW.`,
      },
      {
        tr: R`Modeller COCO'da (günlük fotoğraflardan 80 sınıf) eğitilmiş ağırlıklarla başladı. COCO'da "bina" yoktur, ama kenar, köşe ve doku bilgisi her görüntüde işe yarar: [[transfer-learning|aktarım öğrenmesi]]. Ölçtük: 100 epoch sonunda sıfırdan başlayan model maske mAP50 0,32, önceden eğitilmiş model 0,50 verdi. [[data-augmentation|Veri artırma]]: kuş bakışında "yukarı" yoktur; kareler rastgele döndürüldü (±90°), yatay ve dikey çevrildi, dördü bir mozaikte birleştirildi. Eğitim ve doğrulama ayrımı Bölüm 2'deki gibidir: 528 karede eğitildi, 131 karede ölçüldü.`,
        en: R`The models started from weights trained on COCO (80 classes of everyday photos). COCO has no "building", but edges, corners and textures help in every image: [[transfer-learning|transfer learning]]. We measured it: after 100 epochs the model that started from scratch gave a mask mAP50 of 0.32, the pretrained one 0.50. [[data-augmentation|Data augmentation]]: a view from above has no "up"; tiles were rotated at random (±90°), flipped both ways and joined four at a time in a mosaic. The split into training and validation is as in chapter 2: trained on 528 tiles, measured on 131.`,
      },
    ],
    model: {
      tr: R`100 epoch, batch 4: RTX 2060 ekran kartında 78 dakika. Eğitim sekmesinde bütün eğriler ve bu karenin her kaydedilen epoch'taki tahmini var.`,
      en: R`100 epochs, batch 4: 78 minutes on an RTX 2060 graphics card. The Training tab has all the curves and this tile's prediction at every saved epoch.`,
    },
  },
  metrics: {
    h: { tr: "Ne kadar iyi? Kesinlik, duyarlılık, mAP", en: "How good? Precision, recall, mAP" },
    p: [
      {
        tr: R`Bir eşikte her tespit ya doğrudur (aynı sınıftan bir gerçek nesneyle IoU ≥ 0,5; TP) ya yanlıştır (FP); bulunamayan gerçek nesneler kaçırılmıştır (FN). [[precision|Kesinlik]] tespitlerin ne kadarının doğru olduğunu, [[recall|duyarlılık]] gerçek nesnelerin ne kadarının bulunduğunu söyler.`,
        en: R`At a given threshold every detection is either correct (IoU ≥ 0.5 with a true object of its class; TP) or wrong (FP); the true objects nobody found are missed (FN). [[precision|Precision]] says how many of the detections are right, [[recall|recall]] how many of the true objects were found.`,
      },
      {
        tr: R`Güven eşiği ikisini birbirine karşı değiştirir: eşiği düşürünce daha çok bina bulunur (duyarlılık artar) ama yanlışlar da çoğalır (kesinlik düşer). Bütün eşiklerin (R, P) noktaları PR eğrisini çizer; eğrinin altındaki alan [[average-precision|AP]]'dir (Ultralytics, COCO gibi 101 duyarlılık düzeyinde okur). mAP, sınıfların AP'lerinin ortalamasıdır: mAP50 IoU eşiği 0,5 ile, mAP50-95 ise 0,50'den 0,95'e kadar on eşiğin ortalamasıyla hesaplanır; ikincisi sınırları tam tutturmayı da ödüllendirir.`,
        en: R`The confidence threshold trades one against the other: a lower threshold finds more buildings (recall goes up) but also more mistakes (precision goes down). The (R, P) points of all thresholds draw the PR curve; the area under it is the [[average-precision|AP]] (Ultralytics reads it at 101 recall levels, like COCO). mAP is the mean of the classes' APs: mAP50 uses an IoU threshold of 0.5, mAP50-95 the mean over ten thresholds from 0.50 to 0.95, so it also rewards outlines that fit exactly.`,
      },
      {
        tr: R`Bizim modelin doğrulamadaki sayıları: bina maske AP50 0,69, spor sahası 0,30, ortalaması (mAP50) 0,50; mAP50-95 ise 0,25. Yani bir binayı bulmak, sınırını tam tutturmaktan kolaydır. Spor sahası zayıftır: eğitimde yalnızca 529 saha vardı ve etiketleri OpenStreetMap'ten geliyor; bazı sahalar haritada hiç yok, bazılarında pistin içi ile tamamı karışmış.`,
        en: R`Our model's scores on the validation set: building mask AP50 0.69, sports field 0.30, their mean (mAP50) 0.50; mAP50-95 is 0.25. So finding a building is easier than fitting its outline exactly. Sports fields are weak: there were only 529 of them in training and their labels come from OpenStreetMap; some fields are missing from the map, and for some the inner pitch and the whole track are mixed up.`,
      },
    ],
    model: {
      tr: R`Demo, 51 binalı bir Paris karesinde yalnızca bina sınıfının eğrisini çizer. Tek karenin AP'si bütün doğrulama setininkinden farklıdır; ama hesap aynıdır.`,
      en: R`The demo draws the curve of the building class only, on a Paris tile with 51 buildings. One tile's AP differs from that of the whole validation set; the computation is the same.`,
    },
  },
  read: {
    h: { tr: "Mühendis sonucu nasıl okur?", en: "How does an engineer read the result?" },
    p: [
      {
        tr: R`Maskeden alan: alan = piksel sayısı × bir pikselin zemindeki alanı. SpaceNet kareleri derece cinsinden olduğu için bir piksel Paris'te 0,20 m × 0,30 m, Las Vegas'ta 0,24 m × 0,30 m'dir (doğu-batı boyu enleme bağlıdır). Karedeki yapılaşma oranı = bina maskelerinin alanı / karenin alanı; Tespit sekmesi bunu her kare için hesaplar.`,
        en: R`Area from a mask: area = number of pixels × the ground area of one pixel. SpaceNet tiles are in degrees, so a pixel is 0.20 m × 0.30 m in Paris and 0.24 m × 0.30 m in Las Vegas (the east-west size depends on the latitude). The tile's built-up ratio = the area of the building masks / the area of the tile; the Detection tab computes it for every tile.`,
      },
      {
        tr: R`Bu oran TAKS'a (taban alanı katsayısı) benzer ama aynı değildir: kare bir parsel değildir, yolları ve parkları da içerir; uydu tabanı değil çatıyı görür (saçak taşmaları, eğik çekimde cephe). Hata kaynakları: maske çözünürlüğü (bir prototip pikseli 0,8–1,2 m), gölgeler, siyah çatılar, bitişik nizam yapılar (Hartum'da bitişik evler tek blok görünür). 0,3 m/piksel çözünürlükte bir prototip pikseli 1,22 m'dir; 10 m × 10 m bir binada sınırın her yandan bu kadar kayması alanı %43 küçültür ya da %55 büyütür:`,
        en: R`This ratio is like the Turkish TAKS (building coverage ratio) but not the same: a tile is not a parcel, it contains roads and parks too; and the satellite sees the roof, not the footprint (eaves, and on an oblique shot the facade). Sources of error: mask resolution (one prototype pixel is 0.8–1.2 m), shadows, black roofs, attached buildings (in Khartoum houses that touch look like one block). At 0.3 m per pixel one prototype pixel is 1.22 m; on a 10 m × 10 m building, moving the outline that much on every side shrinks the area by 43% or grows it by 55%:`,
      },
      {
        tr: R`Model başka bir şehre taşınınca kötüleşebilir: çatı malzemesi, renkler, yapı tipi, çözünürlük ve çekim açısı değişir ([[generalization|genelleme]]). İstanbul'da kullanılacaksa İstanbul'dan etiketli karelerle ince ayar yapılmalıdır. Görüntülerin lisansı da önemlidir: bu bölümde Google Haritalar ekran görüntüleri kullanılmadı, çünkü kullanım koşulları görüntülerin indirilip başka amaçla kullanılmasını kısıtlıyor ve eğitilen model herkese açık. SpaceNet CC BY-SA 4.0, OpenStreetMap ODbL ile paylaşılıyor.`,
        en: R`The model can do worse when moved to another city: roof materials, colours, building types, resolution and viewing angle change ([[generalization|generalization]]). To use it in Istanbul it should be fine-tuned on labelled tiles from Istanbul. The licence of the imagery matters too: this chapter used no Google Maps screenshots, because their terms restrict downloading the imagery for other uses, and the trained model is public. SpaceNet is shared under CC BY-SA 4.0, OpenStreetMap under ODbL.`,
      },
    ],
    model: {
      tr: R`Demo bir binanın maske sınırını prototip pikseli adımlarıyla kaydırır (bu Paris karesinde doğu-batı 0,8 m, kuzey-güney 1,2 m): küçük evlerin alanı büyük depolarınkinden çok daha belirsizdir.`,
      en: R`The demo moves a building's mask outline in steps of one prototype pixel (on this Paris tile 0.8 m east-west, 1.2 m north-south): the area of small houses is much less certain than that of large warehouses.`,
    },
  },
};

const BRIDGE: Text = {
  tr: R`<b>İkinci oturum.</b> Ağın bir görüntüden ne ürettiğini gördük: 8400 aday, eşik ve NMS, prototiplerden maskeler. Şimdi bunu nasıl öğrendiğine, ne kadar iyi olduğunun nasıl ölçüldüğüne ve sonucun nasıl okunacağına bakıyoruz.`,
  en: R`<b>Second session.</b> We have seen what the network makes of an image: 8400 candidates, a threshold and NMS, masks from prototypes. Now we look at how it learns this, how we measure how good it is and how to read the result.`,
};

const RECAP = {
  h: { tr: "Özet ve kendinizi sınayın", en: "Summary and self-test" },
  p: {
    tr: R`<b>Özet:</b> YOLO bir görüntüye bir kez bakar: CNN gövdesi üç ölçekte bir ızgara çıkarır ve her hücre bir aday olur (640 piksellik girişte 8400). Her aday bir kutu, sınıf skorları ve 32 maske katsayısı verir. [[confidence-score|Güven eşiği]] boş adayları, [[nms|NMS]] aynı nesnenin kopyalarını atar; maskeler 32 ortak [[mask-prototype|prototipin]] doğrusal birleşimidir. Eğitimde her nesneye uygun adaylar atanır ve kutu, sınıf ve maske kayıpları birlikte küçültülür; önceden eğitilmiş ağırlıklar büyük fark yapar. Sonuç [[iou|IoU]], [[precision|kesinlik]], [[recall|duyarlılık]] ve mAP ile ölçülür; alan gibi mühendislik sonuçları maskenin çözünürlüğü kadar güvenilirdir.`,
    en: R`<b>Summary:</b> YOLO looks at an image once: a CNN backbone produces a grid at three scales and every cell is a candidate (8400 for a 640-pixel input). Every candidate gives a box, class scores and 32 mask coefficients. The [[confidence-score|confidence threshold]] drops empty candidates, [[nms|NMS]] the copies of the same object; masks are linear combinations of 32 shared [[mask-prototype|prototypes]]. In training every object gets suitable candidates, and the box, class and mask losses are lowered together; pretrained weights make a large difference. The result is measured with [[iou|IoU]], [[precision|precision]], [[recall|recall]] and mAP; engineering results such as areas are only as reliable as the mask's resolution.`,
  },
  q: {
    tr: R`<b>Kendinizi sınayın:</b> (1) 640 yerine 512 piksellik bir giriş kaç aday üretir? (2) 3 sınıflı bir modelde her aday kaç sayı verir? (3) Güven eşiği 0,25'ten 0,05'e inerse kesinlik ve duyarlılık nasıl değişir? (4) Maske 1,2 m'lik adımlarla çözümleniyorsa, küçük evlerin mi büyük depoların mı alanı daha güvenilirdir, neden? Önce kendiniz cevaplayın, sonra aşağıdan kontrol edin.`,
    en: R`<b>Test yourself:</b> (1) How many candidates does a 512-pixel input give instead of 640? (2) In a model with 3 classes, how many numbers does each candidate give? (3) If the confidence threshold drops from 0.25 to 0.05, how do precision and recall change? (4) If the mask resolves in steps of 1.2 m, is the area of small houses or of large warehouses more reliable, and why? Answer first, then check below.`,
  },
  a: {
    tr: R`(1) \(64^2 + 32^2 + 16^2 = 4096 + 1024 + 256 = 5376\) (512'nin 8, 16 ve 32'ye bölümü). Yalnızca en ince ızgarayı sayıp 4096 demek tipik bir hatadır. (2) \(4 + 3 + 32 = 39\): kutunun dört uzaklığı, her sınıfa bir skor, 32 maske katsayısı. (3) Duyarlılık artar ya da aynı kalır, kesinlik genellikle düşer: Şanghay karesinde eşiği geçen aday 107'den 281'e, NMS sonrası nesne 15'ten 47'ye çıkar. (4) Büyük depoların: sınırdaki hata çevreyle, yani kenar uzunluğuyla orantılıdır; alan ise kenarın karesiyle büyür. 10 m'lik bir evde 1,2 m'lik kayma alanı yarı yarıya değiştirebilir, 100 m'lik bir depoda yalnızca birkaç yüzde.`,
    en: R`(1) \(64^2 + 32^2 + 16^2 = 4096 + 1024 + 256 = 5376\) (512 divided by 8, 16 and 32). Counting only the finest grid and saying 4096 is a typical mistake. (2) \(4 + 3 + 32 = 39\): the four distances of the box, one score per class, 32 mask coefficients. (3) Recall goes up or stays, precision usually goes down: on the Shanghai tile the candidates above the threshold go from 107 to 281, the objects after NMS from 15 to 47. (4) The warehouses': the error at the outline grows with the perimeter, that is with the side length, while the area grows with its square. On a 10 m house a 1.2 m shift can change the area by half, on a 100 m warehouse only by a few per cent.`,
  },
};

/** Sections in reading order: id, text, formulas after paragraph n, demo. */
const ORDER: [string, Section, Record<number, string | Text>, () => HTMLElement][] = [
  ["y-tasks", S.tasks, {}, tasksDemo],
  [
    "y-iou",
    S.iou,
    {
      0: R`x_1 = c_x - \tfrac{w}{2},\quad y_1 = c_y - \tfrac{h}{2},\quad x_2 = c_x + \tfrac{w}{2},\quad y_2 = c_y + \tfrac{h}{2}`,
      1: R`\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}`,
    },
    iouDemo,
  ],
  [
    "y-grid",
    S.grid,
    {
      0: R`80^2 + 40^2 + 20^2 = 6400 + 1600 + 400 = 8400`,
      1: {
        tr: R`\underbrace{4}_{\text{kutu}} + \underbrace{2}_{\text{sınıf}} + \underbrace{32}_{\text{maske}} = 38`,
        en: R`\underbrace{4}_{\text{box}} + \underbrace{2}_{\text{classes}} + \underbrace{32}_{\text{mask}} = 38`,
      },
    },
    gridDemo,
  ],
  ["y-nms", S.nms, {}, nmsDemo],
  [
    "y-masks",
    S.masks,
    {
      0: {
        tr: R`M = \sigma\Big(\sum_{k=1}^{32} c_k\, P_k\Big), \qquad \text{maske} = \{\, M > 0{,}5 \,\} \cap \text{kutu}`,
        en: R`M = \sigma\Big(\sum_{k=1}^{32} c_k\, P_k\Big), \qquad \text{mask} = \{\, M > 0.5 \,\} \cap \text{box}`,
      },
    },
    masksDemo,
  ],
  ["y-train", S.train, {}, epochDemo],
  ["y-metrics", S.metrics, { 0: R`P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN}` }, metricsDemo],
  [
    "y-read",
    S.read,
    {
      1: {
        tr: R`\begin{aligned} \frac{(10 - 2\cdot 1{,}22)^2}{10^2} &\approx 0{,}57 \;(-\%43) \\ \frac{(10 + 2\cdot 1{,}22)^2}{10^2} &\approx 1{,}55 \;(+\%55) \end{aligned}`,
        en: R`\begin{aligned} \frac{(10 - 2\cdot 1.22)^2}{10^2} &\approx 0.57 \;(-43\%) \\ \frac{(10 + 2\cdot 1.22)^2}{10^2} &\approx 1.55 \;(+55\%) \end{aligned}`,
      },
    },
    areaDemo,
  ],
];

let built = false;

function build(): void {
  resetDemos();
  const panel = $("#panel-theory") as HTMLElement;
  const c = INTRO.compare;
  const table = h(
    "table",
    { class: "yo-table yo-compare" },
    h("thead", {}, h("tr", {}, ...c.head.map((t) => h("th", { scope: "col", text: T(t) })))),
    h(
      "tbody",
      {},
      ...c.rows.map(([k, a, b]) => h("tr", {}, h("th", { scope: "row", text: T(k) }), h("td", { text: T(a) }), h("td", { text: T(b) }))),
    ),
  );
  const intro = h(
    "div",
    { class: "card t-intro pc-intro", id: "y-intro" },
    h("div", { class: "t-intro-text" }, h("h2", { text: T(INTRO.h) }), ...INTRO.p.map((p) => html("p", {}, p))),
    h("div", { class: "pc-intro-side" }, h("div", { class: "mini-title", text: T(c.title) }), h("div", { class: "yo-table-wrap" }, table)),
  );
  const sections = ORDER.map(([id, s, formulas, demo], k) => {
    const paragraphs: (Node | null)[] = [];
    if (id === "y-train") paragraphs.push(html("p", { class: "t-bridge" }, BRIDGE));
    s.p.forEach((p, i) => {
      paragraphs.push(html("p", {}, p));
      const f = formulas[i];
      if (f) paragraphs.push(formula(typeof f === "string" ? f : T(f)));
    });
    paragraphs.push(html("div", { class: "in-model" }, s.model));
    return sectionCard(id, k + 1, s.h, paragraphs, demo());
  });
  const recap = h(
    "div",
    { class: "card t-intro", id: "y-recap" },
    h(
      "div",
      { class: "t-intro-text" },
      h("h2", { text: T(RECAP.h) }),
      html("p", {}, RECAP.p),
      html("p", {}, RECAP.q),
      h("details", { class: "t-answers" }, h("summary", { text: T({ tr: "Cevaplar", en: "Answers" }) }), html("p", {}, RECAP.a)),
      nextChapterCard(),
    ),
  );
  panel.replaceChildren(intro, ...sections, recap);
  built = true;
}

document.addEventListener("DOMContentLoaded", build);
document.addEventListener("lang-change", () => {
  if (built) build();
});
