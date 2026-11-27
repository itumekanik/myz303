/**
 * The YOLO Outputs tab: how to read what Ultralytics writes during and after training
 * (runs/yolo/seg-pretrained/), file by file, with the confusion matrix taught in four steps: a 2 × 2
 * table on a made-up damage survey, a toy tile for detection's 3 × 3 matrix, our model's real matrix
 * and the land-cover model's pixel matrix. Then the P, R, F1 and PR curves.
 *
 * Texts are { tr, en } pairs of HTML; [[key|text]] links a glossary term. Every number about our
 * model comes from data/yolo-outputs.js (tools/export_yolo_assets.py) or Ultralytics 8.4's code
 * (see plans/2026-11-27_yolo-ciktilar.md). The sections are built when the page loads, so links
 * into them work; the demos wait for the data, loaded when the tab is first opened.
 */
import { $, h } from "../../shared/core.js";
import { KIT } from "../../shared/kit.js";
import type { Text } from "../../shared/chapters.ts";
import { T } from "./page.ts";
import { apDemo, petsDemo } from "./outputs-basics.ts";
import {
  batchesDemo,
  consoleDemo,
  curvesDemo,
  folderDemo,
  labelsDemo,
  loadOutputs,
  realMatrixDemo,
  resultsDemo,
  semMatrixDemo,
  surveyDemo,
  toyDemo,
} from "./outputs-demos.ts";

const R = String.raw;
const { html, formula, sectionCard } = KIT as {
  html: (tag: string, attrs: object, content: Text) => HTMLElement;
  formula: (tex: string) => HTMLElement;
  sectionCard: (id: string, num: number, title: Text, paragraphs: (Node | null)[], demo: Node, mathSection?: string | null) => HTMLElement;
};

interface Section {
  id: string;
  h: Text;
  p: (Text | { tex: Text } | { block: Text })[];
  model: Text;
  demo: () => HTMLElement;
  math?: string;
}

const INTRO: { h: Text; p: Text[] } = {
  h: { tr: "Eğitim bitti: hangi dosyaya, nasıl bakılır?", en: "Training is over: which files to read, and how?" },
  p: [
    {
      tr: R`Ultralytics eğitim boyunca bir klasöre dosyalar yazar: ayarlar, her epoch'un ölçümleri, ağırlıklar, grafikler, örnek görüntüler. Bu sekme bizim eğitimimizin (YOLO26n-seg, bina ve spor sahası, 100 epoch) klasörünü dosya dosya okur. Dosyaların çoğu eğitim sırasında yazılır; bir kısmı ise eğitimin sonunda, <code>best.pt</code> ile yapılan son doğrulamada.`,
      en: R`Ultralytics writes files into a folder throughout training: settings, every epoch's scores, weights, charts, sample images. This tab reads the folder of our run (YOLO26n-seg, buildings and sports fields, 100 epochs) file by file. Most files are written during training; some at the very end, in a last validation with <code>best.pt</code>.`,
    },
    {
      tr: R`En çok yanlış okunan dosya [[confusion-matrix|karışıklık matrisidir]]. 6–11. kısımlar onu adım adım kurar. Önce YOLO'dan bağımsız üç kısım: iki sınıflı bir örnekle dört temel kavram (TP, FP, FN, TN), üç sınıflı bir kedi-köpek-fare sınıflayıcısıyla sınıf başına okuma ve ortalamalar, sıralı listelerle AP ve mAP'in elle hesabı. Sonra tespitteki farkı gösteren küçük bir oyuncak kare, bizim modelin gerçek matrisi ve arazi örtüsü modelinin piksel matrisi. 1–5. kısımlar eğitim sırasında ve sonrasındaki öteki dosyalardır; 12. kısım eşik seçimiyle bitirir.`,
      en: R`The file read wrong most often is the [[confusion-matrix|confusion matrix]]. Sections 6–11 build it step by step. First three sections that need no YOLO: the four basic ideas (TP, FP, FN, TN) on a two-class example, reading per class and averaging on a three-class cat-dog-mouse classifier, and AP and mAP worked out by hand on ranked lists. Then a small toy tile that shows what is different in detection, our model's real matrix, and the land-cover model's pixel matrix. Sections 1–5 are the other files from during and after training; section 12 ends with choosing a threshold.`,
    },
  ],
};

const SECTIONS: Section[] = [
  {
    id: "yr-folder",
    h: { tr: "Eğitim klasörü: hangi dosya, ne zaman?", en: "The training folder: which file, when?" },
    p: [
      {
        tr: R`Eğitim <code>runs/yolo/seg-pretrained/</code> klasörüne yazdı (YOLO_SpaceNet.py; Kodlama sekmesi). Dosyaların ne zaman yazıldığını bilmek, ne söylediklerini anlamanın yarısıdır: <code>labels.jpg</code> eğitim başlamadan veriyi özetler; <code>results.csv</code> her epoch'ta bir satır büyür; <code>confusion_matrix.png</code> ve eğriler ise yalnızca en sonda, en iyi ağırlıklarla çizilir.`,
        en: R`Training wrote into <code>runs/yolo/seg-pretrained/</code> (YOLO_SpaceNet.py; Coding tab). Knowing when a file is written is half of knowing what it says: <code>labels.jpg</code> sums up the data before training starts; <code>results.csv</code> grows by a row every epoch; <code>confusion_matrix.png</code> and the curves are drawn only at the very end, with the best weights.`,
      },
      {
        tr: R`<code>weights/</code> klasöründe iki önemli dosya var. <code>last.pt</code> son epoch'un ağırlıklarıdır; eğitim yarıda kesilirse <code>resume=True</code> ile buradan sürer. <code>best.pt</code> doğrulamada en iyi epoch'un ağırlıklarıdır; kullanacağınız ve dışa aktaracağınız model budur. "En iyi"nin ölçüsü 3. kısımda.`,
        en: R`The <code>weights/</code> folder holds two important files. <code>last.pt</code> is the last epoch's weights; an interrupted run resumes from it with <code>resume=True</code>. <code>best.pt</code> is the weights of the best epoch on validation; it is the model you use and export. What "best" means is in section 3.`,
      },
    ],
    model: {
      tr: R`Bizim klasörde ayrıca <code>epoch0.pt</code> … <code>epoch74.pt</code> var: Eğitim sekmesinin "aynı kare, epoch epoch" görünümü için ara ağırlıkları sakladık. Bunlar her biri 32 MB (optimizer durumu da içinde); <code>best.pt</code> ve <code>last.pt</code> ise eğitim sonunda optimizer'dan arındırılıp 6,6 MB'a iner.`,
      en: R`Our folder also has <code>epoch0.pt</code> … <code>epoch74.pt</code>: we kept snapshots for the Training tab's "same tile, epoch by epoch" view. Each is 32 MB (the optimizer state is inside); <code>best.pt</code> and <code>last.pt</code> are stripped of the optimizer at the end and shrink to 6.6 MB.`,
    },
    demo: folderDemo,
  },
  {
    id: "yr-console",
    h: { tr: "Eğitim sürerken: konsoldaki satırlar", en: "While training runs: the console lines" },
    p: [
      {
        tr: R`Eğitim sürerken ekranda her epoch için iki satır görünür. İlki eğitim satırıdır: o epoch'taki yığınların ortalama [[loss|kayıpları]] (kutu, maske, sınıf…), ekran kartı belleği ve ilerleme. İkincisi doğrulama satırıdır: epoch bitince 131 doğrulama karesinde ölçülen [[precision|kesinlik]], [[recall|duyarlılık]] ve [[average-precision|mAP]], kutular (Box) ve maskeler (Mask) için ayrı ayrı.`,
        en: R`While training runs, every epoch shows two lines. The first is the training line: the mean [[loss|losses]] over the epoch's batches (box, mask, class…), the graphics card's memory and progress. The second is the validation line: [[precision|precision]], [[recall|recall]] and [[average-precision|mAP]] measured on the 131 validation tiles when the epoch ends, for boxes (Box) and masks (Mask) separately.`,
      },
      {
        tr: R`Kayıpların mutlak değeri önemli değil; önemli olan zamanla nasıl değiştiği. Kayıpları kendi aralarında da karşılaştırmayın: her birinin ölçeği ve ağırlığı farklıdır. Doğrulama satırındaki P ve R tek bir güven eşiğinde ölçülür; o eşiği Ultralytics seçer (12. kısım).`,
        en: R`The losses' absolute values do not matter; how they change over time does. Do not compare the losses with each other either: each has its own scale and weight. The P and R of the validation line are measured at one confidence threshold, which Ultralytics picks (section 12).`,
      },
      {
        tr: R`Başlangıç satırları da okunmalı: "528 images, 53 backgrounds, 0 corrupt" eğitim karelerinin 53'ünde hiç nesne olmadığını (yalnızca arka plan: modele "burada bir şey yok" demeyi öğretir) ve bozuk etiket olmadığını söyler. <code>args.yaml</code> bütün ayarları saklar; eğitimi tekrarlamak için gereken tek dosyadır.`,
        en: R`Read the first lines too: "528 images, 53 backgrounds, 0 corrupt" says 53 training tiles have no objects (background only: they teach the model to say "nothing here") and no label is broken. <code>args.yaml</code> keeps every setting; it is the one file needed to repeat the run.`,
      },
    ],
    model: {
      tr: R`Bizim eğitimin gerçek satırları. 1. epoch'ta kesinlik 0,76 ama duyarlılık 0,25: model çok az şey bulur, bulduğunun çoğu doğrudur. 100. epoch'ta kesinlik 0,66'ya iner, duyarlılık 0,46'ya çıkar; mAP50-95 0,11'den 0,28'e yükselir.`,
      en: R`The real lines of our run. At epoch 1 precision is 0.76 but recall 0.25: the model finds very little, and most of it is right. By epoch 100 precision drops to 0.66, recall rises to 0.46, and mAP50-95 climbs from 0.11 to 0.28.`,
    },
    demo: consoleDemo,
  },
  {
    id: "yr-results",
    h: { tr: "results.csv: eğitim nasıl gitti, hangi ağırlık?", en: "results.csv: how did training go, which weights?" },
    p: [
      {
        tr: R`<code>results.csv</code>, konsoldaki satırların bir tablo hâlidir: her epoch bir satır, her ölçüm bir sütun. <code>results.png</code> aynı sütunları küçük grafikler olarak çizer (noktalı çizgi yumuşatılmış hâlidir). Okumanın üç kuralı: eğitim ve doğrulama kayıpları <b>birlikte</b> iniyorsa model öğreniyordur; eğitim kaybı inerken doğrulama kaybı yükselmeye başlarsa model eğitim karelerini [[overfitting|ezberliyordur]]; eğriler sonda hâlâ iniyorsa daha uzun eğitim işe yarayabilir.`,
        en: R`<code>results.csv</code> is the console lines as a table: one row per epoch, one column per score. <code>results.png</code> draws the same columns as small charts (the dotted line is a smoothed copy). Three rules for reading it: if the training and validation losses fall <b>together</b>, the model is learning; if the training loss falls while the validation loss starts to rise, the model is [[overfitting|memorising]] the training tiles; if the curves are still falling at the end, training longer may help.`,
      },
      {
        tr: R`Ultralytics her epoch sonunda bir "uygunluk" (fitness) puanı hesaplar ve en yüksek puanlı epoch'un ağırlıklarını <code>best.pt</code> olarak saklar. Bu sürümde segmentasyon için uygunluk:`,
        en: R`At the end of every epoch Ultralytics computes a "fitness" score and saves the weights of the highest-scoring epoch as <code>best.pt</code>. For segmentation in this version the fitness is:`,
      },
      {
        tex: {
          tr: R`\text{uygunluk} = \text{mAP}_{50\text{-}95}^{\text{kutu}} + \text{mAP}_{50\text{-}95}^{\text{maske}}`,
          en: R`\text{fitness} = \text{mAP}_{50\text{-}95}^{\text{box}} + \text{mAP}_{50\text{-}95}^{\text{mask}}`,
        },
      },
      {
        tr: R`Yani "en iyi", en düşük kayıp değil, sınırları en iyi tutturan modeldir. Erken durdurma ([[early-stopping|early stopping]]) da aynı puana bakar: <code>patience</code> epoch boyunca iyileşme olmazsa eğitim durur.`,
        en: R`So "best" is not the lowest loss but the model that fits the outlines best. [[early-stopping|Early stopping]] watches the same score: if it does not improve for <code>patience</code> epochs, training stops.`,
      },
    ],
    model: {
      tr: R`Bizim eğitimde en yüksek uygunluk 88. epoch'ta (0,542); 100. epoch'unki 0,514. <code>best.pt</code> 88. epoch'tur. Kayıplar 100. epoch'ta hâlâ iniyor ve doğrulama kaybı yükselmiyor: ezberleme yok, daha uzun eğitim biraz daha kazandırabilir.`,
      en: R`In our run the highest fitness is at epoch 88 (0.542); epoch 100 has 0.514. <code>best.pt</code> is epoch 88. The losses are still falling at epoch 100 and the validation loss is not rising: no memorising, and training longer could gain a little more.`,
    },
    demo: resultsDemo,
  },
  {
    id: "yr-labels",
    h: { tr: "labels.jpg: verinin özeti", en: "labels.jpg: a summary of the data" },
    p: [
      {
        tr: R`<code>labels.jpg</code> eğitim başlamadan çizilir ve modelin değil, <b>verinin</b> resmidir: her sınıftan kaç nesne var, nesneler karenin neresinde ve ne büyüklükte. Model ancak verinin öğrettiğini öğrenebilir; sorunların çoğu burada önceden görülür.`,
        en: R`<code>labels.jpg</code> is drawn before training starts and is a picture of the <b>data</b>, not the model: how many objects of each class, where they sit in the tile and how big they are. A model can only learn what the data teaches; most problems can be seen here in advance.`,
      },
      {
        tr: R`İlk bakılacak şey [[class-imbalance|sınıf dengesizliği]]. Bir sınıfın örneği çok azsa model o sınıfı az görür ve kötü öğrenir. Çareler: daha çok örnek toplamak (en iyisi), azınlık sınıfını içeren kareleri çoğaltmak, ya da kayıpta azınlık sınıfına ağırlık vermek (Ultralytics'te <code>cls_pw</code> ayarı; varsayılanı 0, yani kapalı). İkincisi boyutlar: çok küçük nesneler (birkaç piksel) için daha büyük giriş boyutu (<code>imgsz</code>) gerekebilir.`,
        en: R`The first thing to look at is [[class-imbalance|class imbalance]]. If a class has very few examples, the model sees it rarely and learns it badly. Remedies: collect more examples (the best), repeat the tiles that contain the rare class, or weight the rare class in the loss (Ultralytics' <code>cls_pw</code> setting; 0, off, by default). Second, the sizes: very small objects (a few pixels) may need a larger input size (<code>imgsz</code>).`,
      },
    ],
    model: {
      tr: R`Eğitimde 10.974 bina ve 529 spor sahası var: yaklaşık 21'e 1. Spor sahasının zayıf AP'sinin (0,30) bir nedeni bu; öteki nedeni etiketlerin OpenStreetMap'ten gelmesi (Teori, 7. kısım).`,
      en: R`Training has 10,974 buildings and 529 sports fields: about 21 to 1. That is one reason for the sports field's weak AP (0.30); the other is that its labels come from OpenStreetMap (Theory, section 7).`,
    },
    demo: labelsDemo,
  },
  {
    id: "yr-batches",
    h: { tr: "Yığın resimleri: model ne gördü, ne buldu?", en: "Batch pictures: what did the model see and find?" },
    p: [
      {
        tr: R`<code>train_batch0.jpg</code>, <code>1</code> ve <code>2</code> eğitimin ilk üç yığınıdır, [[data-augmentation|veri artırma]] uygulanmış hâlde: modelin gerçekte gördüğü budur. Etiket hatalarını en kolay burada yakalarsınız: kaymış çerçeveler, yanlış sınıflar, unutulmuş nesneler. Eğitimi başlatınca ilk iş bu resimlere bakmaktır; bir saat sonra bozuk bir eğitimle karşılaşmaktan iyidir.`,
        en: R`<code>train_batch0.jpg</code>, <code>1</code> and <code>2</code> are the first three training batches, with [[data-augmentation|augmentation]] applied: this is what the model really sees. Label mistakes are easiest to catch here: shifted outlines, wrong classes, forgotten objects. Looking at these pictures is the first thing to do after starting a run; better than finding a broken run an hour later.`,
      },
      {
        tr: R`<code>val_batch0_labels.jpg</code> ile <code>val_batch0_pred.jpg</code> aynı doğrulama karelerinin doğru etiketleri ve tahminleridir. Sayılar bir modelin <b>ne kadar</b> yanıldığını söyler; bu iki resim <b>nasıl</b> yanıldığını gösterir: gölgede kalan binalar mı kaçıyor, bitişik evler tek kutu mu oluyor, çatı gibi görünen bir otopark mı bina sanılıyor?`,
        en: R`<code>val_batch0_labels.jpg</code> and <code>val_batch0_pred.jpg</code> are the true labels and the predictions on the same validation tiles. The numbers say <b>how much</b> a model errs; these two pictures show <b>how</b>: are buildings in shadow missed, do adjacent houses become one box, is a car park that looks like a roof taken for a building?`,
      },
    ],
    model: {
      tr: R`Hartum karelerinde (bu doğrulama yığını) bitişik nizam evler sık sık tek büyük maske oluyor, bazı sahalar hiç bulunmuyor. Mozaiksiz yığın, son 10 epoch'tan (close_mosaic = 10).`,
      en: R`On the Khartoum tiles (this validation batch) adjacent houses often become one large mask, and some fields are not found at all. The batch without mosaic is from the last 10 epochs (close_mosaic = 10).`,
    },
    demo: batchesDemo,
  },
  {
    id: "yr-cm-intro",
    h: { tr: "Karışıklık matrisi I: dört kutu", en: "Confusion matrix I: four boxes" },
    p: [
      {
        tr: R`Bir depremden sonra, bir mahalledeki 100 binanın uydu görüntüsüne bakıp hangilerinin ağır hasarlı olduğunu tahmin eden bir model düşünün. Model her binaya bir hasar skoru verir; skoru eşiği geçen binalar "hasarlı" diye işaretlenir ve oraya bir inceleme ekibi gider. Her bina dört durumdan birindedir:`,
        en: R`Picture a model that looks at satellite images of 100 buildings in a neighbourhood after an earthquake and guesses which are badly damaged. It gives every building a damage score; buildings above a threshold are flagged "damaged", and an inspection team goes there. Every building ends up in one of four cases:`,
      },
      {
        tr: R`[[true-positive|<b>TP</b> (doğru pozitif)]]: hasarlı, işaretlendi. [[false-positive|<b>FP</b> (yanlış pozitif)]]: sağlam ama işaretlendi; ekip boşuna gider ("yanlış alarm"). [[false-negative|<b>FN</b> (yanlış negatif)]]: hasarlı ama işaretlenmedi; gözden kaçtı. [[true-negative|<b>TN</b> (doğru negatif)]]: sağlam, işaretlenmedi. "Pozitif/negatif" modelin <i>dediğidir</i>, "doğru/yanlış" bunun gerçekle uyup uymadığı. Bu dört sayının tablosu [[confusion-matrix|karışıklık matrisidir]]: satırlar tahmin, sütunlar gerçek (Ultralytics'in düzeni; bazı kitaplar tersini kullanır, önce eksenlere bakın).`,
        en: R`[[true-positive|<b>TP</b> (true positive)]]: damaged, flagged. [[false-positive|<b>FP</b> (false positive)]]: sound but flagged; the team goes for nothing (a "false alarm"). [[false-negative|<b>FN</b> (false negative)]]: damaged but not flagged; missed. [[true-negative|<b>TN</b> (true negative)]]: sound, not flagged. "Positive/negative" is what the model <i>says</i>; "true/false" is whether that matches reality. The table of these four numbers is the [[confusion-matrix|confusion matrix]]: rows are the prediction, columns the truth (Ultralytics' layout; some books swap them, so check the axes first).`,
      },
      {
        tex: {
          tr: R`\text{doğruluk} = \frac{TP+TN}{\text{hepsi}}`,
          en: R`\text{accuracy} = \frac{TP+TN}{\text{all}}`,
        },
      },
      {
        tex: {
          tr: R`P = \frac{TP}{TP+FP},\qquad R = \frac{TP}{TP+FN}`,
          en: R`P = \frac{TP}{TP+FP},\qquad R = \frac{TP}{TP+FN}`,
        },
      },
      {
        tex: {
          tr: R`\text{özgüllük} = \frac{TN}{TN+FP},\qquad F_1 = \frac{2PR}{P+R}`,
          en: R`\text{specificity} = \frac{TN}{TN+FP},\qquad F_1 = \frac{2PR}{P+R}`,
        },
      },
      {
        tr: R`[[accuracy|Doğruluk]] dengesiz veride yanıltır: binaların yalnızca 10'u hasarlıysa "hepsi sağlam" diyen işe yaramaz bir model de %90 doğru çıkar. Bu yüzden [[precision|kesinliğe]] ve [[recall|duyarlılığa]] ayrı ayrı bakılır. [[f1-score|F1]] ikisinin harmonik ortalamasıdır: biri küçükse F1 de küçük kalır. Hangisinin daha önemli olduğu işin bedeline bağlıdır: burada bir FN, içinde insan olabilecek hasarlı bir binanın gözden kaçması demek; bir FP ise bir ekibin boşa giden bir saati. Bu iş için yüksek duyarlılık, yani düşük eşik seçilir.`,
        en: R`[[accuracy|Accuracy]] misleads on imbalanced data: if only 10 buildings are damaged, a useless model that says "all sound" is still 90% accurate. That is why [[precision|precision]] and [[recall|recall]] are read separately. [[f1-score|F1]] is their harmonic mean: if either is small, F1 stays small. Which matters more depends on the cost: here an FN means a damaged building, maybe with people inside, is missed; an FP means a team loses an hour. For this job one picks a high recall, i.e. a low threshold.`,
      },
      {
        tr: R`Aynı dört kavram, "evet/hayır" diyen her modelde vardır; değişen yalnızca hataların bedelidir:`,
        en: R`The same four ideas exist in every model that says "yes/no"; only the cost of the mistakes changes:`,
      },
      {
        block: {
          tr: R`<table class="yo-table yo-compare"><thead><tr><th scope="col">İş (pozitif)</th><th scope="col">FP: yanlış alarm</th><th scope="col">FN: kaçırılan</th><th scope="col">Daha pahalı</th></tr></thead><tbody>
<tr><th scope="row">Hasar taraması (ağır hasarlı)</th><td>ekip boşuna gider</td><td>hasarlı bina gözden kaçar</td><td>FN</td></tr>
<tr><th scope="row">Tıbbi test (hasta)</th><td>sağlıklı kişiye ek tetkik</td><td>hastalık atlanır</td><td>çoğunlukla FN</td></tr>
<tr><th scope="row">Spam filtresi (spam)</th><td>önemli e-posta çöpe gider</td><td>spam gelen kutusuna düşer</td><td>FP</td></tr>
<tr><th scope="row">Beton yüzeyde çatlak (çatlak var)</th><td>gereksiz yerinde inceleme</td><td>çatlak kaçar</td><td>FN</td></tr>
</tbody></table>`,
          en: R`<table class="yo-table yo-compare"><thead><tr><th scope="col">Job (positive)</th><th scope="col">FP: false alarm</th><th scope="col">FN: missed</th><th scope="col">Costs more</th></tr></thead><tbody>
<tr><th scope="row">Damage survey (badly damaged)</th><td>a team goes for nothing</td><td>a damaged building is missed</td><td>FN</td></tr>
<tr><th scope="row">Medical test (ill)</th><td>a healthy person gets more tests</td><td>the illness is missed</td><td>usually FN</td></tr>
<tr><th scope="row">Spam filter (spam)</th><td>an important e-mail goes to junk</td><td>spam reaches the inbox</td><td>FP</td></tr>
<tr><th scope="row">Crack in a concrete surface (crack)</th><td>a needless site inspection</td><td>a crack is missed</td><td>FN</td></tr>
</tbody></table>`,
        },
      },
    ],
    model: {
      tr: R`Bu örneğin sayıları uydurmadır. Matrisin mantığı YOLO'da aynıdır; tek fark, 9. kısımda göreceğimiz gibi, tespitte TN'nin sayılamamasıdır.`,
      en: R`The numbers of this example are made up. The matrix works the same way in YOLO; the one difference, as section 9 shows, is that detection cannot count TN.`,
    },
    demo: surveyDemo,
    math: "ym-ap",
  },
  {
    id: "yr-cm-multi",
    h: { tr: "Karışıklık matrisi II: üç sınıf (kedi, köpek, fare)", en: "Confusion matrix II: three classes (cat, dog, mouse)" },
    p: [
      {
        tr: R`YOLO'dan bağımsız bir örnek: 30 fotoğrafı kedi, köpek ya da fare diye sınıflayan bir model. Her fotoğrafın tek bir gerçek sınıfı ve tek bir tahmini vardır; matris 3 × 3'tür. Köşegen doğru tahminlerdir. Bir <b>satır</b>, modelin o sınıf dediği her şey; bir <b>sütun</b>, gerçekte o sınıf olan her şeydir.`,
        en: R`An example without YOLO: a model that sorts 30 photos into cat, dog or mouse. Every photo has one true class and one prediction; the matrix is 3 × 3. The diagonal holds the right predictions. A <b>row</b> is everything the model called that class; a <b>column</b> is everything that really is that class.`,
      },
      {
        tr: R`Birden çok sınıfta TP, FP, FN ve TN <b>sınıf başına</b> okunur: bir sınıf seçilir ve soru "bu sınıf mı, değil mi?" olur (bire karşı hepsi). Matris o sınıf için 2 × 2'ye iner. Kedi için: TP köşegendeki 7; FP kedi satırının geri kalanı, 2 + 1 = 3 ("kedi" dendi ama köpek ya da fare); FN kedi sütununun geri kalanı, 2 + 1 = 3 (gerçek kedi, başka bir şey dendi); TN öteki dört hücre, 9 + 0 + 1 + 7 = 17. Dördünün toplamı 30'dur.`,
        en: R`With several classes, TP, FP, FN and TN are read <b>per class</b>: pick a class and ask "this class or not?" (one against the rest). For that class the matrix shrinks to 2 × 2. For cat: TP is the 7 on the diagonal; FP is the rest of the cat row, 2 + 1 = 3 (called "cat" but a dog or a mouse); FN is the rest of the cat column, 2 + 1 = 3 (really a cat, called something else); TN is the other four cells, 9 + 0 + 1 + 7 = 17. The four add up to 30.`,
      },
      {
        tex: {
          tr: R`P_{\text{kedi}} = \frac{7}{7+3} = 0{,}70,\qquad R_{\text{kedi}} = \frac{7}{7+3} = 0{,}70`,
          en: R`P_{\text{cat}} = \frac{7}{7+3} = 0.70,\qquad R_{\text{cat}} = \frac{7}{7+3} = 0.70`,
        },
      },
      {
        tr: R`Tek bir sayı istenince sınıfların ölçüleri birleştirilir. [[macro-average|Makro ortalama]]: her sınıfın ölçüsü ayrı hesaplanır, sonra eşit ağırlıkla ortalanır; az örnekli sınıf da bir oy taşır. Ağırlıklı ortalama: her sınıf örnek sayısıyla ağırlıklanır. Mikro ortalama: bütün sınıfların TP, FP ve FN'si toplanıp bir kez hesaplanır; her fotoğrafın tek etiketi olduğunda doğruluğa eşittir. Dengesiz veride aralarındaki fark büyür: "Fare az" senaryosunu seçin. YOLO'nun mAP'i bir makro ortalamadır: her sınıfın AP'si eşit ağırlık taşır.`,
        en: R`When one number is wanted, the classes' scores are combined. The [[macro-average|macro average]] computes each class's score separately, then averages them with equal weight: a class with few examples still gets one vote. The weighted average weights each class by its number of examples. The micro average adds up every class's TP, FP and FN and computes once; when every photo has one label it equals the accuracy. On imbalanced data they part: choose the "Few mice" scenario. YOLO's mAP is a macro average: every class's AP carries the same weight.`,
      },
    ],
    model: {
      tr: R`Bu hesap her sınıflayıcıda aynıdır: 3. bölümdeki CNN'in on rakamı da bir 10 × 10 matris verir ve aynı biçimde, sınıf sınıf okunur.`,
      en: R`The arithmetic is the same for every classifier: the CNN of chapter 3, with its ten digits, gives a 10 × 10 matrix read the same way, class by class.`,
    },
    demo: petsDemo,
  },
  {
    id: "yr-ap",
    h: { tr: "AP ve mAP elle: sıralı bir liste", en: "AP and mAP by hand: a ranked list" },
    p: [
      {
        tr: R`Kesinlik ve duyarlılık tek bir eşiğe bağlıdır. [[average-precision|AP]] eşik seçmeden ölçer. Model her fotoğrafa "kedi" skoru verir; fotoğrafları skora göre sıralayıp listeden aşağı inelim. Her adım, eşiği bir sonraki fotoğrafın skoruna indirmek demektir. Her satırda o ana kadar bulunan kedilerle P ve R hesaplanır: \(P = \text{bulunan}/\text{sıra}\), \(R = \text{bulunan}/\text{bütün kediler}\). Bu (R, P) noktaları PR eğrisini çizer.`,
        en: R`Precision and recall depend on one threshold. [[average-precision|AP]] measures without picking one. The model gives every photo a "cat" score; sort the photos by score and walk down the list. Each step lowers the threshold to the next photo's score. On every row P and R are computed with the cats found so far: \(P = \text{found}/\text{rank}\), \(R = \text{found}/\text{all cats}\). These (R, P) points draw the PR curve.`,
      },
      {
        tr: R`AP, PR eğrisinin altındaki alandır. Elle hesaplamanın kısa yolu: her doğru öğenin (✓) sırasındaki kesinliğin ortalaması; listeye hiç girmeyen pozitifler 0 sayılır.`,
        en: R`AP is the area under the PR curve. The short way by hand: the mean of the precision at the rank of every right item (✓); positives that never made the list count as 0.`,
      },
      {
        tex: {
          tr: R`\text{AP} = \frac{1}{N}\sum_{k:\ ✓} P(k) = \frac{1 + 1 + 0{,}75 + 0{,}571}{4} \approx 0{,}83`,
          en: R`\text{AP} = \frac{1}{N}\sum_{k:\ ✓} P(k) = \frac{1 + 1 + 0.75 + 0.571}{4} \approx 0.83`,
        },
      },
      {
        tr: R`Burada \(N\) kedi fotoğraflarının sayısıdır (4). Ultralytics ve COCO eğrinin önce "zarfını" alır (her duyarlılıkta, o noktadan sonraki en yüksek kesinlik), sonra 0, 0,01, …, 1 olmak üzere 101 duyarlılık düzeyinde okur; sonuç çok yakındır (0,83). [[average-precision|mAP]] sınıfların AP'lerinin ortalamasıdır: \((0{,}83 + 0{,}77 + 0{,}42)/3 \approx 0{,}67\).`,
        en: R`Here \(N\) is the number of cat photos (4). Ultralytics and COCO first take the curve's "envelope" (at every recall, the highest precision from there on), then read it at 101 recall levels 0, 0.01, …, 1; the result is very close (0.83). [[average-precision|mAP]] is the mean of the classes' APs: \((0.83 + 0.77 + 0.42)/3 \approx 0.67\).`,
      },
      {
        tr: R`Tespitte tek fark "✓"nin anlamıdır: bir kutu, aynı sınıftan bir gerçek nesneyle [[iou|IoU]]'su eşiği geçiyorsa doğrudur. mAP50 bu eşiği 0,5 alır. mAP50-95 ise AP'yi 0,50, 0,55, …, 0,95 eşikleriyle on kez hesaplayıp ortalar: aynı kutu 0,5'te ✓, 0,8'de ✗ olabilir.`,
        en: R`In detection the one difference is what "✓" means: a box is right if its [[iou|IoU]] with a true object of its class passes a threshold. mAP50 takes that threshold as 0.5. mAP50-95 computes AP ten times, at 0.50, 0.55, …, 0.95, and averages: the same box can be ✓ at 0.5 and ✗ at 0.8.`,
      },
    ],
    model: {
      tr: R`Fare listesinde dört fareden biri listeye hiç girmiyor: duyarlılık en fazla 0,75'e çıkar ve o fare AP'ye 0 olarak girer. Bizim YOLO modelinin spor sahası AP'sinin düşük (0,30) olmasının bir nedeni de budur: sahaların çoğu hiçbir eşikte bulunmuyor.`,
      en: R`In the mouse list one of the four mice never makes it into the list: recall cannot pass 0.75, and that mouse enters the AP as 0. It is also one reason our YOLO model's sports field AP is low (0.30): most fields are not found at any threshold.`,
    },
    demo: apDemo,
    math: "ym-ap",
  },
  {
    id: "yr-cm-toy",
    h: { tr: "Karışıklık matrisi III: tespitte matris", en: "Confusion matrix III: the matrix in detection" },
    p: [
      {
        tr: R`Tespitte "pozitif", bir kutudur. Bir kutunun doğru sayılması için bir gerçek nesneyle [[iou|IoU]]'sunun 0,5'i geçmesi gerekir. Ultralytics eşleştirmeyi şöyle yapar: eşiği geçen kutular skora göre sıralanır; her kutu, henüz alınmamış gerçek nesneler arasından en çok örtüştüğünü alır (IoU > 0,5 ise). Eşleştirme <b>sınıfa bakmaz</b>; sınıf ancak sonra, matrisin hangi hücresine yazılacağını belirler.`,
        en: R`In detection a "positive" is a box. For a box to count as correct, its [[iou|IoU]] with a true object must pass 0.5. Ultralytics matches like this: the boxes above the threshold are sorted by score; each box takes the free true object it overlaps most (if the IoU is above 0.5). The matching <b>ignores the class</b>; the class only decides afterwards which cell of the matrix the pair goes into.`,
      },
      {
        tr: R`Matrise bir "arka plan" satırı ve sütunu eklenir. Arka plan <b>sütunu</b>, hiçbir gerçek nesneyle eşleşmeyen kutulardır (FP): boş zemine konmuş kutular, aynı nesneye ikinci kez konmuş kutular, nesneyi çok kötü çerçeveleyen kutular. Arka plan <b>satırı</b>, hiçbir kutunun bulamadığı gerçek nesnelerdir (FN). Doğru yeri bulup yanlış sınıf veren bir kutu köşegenin dışına düşer: kendi sınıfı için FP, gerçek sınıf için FN sayılır.`,
        en: R`A "background" row and column are added. The background <b>column</b> holds the boxes that match no true object (FP): boxes on empty ground, second boxes on the same object, boxes that frame the object very badly. The background <b>row</b> holds the true objects no box found (FN). A box that finds the right place with the wrong class lands off the diagonal: an FP for its own class and an FN for the true class.`,
      },
      {
        tr: R`Sağ alt köşe (arka plan × arka plan) hep boştur. TN, "orada bir şey yoktu ve model bir şey demedi" demektir; ama bir karede kaç tane "boş yer" vardır? Sayılamaz. Bu yüzden tespitte TN, doğruluk ve özgüllük yoktur; kesinlik ve duyarlılık TN'ye ihtiyaç duymaz.`,
        en: R`The bottom-right corner (background × background) is always empty. A TN would be "nothing was there and the model said nothing"; but how many "empty places" does a tile have? They cannot be counted. So detection has no TN, no accuracy and no specificity; precision and recall do not need TN.`,
      },
    ],
    model: {
      tr: R`Oyuncak karede her kutu türünden bir örnek var: iyi bulunmuş binalar, aynı binaya ikinci kutu (p3), kaymış kutu (p6), yanlış sınıf (p10), boş zemine kutu (p11, p12) ve skoru düşük olduğu için eşiğe göre gelip giden kutular (p5, p8).`,
      en: R`The toy tile has one of every kind of box: buildings found well, a second box on the same building (p3), a shifted box (p6), the wrong class (p10), boxes on empty ground (p11, p12) and boxes that come and go with the threshold because their scores are low (p5, p8).`,
    },
    demo: toyDemo,
  },
  {
    id: "yr-cm-real",
    h: { tr: "Karışıklık matrisi IV: bizim modelin matrisi", en: "Confusion matrix IV: our model's matrix" },
    p: [
      {
        tr: R`<code>confusion_matrix.png</code> eğitimin sonunda, <code>best.pt</code> ile 131 doğrulama karesinde çizilir. Burada en sık yapılan okuma hatası şudur: Ultralytics doğrulamayı [[confidence-score|güven eşiği]] <b>0,001</b> ile yapar (mAP'i bütün PR eğrisinden hesaplamak için en düşük skorlu tespitlere de ihtiyaç vardır) ve matris de bu eşikle doldurulur. Skoru 0,001'i geçen her kutu sayıldığı için arka plan sütunu binlerce FP ile dolar. Bu, modelin kullanıldığı eşikteki davranışı değildir.`,
        en: R`<code>confusion_matrix.png</code> is drawn at the end of training, with <code>best.pt</code> on the 131 validation tiles. The most common mistake in reading it: Ultralytics validates at a [[confidence-score|confidence threshold]] of <b>0.001</b> (computing mAP from the whole PR curve needs even the lowest-scoring detections), and the matrix is filled at that threshold too. Since every box scoring over 0.001 counts, the background column fills up with thousands of FPs. That is not how the model behaves at the threshold it is used with.`,
      },
      {
        tr: R`Kaydırıcıyla eşiği değiştirin: matris her eşikte, Ultralytics'in kendi <code>ConfusionMatrix</code> koduyla yeniden hesaplandı. Kendi eğitiminizde aynısını görmek için doğrulamayı kullanacağınız eşikle tekrarlayın: <code>yolo val model=best.pt conf=0.25</code> (mAP'i ise varsayılan eşikle okuyun). <code>confusion_matrix_normalized.png</code> her <b>sütunu</b> 1'e böler: köşegen, her sınıfın duyarlılığıdır.`,
        en: R`Move the threshold with the slider: the matrix was rebuilt at every threshold with Ultralytics' own <code>ConfusionMatrix</code> code. To see the same for your own run, repeat the validation at the threshold you will use: <code>yolo val model=best.pt conf=0.25</code> (and read the mAP at the default threshold). <code>confusion_matrix_normalized.png</code> divides every <b>column</b> so it adds up to 1: the diagonal is each class's recall.`,
      },
    ],
    model: {
      tr: R`Eşik 0,25'te: 2329 binadan 1520'si bulundu, 807'si kaçtı; 687 kutu boşa (FP). 172 spor sahasından 57'si bulundu. Sınıf karışması çok az (8 saha "bina" dendi): modelin asıl sorunu sınıfları karıştırmak değil, nesneyi bulmak. Eğitim sonunda yazılan resimdeki sayılar burada gördüklerinizden birkaç tane farklı olabilir (2070'e karşı 2069 gibi): eğitimdeki doğrulama ekran kartında yarım duyarlıkla (fp16) çalışır.`,
      en: R`At threshold 0.25: 1520 of 2329 buildings found, 807 missed; 687 boxes for nothing (FP). 57 of 172 sports fields found. Classes are rarely confused (8 fields called "building"): the model's real problem is finding objects, not telling them apart. The picture written at the end of training may differ from what you see here by a few (2070 against 2069, say): the validation during training runs in half precision (fp16) on the graphics card.`,
    },
    demo: realMatrixDemo,
  },
  {
    id: "yr-cm-sem",
    h: { tr: "Karışıklık matrisi V: arazi örtüsü, piksel piksel", en: "Confusion matrix V: land cover, pixel by pixel" },
    p: [
      {
        tr: R`Anlamsal segmentasyon modelinde matris nesneleri değil <b>pikselleri</b> sayar: her piksel için "gerçekte ne" ve "ne dendi". Burada arka plan da sıradan bir sınıftır, bu yüzden boş hücre yoktur. Altı sınıflı bir matriste TP, FP, FN ve TN sınıf başına okunur: bir sınıf için TP köşegendeki hücre, FP satırının geri kalanı, FN sütununun geri kalanı, TN ise öteki her şeydir. Tespitte sayılamayan TN, burada piksellerle sayılabilir.`,
        en: R`In the semantic segmentation model the matrix counts <b>pixels</b>, not objects: for every pixel, "what it really is" and "what was said". Here background is an ordinary class, so there is no empty cell. In a six-class matrix TP, FP, FN and TN are read per class: for one class, TP is its diagonal cell, FP the rest of its row, FN the rest of its column, and TN everything else. The TN that detection cannot count can be counted here, in pixels.`,
      },
      {
        tr: R`Köşegen dışı büyük hücreler en öğretici olanlardır, çünkü modelin hangi sınıfları birbirine benzettiğini gösterirler. Sınıf başına IoU da matristen gelir: \(\text{IoU} = TP/(TP+FP+FN)\). Ultralytics bunu <code>iou_bar_chart.png</code> olarak çizer; ortalaması mIoU'dur.`,
        en: R`The large off-diagonal cells are the most instructive, because they show which classes the model finds alike. The IoU per class comes from the matrix too: \(\text{IoU} = TP/(TP+FP+FN)\). Ultralytics draws it as <code>iou_bar_chart.png</code>; its mean is the mIoU.`,
      },
    ],
    model: {
      tr: R`Gerçek otopark piksellerinin %41'ine "arka plan", gerçek yol piksellerinin %34'üne "arka plan", gerçek su piksellerinin %23'üne "yeşil alan" dendi. Bunların bir kısmı modelin değil etiketlerin hatasıdır: OpenStreetMap'te birçok otopark eksik, yollar sabit genişlikte çizgilerden üretildi, kıyıdaki yeşil alanlar suyla iç içe. Piksellerin %71'i doğru (piksel doğruluğu), mIoU 0,46.`,
      en: R`41% of the true parking pixels were called "background", 34% of the true road pixels "background", 23% of the true water pixels "green area". Part of this is the labels' fault, not the model's: many car parks are missing in OpenStreetMap, roads were made from lines of a fixed width, and green areas on the shore mix with water. 71% of the pixels are right (pixel accuracy), mIoU 0.46.`,
    },
    demo: semMatrixDemo,
  },
  {
    id: "yr-curves",
    h: { tr: "Hangi eşik? P, R, F1 ve PR eğrileri", en: "Which threshold? The P, R, F1 and PR curves" },
    p: [
      {
        tr: R`Eğitimin sonunda dört eğri çizilir, kutular (Box…) ve maskeler (Mask…) için ayrı: <code>P_curve</code> ve <code>R_curve</code> güven eşiğine göre kesinliği ve duyarlılığı, <code>F1_curve</code> ikisinin [[f1-score|F1]]'ini, <code>PR_curve</code> ise eşiği aradan çıkarıp kesinliği duyarlılığa karşı gösterir. PR eğrisinin altındaki alan [[average-precision|AP]]'dir; eşik seçmeden modelin genel başarısını verir.`,
        en: R`Four curves are drawn at the end of training, for boxes (Box…) and masks (Mask…) separately: <code>P_curve</code> and <code>R_curve</code> show precision and recall by confidence threshold, <code>F1_curve</code> their [[f1-score|F1]], and <code>PR_curve</code> leaves out the threshold and shows precision against recall. The area under the PR curve is the [[average-precision|AP]]; it gives the model's overall quality without picking a threshold.`,
      },
      {
        tr: R`Kullanmak için ise bir eşik seçmek gerekir. Eşik yükseldikçe kesinlik artar, duyarlılık düşer; F1 eğrisi ikisinin dengede olduğu yerde tepe yapar. Ultralytics konsoldaki ve tablodaki P ile R'yi, sınıfların ortalama F1'inin en yüksek olduğu eşikte yazar. Bu iyi bir başlangıçtır, ama doğru eşik işin bedeline bağlıdır (6. kısım): bir FN'nin pahalı olduğu işte eşiği bu tepeden aşağı, bir FP'nin pahalı olduğu işte yukarı çekin.`,
        en: R`To use the model, though, one must pick a threshold. As the threshold rises precision goes up and recall down; the F1 curve peaks where the two balance. Ultralytics writes the P and R of the console and the table at the threshold where the classes' mean F1 is highest. That is a good start, but the right threshold depends on the cost of the job (section 6): where an FN is expensive, move the threshold below this peak; where an FP is expensive, above it.`,
      },
    ],
    model: {
      tr: R`Bizim modelde sınıfların ortalama F1'i kutularda eşik 0,26'da, maskelerde 0,33'te en yüksek. Tespit sekmesindeki varsayılan 0,25 bu yüzden makul. Spor sahası eğrisi binanınkinin hep altında: en iyi eşikte bile sahaların çoğu bulunmuyor.`,
      en: R`In our model the classes' mean F1 is highest at threshold 0.26 for boxes and 0.33 for masks. That is why the Detection tab's default 0.25 is reasonable. The sports field curve stays below the building curve everywhere: even at the best threshold, most fields are not found.`,
    },
    demo: curvesDemo,
    math: "ym-ap",
  },
];

const RECAP: { h: Text; list: Text[]; q: Text; a: Text } = {
  h: { tr: "Eğitim bitince: kontrol listesi", en: "When training ends: a checklist" },
  list: [
    {
      tr: R`<code>train_batch0.jpg</code>: etiketler yerinde mi? (Eğitim başlar başlamaz.)`,
      en: R`<code>train_batch0.jpg</code>: are the labels in place? (As soon as training starts.)`,
    },
    {
      tr: R`<code>labels.jpg</code>: sınıflar dengeli mi, nesneler çok mu küçük?`,
      en: R`<code>labels.jpg</code>: are the classes balanced, are the objects very small?`,
    },
    {
      tr: R`<code>results.png</code>: eğitim ve doğrulama kayıpları birlikte iniyor mu, sonda hâlâ iniyor mu?`,
      en: R`<code>results.png</code>: do the training and validation losses fall together, are they still falling at the end?`,
    },
    {
      tr: R`Kullanılacak model <code>weights/best.pt</code>; <code>args.yaml</code>'ı onunla birlikte saklayın.`,
      en: R`The model to use is <code>weights/best.pt</code>; keep <code>args.yaml</code> with it.`,
    },
    {
      tr: R`<code>confusion_matrix.png</code> eşik 0,001 ile çizildi: kullanacağınız eşikte yeniden doğrulayın.`,
      en: R`<code>confusion_matrix.png</code> was drawn at threshold 0.001: validate again at the threshold you will use.`,
    },
    {
      tr: R`F1 eğrisinden bir eşik seçin; FN mi FP mi daha pahalı, ona göre kaydırın.`,
      en: R`Pick a threshold from the F1 curve; shift it by whether an FN or an FP costs more.`,
    },
    {
      tr: R`<code>val_batch*_pred.jpg</code>: hatalara gözle bakın; sayıların söylemediğini gösterir.`,
      en: R`<code>val_batch*_pred.jpg</code>: look at the mistakes; they show what the numbers do not.`,
    },
  ],
  q: {
    tr: R`<b>Kendinizi sınayın.</b> (1) Tespitte karışıklık matrisinin sağ alt köşesi neden boştur? (2) Bir modelin doğruluğu %95; bu iyi bir model olduğunu gösterir mi? (3) <code>best.pt</code> ile <code>last.pt</code> ne zaman aynı dosyadır?`,
    en: R`<b>Test yourself.</b> (1) Why is the bottom-right corner of a detection confusion matrix empty? (2) A model is 95% accurate; does that show it is a good model? (3) When are <code>best.pt</code> and <code>last.pt</code> the same file?`,
  },
  a: {
    tr: R`(1) O köşe TN'dir: "nesne yoktu, kutu da yoktu". Bir karedeki boş yerler sayılamaz. (2) Göstermez: sınıflardan biri %95 ise "hep o sınıf" diyen model de %95 doğrudur. Kesinliğe, duyarlılığa ve matrise bakın. (3) Uygunluk en son epoch'ta en yüksekse (eğitim sonuna kadar iyileşmeye devam ettiyse).`,
    en: R`(1) That corner is TN: "no object and no box". The empty places of a tile cannot be counted. (2) No: if one class is 95% of the data, a model that always says that class is 95% accurate too. Look at precision, recall and the matrix. (3) When the fitness is highest at the last epoch (training kept improving to the end).`,
  },
};

let built = false;

function build(): void {
  const root = $("#yr-root") as HTMLElement;
  const intro = h(
    "div",
    { class: "card t-intro", id: "yr-intro" },
    h("div", { class: "t-intro-text" }, h("h2", { text: T(INTRO.h) }), ...INTRO.p.map((p) => html("p", {}, p))),
  );
  const sections = SECTIONS.map((s, k) => {
    const paragraphs: Node[] = s.p.map((p) =>
      "tex" in p ? formula(T(p.tex)) : "block" in p ? html("div", { class: "yo-table-wrap" }, p.block) : html("p", {}, p),
    );
    paragraphs.push(html("div", { class: "in-model" }, s.model));
    return sectionCard(s.id, k + 1, s.h, paragraphs, s.demo(), s.math ?? null);
  });
  const recap = h(
    "div",
    { class: "card t-intro", id: "yr-recap" },
    h(
      "div",
      { class: "t-intro-text" },
      h("h2", { text: T(RECAP.h) }),
      h("ol", { class: "yr-check" }, ...RECAP.list.map((t) => html("li", {}, t))),
      html("p", {}, RECAP.q),
      h("details", { class: "t-answers" }, h("summary", { text: T({ tr: "Cevaplar", en: "Answers" }) }), html("p", {}, RECAP.a)),
      h("a", {
        href: "#",
        class: "link",
        "data-go": "practice:yx-outputs",
        text: T({ tr: "Alıştırma sekmesinde çıktı okuma soruları →", en: "Questions on reading the outputs in the Practice tab →" }),
      }),
    ),
  );
  root.replaceChildren(h("div", { class: "yr-outputs" }, intro, ...sections, recap));
  built = true;
}

document.addEventListener("DOMContentLoaded", build);
document.addEventListener("tab-shown", (e) => {
  if ((e as CustomEvent).detail === "outputs") void loadOutputs();
});
document.addEventListener("lang-change", () => {
  if (built) build();
});
