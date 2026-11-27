/**
 * The YOLO Coding tab: a walk through the chapter's two Python files, rendered by lesson.js.
 *
 *   YOLO_SpaceNet_data.py   downloads the SpaceNet tiles and builds the two training sets
 *   YOLO_SpaceNet.py        trains, reads the curves, measures, predicts areas, exports
 *
 * The excerpts are cut from the files when the page is built (bundled as text), so they never drift from
 * the code. The terminal outputs are real runs of these files.
 */
import DATA_PY from "../../../../YOLO_SpaceNet_data.py";
import MODEL_PY from "../../../../YOLO_SpaceNet.py";
import { createLesson } from "../../shared/lesson.js";

const R = String.raw;
const D = "YOLO_SpaceNet_data.py";
const M = "YOLO_SpaceNet.py";
const PACKAGES = "ultralytics rasterio shapely certifi onnx onnxslim";

const GPU_STEP = {
  tr: "Menüden **Çalışma zamanı → Çalışma zamanı türünü değiştir** ile **T4 GPU**'yu seçin. Eğitim GPU'da yaklaşık bir saat sürer; CPU'da günler.",
  en: "From the menu, **Runtime → Change runtime type**, choose **T4 GPU**. Training takes about an hour on a GPU; on a CPU, days.",
};

const CODE = [
  {
    id: "yc-plan",
    title: { tr: "Ne yapacağız?", en: "What will we do?" },
    blocks: [
      {
        p: {
          tr: R`Bu bölümün iki modelini sıfırdan, herkese açık veriyle yeniden üreteceğiz. İş iki dosyaya bölünmüş: <code>${D}</code> veriyi hazırlar, <code>${M}</code> modeli eğitir ve kullanır. Akış:`,
          en: R`We will reproduce the two models of this chapter from scratch, from public data. The work is split into two files: <code>${D}</code> prepares the data, <code>${M}</code> trains and uses the model. The flow:`,
        },
      },
      {
        list: [
          {
            tr: R`<b>Veri</b> (SpaceNet 2): dört şehirden uydu kareleri ve elle çizilmiş bina sınırları.`,
            en: R`<b>Data</b> (SpaceNet 2): satellite tiles of four cities and hand-drawn building outlines.`,
          },
          {
            tr: R`<b>Etiket</b>: bina sınırları ve OpenStreetMap'ten spor sahası, yol, yeşil alan, su, otopark; YOLO biçiminde.`,
            en: R`<b>Labels</b>: the building outlines, and sports fields, roads, green, water and parking from OpenStreetMap; in YOLO's format.`,
          },
          {
            tr: R`<b>Kontrol</b>: etiketleri görüntünün üzerine çizip gözle bakmak.`,
            en: R`<b>Check</b>: drawing the labels on the image and looking.`,
          },
          {
            tr: R`<b>Eğitim</b>: COCO'da eğitilmiş YOLO26n'den başlayıp 100 epoch.`,
            en: R`<b>Training</b>: 100 epochs, starting from YOLO26n trained on COCO.`,
          },
          { tr: R`<b>Ölçme</b>: eğriler, mAP, karışıklık matrisi.`, en: R`<b>Measuring</b>: curves, mAP, confusion matrix.` },
          {
            tr: R`<b>Tahmin</b>: yeni bir karede binalar, m² ve yapılaşma oranı.`,
            en: R`<b>Prediction</b>: buildings on a new tile, m² and the built-up ratio.`,
          },
          {
            tr: R`<b>Dışa aktarma</b>: bu sayfanın tarayıcıda çalıştırdığı ONNX dosyası.`,
            en: R`<b>Export</b>: the ONNX file this page runs in the browser.`,
          },
        ],
        ordered: true,
      },
      {
        p: {
          tr: R`Süreler: bütün veriyi indirmek (746 kare, yaklaşık 1,9 GB) bağlantıya göre 15–30 dakika; denemek için <code>--per-city 20</code> ile 80 kare birkaç dakikada iner. Bir model GPU'da yaklaşık bir saatte eğitilir (bizim RTX 2060'ta 78 dakika).`,
          en: R`Times: downloading all the data (746 tiles, about 1.9 GB) takes 15–30 minutes depending on the connection; to try things, <code>--per-city 20</code> fetches 80 tiles in a few minutes. A model trains in about an hour on a GPU (78 minutes on our RTX 2060).`,
        },
      },
      {
        callout: "info",
        title: { tr: "Lisanslar", en: "Licences" },
        text: {
          tr: "SpaceNet görüntüleri ve bina sınırları CC BY-SA 4.0, OpenStreetMap ODbL ile paylaşılıyor; kaynak göstermek gerekir. Ultralytics AGPL-3.0'dır ve onunla eğitilen modelleri de bu lisansın kapsamında sayar.",
          en: "SpaceNet imagery and building outlines are shared under CC BY-SA 4.0, OpenStreetMap under ODbL; both must be credited. Ultralytics is AGPL-3.0 and counts the models trained with it as covered by that licence too.",
        },
      },
      { links: [{ tab: "theory", section: "y-tasks" }, { tab: "train" }] },
    ],
  },
  {
    id: "yc-setup",
    title: { tr: "Ortamı hazırlamak", en: "Setting up" },
    blocks: [
      {
        p: {
          tr: R`İki dosyanın da kütüphaneleri aynıdır: <code>ultralytics</code> (YOLO, PyTorch, OpenCV ve NumPy'yi de getirir), <code>rasterio</code> (coğrafi görüntüler), <code>shapely</code> (çokgenler), <code>certifi</code> (bazı Windows kurulumlarındaki eski sertifikalar için), <code>onnx</code> ve <code>onnxslim</code> (dışa aktarma). En kolayı Colab'dır: GPU hazırdır.`,
          en: R`Both files need the same libraries: <code>ultralytics</code> (YOLO; it brings PyTorch, OpenCV and NumPy along), <code>rasterio</code> (geographic images), <code>shapely</code> (polygons), <code>certifi</code> (for the outdated certificates of some Windows installs), <code>onnx</code> and <code>onnxslim</code> (export). Colab is the easiest: the GPU is ready there.`,
        },
      },
      {
        run: {
          file: D,
          text: DATA_PY,
          packages: PACKAGES,
          writefile: true,
          colab: {
            tr: "Colab'da veri `data/yolo` klasörüne iner; oturum kapanınca silinir. Saklamak için Google Drive'ı bağlayın.",
            en: "In Colab the data goes to the `data/yolo` folder and is deleted when the session ends. Mount Google Drive to keep it.",
          },
          extra: {
            tr: "Yalnızca denemek için: komutun sonuna `--per-city 20` ekleyin.",
            en: "Just to try things: add `--per-city 20` to the command.",
          },
        },
      },
      {
        run: {
          file: M,
          text: MODEL_PY,
          packages: PACKAGES,
          args: "train seg",
          writefile: true,
          colabStep: GPU_STEP,
          install: {
            tr: "Ekran kartınız varsa PyTorch'un CUDA sürümünü kurun (pytorch.org'daki komutla); yoksa eğitim CPU'da çok yavaş olur.",
            en: "If you have a graphics card, install PyTorch's CUDA build (with the command on pytorch.org); otherwise training on the CPU is very slow.",
          },
          extra: {
            tr: "Öteki komutlar: `train sem`, `curves seg`, `val seg`, `predict <görüntü>`, `export seg` (adım adım aşağıda).",
            en: "The other commands: `train sem`, `curves seg`, `val seg`, `predict <image>`, `export seg` (step by step below).",
          },
        },
      },
    ],
  },
  {
    id: "yc-download",
    title: { tr: "Veriyi indirmek", en: "Downloading the data" },
    blocks: [
      {
        p: {
          tr: R`SpaceNet, AWS'de herkese açık bir depoda durur; dosyalar doğrudan HTTP ile indirilir. Her kare 650 × 650 piksel, yaklaşık 0,3 m çözünürlükte bir uydu görüntüsü (GeoTIFF) ve aynı karedeki binaların GeoJSON çokgenleridir. Bizim seçtiğimiz 746 kare dosyanın sonundaki <code>TILES</code> listesindedir.`,
          en: R`SpaceNet sits in a public bucket on AWS; files are downloaded straight over HTTP. Each tile is a 650 × 650-pixel satellite image (GeoTIFF) at about 0.3 m, and the GeoJSON polygons of its buildings. The 746 tiles we chose are in the <code>TILES</code> list at the end of the file.`,
        },
      },
      {
        file: {
          source: D,
          from: "#--------------- Settings",
          to: "#----------------------------------------#",
          notes: [
            {
              at: "VAL_SHARE = 0.2",
              text: {
                tr: "Her şehrin karelerinin beşte biri doğrulamaya ayrılır (Bölüm 2'deki gibi).",
                en: "A fifth of every city's tiles is kept for validation (as in chapter 2).",
              },
            },
            {
              at: "MAX_NODATA = 0.25",
              text: {
                tr: "Uydu şeridinin kenarındaki kareler kısmen siyahtır; dörtte birinden fazlası boşsa kare atlanır.",
                en: "Tiles at the edge of the satellite strip are partly black; a tile more than a quarter empty is skipped.",
              },
            },
          ],
        },
      },
      {
        file: {
          source: D,
          from: "def download_tile(",
          to: "    return size",
          notes: [
            {
              at: "tif.write_bytes(get(",
              text: {
                tr: "16 bit GeoTIFF, yaklaşık 2,5 MB. Coğrafi konumu (dönüşüm) dosyanın içindedir.",
                en: "A 16-bit GeoTIFF, about 2.5 MB. Its geographic position (the transform) is inside the file.",
              },
            },
            {
              at: '"transform": list(src.transform)[:6]',
              text: {
                tr: "Pikselden boylam/enleme geçen afin dönüşüm: etiketleri piksele çevirirken tersi kullanılacak.",
                en: "The affine transform from pixels to longitude/latitude: its inverse turns the labels into pixels.",
              },
            },
            {
              at: "tif.unlink()",
              text: {
                tr: "JPEG'e çevrilen TIF silinir: 1,9 GB indirilir ama diskte yaklaşık 130 MB kalır.",
                en: "The TIF is deleted once converted to JPEG: 1.9 GB are downloaded but about 130 MB stay on disk.",
              },
            },
          ],
        },
      },
    ],
  },
  {
    id: "yc-8bit",
    title: { tr: "Görüntüyü 8 bite çevirmek", en: "Turning the image into 8 bits" },
    blocks: [
      {
        p: {
          tr: R`Uydu görüntüsünün değerleri 0–255 değil, 11 bitlik bir aralıktadır ve çoğu küçük değerlerde toplanır. Doğrudan 255'e bölmek neredeyse siyah bir görüntü verir. Her kanal ayrı ayrı gerilir: en karanlık %2 siyah, en parlak %2 beyaz olur, arası doğrusal dağıtılır.`,
          en: R`A satellite image's values are not 0–255 but in an 11-bit range, and most of them sit at the low end. Dividing straight by 255 gives an almost black image. Every channel is stretched on its own: the darkest 2% become black, the brightest 2% white, and the rest is spread linearly in between.`,
        },
      },
      {
        file: {
          source: D,
          from: "def to_8bit(",
          to: "    return out",
          notes: [
            {
              at: "valid = b[b > 0]",
              text: {
                tr: "0 değeri görüntü yok demektir (şeridin kenarı); yüzdelikler yalnızca gerçek piksellerden hesaplanır.",
                en: "A value of 0 means no image (the edge of the strip); the percentiles come from real pixels only.",
              },
            },
            {
              at: "np.percentile(valid, (2, 98))",
              text: {
                tr: "Uç değerler (parlayan bir çatı, derin bir gölge) germeyi bozmasın diye %2–98.",
                en: "2–98% so that outliers (a shining roof, a deep shadow) do not spoil the stretch.",
              },
            },
          ],
        },
      },
      {
        callout: "warn",
        title: { tr: "Tutarlılık", en: "Consistency" },
        text: {
          tr: "Modele eğitimde nasıl bir görüntü verildiyse çıkarımda da aynısı verilmelidir. Bu sayfa da JPEG kareleri bu dönüşümden sonra kullanır; başka bir germe, modelin gördüğü renkleri değiştirir.",
          en: "The model must get at inference the same kind of image it got in training. This page also uses the JPEG tiles after this conversion; another stretch would change the colours the model sees.",
        },
      },
    ],
  },
  {
    id: "yc-labels",
    title: { tr: "Etiketleri YOLO biçimine çevirmek", en: "Labels in YOLO's format" },
    blocks: [
      {
        p: {
          tr: R`SpaceNet'in bina çokgenleri boylam/enlem cinsindendir. Görüntünün afin dönüşümünün tersiyle piksele çevrilir, kare sınırında kırpılır ve YOLO'nun satır biçimine yazılır: sınıf numarası, ardından köşelerin \(x\, y\) çiftleri görüntü boyuna bölünmüş (0–1 arası). Bir görüntüye bir <code>.txt</code>, bir nesneye bir satır:`,
          en: R`SpaceNet's building polygons are in longitude/latitude. They are turned into pixels with the inverse of the image's affine transform, cut at the tile's edge and written in YOLO's line format: the class number, then the corners' \(x\, y\) pairs divided by the image size (between 0 and 1). One <code>.txt</code> per image, one line per object:`,
        },
      },
      {
        code: "0 0.35668 1.00000 0.35668 0.99788 0.31930 0.99788 0.31930 1.00000\n0 0.57627 0.99283 0.57650 1.00000 0.60455 1.00000 0.60431 0.99226",
        name: "Vegas_img3509.txt",
      },
      {
        file: {
          source: D,
          from: "def to_pixels(",
          to: '    return f"{cls} " + " ".join',
          notes: [
            {
              at: "inverse = ~Affine",
              text: {
                tr: "`~` afin dönüşümün tersini alır: boylam/enlemden piksele.",
                en: "`~` inverts the affine transform: from longitude/latitude to pixels.",
              },
            },
            {
              at: "p.area >= MIN_AREA_PX",
              text: {
                tr: "Kare kenarının kestiği binalardan kalan birkaç piksellik kıymıklar atılır.",
                en: "Slivers of a few pixels, left where the tile edge cuts a building, are dropped.",
              },
            },
          ],
        },
      },
      {
        p: {
          tr: R`İki tuzak: bina olmayan karelerde SpaceNet bir çokgen yerine yer tutucu bir <i>nokta</i> koyar; noktalar atlanır. Köşeler bir de \(z = 0\) taşır; <code>force_2d</code> onu düşürür.`,
          en: R`Two traps: on tiles without buildings SpaceNet puts a placeholder <i>point</i> instead of a polygon; points are skipped. The corners also carry \(z = 0\); <code>force_2d</code> drops it.`,
        },
      },
    ],
  },
  {
    id: "yc-osm",
    title: { tr: "OpenStreetMap'ten öteki sınıflar", en: "The other classes from OpenStreetMap" },
    blocks: [
      {
        p: {
          tr: R`SpaceNet yalnızca binaları içerir. Spor sahası, yol, yeşil alan, su ve otopark OpenStreetMap'ten alınır: Overpass API'ye, karelerin sınırlarına değen nesneler sorulur. Her sınıf birkaç OSM etiketine karşılık gelir.`,
          en: R`SpaceNet has only buildings. Sports fields, roads, green, water and parking come from OpenStreetMap: the Overpass API is asked for the objects touching the tiles' bounds. Each class stands for a few OSM tags.`,
        },
      },
      {
        file: {
          source: D,
          from: "def classify(tags",
          to: "    return None",
        },
      },
      {
        file: {
          source: D,
          from: "# roads are lines in OpenStreetMap",
          to: 'OSM_FILTERS["road"]',
          notes: [
            {
              at: '"motorway": 20',
              text: {
                tr: "OSM'de yollar orta çizgidir; maskeye boya olarak geçmeleri için türüne göre tipik bir genişlikle kalınlaştırılır.",
                en: "Roads in OSM are centre lines; to be painted into a mask they are widened by a typical width for their type.",
              },
            },
          ],
        },
      },
      {
        file: {
          source: D,
          from: "def nodata_mask(",
          to: "            out |= parts == i",
          notes: [
            {
              at: "dark = (img.max(axis=2) < 4)",
              text: {
                tr: 'Görüntüsü olmayan kenarlar: kenara değen, neredeyse tamamen siyah büyük bölgeler. Bunlar maskede 255 ("yok say") olur.',
                en: 'Edges without image: large almost black regions touching the border. They become 255 ("ignore") in the mask.',
              },
            },
          ],
        },
      },
      {
        callout: "warn",
        title: { tr: "Etiketler kusursuz değil", en: "The labels are not perfect" },
        text: {
          tr: "OpenStreetMap gönüllülerin haritasıdır: bazı otoparklar ve havuzlar hiç çizilmemiştir, bazı yollar görüntüden yenidir. Model bu gürültüden öğrenir; Eğitim sekmesindeki zayıf sınıflar (otopark, spor sahası) büyük ölçüde bundandır.",
          en: "OpenStreetMap is a volunteers' map: some car parks and pools were never drawn, some roads are newer than the imagery. The model learns from this noise; the weak classes of the Training tab (parking, sports fields) owe much to it.",
        },
      },
    ],
  },
  {
    id: "yc-check",
    title: { tr: "Etiketleri gözle kontrol etmek", en: "Checking the labels by eye" },
    blocks: [
      {
        p: {
          tr: R`Eğitimden önce mutlaka birkaç kareye bakın: koordinat sistemi, eksen sırası (\(x\)–\(y\) ya da boylam–enlem) ya da ölçek hatası olan etiketler en kolay gözle yakalanır. <code>--check</code> bir karenin çokgenlerini (solda) ve arazi örtüsü maskesini (sağda) çizer.`,
          en: R`Before training, always look at a few tiles: labels with a wrong coordinate system, axis order (\(x\)–\(y\) or longitude–latitude) or scale are easiest to catch by eye. <code>--check</code> draws a tile's polygons (left) and its land-cover mask (right).`,
        },
      },
      { file: { source: D, from: "def check_labels(", to: "    return out" } },
      {
        term: "PowerShell",
        lines: [`$ .venv\\Scripts\\python.exe ${D} --check Shanghai_img4937`, "wrote data\\yolo\\check\\Shanghai_img4937.png"],
      },
    ],
  },
  {
    id: "yc-split",
    title: { tr: "Eğitim ve doğrulama ayrımı, data.yaml", en: "The train/val split and data.yaml" },
    blocks: [
      {
        p: {
          tr: R`Her şehrin kareleri karıştırılır, beşte biri doğrulamaya ayrılır. Ultralytics veri setini bir <code>data.yaml</code> dosyasından bulur: klasör, eğitim ve doğrulama görüntüleri ve sınıf adları. Etiketler, görüntü klasöründeki <code>images</code> yerine <code>labels</code> (anlamsal segmentasyonda <code>masks</code>) yazılarak bulunur.`,
          en: R`Every city's tiles are shuffled and a fifth goes to validation. Ultralytics finds a dataset from a <code>data.yaml</code> file: the folder, the training and validation images and the class names. Labels are found by writing <code>labels</code> (for semantic segmentation <code>masks</code>) instead of <code>images</code> in the image folder.`,
        },
      },
      {
        file: {
          source: D,
          from: "def make_datasets(",
          to: '        print(f"{split}: " + ", ".join',
          notes: [
            {
              at: "rng.shuffle(ms)",
              text: {
                tr: "Ayrım şehir şehir yapılır: her şehir hem eğitimde hem doğrulamada yer alır.",
                en: "The split is made city by city: every city appears in both training and validation.",
              },
            },
            {
              at: "for cls in PAINT:",
              text: {
                tr: "Sınıflar sırayla boyanır; binalar en son boyanır, çünkü sınırları en güvenilir olanlardır.",
                en: "The classes are painted in turn; buildings last, because their outlines are the most reliable.",
              },
            },
            {
              at: "masks_dir: masks",
              text: {
                tr: "Anlamsal segmentasyonda etiket bir PNG maskedir; Ultralytics onu `masks` klasöründe arar.",
                en: "For semantic segmentation the label is a PNG mask; Ultralytics looks for it in the `masks` folder.",
              },
            },
          ],
        },
      },
      {
        term: "PowerShell",
        lines: [
          `$ .venv\\Scripts\\python.exe ${D} --per-city 3`,
          "downloading 12 tiles from SpaceNet (CC BY-SA 4.0)",
          "  12/12 tiles, 0.03 GB",
          "OpenStreetMap: 3 tiles of AOI_5_Khartoum",
          "…",
          "val: images 4, building 111, green 7, sports_field 2, road 22, water 8, parking 16",
          "train: images 8, building 198, road 83, sports_field 23, green 11, water 7, parking 9",
        ],
      },
    ],
  },
  {
    id: "yc-train",
    title: { tr: "Eğitmek", en: "Training" },
    blocks: [
      {
        file: {
          source: M,
          from: "#--------------- Hyperparameters",
          to: "#-----------------------------------------------#",
          notes: [
            {
              at: "BATCH = 4",
              text: {
                tr: "Bir adımda 4 kare. 200'den çok binalı karelerde her bina ayrı bir maske tuttuğu için 8, 6 GB'lık kartı aştı ve eğitimi yavaşlattı.",
                en: "4 tiles per step. Tiles with over 200 buildings hold a mask per building, so 8 overflowed our 6 GB card and slowed training down.",
              },
            },
          ],
        },
      },
      {
        file: {
          source: M,
          from: "def train(task",
          to: '    print(f"training took',
          notes: [
            {
              at: 'YOLO(f"yolo26n-{task}.yaml" if scratch else f"yolo26n-{task}.pt")',
              text: {
                tr: "`.pt`: COCO'da eğitilmiş ağırlıklar (ilk seferde indirilir). `.yaml`: aynı ağ, rastgele ağırlıklarla (Teori 6'daki karşılaştırma).",
                en: "`.pt`: weights trained on COCO (downloaded the first time). `.yaml`: the same network with random weights (the comparison of Theory 6).",
              },
            },
            {
              at: "flipud=0.5,",
              text: {
                tr: 'Kuş bakışında "yukarı" yoktur: dikey çevirme ve ±90° döndürme fotoğraflarda yanlış, burada doğru veri artırmadır.',
                en: 'A view from above has no "up": vertical flips and ±90° turns, wrong for photos, are fair augmentation here.',
              },
            },
            {
              at: "save_period=10,",
              text: {
                tr: "Her 10 epoch'ta bir ağırlıklar saklanır; sonra eski bir epoch'un tahminine bakabilirsiniz (Eğitim sekmesindeki gibi).",
                en: "The weights are kept every 10 epochs; later you can look at an old epoch's predictions (like in the Training tab).",
              },
            },
          ],
        },
      },
      {
        file: {
          source: M,
          from: 'if __name__ == "__main__":',
          to: "    args = parser.parse_args()",
          notes: [
            {
              at: 'if __name__ == "__main__":',
              text: {
                tr: "Windows'ta veri yükleyicinin işçileri bu dosyayı yeniden içe aktarır; bu satır olmasaydı her işçi kendi eğitimini başlatırdı.",
                en: "On Windows the data loader's workers import this file again; without this line every worker would start a training of its own.",
              },
            },
          ],
        },
      },
      {
        links: [
          { tab: "theory", section: "y-train" },
          { tab: "math", section: "ym-cls" },
        ],
      },
    ],
  },
  {
    id: "yc-curves",
    title: { tr: "Eğitim eğrilerini okumak", en: "Reading the training curves" },
    blocks: [
      {
        p: {
          tr: R`Ultralytics her epoch'un kayıplarını ve ölçümlerini <code>results.csv</code>'ye yazar. Eğitim kaybı düşerken doğrulama kaybı yükselmeye başlarsa model eğitim karelerini ezberliyordur (<a href="#" data-chapter="mlp" data-go="theory:ml-general">Bölüm 2</a>).`,
          en: R`Ultralytics writes every epoch's losses and scores to <code>results.csv</code>. If the validation loss starts rising while the training loss falls, the model is memorizing the training tiles (<a href="#" data-chapter="mlp" data-go="theory:ml-general">chapter 2</a>).`,
        },
      },
      { file: { source: M, from: "def curves(", to: '    print(f"plots by Ultralytics' } },
      {
        term: "PowerShell",
        lines: [
          `$ .venv\\Scripts\\python.exe ${M} curves seg`,
          "epoch  train box_loss    val box_loss  train seg_loss    val seg_loss        mAP50(M)     mAP50-95(M)",
          "    1          2.0276          1.8343          3.4553          3.0293          0.2034          0.0810",
          "   10          1.9189          1.9354          2.9190          2.6599          0.3748          0.1764",
          "   50          1.8747          1.8416          2.6938          2.6165          0.4363          0.2149",
          "  100          1.7043          1.7074          2.5164          2.4136          0.4684          0.2369",
          "lowest val/seg_loss: epoch 99; no clear sign of overfitting",
        ],
      },
      { links: [{ tab: "train" }] },
    ],
  },
  {
    id: "yc-val",
    title: { tr: "Ölçmek", en: "Measuring" },
    blocks: [
      { file: { source: M, from: "def val(task", to: '    print(f"confusion matrix' } },
      {
        term: "PowerShell",
        lines: [
          `$ .venv\\Scripts\\python.exe ${M} val seg`,
          "building       mask AP50 0.693   AP50-95 0.353   box AP50 0.706",
          "sports_field   mask AP50 0.301   AP50-95 0.146   box AP50 0.370",
          "mean           mask mAP50 0.497   mAP50-95 0.250",
        ],
      },
      {
        p: {
          tr: R`Ultralytics aynı klasöre karışıklık matrisini de çizer: hangi sınıfın neyle karıştığını ve "arka plan" sütununda kaçırılan nesneleri gösterir. AP'nin nasıl hesaplandığı Matematik 7'de.`,
          en: R`Ultralytics also draws the confusion matrix into the same folder: which class is confused with what, and in its "background" column the objects that were missed. How AP is computed is in Math 7.`,
        },
      },
      {
        links: [
          { tab: "math", section: "ym-ap" },
          { tab: "theory", section: "y-metrics" },
        ],
      },
    ],
  },
  {
    id: "yc-predict",
    title: { tr: "Tahmin ve alan", en: "Prediction and area" },
    blocks: [
      {
        p: {
          tr: R`<code>retina_masks=True</code> maskeleri görüntünün kendi boyunda verir; böylece alan doğrudan görüntü pikseli × yer örnekleme aralığı (GSD) ile hesaplanır. SpaceNet kareleri GSD'lerini JSON'larından bilir; başka bir görüntü için <code>--mpp</code> verin.`,
          en: R`<code>retina_masks=True</code> gives the masks at the image's own size, so the area is simply image pixels × the ground sampling distance (GSD). SpaceNet tiles know their GSD from their JSON; for another image pass <code>--mpp</code>.`,
        },
      },
      { file: { source: M, from: "def metres_per_pixel(", to: '    print(f"drawn into {out}")' } },
      {
        term: "PowerShell",
        lines: [
          `$ .venv\\Scripts\\python.exe ${M} predict data/yolo/seg/images/val/Paris_img637.jpg`,
          "  1  building     score 0.85  area     88.5 m²",
          "  2  building     score 0.85  area    117.3 m²",
          "…",
          " 46  building     score 0.26  area     10.0 m²",
          "one pixel = 0.197 m × 0.301 m; building footprint 5111 m² of 25048 m²: built-up ratio 20.4%",
        ],
      },
      { links: [{ tab: "detect" }, { tab: "math", section: "ym-area" }] },
    ],
  },
  {
    id: "yc-export",
    title: { tr: "Dışa aktarmak", en: "Exporting" },
    blocks: [
      {
        p: {
          tr: R`Bu sayfa modeli PyTorch olmadan, tarayıcıda ONNX Runtime Web ile çalıştırır. ONNX, ağın hesaplarını taşınabilir bir dosyaya yazar. Ağırlıkları float16'ya çevirmek doğruluktan bir şey kaybettirmeden boyutu yarıya indirdi (11,1 MB → 5,6 MB); giriş ve çıkış float32 kaldı ki JavaScript tarafı basit olsun. NMS'i sayfa kendisi yapar.`,
          en: R`This page runs the model in the browser with ONNX Runtime Web, without PyTorch. ONNX writes the network's computations into a portable file. Turning the weights into float16 halved the size (11.1 MB → 5.6 MB) without losing accuracy; inputs and outputs stayed float32 to keep the JavaScript side simple. The page does NMS itself.`,
        },
      },
      { file: { source: M, from: "def export(task", to: '    print(f"{web}:' } },
      {
        term: "PowerShell",
        lines: [
          `$ .venv\\Scripts\\python.exe ${M} export seg`,
          "runs\\yolo\\seg-pretrained\\weights\\seg_web.onnx: 5.6 MB; the chapter's page loads it with ONNX Runtime Web",
        ],
      },
      { links: [{ tab: "detect" }] },
    ],
  },
  {
    id: "yc-files",
    title: { tr: "Tam dosyalar", en: "The whole files" },
    blocks: [
      { p: { tr: R`<code>${D}</code>, veriyi hazırlayan dosya:`, en: R`<code>${D}</code>, the file that prepares the data:` } },
      { file: { source: D, whole: true } },
      {
        p: {
          tr: R`<code>${M}</code>, modeli eğiten ve kullanan dosya:`,
          en: R`<code>${M}</code>, the file that trains and uses the model:`,
        },
      },
      { file: { source: M, whole: true } },
    ],
  },
  {
    id: "yc-exercises",
    title: { tr: "Alıştırmalar", en: "Exercises" },
    blocks: [
      {
        list: [
          {
            tr: R`<b>★ Sıfırdan eğitim.</b> <code>train seg --scratch</code> ile aynı eğitimi rastgele ağırlıklarla yapın ve <code>curves</code> çıktılarını karşılaştırın. Bizim sonucumuz Eğitim sekmesinde: 0,32'ye karşı 0,50.`,
            en: R`<b>★ From scratch.</b> Run the same training with random weights (<code>train seg --scratch</code>) and compare the <code>curves</code> outputs. Our result is in the Training tab: 0.32 against 0.50.`,
          },
          {
            tr: R`<b>★ Daha küçük giriş.</b> <code>IMGSZ = 320</code> ile eğitin. Aday sayısı kaça iner (Matematik 1)? Hangi binalar artık bulunamıyor, neden?`,
            en: R`<b>★ A smaller input.</b> Train with <code>IMGSZ = 320</code>. How many candidates are left (Math 1)? Which buildings can no longer be found, and why?`,
          },
          {
            tr: R`<b>★★ Başka bir şehre taşımak.</b> Yalnızca Las Vegas kareleriyle eğitip Hartum'un doğrulama karelerinde ölçün. AP ne kadar düşüyor? Çatı malzemesi, renk ve yapı düzeni farkı Teori 8'deki "alan değişimi"dir.`,
            en: R`<b>★★ Moving to another city.</b> Train on the Las Vegas tiles only and measure on Khartoum's validation tiles. How far does AP fall? The difference in roof material, colour and layout is the "domain shift" of Theory 8.`,
          },
          {
            tr: R`<b>★★ Daha uzun eğitim.</b> Bizim eğitimde maske mAP50 75. epoch'ta 0,468'e ulaştı ve 100. epoch'a kadar neredeyse hiç artmadı. 200 epoch eğitin: bir şey kazandırıyor mu, <code>curves</code> aşırı uyum belirtisi gösteriyor mu?`,
            en: R`<b>★★ Longer training.</b> In our training the mask mAP50 reached 0.468 at epoch 75 and hardly rose until epoch 100. Train for 200 epochs: does it gain anything, and does <code>curves</code> show signs of overfitting?`,
          },
          {
            tr: R`<b>★★★ Yeni bir sınıf.</b> <code>OSM_FILTERS</code>'a yüzme havuzlarını (<code>leisure=swimming_pool</code>) ayrı bir sınıf olarak ekleyin; etiketleri <code>--check</code> ile kontrol edin. Bu dört şehirde havuz neden az?`,
            en: R`<b>★★★ A new class.</b> Add swimming pools (<code>leisure=swimming_pool</code>) to <code>OSM_FILTERS</code> as a class of their own; check the labels with <code>--check</code>. Why are there few pools in these four cities?`,
          },
        ],
      },
    ],
  },
];

createLesson({
  tab: "code",
  main: "#yc-main",
  toc: "#yc-toc",
  sections: CODE,
  kicker: { tr: "Adım", en: "Step" },
  sources: [
    { text: DATA_PY, name: D },
    { text: MODEL_PY, name: M },
  ],
});
