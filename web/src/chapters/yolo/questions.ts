/**
 * Chapter 4, Practice tab: questions in the order of the Theory sections, then five experiments on the
 * chapter's trained YOLO26n-seg and the Shanghai tile of the Theory tab. tests/unit/practice.test.ts runs
 * every `compute` and `run`, the experiments included (on the same pixels the page decodes), and fails if
 * a stated answer is not what the engine or the model gives.
 */
import { boxIou, candidateCount, nms, type Candidate } from "../../engines/yolo/detect.ts";
import { resizeLinear, type Rgb } from "../../engines/yolo/image.ts";
import { objectsOf, runSegmentation, type YoloModel } from "../../engines/yolo/model.ts";
import type { PracticeSection } from "../../shared/practice.ts";

/** What the experiments need: the model and the decoded tile (loaded when the first one runs, about 12 MB). */
export interface YoloLab {
  model: YoloModel;
  tile: Rgb;
}

const R = String.raw;

/** Buildings and sports fields the model finds on an image. */
async function found(lab: YoloLab, image: Rgb, conf = 0.25, iou = 0.7): Promise<[number, number]> {
  const out = await runSegmentation(lab.model, image);
  const objects = objectsOf(out, 2, image, { conf, iou });
  return [objects.filter((o) => o.cls === 0).length, objects.filter((o) => o.cls === 1).length];
}

/** The image turned by 90° clockwise. */
function rotated(image: Rgb): Rgb {
  const { width: w, height: h, data } = image;
  const out = new Uint8Array(data.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const from = (y * w + x) * 3;
      const to = (x * h + (h - 1 - y)) * 3;
      out[to] = data[from];
      out[to + 1] = data[from + 1];
      out[to + 2] = data[from + 2];
    }
  }
  return { width: h, height: w, data: out };
}

/** The image in shades of grey (the same brightness, no colour). */
function grey(image: Rgb): Rgb {
  const out = new Uint8Array(image.data.length);
  for (let i = 0; i < image.width * image.height; i++) {
    const v = Math.round(0.299 * image.data[3 * i] + 0.587 * image.data[3 * i + 1] + 0.114 * image.data[3 * i + 2]);
    out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = v;
  }
  return { ...image, data: out };
}

/** The five boxes of the NMS question (one class), with their scores. */
const NMS_BOXES: Candidate[] = [
  { index: 0, cls: 0, score: 0.9, box: [0, 0, 10, 10] },
  { index: 1, cls: 0, score: 0.8, box: [1, 1, 11, 11] },
  { index: 2, cls: 0, score: 0.75, box: [20, 0, 30, 10] },
  { index: 3, cls: 0, score: 0.7, box: [22, 1, 32, 11] },
  { index: 4, cls: 0, score: 0.3, box: [5, 5, 15, 15] },
];

export function buildPractice(lab: () => Promise<YoloLab>): PracticeSection[] {
  return [
    {
      id: "yx-tasks",
      title: { tr: "Dört görev", en: "Four tasks" },
      intro: {
        tr: R`Sınıflandırma, tespit, anlamsal segmentasyon ve örnek segmentasyonu aynı kareye farklı sorular sorar.`,
        en: R`Classification, detection, semantic and instance segmentation ask the same tile different questions.`,
      },
      read: { tab: "theory", section: "y-tasks" },
      questions: [
        {
          id: "t-surface",
          kind: "choice",
          q: {
            tr: "Bir karede yolların, suyun ve yeşil alanın ne kadar yer kapladığını ölçmek istiyoruz. Hangi görev en uygunudur?",
            en: "We want to measure how much of a tile roads, water and green areas cover. Which task fits best?",
          },
          options: [
            {
              text: { tr: "Sınıflandırma", en: "Classification" },
              why: {
                tr: "Sınıflandırma yalnızca karede ne olduğunu söyler; ne kadar yer kapladığını söylemez.",
                en: "Classification only says what is in the tile, not how much of it there is.",
              },
            },
            {
              text: { tr: "Nesne tespiti", en: "Object detection" },
              why: {
                tr: "Bir yol ya da nehir için kutu anlamsızdır: kutu, eğri ve uzun bir yüzeyin çoğunu boş yere kaplar.",
                en: "A box makes no sense for a road or a river: it covers mostly empty space around a long, curving surface.",
              },
            },
            { text: { tr: "Anlamsal segmentasyon", en: "Semantic segmentation" } },
            {
              text: { tr: "Örnek segmentasyonu", en: "Instance segmentation" },
              why: {
                tr: "Yollar ve su sayılabilen ayrı nesneler değildir; onları tek tek ayırmaya gerek yok, her pikselin sınıfı yeter.",
                en: "Roads and water are not separate objects to count; there is no need to tell them apart, the class of every pixel is enough.",
              },
            },
          ],
          answer: 2,
          explain: {
            tr: "Her pikselin sınıfını bilince bir sınıfın payı piksel sayısıyla bulunur. Bu bölümün arazi örtüsü modeli tam bunu yapar (Tespit sekmesi, Arazi örtüsü).",
            en: "Knowing every pixel's class, a class's share is a pixel count. That is exactly what this chapter's land-cover model does (Detection tab, Land cover).",
          },
        },
        {
          id: "t-count",
          kind: "choice",
          q: {
            tr: "Bitişik nizamda yan yana duran evleri tek tek saymak ve her birinin alanını bulmak istiyoruz. Hangi görev gerekir?",
            en: "We want to count houses built wall to wall and find the area of each one. Which task is needed?",
          },
          options: [
            { text: { tr: "Örnek segmentasyonu", en: "Instance segmentation" } },
            {
              text: { tr: "Anlamsal segmentasyon", en: "Semantic segmentation" },
              why: {
                tr: 'Anlamsal segmentasyonda bitişik evler tek bir "bina" lekesi olur; kaç ev olduğu kaybolur.',
                en: 'In semantic segmentation, houses that touch become one "building" patch; how many houses there are is lost.',
              },
            },
            {
              text: { tr: "Nesne tespiti", en: "Object detection" },
              why: {
                tr: "Tespit evleri sayar, ama kutunun alanı evin alanı değildir: eğik ya da L biçimli bir evde kutu çok daha büyüktür.",
                en: "Detection counts the houses, but a box's area is not the house's: for a tilted or L-shaped house the box is much larger.",
              },
            },
          ],
          answer: 0,
          explain: {
            tr: "Örnek segmentasyonu hem sayar (her nesneye ayrı maske) hem sınırı çizer (alan maskeden). YOLO26n-seg bunu yapar.",
            en: "Instance segmentation both counts (a mask per object) and draws the outline (the area comes from the mask). YOLO26n-seg does this.",
          },
        },
      ],
    },
    {
      id: "yx-iou",
      title: { tr: "Kutu ve IoU", en: "Boxes and IoU" },
      intro: {
        tr: R`\(\mathrm{IoU} = |A \cap B| / |A \cup B|\); kutu \((x_1, y_1, x_2, y_2)\) ya da \((c_x, c_y, w, h)\) ile yazılır.`,
        en: R`\(\mathrm{IoU} = |A \cap B| / |A \cup B|\); a box is written as \((x_1, y_1, x_2, y_2)\) or \((c_x, c_y, w, h)\).`,
      },
      read: { tab: "theory", section: "y-iou" },
      questions: [
        {
          id: "i-value",
          kind: "number",
          q: {
            tr: R`Kutu A \((0, 0, 10, 10)\), kutu B \((5, 0, 15, 10)\) (köşe yazılışı). IoU kaçtır?`,
            en: R`Box A is \((0, 0, 10, 10)\), box B \((5, 0, 15, 10)\) (corner form). What is their IoU?`,
          },
          label: { tr: "IoU =", en: "IoU =" },
          answer: 1 / 3,
          tolerance: 0.005,
          compute: () => boxIou([0, 0, 10, 10], [5, 0, 15, 10]),
          hint: {
            tr: "Önce kesişimin, sonra birleşimin alanını bulun. Birleşim, ortak kısım iki kez sayılmadan iki alanın toplamıdır.",
            en: "Find the area of the intersection first, then of the union. The union is the two areas added, without counting the shared part twice.",
          },
          mistakes: [
            {
              value: 0.25,
              why: {
                tr: "İki alanın toplamına (200) böldünüz; ortak kısım iki kez sayıldı. Birleşim 100 + 100 − 50 = 150'dir.",
                en: "You divided by the sum of the two areas (200), counting the shared part twice. The union is 100 + 100 − 50 = 150.",
              },
            },
            {
              value: 0.5,
              why: {
                tr: "Kesişimi tek bir kutunun alanına böldünüz. IoU iki kutunun birlikte kapladığı alana böler.",
                en: "You divided the intersection by one box's area. IoU divides by the area the two boxes cover together.",
              },
            },
          ],
          explain: {
            tr: R`Kesişim \(5 \times 10 = 50\), birleşim \(100 + 100 - 50 = 150\): \(\mathrm{IoU} = 50/150 \approx 0{,}33\). Kutuların yarısı örtüşse de IoU 0,5 değildir; bu tespit doğru sayılmaz.`,
            en: R`Intersection \(5 \times 10 = 50\), union \(100 + 100 - 50 = 150\): \(\mathrm{IoU} = 50/150 \approx 0.33\). Though half of each box overlaps, IoU is not 0.5; this detection does not count as correct.`,
          },
        },
        {
          id: "i-corner",
          kind: "number",
          q: {
            tr: R`Bir kutu merkez yazılışıyla \((c_x, c_y, w, h) = (100, 60, 40, 20)\). Sağ kenarı \(x_2\) kaçtır?`,
            en: R`A box in centre form is \((c_x, c_y, w, h) = (100, 60, 40, 20)\). What is its right edge \(x_2\)?`,
          },
          label: { tr: R`\(x_2 =\)`, en: R`\(x_2 =\)` },
          answer: 120,
          compute: () => 100 + 40 / 2,
          hint: { tr: "Merkezden sağ kenara genişliğin yarısı kadar gidilir.", en: "From the centre to the right edge is half the width." },
          mistakes: [
            {
              value: 140,
              why: {
                tr: "Genişliğin tamamını eklediniz; merkezden kenara genişliğin yarısı kadardır.",
                en: "You added the whole width; from the centre to an edge is half of it.",
              },
            },
            {
              value: 80,
              why: {
                tr: "Bu sol kenar \\(x_1 = c_x - w/2\\). Sağ kenar \\(c_x + w/2\\)'dir.",
                en: "That is the left edge \\(x_1 = c_x - w/2\\). The right edge is \\(c_x + w/2\\).",
              },
            },
          ],
          explain: {
            tr: R`\(x_2 = c_x + w/2 = 100 + 20 = 120\); kutu \((80, 50, 120, 70)\).`,
            en: R`\(x_2 = c_x + w/2 = 100 + 20 = 120\); the box is \((80, 50, 120, 70)\).`,
          },
        },
        {
          id: "i-why",
          kind: "choice",
          q: {
            tr: "Bir tahminin doğru olup olmadığına karar verirken neden merkezler arasındaki uzaklık yerine IoU kullanılır?",
            en: "Why decide whether a prediction is right with IoU instead of the distance between the centres?",
          },
          options: [
            {
              text: { tr: "IoU'yu hesaplamak daha hızlı olduğu için", en: "Because IoU is quicker to compute" },
              why: {
                tr: "Uzaklık daha da hızlı hesaplanır. Sebep hız değil, ölçek.",
                en: "A distance is even quicker. The reason is not speed but scale.",
              },
            },
            {
              text: {
                tr: "Uzaklık kutunun boyutunu görmediği için: aynı kayma küçük bir evde büyük, büyük bir depoda küçük bir hatadır",
                en: "Because distance ignores the box's size: the same shift is a big mistake for a small house and a small one for a large warehouse",
              },
            },
            {
              text: { tr: "Merkezler her zaman aynı noktaya düştüğü için", en: "Because the centres always fall on the same point" },
              why: {
                tr: "Merkezler aynı olsa bile kutuların boyları farklı olabilir; uzaklık 0 olur, kutu yine de yanlıştır.",
                en: "Even with the same centre the boxes can differ in size; the distance is 0 and the box is still wrong.",
              },
            },
          ],
          answer: 1,
          explain: {
            tr: "IoU bir orandır, nesnenin büyüklüğünden bağımsızdır; kutunun hem yerini hem boyunu birlikte ölçer.",
            en: "IoU is a ratio, independent of the object's size; it measures the box's place and size together.",
          },
        },
      ],
    },
    {
      id: "yx-grid",
      title: { tr: "Üç ölçekte ızgara", en: "A grid at three scales" },
      intro: {
        tr: R`640 piksellik girişte stride 8, 16 ve 32 ile 80², 40² ve 20² hücre; her hücre bir aday, her aday \(4 + C + 32\) sayı.`,
        en: R`With a 640-pixel input, strides 8, 16 and 32 give 80², 40² and 20² cells; every cell is a candidate, every candidate \(4 + C + 32\) numbers.`,
      },
      read: { tab: "theory", section: "y-grid" },
      questions: [
        {
          id: "g-512",
          kind: "number",
          q: {
            tr: "Giriş 640 yerine 512 piksel olsaydı kaç aday olurdu?",
            en: "How many candidates would a 512-pixel input give instead of 640?",
          },
          label: { tr: "aday sayısı =", en: "number of candidates =" },
          answer: 5376,
          compute: () => candidateCount(512),
          hint: {
            tr: "512'yi 8'e, 16'ya ve 32'ye bölün; her ızgaranın hücre sayısı bunun karesidir.",
            en: "Divide 512 by 8, 16 and 32; each grid has the square of that many cells.",
          },
          mistakes: [
            {
              value: 4096,
              why: {
                tr: "Yalnızca en ince ızgarayı (64²) saydınız. Öteki iki ölçeğin hücreleri de adaydır: 32² ve 16².",
                en: "You counted only the finest grid (64²). The cells of the other two scales are candidates too: 32² and 16².",
              },
            },
            { value: 5120, why: { tr: "En kaba ızgarayı (16² = 256) unuttunuz.", en: "You left out the coarsest grid (16² = 256)." } },
          ],
          explain: {
            tr: R`\(64^2 + 32^2 + 16^2 = 4096 + 1024 + 256 = 5376\), ya da \(21 \cdot 512^2 / 1024\) (Matematik 1).`,
            en: R`\(64^2 + 32^2 + 16^2 = 4096 + 1024 + 256 = 5376\), or \(21 \cdot 512^2 / 1024\) (Math 1).`,
          },
        },
        {
          id: "g-channels",
          kind: "number",
          q: {
            tr: "Bina, spor sahası ve havuz olmak üzere 3 sınıflı bir YOLO26 segmentasyon modelinde her aday kaç sayı verir?",
            en: "In a YOLO26 segmentation model with 3 classes (building, sports field, pool), how many numbers does every candidate give?",
          },
          label: { tr: "sayı =", en: "numbers =" },
          answer: 39,
          compute: () => 4 + 3 + 32,
          hint: { tr: "Kutu, sınıf skorları ve maske katsayıları.", en: "The box, the class scores and the mask coefficients." },
          mistakes: [
            {
              value: 38,
              why: {
                tr: "Bu, bu bölümün 2 sınıflı modelinin sayısı. Her sınıfa bir skor: 3 sınıf, 3 skor.",
                en: "That is the number of this chapter's 2-class model. One score per class: 3 classes, 3 scores.",
              },
            },
            {
              value: 7,
              why: {
                tr: "32 maske katsayısını unuttunuz: segmentasyon modeli her aday için onları da verir.",
                en: "You left out the 32 mask coefficients: a segmentation model gives them for every candidate too.",
              },
            },
            {
              value: 36,
              why: {
                tr: "Sınıf skorlarını unuttunuz: kutu (4) ve maske (32) yanında her sınıfa bir skor vardır.",
                en: "You left out the class scores: besides the box (4) and the mask (32) there is a score per class.",
              },
            },
          ],
          explain: {
            tr: R`\(4 + 3 + 32 = 39\): kutunun dört uzaklığı, sınıf başına bir skor, 32 maske katsayısı.`,
            en: R`\(4 + 3 + 32 = 39\): the box's four distances, a score per class, 32 mask coefficients.`,
          },
        },
        {
          id: "g-anchor",
          kind: "number",
          q: {
            tr: R`40 × 40 ızgarada (stride 16) \((g_x, g_y) = (10, 4)\) hücresinin çapa noktasının \(x\) koordinatı kaç pikseldir?`,
            en: R`On the 40 × 40 grid (stride 16), what is the \(x\) coordinate in pixels of the anchor point of cell \((g_x, g_y) = (10, 4)\)?`,
          },
          label: { tr: R`\(a_x =\)`, en: R`\(a_x =\)` },
          answer: 168,
          compute: () => (10 + 0.5) * 16,
          hint: { tr: "Çapa noktası hücrenin merkezidir.", en: "The anchor point is the centre of the cell." },
          mistakes: [
            {
              value: 160,
              why: {
                tr: "Bu hücrenin sol kenarı. Çapa noktası merkezdedir: \\((g_x + \\tfrac12)\\,s\\).",
                en: "That is the cell's left edge. The anchor point is its centre: \\((g_x + \\tfrac12)\\,s\\).",
              },
            },
            {
              value: 176,
              why: {
                tr: "Bu hücrenin sağ kenarı; merkez yarım hücre geridedir.",
                en: "That is the cell's right edge; the centre is half a cell back.",
              },
            },
          ],
          explain: {
            tr: R`\(a_x = (10 + 0{,}5)\cdot 16 = 168\). Kutunun kenarları buradan \(l\,s\) ve \(r\,s\) kadar uzaktadır (Matematik 2).`,
            en: R`\(a_x = (10 + 0.5)\cdot 16 = 168\). The box's sides are \(l\,s\) and \(r\,s\) away from it (Math 2).`,
          },
        },
      ],
    },
    {
      id: "yx-nms",
      title: { tr: "Güven eşiği ve NMS", en: "The confidence threshold and NMS" },
      intro: {
        tr: "Eşiğin altındaki adaylar atılır; kalanlar skora göre sıralanır, en iyisi tutulur ve onunla IoU'su NMS eşiğini aşan aynı sınıftan adaylar silinir.",
        en: "Candidates below the threshold are dropped; the rest are sorted by score, the best is kept and those of its class overlapping it by more than the NMS threshold are removed.",
      },
      read: { tab: "theory", section: "y-nms" },
      questions: [
        {
          id: "n-count",
          kind: "number",
          q: {
            tr: R`Aynı sınıftan beş aday: A \((0, 0, 10, 10)\) skor 0,9; B \((1, 1, 11, 11)\) 0,8; C \((20, 0, 30, 10)\) 0,75; D \((22, 1, 32, 11)\) 0,7; E \((5, 5, 15, 15)\) 0,3. Güven eşiği 0,4, NMS IoU eşiği 0,5. Kaç kutu kalır?`,
            en: R`Five candidates of one class: A \((0, 0, 10, 10)\) score 0.9; B \((1, 1, 11, 11)\) 0.8; C \((20, 0, 30, 10)\) 0.75; D \((22, 1, 32, 11)\) 0.7; E \((5, 5, 15, 15)\) 0.3. Confidence threshold 0.4, NMS IoU threshold 0.5. How many boxes are left?`,
          },
          label: { tr: "kalan kutu =", en: "boxes left =" },
          answer: 2,
          compute: () =>
            nms(
              NMS_BOXES.filter((c) => c.score > 0.4),
              0.5,
            ).length,
          hint: {
            tr: "Önce eşiğin altındakini atın. Sonra A'yı tutun ve B ile IoU'suna bakın; C ve D için aynısını yapın.",
            en: "First drop the one below the threshold. Then keep A and look at its IoU with B; do the same for C and D.",
          },
          mistakes: [
            {
              value: 3,
              why: {
                tr: "E'nin skoru 0,3: güven eşiğinin (0,4) altında, NMS'e hiç girmez. A, B ile (IoU 0,68), C de D ile (IoU 0,56) örtüştüğü için B ve D silinir.",
                en: "E scores 0.3: below the confidence threshold (0.4), it never reaches NMS. A overlaps B (IoU 0.68) and C overlaps D (IoU 0.56), so B and D go.",
              },
            },
            {
              value: 4,
              why: {
                tr: "Yalnızca güven eşiğini uyguladınız. NMS, A ile çok örtüşen B'yi ve C ile çok örtüşen D'yi de siler.",
                en: "You applied only the confidence threshold. NMS also removes B, which overlaps A a lot, and D, which overlaps C.",
              },
            },
          ],
          explain: {
            tr: R`E eşikte elenir. A tutulur; \(\mathrm{IoU}(A, B) = 81/119 \approx 0{,}68 > 0{,}5\), B silinir. C tutulur; \(\mathrm{IoU}(C, D) = 72/128 \approx 0{,}56 > 0{,}5\), D silinir. Kalan: A ve C.`,
            en: R`E is dropped at the threshold. A is kept; \(\mathrm{IoU}(A, B) = 81/119 \approx 0.68 > 0.5\), B goes. C is kept; \(\mathrm{IoU}(C, D) = 72/128 \approx 0.56 > 0.5\), D goes. Left: A and C.`,
          },
        },
        {
          id: "n-low",
          kind: "choice",
          q: {
            tr: "NMS IoU eşiğini çok düşük (örneğin 0,1) seçersek ne olur?",
            en: "What happens if the NMS IoU threshold is set very low (say 0.1)?",
          },
          options: [
            {
              text: { tr: "Aynı binanın kopyaları kalır", en: "Copies of the same building stay" },
              why: {
                tr: "Kopyalar çok örtüştüğü için düşük eşikte zaten silinir. Kopyalar yüksek eşikte kalır.",
                en: "Copies overlap a lot, so a low threshold removes them anyway. Copies stay with a high threshold.",
              },
            },
            {
              text: { tr: "Model daha yavaş çalışır", en: "The model runs slower" },
              why: {
                tr: "Ağın kendisi aynı kalır; NMS'in eşiği yalnızca son işlemeyi etkiler.",
                en: "The network itself is unchanged; the NMS threshold only affects the post-processing.",
              },
            },
            {
              text: {
                tr: "Birbirine değen iki gerçek bina tek binaya iner: biraz örtüşen kutular da silinir",
                en: "Two real buildings that touch become one: boxes that overlap even a little are removed",
              },
            },
          ],
          answer: 2,
          explain: {
            tr: 'Eşik, "bu kutu aynı nesnenin kopyası mı?" sorusunun sınırıdır. Çok düşükse komşu nesneler de kopya sayılır; çok yüksekse gerçek kopyalar kalır. Tespit sekmesindeki kaydırıcıyla deneyin.',
            en: 'The threshold draws the line for "is this box a copy of the same object?". Too low and neighbouring objects count as copies; too high and real copies stay. Try it with the slider of the Detection tab.',
          },
        },
        {
          id: "n-free",
          kind: "choice",
          q: { tr: "YOLO26'nın bire bir başı neden NMS'e ihtiyaç duymaz?", en: "Why does YOLO26's one-to-one head need no NMS?" },
          options: [
            {
              text: {
                tr: "Eğitimde her nesneye tek bir aday atanır; ağ öteki adaylara düşük skor vermeyi öğrenir",
                en: "In training each object is assigned a single candidate; the network learns to give the other candidates low scores",
              },
            },
            {
              text: { tr: "Yalnızca en kaba ızgarayı kullandığı için", en: "Because it uses only the coarsest grid" },
              why: {
                tr: "Üç ölçeğin hepsi kullanılır; aday sayısı aynıdır (8400).",
                en: "All three scales are used; the number of candidates is the same (8400).",
              },
            },
            {
              text: { tr: "NMS'i ağın içinde, son katmanda yaptığı için", en: "Because it does NMS inside the network, in the last layer" },
              why: {
                tr: "NMS ağın içine taşınmadı; kopyaların hiç oluşmaması öğretildi. Ağ yalnızca en yüksek skorlu adayları seçer.",
                en: "NMS was not moved into the network; the network was taught not to make copies at all. Only the top-scoring candidates are picked.",
              },
            },
          ],
          answer: 0,
          explain: {
            tr: "Bire bir atamada her nesnenin tek bir \"sorumlu\" adayı vardır. Bizim modelde bu baş maske mAP50'de 0,47, NMS'li bire-çok çıktı 0,50 verdi.",
            en: 'With one-to-one assignment every object has a single "responsible" candidate. On our model this head gave a mask mAP50 of 0.47, the one-to-many output with NMS 0.50.',
          },
        },
      ],
    },
    {
      id: "yx-masks",
      title: { tr: "Prototipler ve katsayılar", en: "Prototypes and coefficients" },
      intro: {
        tr: R`Ağ her görüntü için 32 prototip harita (160 × 160) üretir; bir nesnenin maskesi \(\sigma(\sum_k c_k P_k)\), kutuyla kırpılmış.`,
        en: R`For every image the network makes 32 prototype maps (160 × 160); an object's mask is \(\sigma(\sum_k c_k P_k)\), cropped by its box.`,
      },
      read: { tab: "theory", section: "y-masks" },
      questions: [
        {
          id: "m-numbers",
          kind: "number",
          q: { tr: "Bir görüntünün prototipleri toplam kaç sayıdır?", en: "How many numbers do an image's prototypes hold altogether?" },
          label: { tr: "sayı =", en: "numbers =" },
          answer: 819200,
          compute: () => 32 * 160 * 160,
          hint: { tr: "Prototip sayısı × her prototipin piksel sayısı.", en: "The number of prototypes × the pixels of each prototype." },
          mistakes: [
            {
              value: 25600,
              why: {
                tr: "Bu tek bir prototipin piksel sayısı (160 × 160). 32 prototip var.",
                en: "That is the pixels of one prototype (160 × 160). There are 32 prototypes.",
              },
            },
            {
              value: 13107200,
              why: {
                tr: "Prototipleri 640 × 640 aldınız; 640'ın dörtte biri olan 160 × 160'tırlar.",
                en: "You took the prototypes as 640 × 640; they are 160 × 160, a quarter of 640.",
              },
            },
          ],
          explain: {
            tr: R`\(32 \cdot 160 \cdot 160 = 819\,200\). Bunlar bir kez hesaplanır; her nesne yalnızca 32 katsayı ekler.`,
            en: R`\(32 \cdot 160 \cdot 160 = 819{,}200\). They are computed once; every object adds only 32 coefficients.`,
          },
        },
        {
          id: "m-metres",
          kind: "number",
          q: {
            tr: "0,3 m/piksel çözünürlüklü 650 × 650 bir kare 640 × 640'a küçültülüp modele veriliyor. Bir prototip pikseli zeminde kaç metredir?",
            en: "A 650 × 650 tile at 0.3 m per pixel is shrunk to 640 × 640 for the model. How many metres on the ground is one prototype pixel?",
          },
          label: { tr: "metre =", en: "metres =" },
          answer: 1.21875,
          tolerance: 0.01,
          compute: () => ((4 * 650) / 640) * 0.3,
          hint: {
            tr: "Bir prototip pikseli 4 model pikselidir; bir model pikseli de 650/640 kare pikselidir.",
            en: "One prototype pixel is 4 model pixels; one model pixel is 650/640 tile pixels.",
          },
          mistakes: [
            {
              value: 1.2,
              why: {
                tr: "Neredeyse: 650'den 640'a küçültmeyi unuttunuz, bir model pikseli 650/640 kare pikselidir.",
                en: "Almost: you left out the shrink from 650 to 640; one model pixel is 650/640 tile pixels.",
              },
            },
            {
              value: 0.3,
              why: {
                tr: "Bu bir kare pikseli. Prototipler 4 kat düşük çözünürlüklüdür.",
                en: "That is one tile pixel. The prototypes have a quarter of the resolution.",
              },
            },
          ],
          explain: {
            tr: R`\(4 \cdot \frac{650}{640} \cdot 0{,}3 \approx 1{,}22\) m. Maskenin ayrıntısı bu adımla sınırlıdır (Teori 8).`,
            en: R`\(4 \cdot \frac{650}{640} \cdot 0.3 \approx 1.22\) m. The mask's detail is limited to this step (Theory 8).`,
          },
        },
      ],
    },
    {
      id: "yx-train",
      title: { tr: "Eğitim", en: "Training" },
      intro: {
        tr: "Her nesneye adaylar atanır; kutu, sınıf ve maske kayıpları birlikte küçültülür. Önceden eğitilmiş ağırlıklar ve veri artırma büyük fark yapar.",
        en: "Candidates are assigned to every object; the box, class and mask losses are lowered together. Pretrained weights and augmentation make a large difference.",
      },
      read: { tab: "theory", section: "y-train" },
      questions: [
        {
          id: "r-sigmoid",
          kind: "choice",
          q: {
            tr: "YOLO sınıf skorları için neden softmax değil, her sınıfa ayrı bir sigmoid kullanır?",
            en: "Why does YOLO use a separate sigmoid for every class score instead of a softmax?",
          },
          options: [
            {
              text: { tr: "Sigmoid'in türevi daha kolay hesaplandığı için", en: "Because the sigmoid's derivative is easier to compute" },
              why: {
                tr: "İkisinin de türevi kolaydır (çapraz entropiyle ikisinde de p − y). Sebep türev değil, olasılıkların toplamı.",
                en: "Both have easy derivatives (with cross-entropy, p − y in both). The reason is not the derivative but the sum of the probabilities.",
              },
            },
            {
              text: { tr: "Softmax 8400 aday için çok yavaş olduğu için", en: "Because softmax is too slow for 8400 candidates" },
              why: {
                tr: "Softmax birkaç sınıf üzerinde çok ucuzdur. Sorun, olasılıkları toplamı 1 olmaya zorlamasıdır.",
                en: "A softmax over a few classes is very cheap. The problem is that it forces the probabilities to sum to 1.",
              },
            },
            {
              text: {
                tr: "Softmax skorların toplamını 1 yapar; oysa bir adayın hiçbir nesneye ait olmaması (boş zemin) olağandır",
                en: "Softmax makes the scores sum to 1, yet a candidate belonging to no object (empty ground) is the usual case",
              },
            },
            {
              text: { tr: "Sigmoid'in çıktısı 0 ile 1 arasında olmadığı için", en: "Because the sigmoid's output is not between 0 and 1" },
              why: { tr: "Sigmoid'in çıktısı tam da 0 ile 1 arasındadır.", en: "The sigmoid's output is exactly between 0 and 1." },
            },
          ],
          answer: 2,
          explain: {
            tr: '8400 adayın çoğu boş zemine bakar. Ayrı sigmoid\'lerle bir aday her sınıfa "hayır" diyebilir; softmax ise olasılığı mutlaka bir sınıfa dağıtırdı.',
            en: 'Most of the 8400 candidates look at empty ground. With separate sigmoids a candidate can say "no" to every class; a softmax would always hand the probability to some class.',
          },
        },
        {
          id: "r-flip",
          kind: "choice",
          q: {
            tr: "Hangi veri artırma sokak fotoğrafları için kötü bir fikirdir ama kuş bakışı uydu kareleri için uygundur?",
            en: "Which augmentation is a bad idea for street photos but fine for top-down satellite tiles?",
          },
          options: [
            {
              text: { tr: "Parlaklığı biraz değiştirmek", en: "Changing the brightness a little" },
              why: { tr: "Bu ikisinde de uygundur: ışık her fotoğrafta değişir.", en: "That suits both: light changes in every photo." },
            },
            {
              text: { tr: "Görüntüyü yatay çevirmek (sol-sağ)", en: "Flipping the image horizontally (left-right)" },
              why: {
                tr: "Sol-sağ çevirme sokak fotoğraflarında da uygundur: ayna görüntüsündeki bir araba yine arabadır.",
                en: "A left-right flip suits street photos too: a mirrored car is still a car.",
              },
            },
            {
              text: { tr: "Görüntüyü dikey çevirmek ya da 90° döndürmek", en: "Flipping the image vertically or turning it by 90°" },
            },
          ],
          answer: 2,
          explain: {
            tr: 'Sokak fotoğrafında gökyüzü yukarıdadır; baş aşağı bir araba gerçekte görülmez. Kuş bakışında "yukarı" yoktur: aynı mahalle her yönde aynı mahalledir. Bu yüzden bu bölümde flipud = 0,5 ve degrees = 90 kullanıldı.',
            en: 'In a street photo the sky is on top; an upside-down car is never seen. A top-down view has no "up": the same neighbourhood is the same in every direction. That is why this chapter used flipud = 0.5 and degrees = 90.',
          },
        },
      ],
    },
    {
      id: "yx-metrics",
      title: { tr: "Kesinlik, duyarlılık, mAP", en: "Precision, recall, mAP" },
      intro: {
        tr: R`\(P = TP/(TP + FP)\): tespitlerin ne kadarı doğru. \(R = TP/(TP + FN)\): gerçek nesnelerin ne kadarı bulundu.`,
        en: R`\(P = TP/(TP + FP)\): how many of the detections are right. \(R = TP/(TP + FN)\): how many of the true objects were found.`,
      },
      read: { tab: "theory", section: "y-metrics" },
      questions: [
        {
          id: "p-precision",
          kind: "number",
          q: {
            tr: "Bir karede 50 gerçek bina var. Model 40 tespit yapıyor; 30'u doğru (IoU ≥ 0,5). Kesinlik kaçtır?",
            en: "A tile has 50 true buildings. The model makes 40 detections; 30 are correct (IoU ≥ 0.5). What is the precision?",
          },
          label: { tr: R`\(P =\)`, en: R`\(P =\)` },
          answer: 0.75,
          tolerance: 0.005,
          compute: () => 30 / (30 + 10),
          hint: { tr: "Doğru tespitleri tespitlerin hepsine bölün.", en: "Divide the correct detections by all detections." },
          mistakes: [
            {
              value: 0.6,
              why: {
                tr: "Bu duyarlılık: doğru tespitleri gerçek binalara (50) böldünüz. Kesinlik tespitlere (40) böler.",
                en: "That is the recall: you divided the correct detections by the true buildings (50). Precision divides by the detections (40).",
              },
            },
            {
              value: 0.5,
              why: {
                tr: "30'u 60'a (TP + FP + FN) böldünüz. Kaçan binalar kesinliğe girmez.",
                en: "You divided 30 by 60 (TP + FP + FN). Missed buildings do not enter precision.",
              },
            },
          ],
          explain: {
            tr: R`\(TP = 30\), \(FP = 40 - 30 = 10\): \(P = 30/40 = 0{,}75\).`,
            en: R`\(TP = 30\), \(FP = 40 - 30 = 10\): \(P = 30/40 = 0.75\).`,
          },
        },
        {
          id: "p-recall",
          kind: "number",
          q: {
            tr: "Aynı karede (50 gerçek bina, 40 tespit, 30 doğru) duyarlılık kaçtır?",
            en: "On the same tile (50 true buildings, 40 detections, 30 correct), what is the recall?",
          },
          label: { tr: R`\(R =\)`, en: R`\(R =\)` },
          answer: 0.6,
          tolerance: 0.005,
          compute: () => 30 / 50,
          hint: {
            tr: "Doğru tespitleri gerçek nesnelerin sayısına bölün.",
            en: "Divide the correct detections by the number of true objects.",
          },
          mistakes: [
            {
              value: 0.75,
              why: {
                tr: "Bu kesinlik. Duyarlılık gerçek binalara (50) böler: kaçan 20 bina payda girer.",
                en: "That is the precision. Recall divides by the true buildings (50): the 20 missed ones count.",
              },
            },
            {
              value: 0.4,
              why: {
                tr: "Kaçan binaların oranını (20/50) buldunuz; duyarlılık bulunanların oranıdır.",
                en: "You found the share of missed buildings (20/50); recall is the share found.",
              },
            },
          ],
          explain: {
            tr: R`\(FN = 50 - 30 = 20\): \(R = 30/50 = 0{,}6\). Model binaların %60'ını buldu.`,
            en: R`\(FN = 50 - 30 = 20\): \(R = 30/50 = 0.6\). The model found 60% of the buildings.`,
          },
        },
        {
          id: "p-threshold",
          kind: "choice",
          q: {
            tr: "Güven eşiğini 0,25'ten 0,05'e indirirsek genellikle ne olur?",
            en: "What usually happens if the confidence threshold drops from 0.25 to 0.05?",
          },
          options: [
            {
              text: { tr: "Kesinlik ve duyarlılık birlikte artar", en: "Precision and recall both rise" },
              why: {
                tr: "Eşik ikisini birbirine karşı değiştirir; ikisini birden artıran şey daha iyi bir modeldir, eşik değil.",
                en: "The threshold trades one against the other; what raises both is a better model, not a threshold.",
              },
            },
            {
              text: {
                tr: "Duyarlılık artar ya da aynı kalır, kesinlik genellikle düşer",
                en: "Recall rises or stays, precision usually falls",
              },
            },
            {
              text: { tr: "mAP değişir", en: "The mAP changes" },
              why: {
                tr: "mAP bütün eşiklerin üzerinden hesaplanır (PR eğrisinin altındaki alan); tek bir eşik seçmek onu değiştirmez.",
                en: "mAP is computed over all thresholds (the area under the PR curve); picking one threshold does not change it.",
              },
            },
            {
              text: { tr: "Kesinlik artar, duyarlılık düşer", en: "Precision rises, recall falls" },
              why: {
                tr: "Bu eşiği yükseltince olur. Düşük eşikte daha çok tespit kalır: daha çok bina bulunur, ama yanlışlar da çoğalır.",
                en: "That happens when the threshold goes up. A low threshold keeps more detections: more buildings are found, but more mistakes too.",
              },
            },
          ],
          answer: 1,
          explain: {
            tr: "Şanghay karesinde eşiği geçen aday 107'den 281'e, NMS sonrası nesne 15'ten 47'ye çıkar: yeni tespitlerin çoğu yanlıştır.",
            en: "On the Shanghai tile the candidates above the threshold go from 107 to 281, the objects after NMS from 15 to 47: most of the new detections are wrong.",
          },
        },
      ],
    },
    {
      id: "yx-read",
      title: { tr: "Sonucu okumak", en: "Reading the result" },
      intro: {
        tr: R`Alan = piksel sayısı × bir pikselin zemindeki alanı (\(g_x \cdot g_y\)). Sınırdaki bir kayma küçük binalarda büyük bağıl hata yapar.`,
        en: R`Area = number of pixels × the ground area of one pixel (\(g_x \cdot g_y\)). A shift of the outline makes a large relative error on small buildings.`,
      },
      read: { tab: "theory", section: "y-read" },
      questions: [
        {
          id: "a-m2",
          kind: "number",
          q: {
            tr: "Bir binanın maskesi 0,3 m/piksel çözünürlüklü bir görüntüde 2500 piksel kaplıyor. Alanı kaç m²'dir?",
            en: "A building's mask covers 2500 pixels of an image at 0.3 m per pixel. What is its area in m²?",
          },
          label: { tr: "alan (m²) =", en: "area (m²) =" },
          answer: 225,
          tolerance: 0.5,
          compute: () => 2500 * 0.3 * 0.3,
          hint: { tr: "Bir piksel zeminde 0,3 m × 0,3 m'lik bir karedir.", en: "A pixel is a 0.3 m × 0.3 m square on the ground." },
          mistakes: [
            {
              value: 750,
              why: {
                tr: "0,3 ile bir kez çarptınız; bu uzunluktur. Alan için pikselin alanı gerekir: 0,3 × 0,3 = 0,09 m².",
                en: "You multiplied by 0.3 once; that is a length. Area needs the pixel's area: 0.3 × 0.3 = 0.09 m².",
              },
            },
            {
              value: 2500,
              why: {
                tr: "Bu piksel sayısı. Her piksel zeminde 0,09 m²'dir.",
                en: "That is the pixel count. Every pixel is 0.09 m² on the ground.",
              },
            },
          ],
          explain: {
            tr: R`\(2500 \cdot 0{,}3^2 = 2500 \cdot 0{,}09 = 225\ \mathrm{m^2}\).`,
            en: R`\(2500 \cdot 0.3^2 = 2500 \cdot 0.09 = 225\ \mathrm{m^2}\).`,
          },
        },
        {
          id: "a-shift",
          kind: "number",
          q: {
            tr: "10 m × 10 m bir binanın maske sınırı her yandan 1 m dışarı kayıyor. Alan yüzde kaç büyür?",
            en: "The mask outline of a 10 m × 10 m building moves out by 1 m on every side. By what percentage does the area grow?",
          },
          label: { tr: "artış (%) =", en: "increase (%) =" },
          answer: 44,
          tolerance: 0.5,
          compute: () => (((10 + 2 * 1) / 10) ** 2 - 1) * 100,
          hint: {
            tr: "Her yandan 1 m: kenar 10'dan 12'ye çıkar. Alanı karşılaştırın.",
            en: "1 m on every side: the side goes from 10 to 12. Compare the areas.",
          },
          mistakes: [
            {
              value: 20,
              why: {
                tr: "Bu kenarın büyümesi (%20). Alan kenarın karesiyle büyür: 1,2² = 1,44.",
                en: "That is how much the side grows (20%). The area grows with its square: 1.2² = 1.44.",
              },
            },
            {
              value: 40,
              why: {
                tr: "Bu, \\(4\\delta/a\\) yaklaşımı; küçük kaymalar içindir. Tam hesap: \\(12^2/10^2 - 1 = 0{,}44\\).",
                en: "That is the \\(4\\delta/a\\) approximation, for small shifts. Exactly: \\(12^2/10^2 - 1 = 0.44\\).",
              },
            },
            {
              value: 21,
              why: {
                tr: "Kenar her yandan 1 m, yani toplam 2 m büyür: 10'dan 12'ye.",
                en: "The side grows by 1 m on each side, 2 m in all: from 10 to 12.",
              },
            },
          ],
          explain: {
            tr: R`\((12/10)^2 - 1 = 0{,}44\): %44. Aynı 1 m, 100 m'lik bir depoda yalnızca %4'tür.`,
            en: R`\((12/10)^2 - 1 = 0.44\): 44%. The same 1 m is only 4% on a 100 m warehouse.`,
          },
        },
        {
          id: "a-roof",
          kind: "choice",
          q: {
            tr: "Maskeden bulunan alan neden bir binanın taban alanından farklı olabilir?",
            en: "Why can the area found from a mask differ from a building's footprint?",
          },
          options: [
            {
              text: {
                tr: "Uydu çatıyı görür: saçak taşmaları ve eğik çekimde görünen cephe de maskeye girer",
                en: "The satellite sees the roof: the eaves, and on an oblique shot the facade, fall into the mask too",
              },
            },
            {
              text: { tr: "Model binanın kat sayısını alana kattığı için", en: "Because the model adds the number of storeys to the area" },
              why: {
                tr: "Tepeden çekilmiş bir görüntüde kat sayısı görünmez; maske yalnızca görünen yüzeyi kaplar.",
                en: "A top-down image does not show the storeys; the mask covers only the visible surface.",
              },
            },
            {
              text: { tr: "Piksellerin hepsi aynı alanı temsil etmediği için", en: "Because not every pixel stands for the same area" },
              why: {
                tr: "Aynı karede her piksel aynı zemin alanıdır (bizim karelerde Paris'te 0,20 m × 0,30 m). Fark çatı ile taban arasındadır.",
                en: "Within a tile every pixel is the same ground area (0.20 m × 0.30 m in Paris for our tiles). The difference is between roof and footprint.",
              },
            },
          ],
          answer: 0,
          explain: {
            tr: "Bu yüzden karedeki yapılaşma oranı TAKS'a benzer ama aynı değildir: çatı izdüşümüdür ve kare bir parsel değildir.",
            en: "That is why the tile's built-up ratio resembles TAKS (the building coverage ratio) but is not the same: it is the roof's projection, and a tile is not a parcel.",
          },
        },
        {
          id: "a-city",
          kind: "choice",
          q: {
            tr: "Bu bölümün modelini yeniden eğitmeden İstanbul'un uydu görüntülerinde kullanırsak ne beklenir?",
            en: "What should we expect if we use this chapter's model on satellite images of Istanbul without retraining it?",
          },
          options: [
            {
              text: { tr: "Aynı mAP: binalar her şehirde benzer görünür", en: "The same mAP: buildings look alike in every city" },
              why: {
                tr: "Çatı malzemesi, renkler, yapı tipi ve dokusu şehirden şehre değişir; model yalnızca eğitimde gördüklerini iyi bilir.",
                en: "Roof materials, colours, building types and the urban fabric change from city to city; the model only knows well what it saw in training.",
              },
            },
            {
              text: {
                tr: "Model hiç çalışmaz, çünkü İstanbul eğitim verisinde yok",
                en: "The model does not work at all, because Istanbul is not in the training data",
              },
              why: {
                tr: "Model bir şehrin adını bilmez; benzer çatıları yine bulur. Sorun sıfır değil, düşen bir başarıdır.",
                en: "The model does not know a city's name; it still finds roofs that look familiar. The problem is not zero but lower success.",
              },
            },
            {
              text: {
                tr: "Başarı muhtemelen düşer; İstanbul'dan etiketli birkaç yüz kareyle ince ayar yapmak gerekir",
                en: "Success will probably drop; it needs fine-tuning with a few hundred labelled tiles from Istanbul",
              },
            },
          ],
          answer: 2,
          explain: {
            tr: "Buna alan değişimi (domain shift) denir. Önceden eğitilmiş ağırlıkların COCO'dan uydu karelerine taşınması gibi, buradaki model de İstanbul'a az veriyle uyarlanabilir.",
            en: "This is called domain shift. Just as the pretrained weights carried over from COCO to satellite tiles, this model can be adapted to Istanbul with little data.",
          },
        },
      ],
    },
    {
      id: "yx-outputs",
      title: { tr: "Eğitim çıktılarını okumak", en: "Reading the training outputs" },
      intro: {
        tr: R`Karışıklık matrisinde satırlar tahmin, sütunlar gerçektir (Ultralytics'in düzeni). \(F_1 = 2PR/(P+R)\); doğruluk \(= (TP+TN)/\text{hepsi}\).`,
        en: R`In the confusion matrix rows are the prediction, columns the truth (Ultralytics' layout). \(F_1 = 2PR/(P+R)\); accuracy \(= (TP+TN)/\text{all}\).`,
      },
      read: { tab: "outputs", section: "yr-cm-intro" },
      questions: [
        {
          id: "o-accuracy",
          kind: "number",
          q: {
            tr: "Deprem sonrası taramada 100 bina var. Model: TP = 7, FP = 6, FN = 3, TN = 84. Doğruluk kaçtır?",
            en: "A post-earthquake survey covers 100 buildings. The model: TP = 7, FP = 6, FN = 3, TN = 84. What is the accuracy?",
          },
          label: { tr: "doğruluk =", en: "accuracy =" },
          answer: 0.91,
          tolerance: 0.005,
          compute: () => (7 + 84) / 100,
          hint: {
            tr: "Doğru olan iki hücreyi (TP ve TN) toplayıp bütün binalara bölün.",
            en: "Add the two correct cells (TP and TN) and divide by all buildings.",
          },
          mistakes: [
            {
              value: 0.07,
              why: {
                tr: 'Yalnızca TP\'yi saydınız; doğru "sağlam" kararları (TN) da doğrudur.',
                en: 'You counted only TP; the correct "sound" calls (TN) are right too.',
              },
            },
            {
              value: 0.7,
              why: {
                tr: "Bu duyarlılık: TP / (TP + FN). Doğruluk bütün binalara bakar.",
                en: "That is the recall: TP / (TP + FN). Accuracy looks at all buildings.",
              },
            },
          ],
          explain: {
            tr: R`\((7 + 84)/100 = 0{,}91\). Yüksek görünüyor; ama bir sonraki soru bu sayının neden yanıltabileceğini gösteriyor.`,
            en: R`\((7 + 84)/100 = 0.91\). It looks high; the next question shows why this number can mislead.`,
          },
        },
        {
          id: "o-trap",
          kind: "choice",
          q: {
            tr: 'Aynı 100 binadan 10\'u hasarlı. Bir "model" hiçbir şeye bakmadan hepsine "sağlam" diyor. Hangisi doğru?',
            en: '10 of the same 100 buildings are damaged. A "model" calls them all "sound" without looking. Which is true?',
          },
          options: [
            {
              text: { tr: "Doğruluğu %90; iyi bir model", en: "Its accuracy is 90%; a good model" },
              why: {
                tr: "Doğruluk gerçekten %90 ama model tek bir hasarlı binayı bile bulmuyor. Dengesiz veride doğruluk yanıltır.",
                en: "The accuracy is indeed 90%, but the model finds not a single damaged building. On imbalanced data, accuracy misleads.",
              },
            },
            {
              text: {
                tr: "Doğruluğu %90, duyarlılığı 0: hiçbir hasarlı binayı bulmuyor",
                en: "Its accuracy is 90% and its recall 0: it finds no damaged building",
              },
            },
            {
              text: { tr: "Kesinliği 1", en: "Its precision is 1" },
              why: {
                tr: 'Model hiç "hasarlı" demediği için TP + FP = 0: kesinlik tanımsızdır (0/0).',
                en: 'The model never says "damaged", so TP + FP = 0: precision is undefined (0/0).',
              },
            },
            {
              text: { tr: "Özgüllüğü 0", en: "Its specificity is 0" },
              why: {
                tr: 'Tersine: 90 sağlam binanın hepsine "sağlam" dedi, özgüllük TN / (TN + FP) = 90/90 = 1.',
                en: 'The opposite: it called all 90 sound buildings "sound"; specificity TN / (TN + FP) = 90/90 = 1.',
              },
            },
          ],
          answer: 1,
          explain: {
            tr: R`TP = 0, FN = 10, FP = 0, TN = 90. Doğruluk \(90/100\), duyarlılık \(0/10 = 0\). Bir önceki modelden (%91) neredeyse farksız bir doğruluk, ama işe yaramaz bir model.`,
            en: R`TP = 0, FN = 10, FP = 0, TN = 90. Accuracy \(90/100\), recall \(0/10 = 0\). An accuracy hardly different from the previous model's (91%), but a useless model.`,
          },
        },
        {
          id: "o-f1",
          kind: "number",
          q: {
            tr: "Bir modelin kesinliği 0,9, duyarlılığı 0,1. F1 kaçtır?",
            en: "A model has precision 0.9 and recall 0.1. What is its F1?",
          },
          label: { tr: R`\(F_1 =\)`, en: R`\(F_1 =\)` },
          answer: 0.18,
          tolerance: 0.005,
          compute: () => (2 * 0.9 * 0.1) / (0.9 + 0.1),
          hint: { tr: R`\(F_1 = 2PR/(P + R)\).`, en: R`\(F_1 = 2PR/(P + R)\).` },
          mistakes: [
            {
              value: 0.5,
              why: {
                tr: "Aritmetik ortalamayı aldınız. F1 harmonik ortalamadır: küçük olana yakın durur.",
                en: "You took the arithmetic mean. F1 is the harmonic mean: it stays close to the smaller one.",
              },
            },
            {
              value: 0.09,
              why: { tr: "2 katsayısını unuttunuz: \\(2PR/(P+R)\\).", en: "You forgot the factor 2: \\(2PR/(P+R)\\)." },
            },
          ],
          explain: {
            tr: R`\(2 \cdot 0{,}9 \cdot 0{,}1 / (0{,}9 + 0{,}1) = 0{,}18\). Kesinlik ne kadar yüksek olursa olsun, duyarlılık küçükse F1 küçük kalır.`,
            en: R`\(2 \cdot 0.9 \cdot 0.1 / (0.9 + 0.1) = 0.18\). However high the precision, a small recall keeps F1 small.`,
          },
        },
        {
          id: "o-corner",
          kind: "choice",
          q: {
            tr: "YOLO'nun confusion_matrix.png dosyasında arka plan satırı ile arka plan sütununun kesiştiği hücre neden boştur?",
            en: "In YOLO's confusion_matrix.png, why is the cell where the background row and column meet empty?",
          },
          options: [
            {
              text: { tr: "Model arka planı hiç tahmin etmez", en: "The model never predicts background" },
              why: {
                tr: "Arka plan satırı doludur: orası kaçırılan nesnelerdir (FN). Boş olan yalnızca köşe hücresi.",
                en: "The background row is not empty: it holds the missed objects (FN). Only the corner cell is empty.",
              },
            },
            {
              text: { tr: "Eşik 0,001 olduğu için", en: "Because the threshold is 0.001" },
              why: {
                tr: "Eşik FP sayısını değiştirir, köşe hücresini değil; o hücre her eşikte boştur.",
                en: "The threshold changes the number of FPs, not the corner cell; that cell is empty at every threshold.",
              },
            },
            {
              text: {
                tr: 'O hücre TN olurdu, ama bir görüntüdeki "boş yerler" sayılamaz',
                en: 'That cell would be TN, but the "empty places" of an image cannot be counted',
              },
            },
            {
              text: { tr: "Doğrulama karelerinin hepsinde nesne olduğu için", en: "Because every validation tile has objects" },
              why: {
                tr: "Doğrulamada nesnesiz 13 kare var; yine de köşe boştur. Neden sayılamadığıdır.",
                en: "Validation has 13 tiles without objects; the corner is empty all the same. It is about what cannot be counted.",
              },
            },
          ],
          answer: 2,
          explain: {
            tr: 'Tespitte bir "pozitif" bir kutudur; olmayan kutular sayılamaz. Bu yüzden tespitte TN, doğruluk ve özgüllük kullanılmaz; kesinlik ve duyarlılık TN\'siz hesaplanır.',
            en: 'In detection a "positive" is a box; boxes that were never drawn cannot be counted. So detection uses no TN, accuracy or specificity; precision and recall need no TN.',
          },
        },
        {
          id: "o-real-precision",
          kind: "number",
          q: {
            tr: 'Bizim modelin matrisinde, eşik 0,25\'te "bina" satırı: 1520 (gerçek bina), 8 (gerçek saha), 687 (arka plan). Binaların kesinliği kaçtır?',
            en: "In our model's matrix at threshold 0.25, the \"building\" row is: 1520 (true building), 8 (true field), 687 (background). What is the buildings' precision?",
          },
          label: { tr: R`\(P =\)`, en: R`\(P =\)` },
          answer: 0.686,
          tolerance: 0.005,
          compute: () => 1520 / (1520 + 8 + 687),
          hint: {
            tr: 'Satır, modelin "bina" dediği her şeydir. Doğru olanı (köşegen) satırın toplamına bölün.',
            en: "The row is everything the model called a building. Divide the correct part (the diagonal) by the row's total.",
          },
          mistakes: [
            {
              value: 0.653,
              why: {
                tr: "Bu duyarlılık: 1520'yi 2329 gerçek binaya böldünüz. Kesinlik satıra, duyarlılık sütuna bakar.",
                en: "That is the recall: you divided 1520 by the 2329 true buildings. Precision reads the row, recall the column.",
              },
            },
          ],
          explain: {
            tr: R`\(1520/(1520 + 8 + 687) = 1520/2215 \approx 0{,}686\). Ultralytics'in tablosundaki 0,72 biraz farklıdır: o, eşik 0,26'da ve sınıfa göre eşleştirmeyle ölçülür.`,
            en: R`\(1520/(1520 + 8 + 687) = 1520/2215 \approx 0.686\). The 0.72 in Ultralytics' table differs a little: it is measured at threshold 0.26 with class-aware matching.`,
          },
        },
        {
          id: "o-real-recall",
          kind: "number",
          q: {
            tr: 'Aynı matriste "bina" sütunu: 1520 (bina dendi), 2 (saha dendi), 807 (arka plan: bulunamadı). Binaların duyarlılığı kaçtır?',
            en: 'In the same matrix the "building" column is: 1520 (called building), 2 (called field), 807 (background: not found). What is the buildings\' recall?',
          },
          label: { tr: R`\(R =\)`, en: R`\(R =\)` },
          answer: 0.653,
          tolerance: 0.005,
          compute: () => 1520 / (1520 + 2 + 807),
          hint: {
            tr: "Sütun, gerçekte bina olan her şeydir. Doğru bulunanı sütunun toplamına bölün.",
            en: "The column is everything that really is a building. Divide the part found correctly by the column's total.",
          },
          mistakes: [
            {
              value: 0.686,
              why: { tr: "Bu kesinlik (satır). Duyarlılık sütuna bakar.", en: "That is the precision (the row). Recall reads the column." },
            },
          ],
          explain: {
            tr: R`\(1520/(1520 + 2 + 807) = 1520/2329 \approx 0{,}653\): her üç binadan ikisi bulundu. Sütunun toplamı doğrulamadaki bina sayısıdır (2329).`,
            en: R`\(1520/(1520 + 2 + 807) = 1520/2329 \approx 0.653\): two in three buildings were found. The column's total is the number of buildings in validation (2329).`,
          },
        },
        {
          id: "o-thousands",
          kind: "choice",
          q: {
            tr: "Eğitim bitince yazılan confusion_matrix.png'de bina satırının arka plan hücresinde 24.000'den fazla FP var. En olası açıklama?",
            en: "The confusion_matrix.png written at the end of training has more than 24,000 FPs in the building row's background cell. The most likely reason?",
          },
          options: [
            {
              text: { tr: "Model çok kötü; yeniden eğitilmeli", en: "The model is very bad; retrain it" },
              why: {
                tr: "Aynı modelin mAP50'si 0,50 ve eşik 0,25'te yalnızca 687 FP'si var. Sayı modelden değil eşikten geliyor.",
                en: "The same model has a mAP50 of 0.50 and only 687 FPs at threshold 0.25. The number comes from the threshold, not the model.",
              },
            },
            {
              text: {
                tr: "Matris, doğrulamanın güven eşiği olan 0,001 ile dolduruldu",
                en: "The matrix was filled at validation's confidence threshold, 0.001",
              },
            },
            {
              text: { tr: "NMS çalışmadı", en: "NMS did not run" },
              why: {
                tr: "NMS çalıştı (IoU 0,7). Kalan kutuların çoğunun skoru 0,25'in altında; 0,001 eşiği hepsini sayar.",
                en: "NMS ran (IoU 0.7). Most of the boxes left score below 0.25; a threshold of 0.001 counts them all.",
              },
            },
            {
              text: { tr: "IoU eşiği çok yüksek", en: "The IoU threshold is too high" },
              why: {
                tr: "0,5 olağan eşiktir ve mAP50 ile aynıdır. FP'lerin çoğu hiçbir nesneye yakın değil.",
                en: "0.5 is the usual threshold, the same as mAP50's. Most FPs are near no object at all.",
              },
            },
          ],
          answer: 1,
          explain: {
            tr: R`mAP bütün PR eğrisinden hesaplandığı için doğrulama en düşük skorlu tespitleri de tutar ve matris bu eşikle dolar. Kullanacağınız eşikte matrisi görmek için doğrulamayı o eşikle tekrarlayın: <code>yolo val model=best.pt conf=0.25</code>.`,
            en: R`Since mAP is computed from the whole PR curve, validation keeps even the lowest-scoring detections, and the matrix fills at that threshold. To see the matrix at the threshold you will use, repeat the validation at it: <code>yolo val model=best.pt conf=0.25</code>.`,
          },
        },
        {
          id: "o-normalized",
          kind: "choice",
          q: {
            tr: "confusion_matrix_normalized.png'de her sütun 1'e toplanacak biçimde bölünür. Köşegendeki bir sayı neyi gösterir?",
            en: "In confusion_matrix_normalized.png every column is divided so it adds up to 1. What does a number on the diagonal show?",
          },
          options: [
            { text: { tr: "O sınıfın duyarlılığını", en: "That class's recall" } },
            {
              text: { tr: "O sınıfın kesinliğini", en: "That class's precision" },
              why: {
                tr: "Kesinlik için satıra bölmek gerekirdi (modelin o sınıf dediği her şey).",
                en: "Precision would need dividing by the row (everything the model called that class).",
              },
            },
            {
              text: { tr: "O sınıfın AP'sini", en: "That class's AP" },
              why: {
                tr: "AP bütün eşiklerin üzerinden hesaplanır; matris tek bir eşiktedir.",
                en: "AP is computed over all thresholds; the matrix is at one threshold.",
              },
            },
            {
              text: { tr: "Modelin doğruluğunu", en: "The model's accuracy" },
              why: { tr: "Tespitte TN olmadığı için doğruluk hesaplanamaz.", en: "Without TN, detection cannot compute accuracy." },
            },
          ],
          answer: 0,
          explain: {
            tr: "Sütun, gerçekte o sınıftan olan her şeydir; köşegen bunun doğru bulunan payıdır: TP / (TP + FN), yani duyarlılık.",
            en: "A column is everything that really is that class; the diagonal is the share found correctly: TP / (TP + FN), the recall.",
          },
        },
        {
          id: "o-overfit",
          kind: "choice",
          q: {
            tr: "Bir eğitimin results.png'sinde eğitim kaybı sonuna kadar iniyor, doğrulama kaybı ise 60. epoch'tan sonra yükseliyor. Ne olmuştur, ne yapılır?",
            en: "In a run's results.png the training loss falls to the end, while the validation loss rises after epoch 60. What happened, and what to do?",
          },
          options: [
            {
              text: { tr: "Model öğrenmeyi sürdürüyor; last.pt kullanılır", en: "The model keeps learning; use last.pt" },
              why: {
                tr: "Doğrulama kaybının yükselmesi, eğitim karelerinin ezberlendiğini gösterir; son epoch en iyisi değildir.",
                en: "A rising validation loss shows the training tiles are being memorised; the last epoch is not the best.",
              },
            },
            {
              text: { tr: "Öğrenme oranı çok küçük; büyütülür", en: "The learning rate is too small; raise it" },
              why: {
                tr: "Küçük öğrenme oranı iki kaybı birlikte yavaşlatır; ayrışma ezberlemenin işaretidir.",
                en: "A small learning rate slows both losses together; their parting is the sign of memorising.",
              },
            },
            {
              text: { tr: "Doğrulama verisi bozuk", en: "The validation data is broken" },
              why: {
                tr: "Bozuk doğrulama verisi baştan yüksek bir kayıp verirdi; 60 epoch boyunca birlikte inmezlerdi.",
                en: "Broken validation data would give a high loss from the start; the two would not fall together for 60 epochs.",
              },
            },
            {
              text: {
                tr: "Ezberleme (aşırı uyum) başlamış; best.pt kullanılır, erken durdurma ya da daha çok veri düşünülür",
                en: "Memorising (overfitting) has started; use best.pt and consider early stopping or more data",
              },
            },
          ],
          answer: 3,
          explain: {
            tr: "Eğitim kaybı inerken doğrulama kaybının yükselmesi, modelin genellemek yerine eğitim örneklerini ezberlediğini gösterir. best.pt doğrulamada en iyi epoch'u zaten saklar; patience ile erken durdurma boşa geçen epoch'ları keser.",
            en: "A validation loss that rises while the training loss falls shows the model memorising the training examples instead of generalising. best.pt already keeps the best epoch on validation; early stopping with patience cuts the wasted epochs.",
          },
        },
        {
          id: "o-sem-iou",
          kind: "number",
          q: {
            tr: 'Arazi örtüsü matrisinde "yol" sınıfı (milyon piksel): TP = 3,33, satırın geri kalanı (FP) = 1,92, sütunun geri kalanı (FN) = 2,57. Yolun IoU\'su kaçtır?',
            en: 'In the land-cover matrix, the class "road" (million pixels): TP = 3.33, the rest of its row (FP) = 1.92, the rest of its column (FN) = 2.57. What is the road\'s IoU?',
          },
          label: { tr: "IoU =", en: "IoU =" },
          answer: 0.426,
          tolerance: 0.005,
          compute: () => 3.33 / (3.33 + 1.92 + 2.57),
          hint: { tr: R`\(\text{IoU} = TP/(TP + FP + FN)\).`, en: R`\(\text{IoU} = TP/(TP + FP + FN)\).` },
          mistakes: [
            {
              value: 3.33 / (3.33 + 1.92),
              why: { tr: "Bu kesinlik: FN'yi unuttunuz.", en: "That is the precision: you left out FN." },
            },
            {
              value: 3.33 / (3.33 + 2.57),
              why: { tr: "Bu duyarlılık: FP'yi unuttunuz.", en: "That is the recall: you left out FP." },
            },
          ],
          explain: {
            tr: R`\(3{,}33/(3{,}33 + 1{,}92 + 2{,}57) = 3{,}33/7{,}82 \approx 0{,}426\). Kesişim TP'dir; birleşim, \"yol\" denen ve gerçekte yol olan piksellerin hepsidir (TP + FP + FN). Bu, Eğitim sekmesindeki yol IoU'sudur.`,
            en: R`\(3.33/(3.33 + 1.92 + 2.57) = 3.33/7.82 \approx 0.426\). The intersection is TP; the union is every pixel called road or really road (TP + FP + FN). This is the road IoU of the Training tab.`,
          },
        },
      ],
    },
    {
      id: "yx-basics",
      title: { tr: "Üç sınıf, AP ve mAP elle", en: "Three classes, AP and mAP by hand" },
      intro: {
        tr: R`Kedi, köpek, fare: 30 fotoğraf. Satırlar tahmin, sütunlar gerçek. Kedi satırı 7, 2, 1; köpek satırı 2, 9, 0; fare satırı 1, 1, 7.`,
        en: R`Cat, dog, mouse: 30 photos. Rows are the prediction, columns the truth. The cat row is 7, 2, 1; the dog row 2, 9, 0; the mouse row 1, 1, 7.`,
      },
      read: { tab: "outputs", section: "yr-cm-multi" },
      questions: [
        {
          id: "b-dog-precision",
          kind: "number",
          q: {
            tr: "Köpek sınıfının kesinliği kaçtır?",
            en: "What is the dog class's precision?",
          },
          label: { tr: R`\(P =\)`, en: R`\(P =\)` },
          answer: 0.818,
          tolerance: 0.005,
          compute: () => 9 / (2 + 9 + 0),
          hint: {
            tr: 'Kesinlik satıra bakar: köpek satırı, modelin "köpek" dediği her şeydir.',
            en: "Precision reads the row: the dog row is everything the model called a dog.",
          },
          mistakes: [
            {
              value: 0.75,
              why: {
                tr: "Bu duyarlılık: 9'u köpek sütununa (2 + 9 + 1 = 12) böldünüz. Kesinlik satıra böler.",
                en: "That is the recall: you divided 9 by the dog column (2 + 9 + 1 = 12). Precision divides by the row.",
              },
            },
          ],
          explain: {
            tr: R`Köpek satırı 2, 9, 0: model 11 fotoğrafa "köpek" dedi, 9'u doğru. \(P = 9/11 \approx 0{,}818\).`,
            en: R`The dog row is 2, 9, 0: the model called 11 photos "dog", 9 of them rightly. \(P = 9/11 \approx 0.818\).`,
          },
        },
        {
          id: "b-mouse-recall",
          kind: "number",
          q: {
            tr: "Fare sınıfının duyarlılığı kaçtır? (Fare sütunu: 1, 0, 7.)",
            en: "What is the mouse class's recall? (The mouse column: 1, 0, 7.)",
          },
          label: { tr: R`\(R =\)`, en: R`\(R =\)` },
          answer: 0.875,
          tolerance: 0.005,
          compute: () => 7 / (1 + 0 + 7),
          hint: {
            tr: "Duyarlılık sütuna bakar: fare sütunu, gerçekte fare olan her şeydir.",
            en: "Recall reads the column: the mouse column is everything that really is a mouse.",
          },
          mistakes: [
            {
              value: 0.778,
              why: {
                tr: "Bu kesinlik: 7'yi fare satırına (1 + 1 + 7 = 9) böldünüz. Duyarlılık sütuna böler.",
                en: "That is the precision: you divided 7 by the mouse row (1 + 1 + 7 = 9). Recall divides by the column.",
              },
            },
          ],
          explain: {
            tr: R`8 farenin 7'si bulundu: \(R = 7/8 = 0{,}875\). Bir fareye "kedi" dendi.`,
            en: R`7 of the 8 mice were found: \(R = 7/8 = 0.875\). One mouse was called a cat.`,
          },
        },
        {
          id: "b-cat-tn",
          kind: "number",
          q: {
            tr: "Kedi sınıfı için (bire karşı hepsi) TN kaçtır?",
            en: "For the cat class (one against the rest), what is TN?",
          },
          label: { tr: "TN =", en: "TN =" },
          answer: 17,
          compute: () => 30 - 7 - (2 + 1) - (2 + 1),
          hint: {
            tr: "TN, kedi satırının ve kedi sütununun dışında kalan dört hücredir.",
            en: "TN is the four cells outside both the cat row and the cat column.",
          },
          mistakes: [
            {
              value: 20,
              why: {
                tr: "Yalnızca TP'yi ve FP'yi çıkardınız; gerçek kedi olup başka bir şey denen 3 fotoğraf (FN) da TN değildir.",
                en: "You took away only TP and FP; the 3 real cats called something else (FN) are not TN either.",
              },
            },
            {
              value: 23,
              why: {
                tr: "Yalnızca TP'yi çıkardınız. FP (kedi satırının geri kalanı) ve FN (kedi sütununun geri kalanı) da çıkmalı.",
                en: "You took away only TP. FP (the rest of the cat row) and FN (the rest of the cat column) go too.",
              },
            },
          ],
          explain: {
            tr: R`TP = 7, FP = 2 + 1 = 3, FN = 2 + 1 = 3; TN = 30 − 7 − 3 − 3 = 17 = 9 + 0 + 1 + 7. Bunlar ne gerçekte kedi olan ne de "kedi" denen fotoğraflardır.`,
            en: R`TP = 7, FP = 2 + 1 = 3, FN = 2 + 1 = 3; TN = 30 − 7 − 3 − 3 = 17 = 9 + 0 + 1 + 7. These are the photos that are neither really cats nor called "cat".`,
          },
        },
        {
          id: "b-macro",
          kind: "choice",
          q: {
            tr: "28 kedi ve köpek fotoğrafını çok iyi ayıran ama 2 farenin hiçbirini bulamayan bir model için hangi ölçü bu sorunu gösterir?",
            en: "A model tells 28 cat and dog photos apart very well but finds neither of the 2 mice. Which score shows the problem?",
          },
          options: [
            {
              text: { tr: "Doğruluk", en: "Accuracy" },
              why: {
                tr: "Doğruluk 28/30'a yakın kalır: 2 fare toplamın küçük bir parçasıdır.",
                en: "The accuracy stays near 28/30: 2 mice are a small part of the total.",
              },
            },
            {
              text: { tr: "Mikro ortalama F1", en: "Micro-averaged F1" },
              why: {
                tr: "Tek etiketli sınıflamada mikro ortalama doğruluğa eşittir; o da sorunu saklar.",
                en: "With one label per photo the micro average equals the accuracy; it hides the problem too.",
              },
            },
            { text: { tr: "Makro ortalama F1", en: "Macro-averaged F1" } },
            {
              text: { tr: "Ağırlıklı ortalama F1", en: "Weighted-average F1" },
              why: {
                tr: "Ağırlıklı ortalamada fare sınıfının ağırlığı 2/30'dur; kötü sonucu neredeyse görünmez.",
                en: "In the weighted average the mouse class weighs 2/30; its poor result is almost invisible.",
              },
            },
          ],
          answer: 2,
          explain: {
            tr: "Makro ortalama her sınıfa eşit ağırlık verir: farenin F1'i 0 olduğu için ortalama üçte bir düşer. YOLO'nun mAP'i de bu yüzden makro bir ortalamadır.",
            en: "The macro average gives every class the same weight: the mouse's F1 of 0 pulls the mean down by a third. That is also why YOLO's mAP is a macro average.",
          },
        },
        {
          id: "b-ap",
          kind: "number",
          q: {
            tr: 'Model 8 fotoğrafı "kedi" skoruna göre sıralıyor: ✓ ✓ ✗ ✓ ✗ ✗ ✓ ✗ (✓: gerçekten kedi). Bütün kedilerin sayısı 4. Elle AP kaçtır?',
            en: 'The model ranks 8 photos by their "cat" score: ✓ ✓ ✗ ✓ ✗ ✗ ✓ ✗ (✓: really a cat). There are 4 cats in all. What is the AP by hand?',
          },
          label: { tr: "AP =", en: "AP =" },
          answer: 0.83,
          tolerance: 0.005,
          compute: () => (1 / 1 + 2 / 2 + 3 / 4 + 4 / 7) / 4,
          hint: {
            tr: "Her ✓'nin sırasındaki kesinliği (o ana kadarki ✓ sayısı / sıra) bulun, toplayıp 4'e bölün.",
            en: "Find the precision at the rank of every ✓ (the ✓ so far / the rank), add them up and divide by 4.",
          },
          mistakes: [
            {
              value: 0.5,
              why: {
                tr: "Bu, listenin sonundaki kesinlik (4/8). AP bütün listeyi, her ✓'deki kesinliği ortalar.",
                en: "That is the precision at the end of the list (4/8). AP averages the precision at every ✓ over the whole list.",
              },
            },
            {
              value: 0.698,
              why: {
                tr: "Sekiz satırın hepsinin kesinliğini ortaladınız. Yalnızca ✓ satırları sayılır.",
                en: "You averaged the precision of all eight rows. Only the ✓ rows count.",
              },
            },
          ],
          explain: {
            tr: R`✓ sıraları 1, 2, 4, 7: kesinlikler \(1/1, 2/2, 3/4, 4/7\). \(\text{AP} = (1 + 1 + 0{,}75 + 0{,}571)/4 \approx 0{,}83\). Ultralytics'in 101 noktalı yöntemi de 0,83 verir.`,
            en: R`The ✓ ranks are 1, 2, 4, 7: precisions \(1/1, 2/2, 3/4, 4/7\). \(\text{AP} = (1 + 1 + 0.75 + 0.571)/4 \approx 0.83\). Ultralytics' 101-point way also gives 0.83.`,
          },
        },
      ],
    },
    {
      id: "yx-model",
      title: { tr: "Eğitilmiş modelle deneyler", en: "Experiments with the trained model" },
      intro: {
        tr: "Önce tahmininizi seçin; sonra bölümün modeli Teori'deki Şanghay karesinde gerçekten çalışır (ilk deneyde yaklaşık 12 MB yüklenir).",
        en: "Pick your prediction first; then the chapter's model really runs on the Shanghai tile of the Theory tab (about 12 MB to load at the first experiment).",
      },
      read: { tab: "detect" },
      questions: [
        {
          id: "e-conf",
          kind: "model",
          q: {
            tr: "Güven eşiği 0,25'ken model bu karede 11 bina buluyor. Eşiği 0,6'ya çıkarırsak kaç bina kalır?",
            en: "At a confidence threshold of 0.25 the model finds 11 buildings on this tile. How many are left at 0.6?",
          },
          options: [
            { tr: "Hepsi kalır", en: "All of them" },
            { tr: "Birkaçı gider, çoğu kalır", en: "A few go, most stay" },
            { tr: "Çoğu gider, birkaçı kalır", en: "Most go, a few stay" },
            { tr: "Hiçbiri kalmaz", en: "None" },
          ],
          answer: 2,
          run: async () => {
            const data = await lab();
            const [low] = await found(data, data.tile, 0.25);
            const [high] = await found(data, data.tile, 0.6);
            return { answer: high === low ? 0 : high >= 0.6 * low ? 1 : high > 0 ? 2 : 3, values: [low, high] };
          },
          describe: ([low, high]) => ({ tr: `Eşik 0,25'te ${low}, 0,6'da ${high} bina.`, en: `${low} buildings at 0.25, ${high} at 0.6.` }),
          tryIt: { tab: "detect" },
          explain: {
            tr: "Bu karede bulunan binaların çoğunun skoru 0,6'nın altındadır: model bulduğu binaların çoğundan çok emin değildir. Eşik hangi tespitlerin \"yeterince emin\" sayılacağını seçer; mAP'i değiştirmez, kesinlik ile duyarlılık arasındaki dengeyi değiştirir.",
            en: 'Most of the buildings found on this tile score below 0.6: the model is not very sure about most of what it finds. The threshold picks which detections count as "sure enough"; it does not change the mAP, it moves the balance between precision and recall.',
          },
        },
        {
          id: "e-nms",
          kind: "model",
          q: {
            tr: "NMS IoU eşiği 0,7'den 0,9'a çıkarsa bu karede bulunan nesne sayısı ne olur?",
            en: "If the NMS IoU threshold goes from 0.7 to 0.9, what happens to the number of objects found on this tile?",
          },
          options: [
            { tr: "Azalır", en: "It drops" },
            { tr: "Değişmez", en: "It stays the same" },
            { tr: "Artar", en: "It rises" },
          ],
          answer: 2,
          run: async () => {
            const data = await lab();
            const [b7, f7] = await found(data, data.tile, 0.25, 0.7);
            const [b9, f9] = await found(data, data.tile, 0.25, 0.9);
            const before = b7 + f7;
            const after = b9 + f9;
            return { answer: after < before ? 0 : after === before ? 1 : 2, values: [before, after] };
          },
          describe: ([before, after]) => ({
            tr: `IoU eşiği 0,7'de ${before}, 0,9'da ${after} nesne.`,
            en: `${before} objects at an IoU threshold of 0.7, ${after} at 0.9.`,
          }),
          tryIt: { tab: "theory", section: "y-nms" },
          explain: {
            tr: "Yüksek eşikte bir kopyanın silinmesi için tutulan kutuyla neredeyse tamamen örtüşmesi gerekir; biraz kaymış kopyalar kalır ve aynı bina iki kez sayılır.",
            en: "With a high threshold a copy is removed only if it overlaps the kept box almost completely; copies that are a little off stay, and the same building is counted twice.",
          },
        },
        {
          id: "e-rotate",
          kind: "model",
          q: {
            tr: "Kareyi 90° döndürüp modele veriyoruz. Model kaç nesne bulur?",
            en: "We turn the tile by 90° and give it to the model. How many objects does it find?",
          },
          options: [
            { tr: "Neredeyse hiçbirini", en: "Almost none" },
            { tr: "Yarısı kadarını", en: "About half" },
            { tr: "Yaklaşık aynı sayıda", en: "About the same number" },
          ],
          answer: 2,
          run: async () => {
            const data = await lab();
            const [b0, f0] = await found(data, data.tile);
            const [b1, f1] = await found(data, rotated(data.tile));
            const ratio = (b1 + f1) / Math.max(1, b0 + f0);
            return { answer: ratio >= 0.7 ? 2 : ratio >= 0.3 ? 1 : 0, values: [b0, f0, b1, f1] };
          },
          describe: ([b0, f0, b1, f1]) => ({
            tr: `Döndürmeden ${b0} bina ve ${f0} saha, döndürünce ${b1} bina ve ${f1} saha.`,
            en: `${b0} buildings and ${f0} fields as it is, ${b1} buildings and ${f1} fields turned.`,
          }),
          tryIt: { tab: "theory", section: "y-train" },
          explain: {
            tr: 'Model eğitimde kareleri ±90° döndürülmüş ve çevrilmiş gördü (veri artırma): onun için kuş bakışının "yukarısı" yoktur. Sayıların biraz değişmesi normaldir; ızgara, döndürülmüş karede nesnelere farklı hücrelerden bakar.',
            en: 'In training the model saw the tiles turned by up to ±90° and flipped (augmentation): for it, a top-down view has no "up". A small change in the counts is normal; on the turned tile the grid sees the objects from other cells.',
          },
        },
        {
          id: "e-grey",
          kind: "model",
          q: {
            tr: "Karenin renklerini kaldırıp gri tonlamaya çeviriyoruz. Model 4 spor sahasının kaçını bulur?",
            en: "We take the colour out of the tile, leaving shades of grey. How many of the 4 sports fields does the model find?",
          },
          options: [
            { tr: "Hepsini: sahalar şekillerinden tanınır", en: "All of them: fields are recognized by their shape" },
            { tr: "Bir kısmını", en: "Some of them" },
            { tr: "Hiçbirini", en: "None" },
          ],
          answer: 2,
          run: async () => {
            const data = await lab();
            const [b0, f0] = await found(data, data.tile);
            const [b1, f1] = await found(data, grey(data.tile));
            return { answer: f1 >= f0 ? 0 : f1 > 0 ? 1 : 2, values: [f0, f1, b0, b1] };
          },
          describe: ([f0, f1, b0, b1]) => ({
            tr: `Renkliyken ${f0} saha ve ${b0} bina, gri tonlamada ${f1} saha ve ${b1} bina.`,
            en: `${f0} fields and ${b0} buildings in colour, ${f1} fields and ${b1} buildings in grey.`,
          }),
          tryIt: { tab: "detect" },
          explain: {
            tr: "Model sahaları büyük ölçüde renklerinden (yeşil çim, kırmızı pist) tanımayı öğrenmiş; renk gidince hiçbirini bulamaz, binaların da yaklaşık yarısını kaybeder. Ağ, eğitim verisinde işe yarayan her ipucunu kullanır; hangisini kullandığı ancak böyle bir deneyle görülür.",
            en: "The model learned to recognize fields largely by their colour (green grass, a red track); without colour it finds none of them, and loses about half of the buildings too. A network uses every clue that worked in its training data; which one it uses shows only in an experiment like this.",
          },
        },
        {
          id: "e-coarse",
          kind: "model",
          q: {
            tr: "Kareyi 4 kat düşük çözünürlüğe (yaklaşık 1,2 m/piksel, eski bir uydu gibi) indirip modele veriyoruz. Model bulduğu nesnelerin kaçını bulmaya devam eder?",
            en: "We bring the tile down to a quarter of its resolution (about 1.2 m per pixel, like an older satellite) and give it to the model. How many of its objects does it still find?",
          },
          options: [
            {
              tr: "Yaklaşık hepsini: letterbox görüntüyü yine 640'a büyütür",
              en: "About all of them: the letterbox enlarges the image back to 640",
            },
            { tr: "Biraz azını", en: "A few fewer" },
            { tr: "Yarısından azını", en: "Fewer than half" },
          ],
          answer: 2,
          run: async () => {
            const data = await lab();
            const [b0, f0] = await found(data, data.tile);
            const small = resizeLinear(data.tile, Math.round(data.tile.width / 4), Math.round(data.tile.height / 4));
            const [b1, f1] = await found(data, small);
            const ratio = (b1 + f1) / Math.max(1, b0 + f0);
            return { answer: ratio >= 0.85 ? 0 : ratio >= 0.5 ? 1 : 2, values: [b0 + f0, b1 + f1] };
          },
          describe: ([before, after]) => ({
            tr: `Özgün karede ${before}, düşük çözünürlükte ${after} nesne.`,
            en: `${before} objects on the original tile, ${after} at the low resolution.`,
          }),
          tryIt: { tab: "detect" },
          explain: {
            tr: "Büyütmek kaybolan ayrıntıyı geri getirmez: bulanık bir görüntü 640'a büyütülse de bulanıktır. Model 0,3 m/piksel görüntülerle eğitildi; başka bir ölçekte ya da kaynakta başarı düşer (Tespit sekmesindeki uyarı).",
            en: "Enlarging does not bring back lost detail: a blurred image enlarged to 640 is still blurred. The model was trained on 0.3 m per pixel images; at another scale or from another source it does worse (the warning in the Detection tab).",
          },
        },
      ],
    },
  ];
}
