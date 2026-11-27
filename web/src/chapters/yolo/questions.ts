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

/** What the experiments need: the model and the decoded tile (loaded when the first one runs, about 7 MB). */
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
      id: "yx-model",
      title: { tr: "Eğitilmiş modelle deneyler", en: "Experiments with the trained model" },
      intro: {
        tr: "Önce tahmininizi seçin; sonra bölümün modeli Teori'deki Şanghay karesinde gerçekten çalışır (ilk deneyde yaklaşık 7 MB yüklenir).",
        en: "Pick your prediction first; then the chapter's model really runs on the Shanghai tile of the Theory tab (about 7 MB load at the first experiment).",
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
