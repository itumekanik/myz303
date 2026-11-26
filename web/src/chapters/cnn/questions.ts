/**
 * Chapter 3, Practice tab: the questions, in the order of the Theory sections, then three
 * experiments on the trained SimpleCNN that comes with the site. tests/unit/practice.test.ts runs
 * every `compute` and `run`, the experiments included, and fails if a stated answer is not what
 * the model does. The student's own trained model plays no part, so the answers never change.
 */
import type { ImageSet } from "../../engines/cnn/data.ts";
import { forward, softmax, type CnnWeights } from "../../engines/cnn/forward.ts";
import type { PracticeSection } from "../../shared/practice.ts";

/** What the experiments need; loaded only when the first one runs (about 5 MB). */
export interface CnnLab {
  weights: CnnWeights;
  test: ImageSet;
}

const R = String.raw;
const IMAGES = 40; // the experiments use the first 40 test images

const outSize = (h: number, k: number, p: number, s: number): number => Math.floor((h - k + 2 * p) / s) + 1;
/** The edge of the input region one value sees: r grows by (k − 1)·j at each layer, j is the product of the strides so far. */
function receptiveField(layers: readonly [kernel: number, stride: number][]): number {
  let r = 1;
  let j = 1;
  for (const [k, s] of layers) {
    r += (k - 1) * j;
    j *= s;
  }
  return r;
}
const decimal = (value: number, digits: number) => ({ tr: value.toFixed(digits).replace(".", ","), en: value.toFixed(digits) });
/** Lets the page breathe between two images. */
const breathe = (): Promise<void> => new Promise((resolve) => setTimeout(resolve));

function testImage(test: ImageSet, k: number): Float32Array {
  return Float32Array.from(test.pixels.subarray(k * 784, (k + 1) * 784), (v) => v / 255);
}

/** The image moved dx pixels to the right; what falls off the edge is lost. */
function shifted(image: Float32Array, dx: number): Float32Array {
  const out = new Float32Array(784);
  for (let r = 0; r < 28; r++) for (let c = 0; c + dx < 28; c++) out[r * 28 + c + dx] = image[r * 28 + c]!;
  return out;
}

function topClass(probabilities: Float32Array): number {
  let best = 0;
  probabilities.forEach((p, k) => {
    if (p > probabilities[best]!) best = k;
  });
  return best;
}

async function rightAfterShift({ weights, test }: CnnLab, dx: number): Promise<number> {
  let right = 0;
  for (let k = 0; k < IMAGES; k++) {
    if (topClass(forward(weights, shifted(testImage(test, k), dx)).probabilities) === test.labels[k]) right++;
    await breathe();
  }
  return right;
}

/** The share of inked pixels (in percent) that are still inked after the shift. */
function inkStaying({ test }: CnnLab, dx: number): number {
  let inked = 0;
  let stay = 0;
  for (let k = 0; k < IMAGES; k++) {
    const a = testImage(test, k);
    const b = shifted(a, dx);
    for (let i = 0; i < 784; i++) {
      if (a[i]! > 0.5) {
        inked++;
        if (b[i]! > 0.5) stay++;
      }
    }
  }
  return Math.round((100 * stay) / inked);
}

export function buildPractice(lab: () => Promise<CnnLab>): PracticeSection[] {
  return [
    {
      id: "px-conv",
      title: { tr: "Evrişim", en: "Convolution" },
      intro: {
        tr: R`Küçük bir filtre görüntünün üzerinde kayar. Çıktı boyutu: \(H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - k + 2p}{s} \right\rfloor + 1\).`,
        en: R`A small filter slides over the image. The output size is \(H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - k + 2p}{s} \right\rfloor + 1\).`,
      },
      read: { tab: "theory", section: "t-conv" },
      questions: [
        {
          id: "cv-size",
          kind: "number",
          q: {
            tr: R`32×32 bir girişe \(k = 3\), padding \(p = 1\) ve stride \(s = 2\) ile evrişim uygulanıyor. Çıktının yüksekliği kaçtır?`,
            en: R`A convolution with \(k = 3\), padding \(p = 1\) and stride \(s = 2\) is applied to a 32×32 input. What is the height of the output?`,
          },
          label: { tr: "çıktı boyutu =", en: "output size =" },
          answer: 16,
          compute: () => outSize(32, 3, 1, 2),
          hint: {
            tr: R`Sayıları formüle koyun. Bölmenin sonucunu aşağı yuvarlayıp 1 ekleyin.`,
            en: R`Put the numbers into the formula. Round the division down, then add 1.`,
          },
          mistakes: [
            {
              value: 15,
              why: {
                tr: R`Ya sondaki +1'i ya da padding'i (\(2p = 2\)) unutmuş olabilirsiniz: \(\lfloor (32 - 3 + 2)/2 \rfloor + 1\).`,
                en: R`You may have left out either the +1 at the end or the padding (\(2p = 2\)): \(\lfloor (32 - 3 + 2)/2 \rfloor + 1\).`,
              },
            },
            {
              value: 32,
              why: {
                tr: R`Stride'ı unutmuşsunuz: filtre ikişer piksel kayar, bu yüzden 2'ye bölünür.`,
                en: R`You left out the stride: the filter moves two pixels at a time, so divide by 2.`,
              },
            },
            {
              value: 16.5,
              why: {
                tr: R`Bölmenin sonucunu aşağı yuvarlayın: filtrenin konum sayısı bir tam sayıdır.`,
                en: R`Round the division down: the number of filter positions is a whole number.`,
              },
            },
          ],
          explain: {
            tr: R`\(\lfloor (32 - 3 + 2\cdot 1) / 2 \rfloor + 1 = \lfloor 15{,}5 \rfloor + 1 = 16\). Stride 2 boyutu yaklaşık yarıya indirir.`,
            en: R`\(\lfloor (32 - 3 + 2\cdot 1) / 2 \rfloor + 1 = \lfloor 15.5 \rfloor + 1 = 16\). A stride of 2 roughly halves the size.`,
          },
        },
        {
          id: "cv-params",
          kind: "number",
          q: {
            tr: R`Girişinde 3 kanal olan bir evrişim katmanında 16 adet 3×3 filtre var. Bias dahil kaç parametresi vardır?`,
            en: R`A convolution layer with 3 input channels has 16 filters of size 3×3. How many parameters does it have, biases included?`,
          },
          label: { tr: "parametre sayısı =", en: "number of parameters =" },
          answer: 448,
          compute: () => 16 * (3 * 3 * 3) + 16,
          hint: {
            tr: "Her filtre giriş kanallarının hepsine birden bakar: 3 kanal × 3 × 3 ağırlık ve 1 bias.",
            en: "Each filter looks at all the input channels at once: 3 channels × 3 × 3 weights and 1 bias.",
          },
          mistakes: [
            {
              value: 432,
              why: {
                tr: R`Biasları unutmuşsunuz: 16 filtrenin her birinin bir biası var.`,
                en: R`You left out the biases: each of the 16 filters has one.`,
              },
            },
            {
              value: 160,
              why: {
                tr: R`Giriş kanallarını unutmuşsunuz: her filtre 3 kanalın hepsine bakar, yani \(3\cdot 3\cdot 3 = 27\) ağırlık taşır.`,
                en: R`You left out the input channels: each filter looks at all 3 channels, so it has \(3\cdot 3\cdot 3 = 27\) weights.`,
              },
            },
            {
              value: 144,
              why: {
                tr: R`Hem giriş kanallarını hem biasları unutmuşsunuz: bir filtre \(3\cdot 3\cdot 3\) ağırlık ve 1 bias taşır.`,
                en: R`You left out both the input channels and the biases: one filter has \(3\cdot 3\cdot 3\) weights and 1 bias.`,
              },
            },
          ],
          explain: {
            tr: R`Bir filtre \(3\cdot 3\cdot 3 = 27\) ağırlık ve 1 bias taşır. 16 filtre: \(16\cdot 27 + 16 = 448\). Sayı görüntünün boyutuna bağlı değildir.`,
            en: R`One filter has \(3\cdot 3\cdot 3 = 27\) weights and 1 bias. 16 filters: \(16\cdot 27 + 16 = 448\). The number does not depend on the size of the image.`,
          },
        },
        {
          id: "cv-share",
          kind: "choice",
          q: {
            tr: "Bir evrişim katmanı, MLP katmanına göre neden çok daha az parametreyle çalışır?",
            en: "Why does a convolution layer need far fewer parameters than an MLP layer?",
          },
          options: [
            {
              text: { tr: "Görüntüyü önce küçülttüğü için", en: "Because it shrinks the image first" },
              why: {
                tr: "Küçültme havuzlamanın işidir. Evrişimin parametre sayısı görüntünün boyutundan zaten bağımsızdır.",
                en: "Shrinking is what pooling does. The parameter count of a convolution does not depend on the image size anyway.",
              },
            },
            {
              text: {
                tr: "Aynı küçük filtre görüntünün her konumunda yeniden kullanıldığı için",
                en: "Because the same small filter is reused at every position in the image",
              },
            },
            {
              text: { tr: "Ağırlıklarının çoğu sıfır olduğu için", en: "Because most of its weights are zero" },
              why: {
                tr: "Filtre ağırlıkları sıfır olmak zorunda değildir. Fark, her konum için ayrı ağırlık tutulmamasıdır.",
                en: "Filter weights need not be zero. The difference is that no separate weights are kept for each position.",
              },
            },
            {
              text: { tr: "Filtreleri eğitilmediği için", en: "Because its filters are not trained" },
              why: {
                tr: "Filtreler de geri yayılımla öğrenilir. Sobel gibi elle yazılan çekirdekler yalnızca örnektir.",
                en: "Filters are learned with backpropagation too. Hand-written kernels such as Sobel are only examples.",
              },
            },
          ],
          answer: 1,
          explain: {
            tr: R`Buna ağırlık paylaşımı denir. Conv1'in bir filtresi 26 parametreyle 576 konumun hepsine bakar; MLP'nin tek bir nöronu ise 785 parametre taşır.`,
            en: R`This is called weight sharing. One Conv1 filter looks at all 576 positions with 26 parameters, while a single MLP neuron has 785.`,
          },
        },
      ],
    },
    {
      id: "px-act",
      title: { tr: "Aktivasyon", en: "Activation" },
      intro: {
        tr: R`CNN'de aktivasyon bir öznitelik haritasının her pikseline ayrı ayrı uygulanır: haritanın boyutu değişmez, yalnızca değerleri değişir.`,
        en: R`In a CNN the activation is applied to each pixel of a feature map separately: the size of the map stays, only its values change.`,
      },
      read: { tab: "theory", section: "t-act" },
      questions: [
        {
          id: "ac-relu",
          kind: "number",
          q: {
            tr: R`Bir haritada üç değer var: \(-2\), \(0{,}5\) ve \(3\). ReLU'dan sonra bu değerlerin toplamı kaç olur?`,
            en: R`A map has three values: \(-2\), \(0.5\) and \(3\). What is their sum after ReLU?`,
          },
          label: { tr: "toplam =", en: "sum =" },
          answer: 3.5,
          compute: () => [-2, 0.5, 3].reduce((sum, v) => sum + Math.max(0, v), 0),
          hint: {
            tr: R`ReLU negatif değerleri 0 yapar, pozitifleri olduğu gibi bırakır.`,
            en: R`ReLU sets negative values to 0 and leaves positive ones as they are.`,
          },
          mistakes: [
            {
              value: 1.5,
              why: {
                tr: R`ReLU'yu uygulamadan toplamışsınız: \(-2\), ReLU'dan sonra 0 olur.`,
                en: R`You added them without the ReLU: \(-2\) becomes 0 after the ReLU.`,
              },
            },
            {
              value: 5.5,
              why: {
                tr: R`ReLU mutlak değer almaz: negatif değerleri 0 yapar.`,
                en: R`The ReLU does not take the absolute value: it sets negative values to 0.`,
              },
            },
          ],
          explain: {
            tr: R`\(\max(0;\ -2) + \max(0;\ 0{,}5) + \max(0;\ 3) = 0 + 0{,}5 + 3 = 3{,}5\). Geriye yalnızca filtrenin güçlü tepki verdiği yerler kalır.`,
            en: R`\(\max(0, -2) + \max(0, 0.5) + \max(0, 3) = 0 + 0.5 + 3 = 3.5\). Only the places where the filter responds strongly remain.`,
          },
        },
        {
          id: "ac-why",
          kind: "choice",
          q: {
            tr: "Evrişim katmanlarının arasına hiç aktivasyon (ve havuzlama da) koymazsak ne olur?",
            en: "What happens if we put no activation (and no pooling either) between convolution layers?",
          },
          options: [
            {
              text: {
                tr: "Üst üste konan katmanlar tek bir doğrusal katmana eşit olur",
                en: "The stacked layers amount to a single linear layer",
              },
            },
            {
              text: { tr: "Çıktının boyutu büyür", en: "The output gets larger" },
              why: {
                tr: "Aktivasyon boyutu değiştirmez; yalnızca değerleri değiştirir.",
                en: "An activation does not change the size, only the values.",
              },
            },
            {
              text: { tr: "Gradyan hesaplanamaz", en: "The gradient cannot be computed" },
              why: {
                tr: "Doğrusal işlemlerin türevi kolaydır. Sorun, ağın ifade gücünün artmamasıdır.",
                en: "Linear operations are easy to differentiate. The problem is that the network gains no expressive power.",
              },
            },
            {
              text: { tr: "Ağ yalnızca iki sınıfı ayırabilir", en: "The network can only separate two classes" },
              why: {
                tr: "Sınıf sayısı son katmana bağlıdır. Aktivasyonsuz ağ çok sınıflı olabilir ama sınırları doğrusal kalır.",
                en: "The number of classes depends on the last layer. Without activations the network can have many classes, but its boundaries stay linear.",
              },
            },
          ],
          answer: 0,
          explain: {
            tr: "Doğrusal işlemlerin bileşimi yine doğrusaldır. Derinliğin işe yaraması için katmanlar arasında ReLU gibi doğrusal olmayan bir adım gerekir. Max pooling da doğrusal değildir: en büyük değeri seçmek bir ağırlıklı toplam değildir.",
            en: "A composition of linear operations is still linear. For depth to help, a non-linear step such as ReLU is needed between the layers. Max pooling is not linear either: picking the largest value is not a weighted sum.",
          },
        },
      ],
    },
    {
      id: "px-pool",
      title: { tr: "Havuzlama", en: "Pooling" },
      intro: {
        tr: R`2×2 max pooling her bölgeden en büyük değeri alır: yükseklik ve genişlik yarıya iner, kanal sayısı değişmez.`,
        en: R`2×2 max pooling takes the largest value of each region: the height and width are halved, the number of channels stays.`,
      },
      read: { tab: "theory", section: "t-pool" },
      questions: [
        {
          id: "pl-flat",
          kind: "number",
          q: {
            tr: R`28×28 giriş sırayla 5×5 evrişim, 2×2 havuzlama, 5×5 evrişim ve 2×2 havuzlamadan geçiyor (padding 0). Son haritada 64 kanal var. Düzleştirilince kaç değer olur? (Sayıyı ayırıcı kullanmadan yazın.)`,
            en: R`A 28×28 input goes through a 5×5 convolution, 2×2 pooling, a 5×5 convolution and 2×2 pooling (padding 0). The last map has 64 channels. How many values are there after flattening? (Type the number without separators.)`,
          },
          label: { tr: "değer sayısı =", en: "number of values =" },
          answer: 1024,
          compute: () => {
            const size = Math.floor(outSize(Math.floor(outSize(28, 5, 0, 1) / 2), 5, 0, 1) / 2);
            return 64 * size * size;
          },
          hint: {
            tr: "Yüksekliği adım adım izleyin: 28 → 24 → 12 → … Sonra kanal × yükseklik × genişlik.",
            en: "Follow the height step by step: 28 → 24 → 12 → … Then channels × height × width.",
          },
          mistakes: [
            {
              value: 576,
              why: {
                tr: R`Evrişimdeki +1'i unutmuş olabilirsiniz: \(28 - 5 + 1 = 24\), \(23\) değil.`,
                en: R`You may have left out the +1 of the convolution: \(28 - 5 + 1 = 24\), not \(23\).`,
              },
            },
            {
              value: 4096,
              why: {
                tr: R`Son havuzlamayı unutmuşsunuz: \(8\times 8\)'lik harita \(4\times 4\)'e iner.`,
                en: R`You left out the last pooling: the \(8\times 8\) map shrinks to \(4\times 4\).`,
              },
            },
            {
              value: 16,
              why: {
                tr: R`Kanalları unutmuşsunuz: 64 kanalın her biri \(4\times 4\) değer taşır.`,
                en: R`You left out the channels: each of the 64 channels holds \(4\times 4\) values.`,
              },
            },
          ],
          explain: {
            tr: R`\(28 \to 24 \to 12 \to 8 \to 4\). Düzleştirince \(64\cdot 4\cdot 4 = 1024\) değer olur. SimpleCNN'in FC1 katmanının girişi budur.`,
            en: R`\(28 \to 24 \to 12 \to 8 \to 4\). Flattened, that is \(64\cdot 4\cdot 4 = 1024\) values. This is the input of SimpleCNN's FC1 layer.`,
          },
        },
        {
          id: "pl-params",
          kind: "choice",
          q: {
            tr: "2×2 max pooling katmanının kaç öğrenilen parametresi vardır?",
            en: "How many learned parameters does a 2×2 max pooling layer have?",
          },
          options: [
            {
              text: { tr: "4", en: "4" },
              why: {
                tr: "2×2 yalnızca bölgenin boyutudur. Havuzlama bu bölgede ağırlık kullanmaz, en büyük değeri seçer.",
                en: "2×2 is only the size of the region. Pooling uses no weights there; it picks the largest value.",
              },
            },
            {
              text: { tr: "Kanal sayısı kadar", en: "One per channel" },
              why: {
                tr: "Her kanal ayrı havuzlanır ama bunun için öğrenilen bir değer gerekmez.",
                en: "Every channel is pooled separately, but that takes no learned value.",
              },
            },
            { text: { tr: "0", en: "0" } },
          ],
          answer: 2,
          explain: {
            tr: "Havuzlama sabit bir işlemdir: en büyük değeri seçer, öğrenecek bir şeyi yoktur. Buna rağmen hesap yükünü azaltır ve sonraki katmanların algı alanını büyütür.",
            en: "Pooling is a fixed operation: it picks the largest value and has nothing to learn. Even so, it cuts the work and enlarges the receptive field of the next layers.",
          },
        },
      ],
    },
    {
      id: "px-softmax",
      title: { tr: "Softmax ve kayıp", en: "Softmax and the loss" },
      intro: {
        tr: R`Softmax skorları toplamı 1 olan olasılıklara çevirir: \(p_k = e^{z_k} / \sum_j e^{z_j}\). Kayıp, doğru sınıfın olasılığının \(-\log p\) değeridir.`,
        en: R`Softmax turns the scores into probabilities that sum to 1: \(p_k = e^{z_k} / \sum_j e^{z_j}\). The loss is \(-\log p\) of the correct class's probability.`,
      },
      read: { tab: "theory", section: "t-softmax" },
      questions: [
        {
          id: "sm-equal",
          kind: "number",
          q: {
            tr: R`4 sınıflı bir modelin bütün skorları eşit. Çapraz entropi kaybı kaçtır? (Kayıptaki log doğal logaritmadır, ln. İki ondalığa yuvarlayın.)`,
            en: R`All the scores of a 4-class model are equal. What is the cross-entropy loss? (The log in the loss is the natural logarithm, ln. Round to two decimals.)`,
          },
          label: { tr: R`\(L \approx\)`, en: R`\(L \approx\)` },
          answer: 1.39,
          tolerance: 0.006,
          compute: () => -Math.log(softmax(new Float32Array(4))[0]!),
          hint: {
            tr: R`Eşit skorlarda her sınıf \(1/4\) olasılık alır.`,
            en: R`With equal scores every class gets a probability of \(1/4\).`,
          },
          mistakes: [
            {
              value: 0.602,
              why: {
                tr: R`Bu 10 tabanlı logaritma (\(\log_{10} 4 \approx 0{,}60\)). Kayıptaki log doğal logaritmadır, ln.`,
                en: R`That is the base-10 logarithm (\(\log_{10} 4 \approx 0.60\)). The log in the loss is the natural one, ln.`,
              },
            },
            {
              value: 2,
              why: {
                tr: R`Bu 2 tabanlı logaritma (\(\log_2 4 = 2\)). Kayıptaki log doğal logaritmadır, ln.`,
                en: R`That is the base-2 logarithm (\(\log_2 4 = 2\)). The log in the loss is the natural one, ln.`,
              },
            },
            {
              value: 0.25,
              why: {
                tr: R`Bu doğru sınıfın olasılığı. Kayıp onun eksi logaritmasıdır: \(-\ln 0{,}25\).`,
                en: R`That is the correct class's probability. The loss is minus its logarithm: \(-\ln 0.25\).`,
              },
            },
            {
              value: 2.303,
              why: {
                tr: R`Bu 10 sınıflı bir modelin kaybı (\(\ln 10\)). Buradaki modelin 4 sınıfı var.`,
                en: R`That is the loss of a 10-class model (\(\ln 10\)). The model here has 4 classes.`,
              },
            },
          ],
          explain: {
            tr: R`\(p = 1/4\) ve \(L = -\ln 0{,}25 = \ln 4 \approx 1{,}39\). SimpleCNN'in 10 sınıfında aynı hesap \(\ln 10 \approx 2{,}30\) verir: eğitimin başındaki kayıp budur.`,
            en: R`\(p = 1/4\) and \(L = -\ln 0.25 = \ln 4 \approx 1.39\). With SimpleCNN's 10 classes the same computation gives \(\ln 10 \approx 2.30\): the loss at the start of training.`,
          },
        },
        {
          id: "sm-add",
          kind: "predict",
          q: {
            tr: R`Üç sınıfın skorları \((2,\ 1,\ 0)\). Bütün skorlara 5 ekliyoruz. Softmax olasılıklarına ne olur?`,
            en: R`The scores of three classes are \((2,\ 1,\ 0)\). We add 5 to every score. What happens to the softmax probabilities?`,
          },
          options: [
            { tr: "Hepsi büyür", en: "They all grow" },
            { tr: "En büyük skorun olasılığı artar", en: "The probability of the largest score goes up" },
            { tr: "Hiç değişmez", en: "They do not change" },
          ],
          answer: 2,
          run: () => {
            const before = softmax(Float32Array.of(2, 1, 0));
            const after = softmax(Float32Array.of(7, 6, 5));
            const same = before.every((p, k) => Math.abs(p - after[k]!) < 1e-6);
            const text = (lang: "tr" | "en", p: Float32Array) => [...p].map((v) => decimal(v, 3)[lang]).join(" / ");
            return {
              answer: same ? 2 : after[0]! > before[0]! ? 1 : 0,
              result: {
                tr: `Önce ${text("tr", before)}, sonra ${text("tr", after)}.`,
                en: `Before ${text("en", before)}, after ${text("en", after)}.`,
              },
            };
          },
          tryIt: { tab: "theory", section: "t-softmax" },
          explain: {
            tr: R`\(e^{z+5} = e^5\,e^{z}\) olduğu için pay ve paydadaki \(e^5\) sadeleşir. Olasılıkları mutlak skorlar değil, skorların farkları belirler.`,
            en: R`Since \(e^{z+5} = e^5\,e^{z}\), the factor \(e^5\) cancels in the fraction. The differences between the scores set the probabilities, not their absolute values.`,
          },
        },
        {
          id: "sm-grad",
          kind: "number",
          q: {
            tr: R`Model doğru sınıfa \(p_k = 0{,}7\) olasılık veriyor. Kaybın bu sınıfın skoruna göre gradyanı \(p_k - y_k\) kaçtır?`,
            en: R`The model gives the correct class a probability of \(p_k = 0.7\). What is the gradient of the loss with respect to that class's score, \(p_k - y_k\)?`,
          },
          label: { tr: "gradyan =", en: "gradient =" },
          answer: -0.3,
          compute: () => {
            // Two classes with p = 0.7, differentiated numerically
            const loss = (z: number) => -Math.log(Math.exp(z) / (Math.exp(z) + 1));
            const z = Math.log(7 / 3);
            return (loss(z + 1e-6) - loss(z - 1e-6)) / 2e-6;
          },
          hint: { tr: R`Doğru sınıfta \(y_k = 1\)'dir.`, en: R`For the correct class, \(y_k = 1\).` },
          mistakes: [
            {
              value: 0.3,
              why: {
                tr: R`İşarete dikkat: \(p_k - y_k = 0{,}7 - 1\).`,
                en: R`Watch the sign: \(p_k - y_k = 0.7 - 1\).`,
              },
            },
            {
              value: 0.7,
              why: {
                tr: R`Doğru sınıfın \(y_k = 1\) değerini çıkarmayı unutmuşsunuz.`,
                en: R`You did not subtract the correct class's \(y_k = 1\).`,
              },
            },
          ],
          explain: {
            tr: R`\(0{,}7 - 1 = -0{,}3\). Negatif gradyan, bu skor büyürse kaybın azalacağını söyler. Bölüm 2'deki \(\hat y - y\) ile aynı biçimdedir.`,
            en: R`\(0.7 - 1 = -0.3\). A negative gradient says the loss falls if this score grows. It has the same form as \(\hat y - y\) in chapter 2.`,
          },
        },
      ],
    },
    {
      id: "px-opt",
      title: { tr: "Optimizasyon", en: "Optimization" },
      intro: {
        tr: R`Üç yöntem de aynı gradyanı kullanır, ama adımı farklı hesaplar. Momentum: \(v_t = \mu\,v_{t-1} + g_t\), \(\theta_t = \theta_{t-1} - \eta\,v_t\).`,
        en: R`All three methods use the same gradient but compute the step differently. Momentum: \(v_t = \mu\,v_{t-1} + g_t\), \(\theta_t = \theta_{t-1} - \eta\,v_t\).`,
      },
      read: { tab: "theory", section: "t-opt" },
      questions: [
        {
          id: "op-mom",
          kind: "number",
          q: {
            tr: R`Momentum ile bir adım: \(\theta = 5\), \(v = 2\), \(g = 1\), \(\mu = 0{,}9\) ve \(\eta = 0{,}1\). Yeni \(\theta\) kaçtır?`,
            en: R`One step with Momentum: \(\theta = 5\), \(v = 2\), \(g = 1\), \(\mu = 0.9\) and \(\eta = 0.1\). What is the new \(\theta\)?`,
          },
          label: { tr: R`yeni \(\theta =\)`, en: R`new \(\theta =\)` },
          answer: 4.72,
          compute: () => {
            const v = 0.9 * 2 + 1;
            return 5 - 0.1 * v;
          },
          hint: {
            tr: R`Önce yeni hızı \(v\) bulun, sonra \(\theta\)'dan \(\eta\,v\) çıkarın.`,
            en: R`Find the new velocity \(v\) first, then subtract \(\eta\,v\) from \(\theta\).`,
          },
          mistakes: [
            {
              value: 4.9,
              why: {
                tr: R`Bu düz SGD adımı: \(\theta - \eta\,g\). Momentum önce hızı günceller: \(v = \mu\,v + g\).`,
                en: R`That is a plain SGD step: \(\theta - \eta\,g\). Momentum updates the velocity first: \(v = \mu\,v + g\).`,
              },
            },
            {
              value: 4.8,
              why: {
                tr: R`Eski hızı kullanmışsınız. Önce \(v\)'yi güncelleyin: \(v = 0{,}9\cdot 2 + 1 = 2{,}8\).`,
                en: R`You used the old velocity. Update \(v\) first: \(v = 0.9\cdot 2 + 1 = 2.8\).`,
              },
            },
            {
              value: 2.2,
              why: {
                tr: R`\(\eta\) ile çarpmayı unutmuşsunuz: adım \(\eta\,v = 0{,}28\)'dir.`,
                en: R`You did not multiply by \(\eta\): the step is \(\eta\,v = 0.28\).`,
              },
            },
          ],
          explain: {
            tr: R`\(v = 0{,}9\cdot 2 + 1 = 2{,}8\) ve \(\theta = 5 - 0{,}1\cdot 2{,}8 = 4{,}72\). Düz SGD'de adım yalnızca \(0{,}1\) olurdu; biriken hız adımı büyüttü.`,
            en: R`\(v = 0.9\cdot 2 + 1 = 2.8\) and \(\theta = 5 - 0.1\cdot 2.8 = 4.72\). With plain SGD the step would be only \(0.1\); the accumulated velocity made it larger.`,
          },
        },
        {
          id: "op-adam",
          kind: "choice",
          q: { tr: "Adam'ı düz SGD'den ayıran temel fikir nedir?", en: "What is the key idea that sets Adam apart from plain SGD?" },
          options: [
            {
              text: { tr: "Gradyan hesaplamaması", en: "It does not compute gradients" },
              why: {
                tr: "Üç yöntem de aynı gradyanı kullanır. Fark, adımın nasıl hesaplandığıdır.",
                en: "All three methods use the same gradient. The difference is how the step is computed.",
              },
            },
            {
              text: { tr: "Her zaman global minimumu bulması", en: "It always finds the global minimum" },
              why: {
                tr: "Adam da sığ bir çukurda kalabilir. Hiçbir yöntem bunu garanti etmez.",
                en: "Adam can stay in a shallow dip too. No method guarantees this.",
              },
            },
            {
              text: { tr: "Öğrenme oranına ihtiyaç duymaması", en: "It needs no learning rate" },
              why: {
                tr: R`Adam da bir \(\eta\) kullanır; SimpleCNN'de \(\eta = 0{,}001\).`,
                en: R`Adam uses an \(\eta\) too; in SimpleCNN, \(\eta = 0.001\).`,
              },
            },
            {
              text: {
                tr: "Her parametrenin adımını kendi gradyan geçmişine göre ölçeklemesi",
                en: "It scales each parameter's step by that parameter's gradient history",
              },
            },
          ],
          answer: 3,
          explain: {
            tr: R`Adam, gradyanın hareketli ortalamasını gradyan karelerinin ortalamasının köküne böler. Böylece ölçekleri çok farklı parametreler için tek bir \(\eta\) yeter.`,
            en: R`Adam divides the moving average of the gradient by the root of the moving average of its square. So one \(\eta\) suits parameters of very different scales.`,
          },
        },
      ],
    },
    {
      id: "px-drop",
      title: { tr: "Dropout", en: "Dropout" },
      intro: {
        tr: R`Eğitimde değerlerin \(p\) kadarı sıfırlanır, kalanlar \(1/(1-p)\) ile büyütülür. SimpleCNN'de \(p = 0{,}45\) ve \(p = 0{,}35\) kullanılır.`,
        en: R`In training a share \(p\) of the values is set to zero and the rest are scaled by \(1/(1-p)\). SimpleCNN uses \(p = 0.45\) and \(p = 0.35\).`,
      },
      read: { tab: "theory", section: "t-drop" },
      questions: [
        {
          id: "dr-scale",
          kind: "number",
          q: {
            tr: R`\(p = 0{,}35\) iken değeri \(h = 1{,}3\) olan bir hücre sıfırlanmadı. Dropout'tan sonra değeri kaç olur?`,
            en: R`With \(p = 0.35\), a cell with the value \(h = 1.3\) was not dropped. What is its value after Dropout?`,
          },
          label: { tr: "yeni değer =", en: "new value =" },
          answer: 2,
          compute: () => 1.3 / (1 - 0.35),
          hint: { tr: R`Kalan değerler \(1/(1-p)\) ile çarpılır.`, en: R`Values that remain are multiplied by \(1/(1-p)\).` },
          mistakes: [
            {
              value: 1.3,
              why: {
                tr: R`Kalan değerler aynen kalmaz: beklenen değer değişmesin diye \(1/(1 - p)\) ile büyütülür.`,
                en: R`The values that remain are not left as they are: they are scaled by \(1/(1 - p)\) so that the expected value stays the same.`,
              },
            },
            {
              value: 0.845,
              why: {
                tr: R`\(1 - p\) ile çarpmışsınız. Kalan değerler \(1 - p\)'ye bölünür: \(1{,}3 / 0{,}65\).`,
                en: R`You multiplied by \(1 - p\). The values that remain are divided by it: \(1.3 / 0.65\).`,
              },
            },
          ],
          explain: {
            tr: R`\(1{,}3 / (1 - 0{,}35) = 1{,}3 / 0{,}65 = 2\). Böylece beklenen değer aynı kalır: \(0{,}65\cdot 2 = 1{,}3\).`,
            en: R`\(1.3 / (1 - 0.35) = 1.3 / 0.65 = 2\). So the expected value stays the same: \(0.65\cdot 2 = 1.3\).`,
          },
        },
        {
          id: "dr-eval",
          kind: "choice",
          q: { tr: "Dropout çıkarımda neden kapalıdır?", en: "Why is Dropout off at inference?" },
          options: [
            {
              text: { tr: "Çıkarımı hızlandırmak için", en: "To make inference faster" },
              why: {
                tr: "Dropout'un maliyeti çok küçüktür. Asıl neden sonucun rastgele olmamasıdır.",
                en: "Dropout costs very little. The real reason is that the result must not be random.",
              },
            },
            {
              text: {
                tr: "Tahmin rastgele olmasın ve ağın tamamı kullanılsın diye",
                en: "So that the prediction is not random and the whole network is used",
              },
            },
            {
              text: { tr: "Ağırlıklar çıkarımda değişmesin diye", en: "So that the weights do not change at inference" },
              why: {
                tr: "Ağırlıkları değiştiren optimizer'dır; Dropout hiçbir ağırlığı değiştirmez.",
                en: "The optimizer is what changes weights; Dropout changes none.",
              },
            },
            {
              text: { tr: "Test görüntüleri daha küçük olduğu için", en: "Because test images are smaller" },
              why: { tr: "Test görüntüleri de 28×28'dir.", en: "Test images are 28×28 too." },
            },
          ],
          answer: 1,
          explain: {
            tr: R`Eğitimde kalan değerler \(1/(1-p)\) ile büyütüldüğü için her değerin beklenen değeri zaten aynıdır. Bu yüzden çıkarımda hiçbir düzeltme gerekmez.`,
            en: R`In training the remaining values are scaled by \(1/(1-p)\), so every value's expected value is already the same. That is why no correction is needed at inference.`,
          },
        },
      ],
    },
    {
      id: "px-bn",
      title: { tr: "Batch Normalization", en: "Batch Normalization" },
      intro: {
        tr: R`Her kanal önce standartlaştırılır, sonra öğrenilen iki sayıyla ölçeklenip kaydırılır: \(\hat x = (x - \mu_B)/\sigma_B\), \(y = \gamma\,\hat x + \beta\).`,
        en: R`Each channel is first standardized, then scaled and shifted by two learned numbers: \(\hat x = (x - \mu_B)/\sigma_B\), \(y = \gamma\,\hat x + \beta\).`,
      },
      read: { tab: "theory", section: "t-bn" },
      questions: [
        {
          id: "bn-calc",
          kind: "number",
          q: {
            tr: R`\(x = 5\), \(\mu_B = 3\), \(\sigma_B = 2\), \(\gamma = 2\) ve \(\beta = 0{,}5\). BatchNorm'un çıktısı \(y\) kaçtır?`,
            en: R`\(x = 5\), \(\mu_B = 3\), \(\sigma_B = 2\), \(\gamma = 2\) and \(\beta = 0.5\). What is the BatchNorm output \(y\)?`,
          },
          label: { tr: R`\(y =\)`, en: R`\(y =\)` },
          answer: 2.5,
          compute: () => 2 * ((5 - 3) / 2) + 0.5,
          hint: {
            tr: R`Önce \(\hat x\)'i bulun, sonra \(\gamma\) ile çarpıp \(\beta\) ekleyin.`,
            en: R`Find \(\hat x\) first, then multiply by \(\gamma\) and add \(\beta\).`,
          },
          mistakes: [
            {
              value: 2,
              why: {
                tr: R`\(\beta\)'yı eklemeyi unutmuşsunuz: \(y = \gamma\,\hat x + \beta\).`,
                en: R`You did not add \(\beta\): \(y = \gamma\,\hat x + \beta\).`,
              },
            },
            {
              value: 1.5,
              why: {
                tr: R`Ya \(\gamma = 2\) ile çarpmayı unuttunuz ya da \(\sigma\) yerine \(\sigma^2 = 4\)'e böldünüz. Önce \(\hat x = (x - \mu)/\sigma = 1\).`,
                en: R`Either you did not multiply by \(\gamma = 2\), or you divided by \(\sigma^2 = 4\) instead of \(\sigma\). First \(\hat x = (x - \mu)/\sigma = 1\).`,
              },
            },
          ],
          explain: {
            tr: R`\(\hat x = (5 - 3)/2 = 1\) ve \(y = 2\cdot 1 + 0{,}5 = 2{,}5\).`,
            en: R`\(\hat x = (5 - 3)/2 = 1\) and \(y = 2\cdot 1 + 0.5 = 2.5\).`,
          },
        },
        {
          id: "bn-params",
          kind: "number",
          q: {
            tr: "64 kanallı bir haritaya uygulanan BatchNorm katmanının kaç öğrenilen parametresi vardır?",
            en: "How many learned parameters does a BatchNorm layer on a 64-channel map have?",
          },
          label: { tr: "parametre sayısı =", en: "number of parameters =" },
          answer: 128,
          compute: () => 2 * 64,
          hint: {
            tr: R`Her kanalın kendi \(\gamma\) ve \(\beta\) değeri vardır. Ortalama ve varyans öğrenilmez, ölçülür.`,
            en: R`Each channel has its own \(\gamma\) and \(\beta\). The mean and variance are measured, not learned.`,
          },
          mistakes: [
            {
              value: 64,
              why: {
                tr: R`Her kanalda iki öğrenilen sayı var: ölçek \(\gamma\) ve kayma \(\beta\).`,
                en: R`Each channel has two learned numbers: the scale \(\gamma\) and the shift \(\beta\).`,
              },
            },
            {
              value: 256,
              why: {
                tr: R`Hareketli ortalama ve varyans da saklanır ama öğrenilmez: gradyanla değil, batch'lerden sayılarak güncellenir. Öğrenilenler yalnızca \(\gamma\) ile \(\beta\).`,
                en: R`The running mean and variance are stored too, but not learned: they are updated by counting over the batches, not by the gradient. Only \(\gamma\) and \(\beta\) are learned.`,
              },
            },
          ],
          explain: {
            tr: R`\(2\cdot 64 = 128\). Hareketli ortalama ve varyans da modelle saklanır, ama gradyanla güncellenmedikleri için parametre sayılmazlar.`,
            en: R`\(2\cdot 64 = 128\). The running mean and variance are stored with the model too, but they are not updated by gradients, so they do not count as parameters.`,
          },
        },
        {
          id: "bn-eval",
          kind: "choice",
          q: {
            tr: R`Çıkarımda tek bir görüntü gelince BatchNorm \(\mu\) ve \(\sigma\) değerlerini nereden alır?`,
            en: R`At inference, when a single image comes in, where does BatchNorm get \(\mu\) and \(\sigma\) from?`,
          },
          options: [
            {
              text: { tr: "O tek görüntüden hesaplar", en: "It computes them from that one image" },
              why: {
                tr: "O zaman tahmin görüntünün kendi ölçeğine bağlı olurdu ve aynı görüntü başka bir batch'te başka sonuç verirdi.",
                en: "Then the prediction would depend on the image's own scale, and the same image would give another result in another batch.",
              },
            },
            {
              text: { tr: "Test kümesinin tamamından hesaplar", en: "It computes them from the whole test set" },
              why: {
                tr: "Model test verisinden hiçbir şey öğrenmemelidir, tahmin de test kümesinin geri kalanına bağlı olmamalıdır.",
                en: "A model must learn nothing from the test data, and a prediction must not depend on the rest of the test set.",
              },
            },
            {
              text: {
                tr: "Eğitim boyunca biriktirdiği hareketli ortalamalardan alır",
                en: "From the running averages it collected during training",
              },
            },
            {
              text: { tr: "Hiç kullanmaz, BatchNorm çıkarımda kapalıdır", en: "It does not use them: BatchNorm is off at inference" },
              why: {
                tr: "Çıkarımda kapalı olan Dropout'tur. BatchNorm çalışmaya devam eder, yalnızca istatistiklerinin kaynağı değişir.",
                en: "Dropout is what is off at inference. BatchNorm keeps working; only the source of its statistics changes.",
              },
            },
          ],
          answer: 2,
          explain: {
            tr: R`Eğitimde her batch kendi \(\mu_B, \sigma_B\) değerini kullanır ve bunların hareketli ortalaması saklanır. Çıkarımda bu sabit değerler kullanıldığı için aynı görüntü hep aynı sonucu verir.`,
            en: R`In training each batch uses its own \(\mu_B, \sigma_B\), and a running average of them is kept. At inference these fixed values are used, so the same image always gives the same result.`,
          },
        },
      ],
    },
    {
      id: "px-transfer",
      title: { tr: "Hepsi bir arada", en: "Putting it together" },
      intro: {
        tr: R`Bu kısımdaki sorular birkaç katmanı birlikte kullanır ve SimpleCNN'i değiştirince ne olacağını sorar. Algı alanı ile max pooling'in gradyanı Matematik sekmesinde, eğitim eğrileri Eğitim sekmesinde anlatılıyor.`,
        en: R`The questions here use several layers at once and ask what happens when SimpleCNN changes. The receptive field and the gradient of max pooling are in the Math tab, the training curves in the Training tab.`,
      },
      read: { tab: "math", section: "m-pool" },
      questions: [
        {
          id: "t-rf",
          kind: "number",
          q: {
            tr: R`Üst üste üç katman var: 3×3 evrişim (stride 1), 2×2 max pooling (stride 2) ve yine 3×3 evrişim (stride 1). Son katmandaki tek bir değer giriş görüntüsünde kaç × kaç piksellik bir bölgeye bakar? (Bölgenin kenar uzunluğunu yazın.)`,
            en: R`Three layers are stacked: a 3×3 convolution (stride 1), 2×2 max pooling (stride 2) and another 3×3 convolution (stride 1). How large a region of the input image does a single value of the last layer see? (Type the length of the region's edge.)`,
          },
          label: { tr: "kenar =", en: "edge =" },
          answer: 8,
          compute: () => {
            // the same formula gives the 14×14 region of SimpleCNN's Conv2 in the Theory tab
            const simple = receptiveField([
              [5, 1],
              [2, 2],
              [5, 1],
            ]);
            return simple === 14
              ? receptiveField([
                  [3, 1],
                  [2, 2],
                  [3, 1],
                ])
              : NaN;
          },
          hint: {
            tr: R`Kenar her katmanda \((k - 1)\cdot j\) kadar büyür; \(j\), o katmana kadarki stride'ların çarpımıdır. Havuzlamadan sonra \(j = 2\) olur.`,
            en: R`The edge grows by \((k - 1)\cdot j\) at each layer, where \(j\) is the product of the strides so far. After the pooling, \(j = 2\).`,
          },
          mistakes: [
            {
              value: 5,
              why: {
                tr: R`Havuzlamayı atlamışsınız: o da bölgeyi 1 büyütür ve sonraki katmanlar için \(j\)'yi 2'ye çıkarır.`,
                en: R`You skipped the pooling: it grows the region by 1 too and raises \(j\) to 2 for the layers after it.`,
              },
            },
            {
              value: 6,
              why: {
                tr: R`Havuzlamadan sonra evrişimin bir adımı girişte 2 piksel tutar: ikinci evrişim bölgeyi \((3 - 1)\cdot 2 = 4\) büyütür, 2 değil.`,
                en: R`After the pooling, one step of the convolution covers 2 input pixels: the second convolution grows the region by \((3 - 1)\cdot 2 = 4\), not 2.`,
              },
            },
          ],
          explain: {
            tr: R`\(r\): \(1 \to 3\) (ilk evrişim) \(\to 4\) (havuzlama; artık \(j = 2\)) \(\to 4 + (3 - 1)\cdot 2 = 8\). Havuzlama stride'ı ikiye katladığı için sonraki evrişimin her adımı iki kat geniş bir alan kaplar. SimpleCNN'de aynı hesap Conv2 için \(14\) verir. Derin katmanlar bu yüzden rakamın büyük parçalarını görür.`,
            en: R`\(r\): \(1 \to 3\) (first convolution) \(\to 4\) (pooling; now \(j = 2\)) \(\to 4 + (3 - 1)\cdot 2 = 8\). Pooling doubles the stride, so each step of the next convolution covers twice as wide an area. In SimpleCNN the same computation gives \(14\) for Conv2. That is why the deeper layers see large parts of the digit.`,
          },
        },
        {
          id: "t-poolgrad",
          kind: "choice",
          q: {
            tr: R`2×2 max pooling penceresinde dört değer var: 1, 4, 2 ve 3. Geri yayılımda çıkıştan \(0{,}6\) gradyan geliyor. Dört girdi konumuna hangi gradyanlar gider?`,
            en: R`A 2×2 max pooling window holds four values: 1, 4, 2 and 3. In backpropagation a gradient of \(0.6\) comes from the output. Which gradients go to the four input positions?`,
          },
          options: [
            {
              text: {
                tr: R`Dördüne de \(0{,}15\): gradyan eşit paylaşılır`,
                en: R`\(0.15\) to each of the four: the gradient is shared equally`,
              },
              why: {
                tr: "Bu average pooling'in kuralı. Max pooling'de çıktıyı yalnızca en büyük değer belirler.",
                en: "That is the rule of average pooling. In max pooling only the largest value sets the output.",
              },
            },
            { text: { tr: R`4'e \(0{,}6\), öbür üçüne 0`, en: R`\(0.6\) to the 4, 0 to the other three` } },
            {
              text: {
                tr: R`Değerlerle orantılı: 1'e \(0{,}06\), 4'e \(0{,}24\), 2'ye \(0{,}12\), 3'e \(0{,}18\)`,
                en: R`In proportion to the values: \(0.06\) to the 1, \(0.24\) to the 4, \(0.12\) to the 2, \(0.18\) to the 3`,
              },
              why: {
                tr: "Çıktı bir toplam değil, en büyük değerdir. 4 dışındaki değerler biraz değişse çıktı hiç değişmez: türevleri 0'dır.",
                en: "The output is not a sum but the largest value. If the values other than 4 changed a little, the output would not change at all: their derivatives are 0.",
              },
            },
            {
              text: { tr: "Hiçbirine: havuzlamanın öğrenilen parametresi yok", en: "None: pooling has no learned parameters" },
              why: {
                tr: "Parametresi olmaması gradyanın orada durduğu anlamına gelmez. Gradyan havuzlamadan geçip önceki katmana ulaşır; ulaşmasaydı Conv1 hiç öğrenemezdi.",
                en: "Having no parameters does not mean the gradient stops there. It passes through the pooling to the layer before; otherwise Conv1 could never learn.",
              },
            },
          ],
          answer: 1,
          explain: {
            tr: R`Çıktı \(\max(1, 4, 2, 3) = 4\)'tür ve yalnızca 4'e bağlıdır. Türev bir göstergedir: kazanan konum için 1, öbürleri için 0. Gradyan zincir kuralıyla \(0{,}6\cdot 1\) olarak kazanana gider. PyTorch bunun için ileri yayılımda kazananın konumunu saklar.`,
            en: R`The output is \(\max(1, 4, 2, 3) = 4\) and depends only on the 4. The derivative is an indicator: 1 for the winning position, 0 for the others. By the chain rule, \(0.6\cdot 1\) goes to the winner. That is why PyTorch keeps the winner's position in the forward pass.`,
          },
        },
        {
          id: "t-pad",
          kind: "number",
          q: {
            tr: R`SimpleCNN'de yalnızca Conv1'e padding \(p = 2\) ekliyoruz, geri kalan her şey aynı (iki 5×5 evrişim, iki 2×2 havuzlama, son haritada 64 kanal). Düzleştirilince kaç değer olur? (Sayıyı ayırıcı kullanmadan yazın.)`,
            en: R`In SimpleCNN we add padding \(p = 2\) to Conv1 only and keep everything else (two 5×5 convolutions, two 2×2 poolings, 64 channels in the last map). How many values are there after flattening? (Type the number without separators.)`,
          },
          label: { tr: "değer sayısı =", en: "number of values =" },
          answer: 1600,
          compute: () => {
            const size = Math.floor(outSize(Math.floor(outSize(28, 5, 2, 1) / 2), 5, 0, 1) / 2);
            return 64 * size * size;
          },
          hint: {
            tr: R`Yüksekliği adım adım izleyin. Padding 2 ile 5×5 evrişim boyutu korur: \(28 - 5 + 4 + 1 = 28\).`,
            en: R`Follow the height step by step. With padding 2, a 5×5 convolution keeps the size: \(28 - 5 + 4 + 1 = 28\).`,
          },
          mistakes: [
            {
              value: 1024,
              why: {
                tr: R`Bu, padding'siz SimpleCNN'in değeri. Padding 2 ile Conv1'in çıktısı 24×24 değil, 28×28 olur.`,
                en: R`That is the value of SimpleCNN without padding. With padding 2, Conv1's output is 28×28, not 24×24.`,
              },
            },
            {
              value: 3136,
              why: {
                tr: R`Padding yalnızca Conv1'de. Conv2 yine boyutu 4 küçültür: \(14 \to 10\).`,
                en: R`The padding is only in Conv1. Conv2 still takes 4 off the size: \(14 \to 10\).`,
              },
            },
            {
              value: 25,
              why: { tr: R`Kanalları unutmuşsunuz: \(64\cdot 5\cdot 5\).`, en: R`You left out the channels: \(64\cdot 5\cdot 5\).` },
            },
          ],
          explain: {
            tr: R`\(28 \to 28 \to 14 \to 10 \to 5\) ve \(64\cdot 5\cdot 5 = 1600\). Görüntünün kenarındaki pikseller artık filtrenin ortasına da gelir, ama bedeli var: FC1'in girişi 1024'ten 1600'e çıkar ve 512 nöronlu FC1'e \(576\cdot 512 = 294\,912\) yeni ağırlık eklenir.`,
            en: R`\(28 \to 28 \to 14 \to 10 \to 5\) and \(64\cdot 5\cdot 5 = 1600\). The pixels at the edge of the image now reach the middle of the filter too, but at a price: FC1's input grows from 1024 to 1600, and FC1 with its 512 neurons gains \(576\cdot 512 = 294{,}912\) new weights.`,
          },
        },
        {
          id: "t-scale",
          kind: "predict",
          q: {
            tr: R`Üç sınıfın skorları \((2,\ 1,\ 0)\). Bu kez bütün skorları 2 ile çarpıyoruz: \((4,\ 2,\ 0)\). Softmax olasılıklarına ne olur?`,
            en: R`The scores of three classes are \((2,\ 1,\ 0)\). This time we multiply every score by 2: \((4,\ 2,\ 0)\). What happens to the softmax probabilities?`,
          },
          options: [
            { tr: "Hiç değişmez", en: "They do not change" },
            { tr: "En büyük skorun olasılığı artar", en: "The probability of the largest score goes up" },
            { tr: "Olasılıklar birbirine yaklaşır", en: "The probabilities move closer together" },
          ],
          answer: 1,
          run: () => {
            const before = softmax(Float32Array.of(2, 1, 0));
            const after = softmax(Float32Array.of(4, 2, 0));
            const same = before.every((p, k) => Math.abs(p - after[k]!) < 1e-6);
            const text = (lang: "tr" | "en", p: Float32Array) => [...p].map((v) => decimal(v, 3)[lang]).join(" / ");
            return {
              answer: same ? 0 : after[0]! > before[0]! ? 1 : 2,
              result: {
                tr: `Önce ${text("tr", before)}, sonra ${text("tr", after)}.`,
                en: `Before ${text("en", before)}, after ${text("en", after)}.`,
              },
            };
          },
          tryIt: { tab: "theory", section: "t-softmax" },
          explain: {
            tr: R`Olasılıkları skorların farkları belirler. 5 eklemek farkları korumuştu; 2 ile çarpmak ise onları da ikiye katlar: \(2 - 1 = 1\) olan fark \(2\) olur. En büyük skor öne çıkar ve dağılım sivrileşir. Eğitim boyunca skorlar büyüdükçe model daha emin konuşur; bu güven, tamamen siyah bir görüntüde bile yüksek bir olasılık verebilir.`,
            en: R`The differences between the scores set the probabilities. Adding 5 kept the differences; multiplying by 2 doubles them too: the difference \(2 - 1 = 1\) becomes \(2\). The largest score pulls ahead and the distribution gets sharper. As the scores grow during training the model sounds surer; that confidence can give a high probability even to a completely black image.`,
          },
        },
        {
          id: "t-curve",
          kind: "choice",
          q: {
            tr: "SimpleCNN eğitilirken ilk epoch'larda eğitim kaybı doğrulama kaybından yüksek çıkıyor. En olası açıklama hangisi?",
            en: "While SimpleCNN trains, the training loss is higher than the validation loss in the first epochs. What is the most likely explanation?",
          },
          options: [
            {
              text: { tr: "Model eğitim verisine aşırı uyuyor", en: "The model is overfitting the training data" },
              why: {
                tr: "Aşırı uyumda tablo terstir: eğitim kaybı düşük, doğrulama kaybı yüksek olur.",
                en: "Overfitting gives the opposite picture: a low training loss and a high validation loss.",
              },
            },
            {
              text: { tr: "Öğrenme oranı fazla yüksek seçilmiş", en: "The learning rate is set too high" },
              why: {
                tr: R`Fazla yüksek bir \(\eta\) iki kaybı birden bozar; doğrulamayı eğitimden iyi göstermez.`,
                en: R`Too high an \(\eta\) spoils both losses; it does not make validation look better than training.`,
              },
            },
            { text: { tr: "Dropout eğitimde açık, doğrulamada kapalı", en: "Dropout is on in training and off in validation" } },
            {
              text: {
                tr: "Doğrulama görüntüleri daha kolay olanlardan seçilmiş",
                en: "The validation images were picked from the easier ones",
              },
              why: {
                tr: R`Doğrulama görüntüleri eğitim setinden rastgele ayrılır (<code>random_split</code>); daha kolay olmaları beklenmez.`,
                en: R`The validation images are split off the training set at random (<code>random_split</code>); there is no reason for them to be easier.`,
              },
            },
          ],
          answer: 2,
          explain: {
            tr: R`Eğitimde Dropout her adımda nöronların bir kısmını kapatır ve ağ bu eksik hâliyle ölçülür; doğrulamada ağın tamamı çalışır. İkinci bir neden daha var: eğitim kaybı epoch boyunca, model henüz gelişirken ortalanır, doğrulama kaybı ise epoch sonundaki modelle ölçülür. Asıl alarm işareti, doğrulama kaybının yükselmeye başlamasıdır.`,
            en: R`In training, Dropout switches off part of the neurons at every step, and the network is measured in that reduced form; in validation the whole network runs. There is a second reason: the training loss is averaged over the epoch while the model is still improving, but the validation loss is measured with the model at the end of the epoch. The real warning sign is a validation loss that starts to rise.`,
          },
        },
      ],
    },
    {
      id: "px-model",
      title: { tr: "Eğitilmiş modelle deneyler", en: "Experiments on the trained model" },
      intro: {
        tr: R`Bu sorularda siteyle gelen eğitilmiş SimpleCNN gerçekten çalışır. Önce tahmininizi seçin; model ilk çalıştığında dosyalarının yüklenmesi birkaç saniye sürebilir.`,
        en: R`In these questions the trained SimpleCNN that comes with the site really runs. Pick your prediction first; the first run may take a few seconds to load the model's files.`,
      },
      read: { tab: "infer" },
      questions: [
        {
          id: "e-shift",
          kind: "model",
          q: {
            tr: R`İlk 40 test rakamını 2 piksel sağa kaydırıyoruz. Model bunların kaçını hâlâ doğru bilir?`,
            en: R`We move the first 40 test digits 2 pixels to the right. How many of them does the model still get right?`,
          },
          options: [
            { tr: "Yarısından azını", en: "Fewer than half" },
            { tr: "Yaklaşık yarısını", en: "About half" },
            { tr: "Neredeyse hepsini", en: "Nearly all of them" },
          ],
          answer: 2,
          run: async () => {
            const data = await lab();
            const right = await rightAfterShift(data, 2);
            return { answer: right >= 36 ? 2 : right >= 16 ? 1 : 0, values: [right, inkStaying(data, 2)] };
          },
          describe: ([right, stay]) => ({
            tr: `40 rakamın ${right} tanesini doğru bildi. Aynı yerde kalan mürekkepli piksel oranı ise yalnızca %${stay}.`,
            en: `It got ${right} of the 40 digits right, although only ${stay}% of the inked pixels stayed in place.`,
          }),
          tryIt: { tab: "infer" },
          explain: {
            tr: R`Mürekkepli piksellerin yarısından fazlası yer değiştirdi, ama filtreler aynı desenleri yeni konumda da bulur ve max pooling küçük kaymalara dayanıklılık sağlar. Bu dayanıklılık birkaç pikselle sınırlıdır: 4 piksel kaydırınca test setinin tamamında CNN'in doğruluğu da yaklaşık %82'ye düşer. MLP bölümündeki kaydırma deneyiyle karşılaştırın.`,
            en: R`More than half of the inked pixels moved, but the filters find the same patterns at the new position and max pooling adds robustness to small shifts. That robustness only lasts a few pixels: shifted by 4 pixels, the CNN too drops to about 82% on the whole test set. Compare it with the shifting experiment in the MLP chapter.`,
          },
        },
        {
          id: "e-far",
          kind: "model",
          q: {
            tr: R`Aynı 40 rakamı bu kez 8 piksel sağa kaydırıyoruz. Model kaçını doğru bilir?`,
            en: R`This time we move the same 40 digits 8 pixels to the right. How many does the model get right?`,
          },
          options: [
            { tr: "Neredeyse hepsini", en: "Nearly all of them" },
            { tr: "Yaklaşık yarısını ya da biraz fazlasını", en: "About half or a little more" },
            { tr: "Yarısından azını", en: "Fewer than half" },
          ],
          answer: 2,
          run: async () => {
            const right = await rightAfterShift(await lab(), 8);
            return { answer: right >= 36 ? 0 : right >= 20 ? 1 : 2, values: [right] };
          },
          describe: ([right]) => ({
            tr: `40 rakamın ${right} tanesini doğru bildi.`,
            en: `It got ${right} of the 40 digits right.`,
          }),
          tryIt: { tab: "infer" },
          explain: {
            tr: R`CNN küçük kaymalara dayanıklıdır, ama tam bir kayma değişmezliği yoktur. MNIST rakamları ortalanmıştır ve sondaki tam bağlı katmanlar özniteliklerin konumuna bağlıdır. Eğitimde görmediği kadar uzak bir konum modeli şaşırtır.`,
            en: R`A CNN is robust to small shifts but not fully shift invariant. MNIST digits are centered, and the fully connected layers at the end depend on where the features are. A position much farther than anything seen in training confuses the model.`,
          },
        },
        {
          id: "e-black",
          kind: "model",
          q: {
            tr: R`Modele içinde hiçbir şey olmayan, tamamen siyah bir görüntü veriyoruz. Ne olur?`,
            en: R`We give the model a completely black image with nothing in it. What happens?`,
          },
          options: [
            { tr: "Her rakama yaklaşık %10 verir", en: "It gives every digit about 10%" },
            { tr: "Tek bir rakama yüksek olasılık verir", en: "It gives one digit a high probability" },
            { tr: "Cevap veremez", en: "It cannot answer" },
          ],
          answer: 1,
          run: async () => {
            const { weights } = await lab();
            const { probabilities } = forward(weights, new Float32Array(784));
            const digit = topClass(probabilities);
            const peak = probabilities[digit]!;
            return { answer: peak > 0.5 ? 1 : 0, values: [digit, Math.round(peak * 1000) / 10] };
          },
          describe: ([digit, percent]) => ({
            tr: `Model %${decimal(percent!, 1).tr} olasılıkla "${digit}" dedi.`,
            en: `The model said "${digit}" with ${decimal(percent!, 1).en}% probability.`,
          }),
          tryIt: { tab: "infer" },
          explain: {
            tr: R`Softmax her zaman toplamı 1 olan bir dağılım üretir ve modelin "bilmiyorum" diye bir sınıfı yoktur. Yüksek olasılık, modelin haklı olduğu anlamına gelmez; eğitimde görmediği girdilerde de çok emin görünebilir.`,
            en: R`Softmax always produces a distribution that sums to 1, and the model has no "I don't know" class. A high probability does not mean the model is right; it can look very sure about inputs it never saw in training.`,
          },
        },
      ],
    },
  ];
}
