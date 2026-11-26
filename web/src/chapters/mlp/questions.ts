/**
 * Chapter 2, Practice tab: the questions, in the order of the Theory sections. Numbers and
 * predictions are worked out again by `compute` or `run` with the chapter's own engine (net.js):
 * tests/unit/practice.test.ts fails if a stated answer and the engine disagree.
 */
import { NET, type Example, type Net } from "../../engines/net.js";
import type { PracticeSection } from "../../shared/practice.ts";

const R = String.raw;

const XOR: Example[] = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 },
];

/** The same "random" start on every computer (mulberry32). */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const sigmoidNet = (sizes: number[], seed = 1): Net => NET.create(sizes, { act: "sigmoid", out: "sigmoid", rand: seeded(seed) });

function train(net: Net, epochs: number): Net {
  for (let epoch = 0; epoch < epochs; epoch++) NET.trainEpoch(net, XOR, { lr: 2, batch: XOR.length });
  return net;
}

const rightOf4 = (net: Net): number => Math.round(NET.accuracy(net, XOR) * XOR.length);
const decimal = (value: number, digits: number) => ({ tr: value.toFixed(digits).replace(".", ","), en: value.toFixed(digits) });

export const PRACTICE: PracticeSection[] = [
  {
    id: "px-act",
    title: { tr: "Aktivasyonlar", en: "Activations" },
    intro: {
      tr: R`Gizli katmanı öğretmek için türev gerekir. Sigmoid basamağın yumuşak hâlidir ve türevi \(\sigma'(z) = \sigma(z)\,(1 - \sigma(z))\)'dir.`,
      en: R`Teaching the hidden layer takes a derivative. The sigmoid is a smooth step, and its derivative is \(\sigma'(z) = \sigma(z)\,(1 - \sigma(z))\).`,
    },
    read: { tab: "theory", section: "ml-act" },
    questions: [
      {
        id: "a-deriv",
        kind: "number",
        q: { tr: R`Sigmoid'in \(z = 0\) noktasındaki türevi kaçtır?`, en: R`What is the derivative of the sigmoid at \(z = 0\)?` },
        label: { tr: R`\(\sigma'(0) =\)`, en: R`\(\sigma'(0) =\)` },
        answer: 0.25,
        compute: () => NET.ACT.sigmoid.d(0, NET.sigmoid(0)),
        hint: {
          tr: R`Önce \(\sigma(0) = 0{,}5\) değerini \(\sigma(z)\,(1 - \sigma(z))\) formülüne koyun.`,
          en: R`Put \(\sigma(0) = 0.5\) into \(\sigma(z)\,(1 - \sigma(z))\).`,
        },
        explain: {
          tr: R`\(\sigma'(0) = 0{,}5\cdot(1 - 0{,}5) = 0{,}25\). Bu, sigmoid'in en büyük eğimidir: iki uca gidildikçe türev 0'a yaklaşır ve nöron doyar.`,
          en: R`\(\sigma'(0) = 0.5\cdot(1 - 0.5) = 0.25\). This is the sigmoid's largest slope: towards both ends the derivative approaches 0 and the neuron saturates.`,
        },
      },
      {
        id: "a-step",
        kind: "choice",
        q: {
          tr: "Gizli katmanı eğitirken basamak fonksiyonu neden işe yaramaz?",
          en: "Why is the step function no use for training a hidden layer?",
        },
        options: [
          {
            text: { tr: "Çıktısı 0 ile 1 arasında kalmadığı için", en: "Because its output does not stay between 0 and 1" },
            why: {
              tr: "Basamağın çıktısı zaten yalnızca 0 ya da 1'dir. Sorun çıktının aralığı değil, eğimidir.",
              en: "The step's output is only ever 0 or 1. The problem is its slope, not its range.",
            },
          },
          {
            text: {
              tr: "Eğimi her yerde 0 olduğu için geriye taşınacak bir işaret kalmaz",
              en: "Because its slope is 0 everywhere, no signal is left to carry back",
            },
          },
          {
            text: { tr: "Hesaplaması çok yavaş olduğu için", en: "Because it is too slow to compute" },
            why: {
              tr: "Basamak hesaplanması en kolay fonksiyonlardan biridir. Eksik olan, kullanılabilir bir türevdir.",
              en: "The step is one of the easiest functions to compute. What it lacks is a usable derivative.",
            },
          },
          {
            text: { tr: "Yalnızca iki girdiyle çalışabildiği için", en: "Because it works only with two inputs" },
            why: {
              tr: "Aktivasyon tek bir sayıya, \\(z\\)'ye uygulanır; girdi sayısıyla ilgisi yoktur.",
              en: "The activation is applied to a single number, \\(z\\); the number of inputs does not matter.",
            },
          },
        ],
        answer: 1,
        compute: () => ([-2, -0.1, 0.1, 2].every((z) => NET.ACT.step.d(z, NET.ACT.step.f(z)) === 0) ? 1 : -1),
        explain: {
          tr: R`Ağırlığı biraz oynatmak basamağın çıktısını ya hiç değiştirmez ya da birden 0'dan 1'e atlatır. Türev 0 olunca gradyan inişi hangi yöne gideceğini bilemez.`,
          en: R`Nudging a weight either leaves the step's output unchanged or makes it jump from 0 to 1. With a zero derivative, gradient descent cannot tell which way to go.`,
        },
      },
      {
        id: "a-boundary",
        kind: "choice",
        q: {
          tr: "Tek bir nöronda basamağı sigmoidle değiştirmek karar sınırını nasıl etkiler?",
          en: "In a single neuron, how does swapping the step for the sigmoid affect the decision boundary?",
        },
        options: [
          {
            text: { tr: "Sınır eğri olur", en: "The boundary becomes curved" },
            why: {
              tr: R`Eğri sınır gizli katmanla gelir. Tek nöronda \(\hat y > 0{,}5\) koşulu yine \(z > 0\) demektir.`,
              en: R`Curved boundaries come with a hidden layer. In a single neuron, \(\hat y > 0.5\) still means \(z > 0\).`,
            },
          },
          {
            text: { tr: "Sınır kaybolur, her nokta iki sınıfa da girer", en: "The boundary disappears, every point is in both classes" },
            why: {
              tr: R`Çıktı bir olasılık olur ama 0,5 eşiğiyle her nokta yine tek bir sınıfa düşer.`,
              en: R`The output becomes a probability, but with the 0.5 threshold every point still falls into one class.`,
            },
          },
          { text: { tr: "Değiştirmez, sınır yine bir doğrudur", en: "It does not change it: the boundary is still a line" } },
          {
            text: { tr: "Sınır orijinden geçmek zorunda kalır", en: "The boundary has to pass through the origin" },
            why: {
              tr: R`Bias \(b\) hâlâ sınırı kaydırır; aktivasyon bunu değiştirmez.`,
              en: R`The bias \(b\) still shifts the boundary; the activation does not change that.`,
            },
          },
        ],
        answer: 2,
        compute: () => ([-5, -0.5, -1e-3, 1e-3, 0.5, 5].every((z) => NET.sigmoid(z) > 0.5 === z > 0) ? 2 : -1),
        explain: {
          tr: R`\(\sigma(z) > 0{,}5 \iff z > 0\) olduğu için sınır aynı doğrudur. Yeni olan, her ağırlığın kaybı nasıl etkilediğinin artık hesaplanabilmesidir.`,
          en: R`Since \(\sigma(z) > 0.5 \iff z > 0\), the boundary is the same line. What is new is that we can now compute how every weight affects the loss.`,
        },
      },
    ],
  },
  {
    id: "px-hidden",
    title: { tr: "Gizli katman", en: "The hidden layer" },
    intro: {
      tr: R`Gizli nöronlar her noktaya yeni koordinatlar verir: \(h_1, h_2\). Çıktı nöronu bu gizli uzayda bir perceptron gibi çalışır.`,
      en: R`The hidden neurons give every point new coordinates: \(h_1, h_2\). The output neuron works like a perceptron in this hidden space.`,
    },
    read: { tab: "theory", section: "ml-hidden" },
    questions: [
      {
        id: "h-xor",
        kind: "predict",
        q: {
          tr: R`XOR'u aynı öğrenme oranıyla (\(\eta = 2\)) ve 3000 epoch boyunca iki ağla eğitiyoruz: gizli katmansız (2 → 1) ve iki gizli nöronlu (2 → 2 → 1). Bu başlangıç ağırlıklarıyla hangisi dört noktayı da doğru sınıflar?`,
          en: R`We train XOR with the same learning rate (\(\eta = 2\)) for 3000 epochs on two networks: without a hidden layer (2 → 1) and with two hidden neurons (2 → 2 → 1). With these starting weights, which one classifies all four points correctly?`,
        },
        options: [
          { tr: "Yalnızca gizli katmansız ağ", en: "Only the network without a hidden layer" },
          { tr: "Yalnızca iki gizli nöronlu ağ", en: "Only the network with two hidden neurons" },
          { tr: "İkisi de", en: "Both" },
          { tr: "Hiçbiri", en: "Neither" },
        ],
        answer: 1,
        run: () => {
          const flat = rightOf4(train(sigmoidNet([2, 1]), 3000));
          const hidden = rightOf4(train(sigmoidNet([2, 2, 1]), 3000));
          const answer = flat === 4 ? (hidden === 4 ? 2 : 0) : hidden === 4 ? 1 : 3;
          return {
            answer,
            result: {
              tr: `Gizli katmansız ağ 4 noktanın ${flat} tanesini, iki gizli nöronlu ağ ${hidden} tanesini doğru sınıfladı.`,
              en: `The network without a hidden layer got ${flat} of the 4 points right, the one with two hidden neurons got ${hidden}.`,
            },
          };
        },
        tryIt: { tab: "play" },
        explain: {
          tr: R`Gizli katmansız ağın sınırı tek bir doğrudur ve XOR'u ayıramaz. Gizli katman noktaları, bir doğrunun ayırabileceği yeni bir gösterime taşır. Gizli katmanlı ağ da bazı başlangıçlarda takılabilir (Teori 2, "Bazen takılır").`,
          en: R`Without a hidden layer the boundary is a single line, which cannot separate XOR. The hidden layer moves the points into a new representation that a line can separate. The network with a hidden layer can still get stuck from some starts (Theory 2, "Sometimes it gets stuck").`,
        },
      },
      {
        id: "h-count",
        kind: "number",
        q: {
          tr: "2 girdi, 2 gizli nöron ve 1 çıktı nöronu olan ağda toplam kaç parametre (ağırlık ve bias) vardır?",
          en: "A network has 2 inputs, 2 hidden neurons and 1 output neuron. How many parameters (weights and biases) does it have in all?",
        },
        label: { tr: "parametre sayısı =", en: "number of parameters =" },
        answer: 9,
        compute: () => NET.paramCount([2, 2, 1]),
        hint: {
          tr: "Her nöronun, girdi sayısı kadar ağırlığı ve bir biası vardır. Üç nöronu ayrı ayrı sayın.",
          en: "Each neuron has one weight per input and one bias. Count the three neurons one by one.",
        },
        explain: {
          tr: R`Her gizli nöron: 2 ağırlık + 1 bias = 3. Çıktı nöronu: 2 ağırlık + 1 bias = 3. Toplam \(3 + 3 + 3 = 9\).`,
          en: R`Each hidden neuron: 2 weights + 1 bias = 3. The output neuron: 2 weights + 1 bias = 3. In all, \(3 + 3 + 3 = 9\).`,
        },
      },
      {
        id: "h-space",
        kind: "choice",
        q: { tr: "Gizli katman XOR'u çözerken aslında ne yapar?", en: "What does the hidden layer actually do to solve XOR?" },
        options: [
          {
            text: {
              tr: "Noktaları, tek bir doğruyla ayrılabilecekleri yeni koordinatlara taşır",
              en: "It moves the points to new coordinates where a single line separates them",
            },
          },
          {
            text: { tr: "Eğitim örneklerini bir tabloda ezberler", en: "It memorizes the training examples in a table" },
            why: {
              tr: "Ağda tablo yoktur; yalnızca ağırlıklar vardır. Bu ağırlıklar veriyi yeni bir gösterime çevirir.",
              en: "There is no table in the network, only weights. Those weights turn the data into a new representation.",
            },
          },
          {
            text: { tr: "Her nöron için ayrı bir öğrenme oranı seçer", en: "It picks a separate learning rate for each neuron" },
            why: {
              tr: "Öğrenme oranı eğitimin bir ayarıdır. Gizli katmanın işi veriyi dönüştürmektir.",
              en: "The learning rate is a setting of the training. The hidden layer's job is to transform the data.",
            },
          },
          {
            text: { tr: "Girdilerin sayısını dörde çıkarır", en: "It raises the number of inputs to four" },
            why: {
              tr: R`Bu ağda gizli uzay iki boyutludur (\(h_1, h_2\)), yani boyut artmaz. Değişen, noktaların konumudur.`,
              en: R`The hidden space here has two dimensions (\(h_1, h_2\)), so the size does not grow. What changes is where the points are.`,
            },
          },
        ],
        answer: 0,
        explain: {
          tr: R`Gizli katman veriyi sonraki katmanın işini kolaylaştıran bir gösterime çevirir. Bu gösterimi biz tasarlamayız, ağ onu geri yayılımla kendisi öğrenir.`,
          en: R`The hidden layer turns the data into a representation that makes the next layer's job easy. We do not design it; the network learns it by itself with backpropagation.`,
        },
      },
    ],
  },
  {
    id: "px-loss",
    title: { tr: "Kayıp ve gradyan inişi", en: "The loss and gradient descent" },
    intro: {
      tr: R`İkili çapraz entropi: \(L = -\big[y\ln\hat y + (1-y)\ln(1-\hat y)\big]\). Gradyan inişi her adımda eğimin tersine gider: \(w \leftarrow w - \eta\,L'(w)\).`,
      en: R`Binary cross-entropy: \(L = -\big[y\ln\hat y + (1-y)\ln(1-\hat y)\big]\). Gradient descent steps against the slope every time: \(w \leftarrow w - \eta\,L'(w)\).`,
    },
    read: { tab: "theory", section: "ml-loss" },
    questions: [
      {
        id: "l-bce",
        kind: "number",
        q: {
          tr: R`Doğru sınıf \(y = 1\) ve model \(\hat y = 0{,}5\) diyor. Çapraz entropi kaybı kaçtır? (İki ondalığa yuvarlayın.)`,
          en: R`The right class is \(y = 1\) and the model says \(\hat y = 0.5\). What is the cross-entropy loss? (Round to two decimals.)`,
        },
        label: { tr: R`\(L \approx\)`, en: R`\(L \approx\)` },
        answer: 0.69,
        tolerance: 0.006,
        compute: () => NET.lossOne(0.5, 1),
        hint: { tr: R`\(y = 1\) iken formül \(L = -\ln\hat y\)'ye iner.`, en: R`With \(y = 1\) the formula becomes \(L = -\ln\hat y\).` },
        explain: {
          tr: R`\(L = -\ln 0{,}5 = \ln 2 \approx 0{,}69\). Bu, hiçbir şey bilmeyen bir modelin kaybıdır: iki sınıfa da eşit olasılık veriyor.`,
          en: R`\(L = -\ln 0.5 = \ln 2 \approx 0.69\). This is the loss of a model that knows nothing: it gives both classes the same probability.`,
        },
      },
      {
        id: "l-grad",
        kind: "number",
        q: {
          tr: R`Çıktı sigmoid, kayıp çapraz entropi. \(y = 1\) ve \(\hat y = 0{,}8\) için çıktı nöronunun hatası \(\delta_o = \partial L / \partial z_o\) kaçtır?`,
          en: R`The output is a sigmoid and the loss is cross-entropy. For \(y = 1\) and \(\hat y = 0.8\), what is the output neuron's error \(\delta_o = \partial L / \partial z_o\)?`,
        },
        label: { tr: R`\(\delta_o =\)`, en: R`\(\delta_o =\)` },
        answer: -0.2,
        compute: () => {
          const net = sigmoidNet([1, 1]);
          net.layers[0]!.W = [[0]];
          net.layers[0]!.b = [Math.log(4)]; // σ(ln 4) = 0.8
          return NET.gradient(net, [{ x: [1], y: 1 }])[0]!.b[0]!;
        },
        hint: {
          tr: R`Sigmoid ile çapraz entropi birlikte çok basit bir türev verir: \(\hat y - y\).`,
          en: R`The sigmoid and cross-entropy together give a very simple derivative: \(\hat y - y\).`,
        },
        explain: {
          tr: R`\(\delta_o = \hat y - y = 0{,}8 - 1 = -0{,}2\). Negatif işaret, \(z_o\) büyürse kaybın azalacağını söyler.`,
          en: R`\(\delta_o = \hat y - y = 0.8 - 1 = -0.2\). The negative sign says the loss falls if \(z_o\) grows.`,
        },
      },
      {
        id: "l-mse",
        kind: "predict",
        q: {
          tr: R`Tek ağırlıklı model \(\hat y = \sigma(w x)\) çok yanlış durumda: \(x = 1\), \(y = 1\), \(w = -6\). Hangi kaybın gradyanı daha büyüktür?`,
          en: R`The one-weight model \(\hat y = \sigma(w x)\) is very wrong: \(x = 1\), \(y = 1\), \(w = -6\). Which loss has the larger gradient?`,
        },
        options: [
          { tr: "Çapraz entropi", en: "Cross-entropy" },
          { tr: "Ortalama kare hata (MSE)", en: "Mean squared error (MSE)" },
          { tr: "İkisi yaklaşık aynı", en: "About the same" },
        ],
        answer: 0,
        run: () => {
          const net = sigmoidNet([1, 1]);
          net.layers[0]!.W = [[-6]];
          net.layers[0]!.b = [0];
          const data = [{ x: [1], y: 1 }];
          const bce = Math.abs(NET.gradient(net, data, "bce")[0]!.W[0]![0]!);
          const mse = Math.abs(NET.gradient(net, data, "mse")[0]!.W[0]![0]!);
          const a = decimal(bce, 3);
          const b = decimal(mse, 3);
          return {
            answer: bce > 2 * mse ? 0 : mse > 2 * bce ? 1 : 2,
            result: {
              tr: `Gradyanın büyüklüğü: çapraz entropide ${a.tr}, MSE'de ${b.tr}.`,
              en: `Size of the gradient: ${a.en} for cross-entropy, ${b.en} for MSE.`,
            },
          };
        },
        tryIt: { tab: "theory", section: "ml-loss" },
        explain: {
          tr: R`Sigmoid doyduğunda MSE'nin gradyanındaki \(\hat y\,(1-\hat y)\) çarpanı neredeyse 0 olur ve öğrenme yavaşlar. Çapraz entropide bu çarpan sadeleşir, gradyan \(\hat y - y\) kalır.`,
          en: R`When the sigmoid saturates, the factor \(\hat y\,(1-\hat y)\) in the MSE gradient is almost 0 and learning slows down. In cross-entropy that factor cancels, leaving the gradient \(\hat y - y\).`,
        },
      },
    ],
  },
  {
    id: "px-backprop",
    title: { tr: "Geri yayılım", en: "Backpropagation" },
    intro: {
      tr: R`Hata kayıptan geriye doğru taşınır: \(\delta = \partial L / \partial z\). Bir ağırlığın gradyanı, girdisiyle hedef nöronun \(\delta\)'sının çarpımıdır.`,
      en: R`The error is carried back from the loss: \(\delta = \partial L / \partial z\). A weight's gradient is its input times the \(\delta\) of the neuron it feeds.`,
    },
    read: { tab: "theory", section: "ml-backprop" },
    questions: [
      {
        id: "bp-zero",
        kind: "choice",
        q: {
          tr: R`2 → 2 → 1 ağında \(\mathbf x = (1, 0)\) örneği için hangi gizli ağırlıkların gradyanı 0 olur?`,
          en: R`In a 2 → 2 → 1 network, which hidden weights get a zero gradient for the example \(\mathbf x = (1, 0)\)?`,
        },
        options: [
          {
            text: { tr: R`\(x_1\)'den gelen ağırlıklar`, en: R`The weights coming from \(x_1\)` },
            why: {
              tr: R`Gradyan girdiyle çarpılır ve \(x_1 = 1\)'dir, yani bu ağırlıklar güncellenir.`,
              en: R`The gradient is multiplied by the input, and \(x_1 = 1\), so these weights are updated.`,
            },
          },
          { text: { tr: R`\(x_2\)'den gelen ağırlıklar`, en: R`The weights coming from \(x_2\)` } },
          {
            text: { tr: "Bütün gizli ağırlıklar", en: "All the hidden weights" },
            why: {
              tr: R`\(x_1 = 1\) olduğu için \(x_1\)'in ağırlıklarının gradyanı genelde 0 değildir.`,
              en: R`Since \(x_1 = 1\), the gradients of the weights from \(x_1\) are generally not 0.`,
            },
          },
          {
            text: { tr: "Hiçbiri", en: "None of them" },
            why: {
              tr: R`\(\partial L/\partial w_{ji} = \delta_j\,x_i\) formülünde \(x_2 = 0\) çarpanına bakın.`,
              en: R`Look at the factor \(x_2 = 0\) in \(\partial L/\partial w_{ji} = \delta_j\,x_i\).`,
            },
          },
        ],
        answer: 1,
        compute: () => {
          const grad = NET.gradient(sigmoidNet([2, 2, 1], 3), [{ x: [1, 0], y: 1 }])[0]!.W;
          return grad.every((row) => row[1] === 0 && row[0] !== 0) ? 1 : -1;
        },
        explain: {
          tr: R`\(\partial L/\partial w_{ji} = \delta_j\,x_i\) ve \(x_2 = 0\). Girdisi 0 olan ağırlık bu örnekte çıktıyı etkilemediği için güncellenmez.`,
          en: R`\(\partial L/\partial w_{ji} = \delta_j\,x_i\) and \(x_2 = 0\). A weight whose input is 0 has no effect on the output for this example, so it is not updated.`,
        },
      },
      {
        id: "bp-delta",
        kind: "number",
        q: {
          tr: R`\(y = 1\), \(\hat y = 0{,}75\), çıktı ağırlığı \(v_1 = 2\) ve gizli çıktı \(h_1 = 0{,}5\). Gizli nöronun hatası \(\delta_1 = \delta_o\, v_1\, h_1 (1 - h_1)\) kaçtır?`,
          en: R`\(y = 1\), \(\hat y = 0.75\), the output weight is \(v_1 = 2\) and the hidden output is \(h_1 = 0.5\). What is the hidden neuron's error \(\delta_1 = \delta_o\, v_1\, h_1 (1 - h_1)\)?`,
        },
        label: { tr: R`\(\delta_1 =\)`, en: R`\(\delta_1 =\)` },
        answer: -0.125,
        compute: () => {
          const net = sigmoidNet([1, 1, 1]);
          net.layers[0]!.W = [[0]];
          net.layers[0]!.b = [0]; // h = σ(0) = 0.5
          net.layers[1]!.W = [[2]];
          net.layers[1]!.b = [Math.log(3) - 1]; // z_o = 2·0.5 + c = ln 3, so ŷ = 0.75
          return NET.backward(net, NET.forward(net, [1]), 1).deltas[0]![0]!;
        },
        hint: {
          tr: R`Önce \(\delta_o = \hat y - y\)'yi bulun, sonra üç çarpanla çarpın.`,
          en: R`First find \(\delta_o = \hat y - y\), then multiply by the three factors.`,
        },
        explain: {
          tr: R`\(\delta_o = 0{,}75 - 1 = -0{,}25\). \(\delta_1 = -0{,}25\cdot 2\cdot 0{,}5\cdot 0{,}5 = -0{,}125\). Sigmoid'in eğimi en fazla 0,25 olduğu için, ağırlıklar büyük değilse hata geriye giderken küçülür.`,
          en: R`\(\delta_o = 0.75 - 1 = -0.25\). \(\delta_1 = -0.25\cdot 2\cdot 0.5\cdot 0.5 = -0.125\). The sigmoid's slope is at most 0.25, so, if the weights are not large, the error shrinks as it goes back.`,
        },
      },
      {
        id: "bp-sym",
        kind: "predict",
        q: {
          tr: R`2 → 2 → 1 ağının iki gizli nöronunu tamamen aynı başlatıyoruz: giren ağırlıkları, biasları ve çıktıya giden ağırlıkları birebir aynı. Sonra XOR'u 2000 epoch eğitiyoruz. Ne olur?`,
          en: R`We start the two hidden neurons of a 2 → 2 → 1 network exactly alike: the same incoming weights, the same biases and the same weights to the output. Then we train XOR for 2000 epochs. What happens?`,
        },
        options: [
          { tr: "Rastgele başlangıçtaki gibi öğrenir", en: "It learns as it does from a random start" },
          { tr: "İki nöron hep aynı kalır ve ağ XOR'u çözemez", en: "The two neurons stay identical and the network cannot solve XOR" },
          { tr: "Nöronlardan biri kendiliğinden sıfırlanır", en: "One of the neurons resets itself to zero" },
        ],
        answer: 1,
        run: () => {
          const net = sigmoidNet([2, 2, 1]);
          net.layers[0]!.W = [
            [0.5, -0.3],
            [0.5, -0.3],
          ];
          net.layers[0]!.b = [0.1, 0.1];
          net.layers[1]!.W = [[0.4, 0.4]];
          net.layers[1]!.b = [-0.2];
          train(net, 2000);
          const [first, second] = net.layers[0]!.W;
          const same = first![0] === second![0] && first![1] === second![1] && net.layers[0]!.b[0] === net.layers[0]!.b[1];
          const right = rightOf4(net);
          return {
            answer: same && right < 4 ? 1 : right === 4 ? 0 : 2,
            result: same
              ? {
                  tr: `2000 epoch sonra iki nöronun ağırlıkları hâlâ birebir aynı. Ağ 4 noktanın ${right} tanesini doğru sınıflıyor.`,
                  en: `After 2000 epochs the two neurons' weights are still exactly the same. The network gets ${right} of the 4 points right.`,
                }
              : {
                  tr: `Ağırlıklar ayrıştı. Ağ 4 noktanın ${right} tanesini doğru sınıflıyor.`,
                  en: `The weights drifted apart. The network gets ${right} of the 4 points right.`,
                },
          };
        },
        tryIt: { tab: "math", section: "mm-train" },
        explain: {
          tr: R`Aynı başlayan iki nöron aynı \(\delta\)'yı, dolayısıyla aynı gradyanı alır ve hiçbir zaman ayrışamaz. Ağ fiilen tek gizli nöronlu kalır. Ağırlıkları bu yüzden rastgele başlatırız.`,
          en: R`Two neurons that start the same get the same \(\delta\), so the same gradient, and can never drift apart. The network is in effect left with one hidden neuron. That is why we start the weights at random.`,
        },
      },
      {
        id: "bp-why",
        kind: "choice",
        q: {
          tr: "Gradyanları sondan başa hesaplamanın asıl avantajı nedir?",
          en: "What is the main advantage of computing the gradients from the end to the start?",
        },
        options: [
          {
            text: { tr: "Ağın daha az ağırlığa ihtiyaç duyması", en: "The network needs fewer weights" },
            why: {
              tr: "Ağırlık sayısı ağın yapısına bağlıdır; hesap sırası onu değiştirmez.",
              en: "The number of weights depends on the architecture; the order of the computation does not change it.",
            },
          },
          {
            text: { tr: "İleri hesaba hiç gerek kalmaması", en: "The forward pass is no longer needed" },
            why: {
              tr: "Geri hesap, ileri hesabın bulduğu değerleri kullanır; önce ileri hesap yapılır.",
              en: "The backward pass uses the values found by the forward pass, which comes first.",
            },
          },
          {
            text: {
              tr: "Bütün gradyanları yaklaşık iki ileri hesap maliyetiyle bulması",
              en: "It finds all the gradients for about the cost of two forward passes",
            },
          },
          {
            text: { tr: "Hatanın geriye giderken büyümesi", en: "The error grows on its way back" },
            why: {
              tr: "Sigmoid'de tersi olur: eğim en fazla 0,25 olduğu için hata genelde küçülür.",
              en: "With the sigmoid the opposite happens: the slope is at most 0.25, so the error usually shrinks.",
            },
          },
        ],
        answer: 2,
        explain: {
          tr: R`Her ara sonuç bir kez kullanılır, bu yüzden geri hesap, ağ ne kadar büyük olursa olsun, yaklaşık iki ileri hesap tutar. Her ağırlığı tek tek oynatmak ise en az ağırlık sayısı kadar ileri hesap ister: SimpleCNN için 582 bin.`,
          en: R`Every intermediate result is used once, so the backward pass costs about two forward passes, however large the network is. Nudging each weight in turn would take at least as many forward passes as there are weights: 582 thousand for SimpleCNN.`,
        },
      },
    ],
  },
  {
    id: "px-capacity",
    title: { tr: "Kaç nöron yeter?", en: "How many neurons?" },
    intro: {
      tr: R`Her gizli tanh nöronu bir "yumuşak basamak" üretir, çıktı ise bunların ağırlıklı toplamıdır.`,
      en: R`Each hidden tanh neuron makes a "soft step", and the output is their weighted sum.`,
    },
    read: { tab: "theory", section: "ml-capacity" },
    questions: [
      {
        id: "c-bump",
        kind: "choice",
        q: {
          tr: "Tek gizli katmanlı bir tanh ağıyla bir tepe (önce yükselip sonra inen eğri) çizmek için en az kaç gizli nöron gerekir?",
          en: "With one hidden layer of tanh neurons, how many hidden neurons does it take at least to draw a bump (a curve that rises and then falls)?",
        },
        options: [
          {
            text: { tr: "1", en: "1" },
            why: {
              tr: "Tek tanh nöronu yalnızca tek yönde değişir. Tepe ise hem yükselip hem inmek zorundadır.",
              en: "A single tanh neuron changes in one direction only. A bump has to rise and fall.",
            },
          },
          { text: { tr: "2", en: "2" } },
          {
            text: { tr: "4", en: "4" },
            why: {
              tr: "Dört nöronla da yapılır, ama en az gereken sayı daha küçüktür: biri yukarı, biri aşağı iki basamak yeter.",
              en: "Four neurons can do it too, but the least is smaller: one step up and one step down are enough.",
            },
          },
          {
            text: { tr: "Sonsuz sayıda", en: "Infinitely many" },
            why: {
              tr: "Sonsuz nöron gerekmez: iki yumuşak basamağın farkı bir tepe verir.",
              en: "It does not take infinitely many: the difference of two soft steps is a bump.",
            },
          },
        ],
        answer: 1,
        compute: () => {
          const bump = (x: number) => Math.tanh(4 * (x + 1)) - Math.tanh(4 * (x - 1));
          return bump(0) > bump(-3) + 1 && bump(0) > bump(3) + 1 ? 1 : -1;
        },
        explain: {
          tr: R`Biri yukarı, biri aşağı giden iki yumuşak basamağın toplamı bir tepe yapar: örneğin \(\tanh(k(x+1)) - \tanh(k(x-1))\).`,
          en: R`Two soft steps, one up and one down, add up to a bump: for example \(\tanh(k(x+1)) - \tanh(k(x-1))\).`,
        },
      },
      {
        id: "c-params",
        kind: "number",
        q: {
          tr: R`1 girdi, \(H = 5\) gizli nöron ve 1 çıktı nöronu olan ağda kaç parametre vardır?`,
          en: R`A network has 1 input, \(H = 5\) hidden neurons and 1 output neuron. How many parameters does it have?`,
        },
        label: { tr: "parametre sayısı =", en: "number of parameters =" },
        answer: 16,
        compute: () => NET.paramCount([1, 5, 1]),
        hint: {
          tr: "Her gizli nöronda 1 ağırlık ve 1 bias, çıktı nöronunda 5 ağırlık ve 1 bias vardır.",
          en: "Each hidden neuron has 1 weight and 1 bias; the output neuron has 5 weights and 1 bias.",
        },
        explain: {
          tr: R`Gizli katman: \(5\cdot(1 + 1) = 10\). Çıktı: \(5 + 1 = 6\). Toplam \(10 + 6 = 16\).`,
          en: R`Hidden layer: \(5\cdot(1 + 1) = 10\). Output: \(5 + 1 = 6\). In all, \(10 + 6 = 16\).`,
        },
      },
      {
        id: "c-theorem",
        kind: "choice",
        q: {
          tr: "Evrensel yaklaşım teoremi aşağıdakilerden hangisini garanti etmez?",
          en: "Which of these does the universal approximation theorem not guarantee?",
        },
        options: [
          {
            text: { tr: "Tek gizli katmanın yeterli olabileceğini", en: "That one hidden layer can be enough" },
            why: {
              tr: "Teoremin söylediği tam olarak budur: tek gizli katman, yeterince nöronla yeter.",
              en: "That is exactly what the theorem says: one hidden layer is enough, with enough neurons.",
            },
          },
          {
            text: { tr: "Sürekli fonksiyonlara yaklaşılabileceğini", en: "That continuous functions can be approximated" },
            why: {
              tr: "Teorem kapalı bir aralıktaki her sürekli fonksiyon için bunu söyler.",
              en: "The theorem says this for every continuous function on a closed interval.",
            },
          },
          {
            text: { tr: "Yaklaşımın istenen kadar iyi yapılabileceğini", en: "That the approximation can be made as close as we like" },
            why: {
              tr: "Yeterince nöronla hata istenen kadar küçültülebilir; teorem bunu söyler.",
              en: "With enough neurons the error can be made as small as we like; the theorem says so.",
            },
          },
          { text: { tr: "Gradyan inişinin o ağırlıkları bulacağını", en: "That gradient descent will find those weights" } },
        ],
        answer: 3,
        explain: {
          tr: "Teorem iyi ağırlıkların var olduğunu söyler, ama kaç nöron gerektiğini ya da eğitimin onları bulacağını söylemez. Eğitim verisine mükemmel uymak da iyi genelleme demek değildir.",
          en: "The theorem says good weights exist, but not how many neurons it takes or that training will find them. Fitting the training data perfectly does not mean good generalization either.",
        },
      },
    ],
  },
  {
    id: "px-images",
    title: { tr: "Görüntülerde MLP", en: "The MLP on images" },
    intro: {
      tr: R`Bir MLP görüntüyü düzleştirir: her piksel ayrı bir girdidir ve her gizli nöron her piksele ayrı bir ağırlık verir.`,
      en: R`An MLP flattens the image: every pixel is a separate input, and every hidden neuron gives every pixel its own weight.`,
    },
    read: { tab: "theory", section: "ml-images" },
    questions: [
      {
        id: "i-params",
        kind: "number",
        q: {
          tr: R`10×10 piksellik bir görüntü, 8 nöronlu bir gizli katmana giriyor. Bu katmanda kaç parametre (ağırlık ve bias) vardır?`,
          en: R`A 10×10-pixel image goes into a hidden layer of 8 neurons. How many parameters (weights and biases) does this layer have?`,
        },
        label: { tr: "parametre sayısı =", en: "number of parameters =" },
        answer: 808,
        compute: () => NET.paramCount([100, 8]),
        hint: {
          tr: R`Düzleştirilince 100 girdi olur. Her nöronda 100 ağırlık ve 1 bias vardır.`,
          en: R`Flattened, that is 100 inputs. Each neuron has 100 weights and 1 bias.`,
        },
        explain: {
          tr: R`\(100\cdot 8 + 8 = 808\). Parametre sayısı görüntünün piksel sayısıyla doğrudan büyür: MNIST'te 512 nöronlu tek bir katman 401.920 parametre demektir.`,
          en: R`\(100\cdot 8 + 8 = 808\). The parameters grow directly with the number of pixels: on MNIST a single layer of 512 neurons has 401,920 parameters.`,
        },
      },
      {
        id: "i-shift",
        kind: "predict",
        q: {
          tr: "5×5 görüntüde tek piksel kalınlığında, 5 piksellik dikey bir çizgi var. Çizgiyi bir piksel sağa kaydırıyoruz. Mürekkepli piksellerden kaçı aynı girdide kalır?",
          en: "A 5×5 image has a vertical line, one pixel wide and 5 pixels long. We move the line one pixel to the right. How many of the inked pixels stay on the same input?",
        },
        options: [
          { tr: "Hepsi", en: "All of them" },
          { tr: "Yaklaşık yarısı", en: "About half" },
          { tr: "Hiçbiri", en: "None" },
        ],
        answer: 2,
        run: () => {
          const line = (column: number) => Array.from({ length: 25 }, (_, i) => (i % 5 === column ? 1 : 0));
          const before = line(2);
          const after = line(3);
          const stay = before.filter((v, i) => v === 1 && after[i] === 1).length;
          return {
            answer: stay === 5 ? 0 : stay === 0 ? 2 : 1,
            result: {
              tr: `5 mürekkepli pikselin ${stay} tanesi aynı girdide kaldı.`,
              en: `${stay} of the 5 inked pixels stayed on the same input.`,
            },
          };
        },
        tryIt: { tab: "theory", section: "ml-images" },
        explain: {
          tr: R`Göz için aynı çizgidir, ama MLP için tamamen başka girdiler ve başka ağırlıklar devreye girer. Ağ aynı deseni her konumda ayrı ayrı öğrenmek zorunda kalır.`,
          en: R`To the eye it is the same line, but for the MLP entirely different inputs and weights come into play. The network has to learn the same pattern separately at every position.`,
        },
      },
      {
        id: "i-perm",
        kind: "choice",
        q: {
          tr: "Piksellerin sırasını her görüntüde aynı şekilde karıştırıp MLP'yi bu görüntülerle eğitirsek ne olur?",
          en: "If we shuffle the order of the pixels the same way in every image and train an MLP on those images, what happens?",
        },
        options: [
          {
            text: { tr: "Hiç öğrenemez", en: "It cannot learn at all" },
            why: {
              tr: "MLP için her piksel yalnızca numaralı bir girdidir; numaraların sırası öğrenmeyi engellemez.",
              en: "To an MLP every pixel is just a numbered input; the order of the numbers does not stop it learning.",
            },
          },
          { text: { tr: "Karıştırılmamış görüntülerdeki başarıya ulaşır", en: "It does as well as on the unshuffled images" } },
          {
            text: { tr: "Başarısı yaklaşık yarıya düşer", en: "Its accuracy falls by about half" },
            why: {
              tr: "Karıştırma her görüntüde aynı olduğu için bilgi kaybolmaz; MLP komşuluğu zaten kullanmıyordu.",
              en: "The shuffle is the same in every image, so no information is lost; the MLP was not using neighbours anyway.",
            },
          },
          {
            text: { tr: "Çok daha başarılı olur", en: "It does much better" },
            why: {
              tr: "Karıştırma yeni bilgi eklemez; yalnızca girdilerin numaralarını değiştirir.",
              en: "Shuffling adds no information; it only renumbers the inputs.",
            },
          },
        ],
        answer: 1,
        explain: {
          tr: R`MLP hangi pikselin hangisinin yanında olduğunu hiç kullanmaz. Evrişim bu yüzden farklıdır: küçük bir filtre komşu piksellere bakar ve aynı ağırlıkları görüntünün her yerinde kullanır.`,
          en: R`An MLP never uses which pixel is next to which. That is what makes convolution different: a small filter looks at neighbouring pixels and uses the same weights everywhere in the image.`,
        },
      },
    ],
  },
];
