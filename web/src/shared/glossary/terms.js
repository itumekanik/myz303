/*
 * Glossary: short definitions of the terms worth learning properly.
 *
 * - In explanation texts, [[key|text]] becomes a term link. Clicking it opens a small term
 *   card next to it; terms inside the card can be followed too (with a back button).
 * - The "Glossary" button in the top bar opens a searchable side panel with every term,
 *   grouped by topic, each with a link to the Theory or Coding section that shows it.
 *
 * Every chapter page loads it. A term's places may be on another chapter's page
 * ({ chapter, tab, section }; no chapter means the CNN page); they are listed with this page's first.
 *
 * Uses the helpers from i18n.js and core.js (L, h, $, nav, CHAPTER, TAB_NAMES, goToChapterPlace).
 */
import { hooks } from "../page.ts";
import { I18N, L } from "../i18n.js";
import { CHAPTER, PLACE_TITLES, TAB_NAMES, chapterOf } from "../core.js";


export const GLOSSARY_GROUPS = [
  { key: "neuron", title: { tr: "Yapay nöron ve perceptron", en: "The artificial neuron and the perceptron" } },
  { key: "mlp", title: { tr: "Çok katmanlı ağlar", en: "Multilayer networks" } },
  { key: "train", title: { tr: "Eğitim süreci", en: "The training process" } },
  { key: "data", title: { tr: "Veri ve değerlendirme", en: "Data and evaluation" } },
  { key: "layers", title: { tr: "Model ve katmanlar", en: "The model and its layers" } },
  { key: "run", title: { tr: "Modeli çalıştırmak", en: "Running the model" } },
  { key: "detect", title: { tr: "Nesne tespiti ve segmentasyon", en: "Object detection and segmentation" } },
];

/*
 * name: shown in the current language, the other language is shown next to it.
 * aka:  extra words the search should find.
 * def:  the definition (static markup; [[key|text]] links another term).
 * model: how it looks in SimpleCNN (optional; modelLabel replaces the "In SimpleCNN" label).
 * see:  related terms (optional; ones already linked in the definition are skipped).
 * link: where the pages show it (optional): { chapter?, tab, section? } or a list of them.
 */
export const GLOSSARY = {
  // ------------------------------------------------------------------ the neuron (chapter 1)
  perceptron: {
    group: "neuron",
    name: { tr: "Perceptron", en: "Perceptron" },
    aka: ["algılayıcı", "rosenblatt"],
    def: {
      tr: "Frank Rosenblatt'ın 1958'de tanıttığı, örneklerden öğrenen ilk yapay nöron. Girdileri [[weight|ağırlıklarla]] çarpıp toplar, [[bias|bias]] ekler ve [[step-function|basamak fonksiyonuyla]] 0 ya da 1 der. Ağırlıklarını [[perceptron-rule|perceptron kuralıyla]] öğrenir. Karar sınırı düz olduğu için yalnızca [[linear-separability|doğrusal ayrılabilir]] verileri ayırabilir.",
      en: "The first artificial neuron that learns from examples, introduced by Frank Rosenblatt in 1958. It multiplies its inputs by [[weight|weights]], adds them up, adds a [[bias|bias]] and says 0 or 1 with a [[step-function|step function]]. It learns its weights with the [[perceptron-rule|perceptron rule]]. Its boundary is straight, so it can only separate [[linear-separability|linearly separable]] data.",
    },
    see: ["artificial-neuron", "decision-boundary", "xor-problem"],
    link: [{ chapter: "perceptron", tab: "theory", section: "p-intro" }, { chapter: "perceptron", tab: "code", section: "pc-class" }],
  },
  "artificial-neuron": {
    group: "neuron",
    name: { tr: "Yapay nöron", en: "Artificial neuron" },
    aka: ["nöron", "neuron", "birim", "unit", "node"],
    def: {
      tr: "Sinir ağlarının temel hesap birimi: girdilerin ağırlıklı toplamına [[bias|bias]] ekler ve sonucu bir [[activation|aktivasyon fonksiyonundan]] geçirir: \\(a = f(\\mathbf w\\cdot\\mathbf x + b)\\). [[perceptron|Perceptron]] basamak fonksiyonlu bir yapay nörondur; bugünkü ağlarda genellikle [[relu|ReLU]] kullanılır.",
      en: "The basic unit of computation in neural networks: it adds a [[bias|bias]] to the weighted sum of its inputs and passes the result through an [[activation|activation function]]: \\(a = f(\\mathbf w\\cdot\\mathbf x + b)\\). The [[perceptron|perceptron]] is an artificial neuron with a step function; today's networks mostly use [[relu|ReLU]].",
    },
    model: {
      tr: "FC1 katmanında 512, FC2'de 10 yapay nöron var; her biri yukarıdaki hesabı yapar.",
      en: "The FC1 layer has 512 artificial neurons and FC2 has 10; each does the computation above.",
    },
    link: { chapter: "perceptron", tab: "theory", section: "p-neuron" },
  },
  weight: {
    group: "neuron",
    name: { tr: "Ağırlık", en: "Weight" },
    aka: ["weights", "ağırlıklar", "w"],
    def: {
      tr: "Bir girdinin nörona ne kadar ve hangi yönde etki ettiğini belirleyen sayı. Girdi ağırlığıyla çarpılır: büyük pozitif ağırlık girdiyi \"evet\" tarafına, negatif ağırlık \"hayır\" tarafına iter, sıfıra yakın ağırlık girdiyi önemsiz kılar. Ağırlıklar eğitimde öğrenilen [[parameter|parametrelerdir]].",
      en: "The number that sets how strongly, and in which direction, an input affects a neuron. The input is multiplied by its weight: a large positive weight pushes the input towards \"yes\", a negative weight towards \"no\", a weight near zero makes the input unimportant. Weights are the [[parameter|parameters]] learned in training.",
    },
    modelLabel: { tr: "Geometrisi", en: "Its geometry" },
    model: {
      tr: "Ağırlık vektörü \\(\\mathbf w\\) [[decision-boundary|karar sınırına]] diktir ve 1 sınıfının tarafını gösterir.",
      en: "The weight vector \\(\\mathbf w\\) is perpendicular to the [[decision-boundary|decision boundary]] and points to the side of class 1.",
    },
    see: ["bias"],
    link: [{ chapter: "perceptron", tab: "theory", section: "p-neuron" }, { chapter: "perceptron", tab: "math", section: "pm-notation" }],
  },
  bias: {
    group: "neuron",
    name: { tr: "Bias (yanlılık)", en: "Bias" },
    aka: ["b", "eşik", "threshold", "sapma"],
    def: {
      tr: "Ağırlıklı toplama eklenen sabit sayı: \\(z = \\mathbf w\\cdot\\mathbf x + b\\). Nöronun eşiğini belirler: \\(b = -0{,}7\\) ise ağırlıklı toplamın 0,7'yi aşması gerekir. Geometride [[decision-boundary|karar sınırını]] kendine paralel kaydırır. Değeri hep 1 olan bir girdinin ağırlığı gibi düşünülebilir ve o da eğitimde öğrenilir.",
      en: "The constant added to the weighted sum: \\(z = \\mathbf w\\cdot\\mathbf x + b\\). It sets the neuron's threshold: with \\(b = -0.7\\) the weighted sum has to exceed 0.7. Geometrically it shifts the [[decision-boundary|decision boundary]] parallel to itself. It can be seen as the weight of an input that is always 1, and it is learned in training too.",
    },
    see: ["weight", "parameter"],
    link: [{ chapter: "perceptron", tab: "theory", section: "p-neuron" }, { chapter: "perceptron", tab: "math", section: "pm-notation" }],
  },
  "step-function": {
    group: "neuron",
    name: { tr: "Basamak fonksiyonu", en: "Step function" },
    aka: ["heaviside", "eşik fonksiyonu", "threshold function", "H(z)"],
    def: {
      tr: "Girdisi pozitifse 1, değilse 0 veren [[activation|aktivasyon fonksiyonu]]: \\(H(z) = 1\\) (\\(z > 0\\)), \\(H(z) = 0\\) (\\(z \\le 0\\)). [[perceptron|Perceptronun]] kararını verir. Eğimi \\(z = 0\\) dışında her yerde 0'dır (orada tanımsızdır); bu yüzden [[gradient|gradyan]] taşımaz; bu yüzden çok katmanlı ağlarda sigmoid ya da [[relu|ReLU]] kullanılır.",
      en: "The [[activation|activation function]] that gives 1 if its input is positive and 0 otherwise: \\(H(z) = 1\\) (\\(z > 0\\)), \\(H(z) = 0\\) (\\(z \\le 0\\)). It makes the [[perceptron|perceptron]]'s decision. Its slope is 0 everywhere except at \\(z = 0\\), where it is undefined, so it carries no [[gradient|gradient]]; that is why multilayer networks use the sigmoid or [[relu|ReLU]].",
    },
    link: [{ chapter: "perceptron", tab: "theory", section: "p-neuron" }, { chapter: "perceptron", tab: "math", section: "pm-gd" }],
  },
  "logic-gate": {
    group: "neuron",
    name: { tr: "Mantık kapısı", en: "Logic gate" },
    aka: ["and", "or", "nand", "nor", "kapı", "gate", "truth table", "doğruluk tablosu"],
    def: {
      tr: "0/1 girdilerden 0/1 çıktı üreten basit kural. İki girdili kapının dört olası girdisi vardır: <b>AND</b> ikisi de 1 ise, <b>OR</b> en az biri 1 ise, <b>NAND</b> ikisi birden 1 değilse, <b>XOR</b> yalnızca biri 1 ise 1 verir. Tek bir [[perceptron|perceptron]] AND, OR ve NAND'ı öğrenebilir ama XOR'u öğrenemez ([[xor-problem|XOR problemi]]).",
      en: "A simple rule that turns 0/1 inputs into a 0/1 output. A two-input gate has four possible inputs: <b>AND</b> gives 1 if both are 1, <b>OR</b> if at least one is 1, <b>NAND</b> if not both are 1, <b>XOR</b> if exactly one is 1. A single [[perceptron|perceptron]] can learn AND, OR and NAND, but not XOR (the [[xor-problem|XOR problem]]).",
    },
    link: [{ chapter: "perceptron", tab: "theory", section: "p-neuron" }, { chapter: "perceptron", tab: "code", section: "pc-data" }],
  },
  "decision-boundary": {
    group: "neuron",
    name: { tr: "Karar sınırı", en: "Decision boundary" },
    aka: ["sınır", "boundary", "ayırıcı", "separator", "doğrusal sınıflandırıcı", "linear classifier"],
    def: {
      tr: "Girdi uzayında modelin kararının değiştiği yer. Perceptronda \\(\\mathbf w\\cdot\\mathbf x + b = 0\\) olan noktalardır: iki girdide bir doğru, çok girdide bir [[hyperplane|hiper düzlem]]. Bir yanı 1, öbür yanı 0 sınıfıdır. [[weight|Ağırlık vektörü]] sınıra diktir; [[bias|bias]] onu kaydırır. Sınırı düz olan modellere doğrusal sınıflandırıcı denir.",
      en: "Where the model's decision changes in the input space. For a perceptron it is the set of points with \\(\\mathbf w\\cdot\\mathbf x + b = 0\\): a line with two inputs, a [[hyperplane|hyperplane]] with many. One side is class 1, the other class 0. The [[weight|weight vector]] is perpendicular to it; the [[bias|bias]] shifts it. Models with a straight boundary are called linear classifiers.",
    },
    link: [{ chapter: "perceptron", tab: "theory", section: "p-boundary" }, { chapter: "perceptron", tab: "math", section: "pm-geometry" }],
  },
  hyperplane: {
    group: "neuron",
    name: { tr: "Hiper düzlem", en: "Hyperplane" },
    aka: ["hiperdüzlem", "düzlem", "plane"],
    def: {
      tr: "Doğrunun çok boyutlu karşılığı: \\(n\\) boyutlu uzayda \\(\\mathbf w\\cdot\\mathbf x + b = 0\\) denklemini sağlayan \\(n-1\\) boyutlu düz yüzey. 2 boyutta doğru, 3 boyutta düzlemdir. Bir perceptronun [[decision-boundary|karar sınırı]] her zaman bir hiper düzlemdir; 784 pikselli bir görüntüde 783 boyutludur.",
      en: "The many-dimensional version of a line: the flat \\(n-1\\)-dimensional surface of points satisfying \\(\\mathbf w\\cdot\\mathbf x + b = 0\\) in an \\(n\\)-dimensional space. It is a line in 2 dimensions and a plane in 3. A perceptron's [[decision-boundary|decision boundary]] is always a hyperplane; for an image of 784 pixels it has 783 dimensions.",
    },
    link: { chapter: "perceptron", tab: "math", section: "pm-geometry" },
  },
  "perceptron-rule": {
    group: "neuron",
    name: { tr: "Perceptron öğrenme kuralı", en: "Perceptron learning rule" },
    aka: ["öğrenme kuralı", "learning rule", "perceptron kuralı", "rosenblatt kuralı", "online"],
    def: {
      tr: "Rosenblatt'ın ağırlık güncelleme kuralı. Örnekler tek tek gösterilir; cevap doğruysa hiçbir şey değişmez, yanlışsa \\(\\mathbf w \\leftarrow \\mathbf w + \\eta\\,(y - \\hat y)\\,\\mathbf x\\) ve \\(b \\leftarrow b + \\eta\\,(y - \\hat y)\\). Her düzeltme o örneğin \\(z\\)'sini doğru yöne iter. Aslında \"perceptron ölçütü\" denen bir kaybın [[sgd|stokastik gradyan inişidir]].",
      en: "Rosenblatt's weight update rule. The examples are shown one at a time; if the answer is right nothing changes, if it is wrong \\(\\mathbf w \\leftarrow \\mathbf w + \\eta\\,(y - \\hat y)\\,\\mathbf x\\) and \\(b \\leftarrow b + \\eta\\,(y - \\hat y)\\). Each correction pushes that example's \\(z\\) the right way. It is in fact [[sgd|stochastic gradient descent]] on a loss called the \"perceptron criterion\".",
    },
    see: ["learning-rate", "epoch", "linear-separability"],
    link: [{ chapter: "perceptron", tab: "theory", section: "p-rule" }, { chapter: "perceptron", tab: "math", section: "pm-rule" }, { chapter: "perceptron", tab: "code", section: "pc-fit" }],
  },
  "linear-separability": {
    group: "neuron",
    name: { tr: "Doğrusal ayrılabilirlik", en: "Linear separability" },
    aka: ["doğrusal ayrılabilir", "linearly separable", "yakınsama", "convergence", "novikoff"],
    def: {
      tr: "İki sınıfın tek bir [[hyperplane|hiper düzlemle]] (iki boyutta tek bir doğruyla) ayrılabilmesi. Perceptron yakınsama teoremine göre veri doğrusal ayrılabilirse [[perceptron-rule|perceptron kuralı]] sonlu sayıda düzeltmeden sonra bütün örnekleri doğru sınıflandırır; değilse (örneğin XOR) sonsuza kadar düzeltme yapar.",
      en: "Two classes being separable by one [[hyperplane|hyperplane]] (one line in two dimensions). By the perceptron convergence theorem, if the data is linearly separable, the [[perceptron-rule|perceptron rule]] classifies every example correctly after a finite number of corrections; if not (XOR, for example), it keeps correcting forever.",
    },
    see: ["margin", "xor-problem"],
    link: [{ chapter: "perceptron", tab: "theory", section: "p-converge" }, { chapter: "perceptron", tab: "math", section: "pm-converge" }],
  },
  margin: {
    group: "neuron",
    name: { tr: "Marj", en: "Margin" },
    aka: ["boşluk", "gap", "gamma", "γ", "aralık"],
    def: {
      tr: "Bir [[decision-boundary|karar sınırı]] ile ona en yakın örnek arasındaki uzaklık. Marj büyükse sınıflar birbirinden belirgin biçimde ayrıdır. Perceptronun yaptığı düzeltme sayısı en fazla \\((R/\\gamma)^2\\)'dir (\\(\\gamma\\) marj, \\(R\\) en uzak örneğin, bias için eklenen 1 bileşeniyle, orijine uzaklığı; sınır ağırlıklar sıfırdan başlayınca geçerlidir; kesin tanımlar perceptron bölümünün Matematik sekmesinde): marj küçüldükçe öğrenme uzar.",
      en: "The distance between a [[decision-boundary|decision boundary]] and the example closest to it. A large margin means the classes are clearly apart. The number of corrections the perceptron makes is at most \\((R/\\gamma)^2\\) (\\(\\gamma\\) the margin, \\(R\\) the distance of the farthest example from the origin, with the extra 1 for the bias; the bound holds when the weights start at zero; the perceptron chapter's Math tab defines both exactly): the smaller the margin, the longer learning takes.",
    },
    link: [{ chapter: "perceptron", tab: "theory", section: "p-converge" }, { chapter: "perceptron", tab: "math", section: "pm-converge" }],
  },
  "xor-problem": {
    group: "neuron",
    name: { tr: "XOR problemi", en: "The XOR problem" },
    aka: ["xor", "exclusive or", "ya da", "minsky", "papert", "yapay zekâ kışı", "ai winter"],
    def: {
      tr: "XOR ([[logic-gate|mantık kapısı]]) doğrusal ayrılabilir olmadığı için tek bir [[perceptron|perceptronun]] onu öğrenememesi. Minsky ve Papert 1969'da bu tür sınırları gösterdi ve sinir ağlarına ilgi uzun süre azaldı. Çözüm perceptronları katman katman bağlamaktır (XOR = AND(OR, NAND)); bu [[hidden-layer|gizli katmanların]] eğitilmesi için de geri yayılım gerekir.",
      en: "A single [[perceptron|perceptron]] cannot learn XOR (a [[logic-gate|logic gate]]) because XOR is not linearly separable. Minsky and Papert showed limits like this in 1969, and interest in neural networks fell for a long time. The way out is to connect perceptrons in layers (XOR = AND(OR, NAND)); training those [[hidden-layer|hidden layers]] then needs backpropagation.",
    },
    see: ["linear-separability"],
    link: [{ chapter: "perceptron", tab: "theory", section: "p-xor" }, { chapter: "perceptron", tab: "math", section: "pm-xor" }],
  },
  "hidden-layer": {
    group: "neuron",
    name: { tr: "Gizli katman", en: "Hidden layer" },
    aka: ["hidden", "ara katman", "mlp", "çok katmanlı"],
    def: {
      tr: "Girdi ile çıktı arasında kalan katman; değerleri doğrudan gözlenmez ve onlar için bir \"doğru cevap\" yoktur. Veriyi, sonraki katmanın işini kolaylaştıran yeni bir gösterime çevirir: XOR'un dört noktası gizli katmanda doğrusal ayrılabilir hâle gelir. Ağırlıkları [[backprop|geri yayılımla]] öğrenilir.",
      en: "A layer between the input and the output; its values are not observed directly and there is no \"right answer\" for them. It turns the data into a new representation that makes the next layer's job easier: the four points of XOR become linearly separable in the hidden layer. Its weights are learned with [[backprop|backpropagation]].",
    },
    model: {
      tr: "Evrişim blokları ve FC1 gizli katmanlardır; yalnızca FC2'nin 10 çıktısının doğru cevabı bilinir.",
      en: "The convolution blocks and FC1 are hidden layers; only the 10 outputs of FC2 have a known right answer.",
    },
    link: [{ chapter: "perceptron", tab: "theory", section: "p-xor" }, { chapter: "mlp", tab: "theory", section: "ml-hidden" }],
  },
  // ------------------------------------------------------------------ multilayer networks (chapter 2)
  mlp: {
    group: "mlp",
    name: { tr: "Çok katmanlı ağ (MLP)", en: "Multilayer perceptron (MLP)" },
    aka: ["mlp", "multilayer perceptron", "çok katmanlı perceptron", "çok katmanlı algılayıcı", "çka", "ysa", "yapay sinir ağı", "ann", "artificial neural network", "ileri beslemeli ağ", "feedforward network", "dense network", "yoğun ağ"],
    def: {
      tr: "Nöronların katmanlar hâlinde dizildiği ve her katmanın bir öncekinin bütün çıktılarına bağlandığı ([[fully-connected|tam bağlı]]) ağ: girdi → bir ya da daha fazla [[hidden-layer|gizli katman]] → çıktı. Katmanlar arasında doğrusal olmayan bir [[activation|aktivasyon]] vardır; ağırlıklar [[backprop|geri yayılım]] ve [[gradient-descent|gradyan inişiyle]] öğrenilir. Adındaki \"perceptron\" tarihseldir: nöronları basamak yerine sigmoid, tanh ya da ReLU kullanır. Türkçe kaynaklarda çok katmanlı algılayıcı (ÇKA) da denir; derslerde çoğu zaman kısaca YSA (yapay sinir ağı, ANN) diye anılan klasik ağdır. YSA ise bütün ailenin adıdır: CNN de RNN de birer YSA'dır.",
      en: "A network whose neurons are arranged in layers, each layer connected to all the outputs of the previous one ([[fully-connected|fully connected]]): input → one or more [[hidden-layer|hidden layers]] → output. A nonlinear [[activation|activation]] sits between the layers; the weights are learned with [[backprop|backpropagation]] and [[gradient-descent|gradient descent]]. The \"perceptron\" in its name is historical: its neurons use a sigmoid, tanh or ReLU instead of the step. It is the classic network that courses often simply call an ANN (artificial neural network); ANN is the name of the whole family, though: CNNs and RNNs are ANNs too.",
    },
    model: {
      tr: "SimpleCNN'in sonundaki FC1 ve FC2 katmanları bir MLP'dir: evrişim katmanlarının çıkardığı 1024 özellikten 10 sınıfa karar verir.",
      en: "The FC1 and FC2 layers at the end of SimpleCNN are an MLP: it decides among the 10 classes from the 1024 features the convolution layers extract.",
    },
    link: [{ chapter: "mlp", tab: "theory", section: "ml-intro" }, { chapter: "mlp", tab: "play" }],
  },
  sigmoid: {
    group: "mlp",
    name: { tr: "Sigmoid", en: "Sigmoid" },
    aka: ["sigmoid", "lojistik", "logistic", "σ", "s eğrisi", "s-curve"],
    def: {
      tr: "σ(z) = 1 / (1 + e^(−z)): her sayıyı 0 ile 1 arasına sıkıştıran, S biçimli bir [[activation|aktivasyon]]. [[step-function|Basamak fonksiyonunun]] yumuşatılmış hâlidir ve her yerde türevlidir: σ'(z) = σ(z)(1 − σ(z)), en fazla 0,25. Çıktısı bir olasılık gibi okunur; bu yüzden iki sınıflı problemlerde çıktı nöronunda kullanılır. Gizli katmanlarda ise iki uçta türevi 0'a indiği için öğrenmeyi yavaşlatır ([[vanishing-gradient|kaybolan gradyan]]).",
      en: "σ(z) = 1 / (1 + e^(−z)): an S-shaped [[activation|activation]] that squeezes every number between 0 and 1. It is a smoothed [[step-function|step function]] with a derivative everywhere: σ'(z) = σ(z)(1 − σ(z)), at most 0.25. Its output reads like a probability, so it is used at the output neuron for two-class problems. In hidden layers it slows learning, because its derivative falls to 0 at both ends ([[vanishing-gradient|vanishing gradient]]).",
    },
    model: {
      tr: "SimpleCNN sigmoid kullanmaz: gizli katmanlarda ReLU, çıktıda 10 sınıf için softmax.",
      en: "SimpleCNN does not use the sigmoid: ReLU in the hidden layers, softmax over the 10 classes at the output.",
    },
    link: [{ chapter: "mlp", tab: "theory", section: "ml-act" }, { chapter: "mlp", tab: "math", section: "mm-act" }],
  },
  tanh: {
    group: "mlp",
    name: { tr: "tanh (hiperbolik tanjant)", en: "tanh (hyperbolic tangent)" },
    aka: ["tanh", "hiperbolik tanjant", "hyperbolic tangent"],
    def: {
      tr: "tanh z = (e^z − e^(−z)) / (e^z + e^(−z)): [[sigmoid|sigmoid]]'in −1 ile 1 arasına gerilmiş hâli (tanh z = 2σ(2z) − 1). Çıktısı sıfır merkezli olduğu için gizli katmanlarda genelde sigmoid'den iyi çalışır. Türevi 1 − tanh² z, en fazla 1; o da iki uçta 0'a iner.",
      en: "tanh z = (e^z − e^(−z)) / (e^z + e^(−z)): the [[sigmoid|sigmoid]] stretched to lie between −1 and 1 (tanh z = 2σ(2z) − 1). Its output is centred on zero, so in hidden layers it usually works better than the sigmoid. Its derivative is 1 − tanh² z, at most 1; it too falls to 0 at both ends.",
    },
    link: [{ chapter: "mlp", tab: "theory", section: "ml-act" }, { chapter: "mlp", tab: "code", section: "mc-act" }],
  },
  "chain-rule": {
    group: "mlp",
    name: { tr: "Zincir kuralı", en: "Chain rule" },
    aka: ["chain rule", "zincir", "türev", "derivative", "bileşke fonksiyon", "composite function"],
    def: {
      tr: "Bileşke bir fonksiyonun türevi, iç içe geçmiş fonksiyonların türevlerinin çarpımıdır: y = f(g(x)) ise dy/dx = f'(g(x)) · g'(x). Bir değişken sonuca birden fazla yoldan etki ediyorsa yolların katkıları toplanır. [[backprop|Geri yayılım]], zincir kuralının bir ağın bütün ağırlıklarına düzenli bir sırayla, sondan başa doğru uygulanmasıdır.",
      en: "The derivative of a composite function is the product of the derivatives of the nested functions: if y = f(g(x)), then dy/dx = f'(g(x)) · g'(x). If a variable affects the result along several paths, the contributions of the paths are added. [[backprop|Backpropagation]] is the chain rule applied to all the weights of a network in an orderly way, from the end to the start.",
    },
    link: [{ chapter: "mlp", tab: "theory", section: "ml-backprop" }, { chapter: "mlp", tab: "math", section: "mm-backprop" }],
  },
  "vanishing-gradient": {
    group: "mlp",
    name: { tr: "Kaybolan gradyan", en: "Vanishing gradient" },
    aka: ["vanishing gradient", "sönen gradyan", "doyma", "saturation", "exploding gradient", "patlayan gradyan"],
    def: {
      tr: "Derin bir ağda hata sinyalinin geriye giderken her katmanda bir aktivasyon türevi ve ağırlıklarla çarpılıp giderek küçülmesi; ilk katmanlar neredeyse hiç öğrenmez. [[sigmoid|Sigmoid]]'in türevi en fazla 0,25 olduğu için 10 katmanda türev çarpanlarının çarpımı en fazla milyonda birdir; ağırlıklar büyük değilse sinyal de o kadar küçülür. [[relu|ReLU]], dikkatli seçilmiş başlangıç ağırlıkları ve [[batchnorm|batch normalization]] bu sorunu azaltır. Tersi, sinyalin katman katman büyümesi, patlayan gradyandır.",
      en: "In a deep network, the error signal is multiplied by an activation derivative and weights at every layer on its way back and keeps shrinking; the first layers barely learn. The [[sigmoid|sigmoid]]'s derivative is at most 0.25, so over 10 layers the product of the derivative factors is at most a millionth; unless the weights are large, the signal shrinks as much. [[relu|ReLU]], carefully chosen initial weights and [[batchnorm|batch normalization]] reduce the problem. The opposite, a signal that grows layer by layer, is the exploding gradient.",
    },
    model: {
      tr: "SimpleCNN bu yüzden ReLU ve BatchNorm kullanır.",
      en: "That is why SimpleCNN uses ReLU and BatchNorm.",
    },
    link: { chapter: "mlp", tab: "math", section: "mm-act" },
  },
  "local-minimum": {
    group: "mlp",
    name: { tr: "Yerel minimum", en: "Local minimum" },
    aka: ["local minimum", "yerel en küçük", "plato", "plateau", "takılmak", "stuck"],
    def: {
      tr: "Kaybın, yakınındaki bütün noktalardakinden küçük olduğu ama en küçük değerin (global minimum) olmadığı bir nokta. Orada gradyan sıfırdır ve [[gradient-descent|gradyan inişi]] durur. Küçük ağlarda eğitim bazen böyle bir yere ya da düz bir bölgeye (plato) takılır; XOR için 2 gizli nöronlu ağda görülen takılma, nöronların doyduğu çok düz bir bölgedir; farklı bir rastgele başlangıç ya da daha fazla nöron genelde çözer. Büyük ağlarda kötü yerel minimumlar daha az sorun olur.",
      en: "A point where the loss is smaller than at every nearby point but is not the smallest value (the global minimum). The gradient is zero there and [[gradient-descent|gradient descent]] stops. In small networks training sometimes gets stuck at such a place or on a flat region (a plateau); the stuck runs of the XOR network with 2 hidden neurons sit in a very flat region where the neurons saturate; a different random start or more neurons usually fixes it. In large networks bad local minima are less of a problem.",
    },
    link: { chapter: "mlp", tab: "theory", section: "ml-hidden" },
  },
  "universal-approximation": {
    group: "mlp",
    name: { tr: "Evrensel yaklaşım teoremi", en: "Universal approximation theorem" },
    aka: ["universal approximation", "evrensel yaklaşım", "cybenko", "hornik"],
    def: {
      tr: "Tek gizli katmanlı bir ağın, yeterince nöronla, kapalı bir aralıktaki her sürekli fonksiyona istenen yakınlıkta yaklaşabileceğini söyleyen teorem (Cybenko 1989 sigmoid için; Hornik 1991; Leshno ve arkadaşları 1993 polinom olmayan her aktivasyon için). Kaç nöron gerektiğini, gradyan inişinin o ağırlıkları bulacağını ya da ağın iyi [[generalization|genelleyeceğini]] söylemez.",
      en: "The theorem that a network with one hidden layer can, with enough neurons, approximate any continuous function on a closed interval as closely as we like (Cybenko 1989 for the sigmoid; Hornik 1991; Leshno and colleagues 1993 for every non-polynomial activation). It does not say how many neurons it takes, that gradient descent will find those weights, or that the network will [[generalization|generalize]] well.",
    },
    link: [{ chapter: "mlp", tab: "theory", section: "ml-capacity" }, { chapter: "mlp", tab: "math", section: "mm-universal" }],
  },
  // ------------------------------------------------------------------ training
  hyperparameter: {
    group: "train",
    name: { tr: "Hiperparametre", en: "Hyperparameter" },
    aka: ["hyperparameters", "ayar", "settings"],
    def: {
      tr: "Eğitimden önce elle seçilen, modelin kendisinin öğrenmediği değer: [[epoch|epoch]] sayısı, [[learning-rate|öğrenme oranı]], [[batch|batch]] boyutu, [[dropout|dropout]] oranı, katman ve filtre sayıları… Modelin eğitimle öğrendiği değerlere ise [[parameter|parametre]] denir.",
      en: "A value chosen by hand before training, which the model does not learn: the number of [[epoch|epochs]], the [[learning-rate|learning rate]], the [[batch|batch]] size, the [[dropout|dropout]] rate, the number of layers and filters… The values the model does learn in training are called [[parameter|parameters]].",
    },
    model: {
      tr: "Hiperparametreler [[val-set|doğrulama]] sonucuna bakılarak seçilir, test sonucuna bakılarak değil.",
      en: "Hyperparameters are chosen by the [[val-set|validation]] result, never by the test result.",
    },
    link: { tab: "code", section: "c-config" },
  },
  epoch: {
    group: "train",
    name: { tr: "Epoch", en: "Epoch" },
    aka: ["epok", "devir", "tur", "epochs"],
    def: {
      tr: "Eğitim verisinin tamamının modelden bir kez geçirilmesi. Veri [[batch|batch]]'ler hâlinde verilirse bir epoch, batch sayısı kadar [[step|eğitim adımından]] oluşur (perceptronda her örnek ayrı bir adımdır). Genellikle her epoch sonunda model [[val-set|doğrulama setinde]] ölçülür.",
      en: "One pass of the whole training set through the model. When the data comes in [[batch|batches]], one epoch consists of as many [[step|training steps]] as there are batches (for the perceptron every example is its own step). The model is usually measured on the [[val-set|validation set]] after every epoch.",
    },
    model: {
      tr: "48.000 eğitim görüntüsü / 256 = 188 batch, yani epoch başına 188 adım.",
      en: "48,000 training images / 256 = 188 batches, i.e. 188 steps per epoch.",
    },
    link: [{ tab: "code", section: "c-train-loop" }, { chapter: "perceptron", tab: "theory", section: "p-rule" }],
  },
  batch: {
    group: "train",
    name: { tr: "Batch (mini-batch)", en: "Batch (mini-batch)" },
    aka: ["batch size", "batch boyutu", "yığın", "mini batch", "minibatch"],
    def: {
      tr: "Modele aynı anda verilen küçük görüntü grubu. Gruptaki görüntülerin [[loss|kaybının]] ortalaması alınır ve ağırlıklar bu ortalamanın [[gradient|gradyanıyla]] bir kez güncellenir. Tüm veriyle tek adım atmak çok yavaş, her görüntüyle ayrı adım atmak çok gürültülü olurdu; batch ikisi arasında bir dengedir.",
      en: "A small group of images given to the model at once. The [[loss|losses]] of the group are averaged and the weights are updated once with the [[gradient|gradient]] of that average. One step over all the data would be too slow and one step per image too noisy; a batch is the balance between the two.",
    },
    model: {
      tr: "Batch boyutu 256; bir batch [[tensor|tensörünün]] şekli (256, 1, 28, 28).",
      en: "The batch size is 256; a batch [[tensor|tensor]] has the shape (256, 1, 28, 28).",
    },
    see: ["epoch", "step"],
    link: { tab: "code", section: "c-data" },
  },
  step: {
    group: "train",
    name: { tr: "Eğitim adımı (iterasyon)", en: "Training step (iteration)" },
    aka: ["iteration", "iterasyon", "adım", "step", "update", "güncelleme"],
    def: {
      tr: "Tek bir [[batch|batch]] için yapılan işlerin tamamı: [[forward-pass|ileri yayılım]], [[loss|kaybın]] hesaplanması, [[backprop|geri yayılım]] ve [[optimizer|optimizer]]'ın ağırlıkları bir kez güncellemesi. Eğitim, bu adımın binlerce kez tekrarlanmasıdır.",
      en: "Everything done for one [[batch|batch]]: the [[forward-pass|forward pass]], computing the [[loss|loss]], [[backprop|backpropagation]], and one weight update by the [[optimizer|optimizer]]. Training is this step repeated thousands of times.",
    },
    see: ["epoch"],
    link: { tab: "code", section: "c-train-epoch" },
  },
  "forward-pass": {
    group: "train",
    name: { tr: "İleri yayılım", en: "Forward pass" },
    aka: ["forward", "ileri geçiş", "model(x)"],
    def: {
      tr: "Girişin katmanlardan sırayla geçirilip çıktının hesaplanması. PyTorch'ta <code>model(x)</code> çağrısı, modelin <code>forward</code> metodunu çalıştırır. Eğitimde ileri yayılımı [[backprop|geri yayılım]] izler; [[inference|çıkarımda]] yalnızca ileri yayılım yapılır.",
      en: "Passing the input through the layers in order to compute the output. In PyTorch, calling <code>model(x)</code> runs the model's <code>forward</code> method. In training the forward pass is followed by [[backprop|backpropagation]]; [[inference|inference]] is the forward pass alone.",
    },
    model: {
      tr: "1×28×28 → 32×24×24 → 32×12×12 → 64×8×8 → 64×4×4 → 1024 → 512 → 10 [[logit|logit]].",
      en: "1×28×28 → 32×24×24 → 32×12×12 → 64×8×8 → 64×4×4 → 1024 → 512 → 10 [[logit|logits]].",
    },
    link: [{ tab: "code", section: "c-forward" }, { chapter: "mlp", tab: "math", section: "mm-forward" }],
  },
  loss: {
    group: "train",
    name: { tr: "Kayıp (loss) fonksiyonu", en: "Loss function" },
    aka: ["loss", "kayıp", "maliyet", "cost", "hata", "error", "objective", "amaç fonksiyonu"],
    def: {
      tr: "Modelin tahminlerinin ne kadar yanlış olduğunu tek bir sayıyla ölçer: küçük kayıp iyi tahmin demektir. Eğitimin amacı bu sayıyı küçültmektir; [[gradient|gradyanlar]] da bu sayıdan hesaplanır. [[accuracy|Doğruluktan]] farklı olarak modelin ne kadar emin olduğunu da hesaba katar.",
      en: "Measures with a single number how wrong the model's predictions are: a small loss means good predictions. Training aims to make this number smaller, and the [[gradient|gradients]] are computed from it. Unlike [[accuracy|accuracy]], it also takes into account how confident the model is.",
    },
    model: {
      tr: "Kayıp fonksiyonu [[cross-entropy|cross-entropy]]. Eğitim yaklaşık 2,3'ten başlar: rastgele tahmin yapan bir modelin kaybı (ln 10).",
      en: "The loss function is [[cross-entropy|cross-entropy]]. Training starts at about 2.3: the loss of a model that guesses at random (ln 10).",
    },
    link: [{ tab: "theory", section: "t-softmax" }, { chapter: "perceptron", tab: "math", section: "pm-gd" }, { chapter: "mlp", tab: "theory", section: "ml-loss" }],
  },
  "cross-entropy": {
    group: "train",
    name: { tr: "Çapraz entropi (cross-entropy)", en: "Cross-entropy" },
    aka: ["CrossEntropyLoss", "log loss", "negative log likelihood", "nll", "capraz entropi"],
    def: {
      tr: "Sınıflandırmanın standart kayıp fonksiyonu: doğru sınıfa verilen olasılığın negatif logaritması, <span class=\"g-formula\">\\(L = -\\log p_{\\text{doğru}}\\)</span>. Doğru sınıfa 0,9 olasılık verilirse kayıp 0,11; 0,1 verilirse 2,3'tür. Yanlış bir sınıftan emin olmak en ağır cezayı alır.",
      en: "The standard loss for classification: the negative logarithm of the probability given to the correct class, <span class=\"g-formula\">\\(L = -\\log p_{\\text{correct}}\\)</span>. A probability of 0.9 on the correct class gives a loss of 0.11; 0.1 gives 2.3. Being sure of a wrong class is punished hardest.",
    },
    model: {
      tr: "PyTorch'un <code>nn.CrossEntropyLoss</code>'u [[softmax|softmax]]'i kendi içinde uygular; bu yüzden model ham [[logit|logit]] döndürür.",
      en: "PyTorch's <code>nn.CrossEntropyLoss</code> applies [[softmax|softmax]] itself, so the model returns raw [[logit|logits]].",
    },
    link: [{ tab: "theory", section: "t-softmax" }, { chapter: "mlp", tab: "math", section: "mm-loss" }],
  },
  gradient: {
    group: "train",
    name: { tr: "Gradyan", en: "Gradient" },
    aka: ["türev", "derivative", "grad", "∇", "gradyen"],
    def: {
      tr: "Kaybın her ağırlığa göre türevi: bir ağırlık biraz artarsa kaybın hangi yönde ve ne kadar değişeceğini söyler. Tüm ağırlıkların gradyanları birlikte, kaybın en hızlı arttığı yönü gösterir; [[gradient-descent|gradyan inişi]] bunun tersine gider.",
      en: "The derivative of the loss with respect to each weight: it tells in which direction, and how much, the loss changes when that weight grows a little. Together, the gradients of all weights point to where the loss grows fastest; [[gradient-descent|gradient descent]] goes the opposite way.",
    },
    model: {
      tr: "<code>loss.backward()</code> çağrısından sonra her ağırlığın gradyanı kendi <code>.grad</code> alanındadır.",
      en: "After <code>loss.backward()</code>, each weight's gradient is in its own <code>.grad</code> field.",
    },
    see: ["backprop"],
    link: { tab: "theory", section: "t-opt" },
  },
  backprop: {
    group: "train",
    name: { tr: "Geri yayılım (backpropagation)", en: "Backpropagation" },
    aka: ["backward", "backprop", "zincir kuralı", "chain rule", "loss.backward"],
    def: {
      tr: "Tüm [[gradient|gradyanları]] verimli hesaplayan yöntem. Kayıptan başlayıp katmanlar boyunca geriye doğru ilerler; türev, zincir kuralıyla her katmandan bir öncekine aktarılır. Böylece yüz binlerce ağırlığın gradyanı, bir [[forward-pass|ileri yayılımla]] aynı mertebede bir maliyetle bulunur.",
      en: "The method that computes all [[gradient|gradients]] efficiently. It starts at the loss and moves backward through the layers, passing the derivative from each layer to the one before it with the chain rule. That gives the gradients of hundreds of thousands of weights at a cost of the same order as a [[forward-pass|forward pass]].",
    },
    model: {
      tr: "PyTorch'ta tek satırdır: <code>loss.backward()</code>.",
      en: "In PyTorch it is a single line: <code>loss.backward()</code>.",
    },
    link: [{ tab: "theory", section: "t-opt" }, { chapter: "mlp", tab: "theory", section: "ml-backprop" }, { chapter: "mlp", tab: "math", section: "mm-backprop" }],
  },
  "gradient-descent": {
    group: "train",
    name: { tr: "Gradyan inişi", en: "Gradient descent" },
    aka: ["gradient descent", "iniş", "descent", "eğim inişi"],
    def: {
      tr: "Ağırlıkları [[gradient|gradyanın]] tersi yönünde küçük adımlarla kaydırarak kaybı azaltma yöntemi: <span class=\"g-formula\">\\(\\theta \\leftarrow \\theta - \\eta\\, \\nabla L\\)</span>. Adımın büyüklüğünü [[learning-rate|öğrenme oranı]] η belirler. Bir vadiden aşağı, her an en dik iniş yönünde yürümeye benzer.",
      en: "Lowering the loss by moving the weights in small steps opposite to the [[gradient|gradient]]: <span class=\"g-formula\">\\(\\theta \\leftarrow \\theta - \\eta\\, \\nabla L\\)</span>. The [[learning-rate|learning rate]] η sets the step size. It is like walking down a valley, always in the steepest downhill direction.",
    },
    see: ["sgd", "adam"],
    link: [{ tab: "theory", section: "t-opt" }, { chapter: "perceptron", tab: "math", section: "pm-gd" }, { chapter: "mlp", tab: "theory", section: "ml-loss" }],
  },
  "learning-rate": {
    group: "train",
    name: { tr: "Öğrenme oranı (learning rate)", en: "Learning rate" },
    aka: ["lr", "η", "eta", "adım büyüklüğü", "step size", "ogrenme orani"],
    def: {
      tr: "[[gradient-descent|Gradyan inişinde]] her adımın ne kadar büyük olacağını belirleyen [[hyperparameter|hiperparametre]] (η). Çok küçükse eğitim çok yavaş ilerler; çok büyükse kayıp salınır ya da ıraksar. Genellikle 0,1 ile 0,0001 arasındaki değerler denenir.",
      en: "The [[hyperparameter|hyperparameter]] (η) that sets how big each [[gradient-descent|gradient descent]] step is. Too small and training crawls; too large and the loss oscillates or diverges. Values between 0.1 and 0.0001 are typical.",
    },
    model: { tr: "[[adam|Adam]] ile η = 0,001.", en: "η = 0.001 with [[adam|Adam]]." },
    link: [{ tab: "theory", section: "t-opt" }, { chapter: "perceptron", tab: "theory", section: "p-rule" }, { chapter: "mlp", tab: "theory", section: "ml-loss" }],
  },
  optimizer: {
    group: "train",
    name: { tr: "Optimizer (eniyileyici)", en: "Optimizer" },
    aka: ["optimizasyon", "optimization", "eniyileme", "torch.optim", "optimiser"],
    def: {
      tr: "[[gradient|Gradyanları]] kullanarak ağırlıkları güncelleyen algoritma. En basiti [[sgd|SGD]]'dir; [[momentum|momentum]] ve [[adam|Adam]] gibi yöntemler geçmiş adımların bilgisini de kullanarak daha hızlı ve kararlı ilerler. <code>optimizer.step()</code> güncellemeyi yapar, <code>optimizer.zero_grad()</code> gradyanları sıfırlar.",
      en: "The algorithm that updates the weights using the [[gradient|gradients]]. The simplest is [[sgd|SGD]]; methods such as [[momentum|momentum]] and [[adam|Adam]] also use information from earlier steps to move faster and more steadily. <code>optimizer.step()</code> makes the update, <code>optimizer.zero_grad()</code> resets the gradients.",
    },
    model: {
      tr: "<code>optim.Adam(model.parameters(), lr=0.001)</code>",
      en: "<code>optim.Adam(model.parameters(), lr=0.001)</code>",
    },
    link: { tab: "code", section: "c-setup-train" },
  },
  sgd: {
    group: "train",
    name: { tr: "SGD (stokastik gradyan inişi)", en: "SGD (stochastic gradient descent)" },
    aka: ["stochastic gradient descent", "stokastik"],
    def: {
      tr: "[[gradient-descent|Gradyan inişinin]], tüm veri yerine her adımda tek bir [[batch|batch]] ile yapılan hâli. Gradyan tek bir batch'ten hesaplandığı için gürültülüdür (\"stokastik\" = rastgele), ama adımlar çok daha ucuzdur; bu gürültü bazen daha iyi çözümler bulmaya bile yardım eder.",
      en: "[[gradient-descent|Gradient descent]] done with one [[batch|batch]] per step instead of all the data. The gradient comes from a single batch, so it is noisy (\"stochastic\" = random), but the steps are much cheaper, and the noise sometimes even helps find better solutions.",
    },
    see: ["momentum", "adam"],
    link: [{ tab: "theory", section: "t-opt" }, { chapter: "perceptron", tab: "math", section: "pm-gd" }, { chapter: "mlp", tab: "math", section: "mm-train" }],
  },
  momentum: {
    group: "train",
    name: { tr: "Momentum", en: "Momentum" },
    aka: ["ivme", "heavy ball"],
    def: {
      tr: "[[optimizer|Optimizer]]'ın, geçmiş adımların yönünü bir hız gibi biriktirip yeni adıma eklemesi. Tutarlı yönlerde hızlanır; ileri geri salınan yönlerde ise salınımlar birbirini sönümler. Yokuş aşağı yuvarlanan ağır bir topa benzer.",
      en: "The [[optimizer|optimizer]] keeps a running velocity of the past steps' directions and adds it to each new step. It speeds up along consistent directions, while back-and-forth oscillations cancel out. Like a heavy ball rolling downhill.",
    },
    see: ["sgd", "adam"],
    link: { tab: "theory", section: "t-opt" },
  },
  adam: {
    group: "train",
    name: { tr: "Adam", en: "Adam" },
    aka: ["adaptive moment estimation"],
    def: {
      tr: "En yaygın kullanılan [[optimizer|optimizer]]. [[momentum|Momentuma]] ek olarak her ağırlığın gradyanının ne kadar büyük olduğunu da izler ve adımı ağırlık başına ölçekler: büyük gradyanlı ağırlıklar temkinli, küçük gradyanlılar daha cesur adımlar atar. Bu yüzden [[learning-rate|öğrenme oranı]] seçimine daha az duyarlıdır.",
      en: "The most widely used [[optimizer|optimizer]]. On top of [[momentum|momentum]], it tracks how large each weight's gradient is and scales the step per weight: weights with large gradients take careful steps, those with small gradients bolder ones. That makes it less sensitive to the choice of [[learning-rate|learning rate]].",
    },
    model: {
      tr: "lr = 0,001; β₁ = 0,9 ve β₂ = 0,999 (PyTorch varsayılanları).",
      en: "lr = 0.001; β₁ = 0.9 and β₂ = 0.999 (PyTorch defaults).",
    },
    see: ["sgd"],
    link: { tab: "theory", section: "t-opt" },
  },
  "moving-average": {
    group: "train",
    name: { tr: "Hareketli ortalama", en: "Moving average" },
    aka: ["EMA", "üstel hareketli ortalama", "exponential moving average", "yumuşatma", "smoothing", "ortalama", "average"],
    def: {
      tr: "Gürültülü bir dizinin genel eğilimini görmek için her noktada son değerlerin ortalamasını almak. Burada kullanılan <b>üstel</b> hareketli ortalamada her yeni değer ortalamaya küçük bir payla katılır, eski değerlerin etkisi giderek azalır: <span class=\"g-formula\">\\(m \\leftarrow 0{,}95\\, m + 0{,}05\\, x\\)</span> BatchNorm'un sakladığı ortalama ve varyans da böyle güncellenir; orada katsayılar 0,9 ve 0,1'dir.",
      en: "Averaging the most recent values at each point to see the overall trend of a noisy series. In the <b>exponential</b> moving average used here, each new value enters with a small share and older values fade away gradually: <span class=\"g-formula\">\\(m \\leftarrow 0.95\\, m + 0.05\\, x\\)</span> BatchNorm's stored mean and variance are updated the same way, with factors 0.9 and 0.1.",
    },
    model: {
      tr: "Eğitim sekmesindeki <b>eğitim kaybı</b> kutusu ve batch kaybı grafiğindeki <b>mavi çizgi</b> bu ortalamadır. İlk değerlerdeki sıfıra çekilme düzeltilir (TensorBoard'daki gibi).",
      en: "The <b>training loss</b> tile and the <b>blue line</b> of the batch loss chart on the Training tab are this average. The pull towards zero at the start is corrected (as in TensorBoard).",
    },
    see: ["log-scale"],
  },
  seed: {
    group: "train",
    name: { tr: "Seed (rastgelelik tohumu)", en: "Random seed" },
    aka: ["tohum", "random seed", "rastgele", "manual_seed", "tekrarlanabilirlik", "reproducibility"],
    def: {
      tr: "Rastgele sayı üretecinin başlangıç değeri. Bilgisayardaki \"rastgele\" sayılar aslında bu değerden başlayan belirli bir dizidir. Aynı seed her seferinde aynı diziyi, dolayısıyla aynı ilk ağırlıkları, aynı veri bölünmesini, aynı karıştırma sırasını ve aynı [[dropout|dropout]] maskelerini verir.",
      en: "The starting value of the random number generator. \"Random\" numbers on a computer are really a fixed sequence that starts from this value. The same seed gives the same sequence every time, and so the same initial weights, the same data split, the same shuffling order and the same [[dropout|dropout]] masks.",
    },
    model: {
      tr: "seed = 42; <code>torch.manual_seed(42)</code> CPU'nun ve tüm GPU'ların üreteçlerini sabitler.<span data-engine=\"browser\"> Tarayıcıdaki eğitim başka bir üreteç kullanır; orada aynı seed sonuçları birbirine çok yakın tutar, ama birebir aynı yapmaz.</span>",
      en: "seed = 42; <code>torch.manual_seed(42)</code> fixes the generators of the CPU and all GPUs.<span data-engine=\"browser\"> Training in the browser uses another generator; there the same seed keeps results very close to each other, but not identical.</span>",
    },
    link: { tab: "code", section: "c-seed" },
  },
  "train-eval": {
    group: "train",
    name: { tr: "Eğitim ve değerlendirme modu", en: "Training and evaluation mode" },
    aka: ["model.train", "model.eval", "train mode", "eval mode", "no_grad", "eğitim modu", "değerlendirme modu"],
    def: {
      tr: "Bazı katmanlar eğitimde ve tahminde farklı davranır. <code>model.train()</code> modunda [[dropout|Dropout]] değerleri rastgele sıfırlar ve [[batchnorm|BatchNorm]] o batch'in istatistiklerini kullanır. <code>model.eval()</code> modunda Dropout kapanır ve BatchNorm eğitimde biriktirdiği ortalamaları kullanır; böylece aynı giriş her zaman aynı çıktıyı verir.",
      en: "Some layers behave differently in training and in prediction. In <code>model.train()</code> mode, [[dropout|Dropout]] zeroes random values and [[batchnorm|BatchNorm]] uses the batch's own statistics. In <code>model.eval()</code> mode Dropout is off and BatchNorm uses the averages it collected during training, so the same input always gives the same output.",
    },
    model: {
      tr: "Doğrulama, test ve [[inference|çıkarım]] <code>model.eval()</code> ve <code>torch.no_grad()</code> ile yapılır; ikincisi gradyan takibini kapatarak zaman ve bellek kazandırır.",
      en: "Validation, test and [[inference|inference]] run with <code>model.eval()</code> and <code>torch.no_grad()</code>; the latter turns off gradient tracking to save time and memory.",
    },
    link: { tab: "code", section: "c-evaluate" },
  },

  // ------------------------------------------------------------------ data and evaluation
  mnist: {
    group: "data",
    name: { tr: "MNIST", en: "MNIST" },
    aka: ["veri seti", "dataset", "rakam", "digits", "el yazısı", "handwritten"],
    def: {
      tr: "El yazısı rakamlardan (0–9) oluşan klasik bir veri seti: 60.000 eğitim ve 10.000 test görüntüsü. Her görüntü 28×28 piksel, gri tonlamalıdır ve rakam ortadadır. Küçük ve temiz olduğu için yeni yöntemleri denemenin ilk durağıdır.",
      en: "A classic dataset of handwritten digits (0–9): 60,000 training and 10,000 test images. Each image is 28×28 pixels in grayscale, with the digit in the middle. Being small and clean, it is the first stop for trying out new methods.",
    },
    model: {
      tr: "60.000 eğitim görüntüsü 48.000 [[train-set|eğitim]] ve 12.000 [[val-set|doğrulama]] görüntüsü olarak bölünür.",
      en: "The 60,000 training images are split into 48,000 [[train-set|training]] and 12,000 [[val-set|validation]] images.",
    },
    link: { tab: "code", section: "c-data" },
  },
  "train-set": {
    group: "data",
    name: { tr: "Eğitim seti", en: "Training set" },
    aka: ["training data", "eğitim verisi", "train", "train set"],
    def: {
      tr: "Modelin ağırlıklarını güncellemek için kullanılan veri. Model bu görüntüleri her [[epoch|epoch]]'ta yeniden görür. Eğitim setindeki başarı tek başına pek bir şey söylemez: model bu görüntüleri ezberlemiş olabilir ([[overfitting|aşırı uyum]]).",
      en: "The data used to update the model's weights. The model sees these images again in every [[epoch|epoch]]. Success on the training set alone says little: the model may have memorized these images ([[overfitting|overfitting]]).",
    },
    model: { tr: "48.000 görüntü; her epoch'ta karıştırılır.", en: "48,000 images, shuffled every epoch." },
    see: ["val-set", "test-set"],
    link: { tab: "code", section: "c-data" },
  },
  "val-set": {
    group: "data",
    name: { tr: "Doğrulama seti", en: "Validation set" },
    aka: ["validation", "doğrulama", "val", "dev set", "geçerleme", "validation data"],
    def: {
      tr: "Eğitimde kullanılmayan ve modelin eğitim sırasında ne kadar iyi [[generalization|genellediğini]] ölçmeye yarayan veri. Her epoch sonunda bakılır; [[hyperparameter|hiperparametreler]] ve eğitimin ne zaman durdurulacağı buna göre seçilir. Bu seçimler doğrulama setine göre yapıldığı için son ölçüm ayrı bir [[test-set|test setiyle]] yapılır.",
      en: "Data not used for training, which measures how well the model [[generalization|generalizes]] while it trains. It is checked after every epoch, and the [[hyperparameter|hyperparameters]] and when to stop training are chosen by it. Because those choices are tuned to it, the final measurement uses a separate [[test-set|test set]].",
    },
    model: { tr: "Eğitim görüntülerinden ayrılan 12.000 görüntü.", en: "12,000 images split off from the training images." },
    see: ["train-set", "early-stopping"],
    link: [{ chapter: "mlp", tab: "theory", section: "ml-general" }, { tab: "code", section: "c-data" }],
  },
  "test-set": {
    group: "data",
    name: { tr: "Test seti", en: "Test set" },
    aka: ["test", "sınama", "test data", "test verisi"],
    def: {
      tr: "Yalnızca en sonda, bir kez kullanılan veri. Ne eğitimde ne de hiperparametre seçiminde kullanıldığı için, modelin hiç görmediği yeni verideki başarısının en dürüst tahminini verir. Test sonucuna bakıp modeli değiştirmek bu dürüstlüğü bozar.",
      en: "Data used only once, at the very end. Since it plays no part in training or in choosing hyperparameters, it gives the most honest estimate of how the model does on new data it has never seen. Changing the model after looking at the test result spoils that honesty.",
    },
    model: {
      tr: "MNIST'in ayrı 10.000 test görüntüsü; SimpleCNN varsayılan ayarlarla %99,29 [[accuracy|doğruluğa]] ulaşır.",
      en: "MNIST's separate 10,000 test images; with the default settings SimpleCNN reaches 99.29% [[accuracy|accuracy]].",
    },
    see: ["train-set", "val-set"],
    link: [{ chapter: "mlp", tab: "theory", section: "ml-general" }, { tab: "code", section: "c-evaluate" }],
  },
  accuracy: {
    group: "data",
    name: { tr: "Doğruluk (accuracy)", en: "Accuracy" },
    aka: ["başarı", "isabet", "doğruluk oranı", "dogruluk"],
    def: {
      tr: "Doğru sınıflandırılan örneklerin yüzdesi. Yalnızca en yüksek skorun doğru sınıfta olup olmadığına bakar: model %51 de emin olsa %99 da emin olsa aynı sayılır. Bu yüzden eğitimi [[loss|kayıp]] yönlendirir; doğruluk ise sonucu anlamayı kolaylaştırır.",
      en: "The percentage of examples classified correctly. It only checks whether the highest score is on the correct class: the model being 51% or 99% sure counts the same. That is why the [[loss|loss]] drives training, while accuracy makes the result easy to understand.",
    },
    link: { tab: "code", section: "c-evaluate" },
  },
  overfitting: {
    group: "data",
    name: { tr: "Aşırı uyum (overfitting)", en: "Overfitting" },
    aka: ["aşırı öğrenme", "ezberleme", "ezber", "memorization", "overfit"],
    def: {
      tr: "Modelin genel kuralları öğrenmek yerine eğitim verisini ezberlemesi. Belirtisi: [[train-set|eğitim]] kaybı düşmeye devam ederken [[val-set|doğrulama]] kaybının yükselmesi. Daha çok veri, [[dropout|dropout]] gibi düzenlileştirme yöntemleri ya da eğitimi erken durdurmak bunu önlemeye yardım eder.",
      en: "The model memorizes the training data instead of learning general rules. The sign: the [[train-set|training]] loss keeps going down while the [[val-set|validation]] loss goes up. More data, regularization such as [[dropout|dropout]], or stopping training early help prevent it.",
    },
    see: ["underfitting", "generalization", "early-stopping"],
    link: [{ chapter: "mlp", tab: "theory", section: "ml-general" }, { tab: "theory", section: "t-drop" }],
  },
  underfitting: {
    group: "data",
    name: { tr: "Yetersiz öğrenme (underfitting)", en: "Underfitting" },
    aka: ["underfit", "eksik öğrenme"],
    def: {
      tr: "Modelin eğitim verisini bile yeterince öğrenememesi: hem eğitim hem doğrulama kaybı yüksek kalır. Nedeni çok kısa bir eğitim, çok küçük bir model ya da uygun olmayan bir [[learning-rate|öğrenme oranı]] olabilir.",
      en: "The model does not learn even the training data well enough: both the training and the validation loss stay high. Training too briefly, a model that is too small, or an unsuitable [[learning-rate|learning rate]] can cause it.",
    },
    see: ["overfitting"],
    link: { chapter: "mlp", tab: "theory", section: "ml-general" },
  },
  "early-stopping": {
    group: "data",
    name: { tr: "Erken durdurma", en: "Early stopping" },
    aka: ["early stopping", "erken durdurmak", "durdurma", "patience"],
    def: {
      tr: "Eğitimi, [[val-set|doğrulama]] hatasının en düşük olduğu anda durdurmak ya da o andaki ağırlıklara geri dönmek. Eğitim hatası düşmeye devam ederken doğrulama hatası yükselmeye başladıysa ağ [[overfitting|aşırı uyuyordur]]; o andan sonraki epoch'lar genellemeyi bozar. Durma anı doğrulama setine bakılarak seçildiği için asıl sonuç [[test-set|test setinde]] ölçülür.",
      en: "Stopping training when the [[val-set|validation]] error is lowest, or going back to the weights of that moment. If the validation error starts rising while the training error keeps falling, the network is [[overfitting|overfitting]]; the epochs after that make it generalize worse. Since the moment to stop is chosen by looking at the validation set, the real result is measured on the [[test-set|test set]].",
    },
    see: ["overfitting", "val-set"],
    link: { chapter: "mlp", tab: "theory", section: "ml-general" },
  },
  generalization: {
    group: "data",
    name: { tr: "Genelleme", en: "Generalization" },
    aka: ["generalize", "genelleştirme", "generalisation"],
    def: {
      tr: "Modelin eğitimde görmediği yeni verilerde de doğru tahmin yapabilmesi; makine öğrenmesinin asıl amacı budur. [[val-set|Doğrulama]] ve [[test-set|test]] setleri tam olarak bunu ölçmek için ayrılır.",
      en: "The model's ability to predict correctly on new data it did not see in training; this is the real goal of machine learning. The [[val-set|validation]] and [[test-set|test]] sets are set aside exactly to measure it.",
    },
    see: ["overfitting"],
    link: { chapter: "mlp", tab: "theory", section: "ml-general" },
  },
  "log-scale": {
    group: "data",
    name: { tr: "Logaritmik eksen", en: "Logarithmic axis" },
    aka: ["log", "logaritma", "log scale", "logaritmik ölçek"],
    def: {
      tr: "Eşit aralıkların eşit <i>oranları</i> gösterdiği eksen: 1'den 0,1'e inmek, 0,1'den 0,01'e inmekle aynı yeri kaplar. Kayıp birkaç basamak boyunca küçüldüğü için sonlardaki küçük ama önemli iyileşmeler ancak böyle bir eksende görünür.",
      en: "An axis where equal distances mean equal <i>ratios</i>: going from 1 to 0.1 takes the same space as going from 0.1 to 0.01. Because the loss shrinks over several orders of magnitude, the small but important late improvements are visible only on such an axis.",
    },
    model: { tr: "Batch kaybı grafiğinin y ekseni.", en: "The y axis of the batch loss chart." },
  },

  // ------------------------------------------------------------------ the model and its layers
  cnn: {
    group: "layers",
    name: { tr: "Evrişimli sinir ağı (CNN)", en: "Convolutional neural network (CNN)" },
    aka: ["cnn", "convnet", "sinir ağı", "neural network", "derin öğrenme", "deep learning", "model"],
    def: {
      tr: "Görüntüler için tasarlanmış sinir ağı. [[convolution|Evrişim]] katmanları küçük filtreleri tüm görüntüde kaydırarak kenar, köşe gibi yerel desenleri bulur, [[pooling|havuzlama]] haritaları küçültür. Sonraki katmanlar bu desenleri birleştirerek daha karmaşık şekilleri tanır; en sondaki [[fully-connected|tam bağlı]] katmanlar da sınıfa karar verir.",
      en: "A neural network designed for images. [[convolution|Convolution]] layers slide small filters over the whole image to find local patterns such as edges and corners, and [[pooling|pooling]] shrinks the maps. Later layers combine these patterns into more complex shapes, and the [[fully-connected|fully connected]] layers at the end decide the class.",
    },
    model: {
      tr: "2 evrişim bloğu ve 2 tam bağlı katman; toplam 582.218 [[parameter|parametre]].",
      en: "2 convolution blocks and 2 fully connected layers; 582,218 [[parameter|parameters]] in total.",
    },
    link: { tab: "theory", section: "t-schematic" },
  },
  tensor: {
    group: "layers",
    name: { tr: "Tensör", en: "Tensor" },
    aka: ["torch.tensor", "dizi", "array", "shape", "şekil", "boyut", "tensor"],
    def: {
      tr: "Çok boyutlu sayı dizisi; PyTorch'taki her veri ve her ağırlık bir tensördür. Tek bir sayı 0, bir vektör 1, bir matris 2 boyutlu bir tensördür. <b>Şekli</b> (shape) her boyuttaki eleman sayısını verir. Tensörler [[gpu|GPU]]'da çok hızlı işlenebilir.",
      en: "A multi-dimensional array of numbers; every piece of data and every weight in PyTorch is a tensor. A single number is a 0-dimensional tensor, a vector 1-dimensional, a matrix 2-dimensional. Its <b>shape</b> gives the number of elements along each dimension. Tensors can be processed very fast on a [[gpu|GPU]].",
    },
    model: {
      tr: "Bir batch: (256, 1, 28, 28) = (batch, [[channel|kanal]], yükseklik, genişlik).",
      en: "One batch: (256, 1, 28, 28) = (batch, [[channel|channels]], height, width).",
    },
    link: { tab: "code", section: "c-data" },
  },
  parameter: {
    group: "layers",
    name: { tr: "Parametre (ağırlık ve bias)", en: "Parameter (weight and bias)" },
    aka: ["ağırlık", "weight", "weights", "bias", "sapma", "parameters", "öğrenilebilir", "learnable"],
    def: {
      tr: "Modelin eğitimle öğrendiği sayılar. <b>Ağırlıklar</b> her girişin ne kadar önemli olduğunu, <b>bias</b> ise sonuca eklenen sabit kaymayı belirler. Eğitimden önce rastgele başlarlar; [[optimizer|optimizer]] her adımda onları biraz değiştirir.",
      en: "The numbers the model learns through training. <b>Weights</b> set how important each input is, and the <b>bias</b> is a constant offset added to the result. They start out random, and the [[optimizer|optimizer]] changes them a little at every step.",
    },
    model: {
      tr: "582.218 parametre; bunların %90'ı (524.800) FC1 katmanındadır.",
      en: "582,218 parameters; 90% of them (524,800) are in the FC1 layer.",
    },
    see: ["hyperparameter"],
    link: [{ tab: "code", section: "c-model" }, { chapter: "perceptron", tab: "math", section: "pm-notation" }, { chapter: "mlp", tab: "math", section: "mm-forward" }],
  },
  convolution: {
    group: "layers",
    name: { tr: "Evrişim (convolution)", en: "Convolution" },
    aka: ["conv", "conv2d", "konvolüsyon", "evrisim", "nn.Conv2d"],
    def: {
      tr: "Küçük bir ağırlık matrisini ([[kernel|filtre]]) görüntünün üzerinde kaydırıp her konumda, filtre ile altındaki piksellerin çarpımlarını toplama işlemi. Sonuç bir [[feature-map|öznitelik haritasıdır]]: filtrenin aradığı desen görüntünün neresindeyse orada büyük değerler çıkar. Aynı filtre her konumda kullanıldığı için az ağırlıkla çok iş görür.",
      en: "Sliding a small weight matrix (a [[kernel|filter]]) over the image and, at each position, summing the products of the filter and the pixels under it. The result is a [[feature-map|feature map]]: large values appear wherever the filter's pattern is in the image. Because the same filter is used at every position, it does a lot with few weights.",
    },
    model: {
      tr: "conv1: 1 → 32 [[channel|kanal]], conv2: 32 → 64 kanal; ikisi de 5×5, [[stride|stride]] 1, [[padding|dolgu]] yok.",
      en: "conv1: 1 → 32 [[channel|channels]], conv2: 32 → 64 channels; both 5×5, [[stride|stride]] 1, no [[padding|padding]].",
    },
    see: ["receptive-field"],
    link: { tab: "theory", section: "t-conv" },
  },
  kernel: {
    group: "layers",
    name: { tr: "Filtre (çekirdek, kernel)", en: "Filter (kernel)" },
    aka: ["kernel", "çekirdek", "filter", "filtre", "kernel_size", "filters"],
    def: {
      tr: "[[convolution|Evrişimde]] görüntünün üzerinde kaydırılan küçük ağırlık matrisi. Her filtre bir desen arar (örneğin yatay bir kenar); bir katmandaki filtre sayısı, çıktısındaki [[channel|kanal]] sayısını belirler. Filtrelerin değerleri elle verilmez, eğitimle öğrenilir.",
      en: "The small weight matrix slid over the image in a [[convolution|convolution]]. Each filter looks for one pattern (a horizontal edge, say); the number of filters in a layer sets the number of [[channel|channels]] in its output. The filter values are not set by hand but learned in training.",
    },
    model: {
      tr: "conv1'de 32 adet 5×5 filtre (Eğitim sekmesinde görünenler), conv2'de 64 adet 32×5×5 filtre.",
      en: "conv1 has 32 filters of 5×5 (the ones shown on the Training tab), conv2 has 64 filters of 32×5×5.",
    },
    see: ["feature-map"],
    link: { tab: "theory", section: "t-conv" },
  },
  "feature-map": {
    group: "layers",
    name: { tr: "Öznitelik haritası (feature map)", en: "Feature map" },
    aka: ["aktivasyon haritası", "activation map", "harita", "map", "feature", "öznitelik", "özellik haritası"],
    def: {
      tr: "Bir filtrenin, görüntünün her konumundaki tepkisini gösteren 2B dizi; bir katmanın çıktısındaki her [[channel|kanal]] bir öznitelik haritasıdır. İlk katmanlarınki kenar ve çizgileri, sonraki katmanlarınki daha karmaşık parçaları gösterir.",
      en: "A 2D array showing a filter's response at every position of the image; each [[channel|channel]] of a layer's output is a feature map. Those of the first layers show edges and strokes, those of later layers more complex parts.",
    },
    model: {
      tr: "conv1'den sonra 32 harita (24×24), conv2'den sonra 64 harita (8×8). Çıkarım sekmesinde her katmanınkini görebilirsiniz.",
      en: "32 maps (24×24) after conv1, 64 maps (8×8) after conv2. The Inference tab shows those of every layer.",
    },
    link: { tab: "theory", section: "t-conv" },
  },
  channel: {
    group: "layers",
    name: { tr: "Kanal", en: "Channel" },
    aka: ["channels", "in_channels", "out_channels", "derinlik", "depth"],
    def: {
      tr: "Bir görüntünün ya da katman çıktısının üst üste duran \"katmanları\". Gri bir görüntünün 1, renkli bir görüntünün 3 (kırmızı, yeşil, mavi) kanalı vardır. Evrişim katmanının çıktısında her [[kernel|filtre]] bir kanal üretir; sonraki katmanın filtreleri tüm kanallara birlikte bakar.",
      en: "The stacked \"layers\" of an image or of a layer's output. A grayscale image has 1 channel, a color image 3 (red, green, blue). In a convolution layer's output every [[kernel|filter]] produces one channel; the next layer's filters look at all channels together.",
    },
    model: { tr: "1 → 32 → 64 kanal.", en: "1 → 32 → 64 channels." },
    see: ["feature-map"],
    link: { tab: "theory", section: "t-conv" },
  },
  stride: {
    group: "layers",
    name: { tr: "Stride (kaydırma adımı)", en: "Stride" },
    aka: ["kaydırma", "adım"],
    def: {
      tr: "Filtrenin ya da havuzlama penceresinin her seferinde kaç piksel kaydırıldığı. Stride 1 her konuma bakar; stride 2 birer konum atlayarak ilerler ve çıktının enini ve boyunu yarıya indirir.",
      en: "How many pixels the filter or the pooling window moves each time. A stride of 1 visits every position; a stride of 2 skips every other one and halves the width and height of the output.",
    },
    model: { tr: "Evrişimlerde 1, [[pooling|havuzlamada]] 2.", en: "1 in the convolutions, 2 in [[pooling|pooling]]." },
    see: ["padding"],
    link: { tab: "theory", section: "t-conv" },
  },
  padding: {
    group: "layers",
    name: { tr: "Dolgu (padding)", en: "Padding" },
    aka: ["dolgu", "pad", "zero padding", "sıfır dolgu"],
    def: {
      tr: "Evrişimden önce görüntünün kenarlarına sıfırlar eklemek. Dolgu yoksa k×k'lık bir filtre çıktıyı her yönde k − 1 piksel küçültür; uygun bir dolguyla boyut korunur ve kenardaki pikseller de eşit ilgi görür.",
      en: "Adding zeros around the edges of the image before a convolution. Without padding, a k×k filter shrinks the output by k − 1 pixels in each direction; with suitable padding the size is kept and the edge pixels get equal attention.",
    },
    model: {
      tr: "Dolgu kullanılmaz: 5×5 filtreler 28×28'i 24×24'e, 12×12'yi 8×8'e indirir.",
      en: "No padding is used: the 5×5 filters shrink 28×28 to 24×24 and 12×12 to 8×8.",
    },
    see: ["stride"],
    link: { tab: "theory", section: "t-conv" },
  },
  "receptive-field": {
    group: "layers",
    name: { tr: "Algı alanı (receptive field)", en: "Receptive field" },
    aka: ["receptive", "görüş alanı", "alıcı alan"],
    def: {
      tr: "Bir öznitelik haritasındaki tek bir değerin, giriş görüntüsünün ne kadar büyük bir bölgesine bağlı olduğu. Her evrişim ve [[pooling|havuzlama]] bu alanı büyütür; bu yüzden derin katmanlar küçük filtrelerle bile görüntünün geniş parçalarına bakabilir.",
      en: "How large a region of the input image a single value of a feature map depends on. Every convolution and [[pooling|pooling]] makes it larger, which is why deep layers see large parts of the image even with small filters.",
    },
    model: {
      tr: "conv1'den sonra 5×5, conv2'den sonra 14×14, son havuzlamadan sonra 16×16 piksel.",
      en: "5×5 after conv1, 14×14 after conv2 and 16×16 pixels after the last pooling.",
    },
    link: { tab: "theory", section: "t-pool" },
  },
  pooling: {
    group: "layers",
    name: { tr: "Havuzlama (pooling)", en: "Pooling" },
    aka: ["max pooling", "maxpool", "average pooling", "havuz", "alt örnekleme", "downsampling", "MaxPool2d"],
    def: {
      tr: "Öznitelik haritasını küçük bölgelere ayırıp her bölgeden tek bir değer alarak küçültme. <b>Max pooling</b> en büyük değeri seçer, böylece desen bölgenin neresinde olursa olsun korunur. Hesap yükünü azaltır, [[receptive-field|algı alanını]] büyütür ve küçük kaymalara karşı dayanıklılık kazandırır. Öğrenilecek ağırlığı yoktur.",
      en: "Shrinking a feature map by dividing it into small regions and keeping one value from each. <b>Max pooling</b> keeps the largest, so the pattern survives wherever it is in the region. It reduces computation, enlarges the [[receptive-field|receptive field]] and adds robustness to small shifts. It has no weights to learn.",
    },
    model: {
      tr: "2×2 max pooling, [[stride|stride]] 2: 24×24 → 12×12 ve 8×8 → 4×4.",
      en: "2×2 max pooling with [[stride|stride]] 2: 24×24 → 12×12 and 8×8 → 4×4.",
    },
    link: { tab: "theory", section: "t-pool" },
  },
  activation: {
    group: "layers",
    name: { tr: "Aktivasyon fonksiyonu", en: "Activation function" },
    aka: ["activation", "doğrusal olmayan", "nonlinearity", "non-linear", "sigmoid", "tanh", "etkinleştirme"],
    def: {
      tr: "Katmanların çıktısına uygulanan doğrusal olmayan fonksiyon. Onsuz, üst üste kaç katman konursa konsun ağ tek bir doğrusal işleme eşit olurdu ve eğri sınırlar çizemezdi. En yaygını [[relu|ReLU]]'dur; sigmoid ve tanh daha eski örneklerdir.",
      en: "A non-linear function applied to the output of the layers. Without it, any stack of layers would equal a single linear operation and could not draw curved boundaries. The most common one is [[relu|ReLU]]; sigmoid and tanh are older examples.",
    },
    link: [{ tab: "theory", section: "t-act" }, { chapter: "perceptron", tab: "theory", section: "p-neuron" }, { chapter: "mlp", tab: "theory", section: "ml-act" }],
  },
  relu: {
    group: "layers",
    name: { tr: "ReLU", en: "ReLU" },
    aka: ["rectified linear unit", "doğrultulmuş doğrusal birim", "F.relu"],
    def: {
      tr: "En basit ve en yaygın [[activation|aktivasyon fonksiyonu]]: <span class=\"g-formula\">\\(\\mathrm{ReLU}(z) = \\max(0, z)\\)</span>. Negatif değerleri sıfırlar, pozitifleri olduğu gibi bırakır. Hesabı çok ucuzdur ve pozitif bölgede gradyanı 1 olduğu için derin ağlar onunla hızlı öğrenir.",
      en: "The simplest and most common [[activation|activation function]]: <span class=\"g-formula\">\\(\\mathrm{ReLU}(z) = \\max(0, z)\\)</span>. It sets negative values to zero and leaves positive ones unchanged. It is very cheap to compute, and since its gradient is 1 for positive inputs, deep networks learn quickly with it.",
    },
    model: {
      tr: "Her [[batchnorm|BatchNorm]]'dan sonra ve FC1'den sonra.",
      en: "After every [[batchnorm|BatchNorm]] and after FC1.",
    },
    link: [{ tab: "theory", section: "t-act" }, { chapter: "mlp", tab: "math", section: "mm-act" }],
  },
  batchnorm: {
    group: "layers",
    name: { tr: "Batch Normalization (BatchNorm)", en: "Batch normalization (BatchNorm)" },
    aka: ["batch norm", "bn", "normalizasyon", "normalization", "normalleştirme", "BatchNorm2d"],
    def: {
      tr: "Her [[channel|kanalı]], [[batch|batch]] üzerinde ortalaması 0, standart sapması 1 olacak şekilde normalleştirir, sonra öğrenilen bir ölçek (γ) ve kayma (β) uygular. Eğitim hızlanır ve daha büyük öğrenme oranları kullanılabilir. Önerenler bunu katmanlara giden değerlerin dağılımını kararlı tutmasına bağladı; Santurkar ve ark. (2018) ise yararın daha çok kayıp yüzeyini düzgünleştirmesinden geldiğini öne sürdü. Eğitimde ve değerlendirmede farklı davranır ([[train-eval|eğitim ve değerlendirme modu]]).",
      en: "Normalizes each [[channel|channel]] to mean 0 and standard deviation 1 over the [[batch|batch]], then applies a learned scale (γ) and shift (β). Training is faster and larger learning rates can be used. Its authors put this down to keeping the distribution of the values reaching each layer steady; Santurkar et al. (2018) argued that the benefit comes more from smoothing the loss surface. It behaves differently in training and in evaluation ([[train-eval|training and evaluation mode]]).",
    },
    model: {
      tr: "Her evrişimden sonra, ReLU'dan önce: <code>BatchNorm2d(32)</code> ve <code>BatchNorm2d(64)</code>.",
      en: "After every convolution, before ReLU: <code>BatchNorm2d(32)</code> and <code>BatchNorm2d(64)</code>.",
    },
    link: { tab: "theory", section: "t-bn" },
  },
  dropout: {
    group: "layers",
    name: { tr: "Dropout", en: "Dropout" },
    aka: ["seyreltme", "bırakma", "drop", "regularization", "düzenlileştirme"],
    def: {
      tr: "Eğitim sırasında her adımda değerlerin rastgele bir kısmını (oran <i>p</i>) sıfırlayan, kalanları 1/(1 − p) ile büyüten katman. Hiçbir nöron belirli komşularının orada olacağına güvenemediği için her öznitelik tek başına da işe yarar olmak zorundadır ve [[overfitting|aşırı uyum]] azalır. Değerlendirmede kapalıdır.",
      en: "A layer that, during training, zeroes a random fraction <i>p</i> of the values at every step and scales the rest by 1/(1 − p). No neuron can count on particular neighbours being there, so every feature has to be useful on its own too, and [[overfitting|memorization]] decreases. It is off during evaluation.",
    },
    model: {
      tr: "Birinci havuzlamadan sonra p = 0,45, ikinciden sonra p = 0,35.",
      en: "p = 0.45 after the first pooling, p = 0.35 after the second.",
    },
    see: ["train-eval"],
    link: { tab: "theory", section: "t-drop" },
  },
  flatten: {
    group: "layers",
    name: { tr: "Düzleştirme (flatten)", en: "Flatten" },
    aka: ["düzleştir", "reshape", "view", "torch.flatten"],
    def: {
      tr: "Çok boyutlu bir [[tensor|tensörü]], değerlerini sırayla dizerek tek boyutlu bir vektöre çevirmek. Evrişim bloklarının çıktısını [[fully-connected|tam bağlı]] katmana verebilmek için gerekir; hiçbir değer değişmez, yalnızca şekil değişir.",
      en: "Turning a multi-dimensional [[tensor|tensor]] into a one-dimensional vector by laying its values out in order. It is needed to feed the output of the convolution blocks to a [[fully-connected|fully connected]] layer; no value changes, only the shape.",
    },
    model: {
      tr: "64×4×4 = 1024 değer: <code>torch.flatten(x, start_dim=1)</code>.",
      en: "64×4×4 = 1024 values: <code>torch.flatten(x, start_dim=1)</code>.",
    },
    link: [{ tab: "code", section: "c-forward" }, { chapter: "mlp", tab: "theory", section: "ml-images" }],
  },
  "fully-connected": {
    group: "layers",
    name: { tr: "Tam bağlı katman (fully connected)", en: "Fully connected layer" },
    aka: ["linear", "dense", "fc", "nn.Linear", "yoğun katman", "tam bağlantılı", "doğrusal katman"],
    def: {
      tr: "Her çıkış nöronunun tüm girişlere bağlı olduğu katman: her çıkış, girişlerin ağırlıklı toplamı artı bir bias'tır, <span class=\"g-formula\">\\(y = Wx + b\\)</span>. CNN'de evrişim bloklarının bulduğu öznitelikleri birleştirir ve sınıfa karar verir.",
      en: "A layer where every output neuron is connected to every input: each output is a weighted sum of the inputs plus a bias, <span class=\"g-formula\">\\(y = Wx + b\\)</span>. In a CNN it combines the features found by the convolution blocks and decides the class.",
    },
    model: {
      tr: "FC1: 1024 → 512 (ardından ReLU), FC2: 512 → 10 [[logit|logit]].",
      en: "FC1: 1024 → 512 (followed by ReLU), FC2: 512 → 10 [[logit|logits]].",
    },
    see: ["flatten"],
    link: { tab: "code", section: "c-model" },
  },
  logit: {
    group: "layers",
    name: { tr: "Logit (ham skor)", en: "Logit (raw score)" },
    aka: ["logits", "skor", "score", "ham çıktı", "raw output"],
    def: {
      tr: "Modelin son katmanının, [[softmax|softmax]]'ten önceki ham çıktısı: her sınıf için sınırsız bir sayı. Büyük olan daha olası demektir ama bu değerler olasılık değildir; toplamları 1 etmez ve negatif olabilirler.",
      en: "The raw output of the model's last layer, before [[softmax|softmax]]: an unbounded number for each class. A larger value means more likely, but these values are not probabilities; they do not add up to 1 and can be negative.",
    },
    model: {
      tr: "Her görüntü için 10 logit; en büyüğü tahmin edilen rakamdır.",
      en: "10 logits per image; the largest one is the predicted digit.",
    },
    see: ["cross-entropy"],
    link: { tab: "theory", section: "t-softmax" },
  },
  softmax: {
    group: "layers",
    name: { tr: "Softmax", en: "Softmax" },
    aka: ["olasılık", "probability", "probabilities", "olasılıklar"],
    def: {
      tr: "[[logit|Logitleri]], toplamı 1 olan olasılıklara çeviren fonksiyon: <span class=\"g-formula\">\\(p_i = e^{z_i} \\big/ \\textstyle\\sum_j e^{z_j}\\)</span>. Üstel fonksiyon büyük logitleri daha da öne çıkarır; sıralama değişmez, bu yüzden tahmin edilen sınıf da değişmez.",
      en: "The function that turns [[logit|logits]] into probabilities that add up to 1: <span class=\"g-formula\">\\(p_i = e^{z_i} \\big/ \\textstyle\\sum_j e^{z_j}\\)</span>. The exponential makes large logits stand out even more; the order does not change, and so neither does the predicted class.",
    },
    model: {
      tr: "Çıkarım sekmesindeki olasılık çubukları softmax çıktısıdır.",
      en: "The probability bars on the Inference tab are the softmax output.",
    },
    see: ["cross-entropy"],
    link: { tab: "theory", section: "t-softmax" },
  },

  // ------------------------------------------------------------------ running the model
  inference: {
    group: "run",
    name: { tr: "Çıkarım (inference)", en: "Inference" },
    aka: ["tahmin", "prediction", "predict", "çıkarsama", "cikarim"],
    def: {
      tr: "Eğitilmiş modeli yeni bir girdiye uygulayıp tahmin almak. Yalnızca [[forward-pass|ileri yayılım]] yapılır, ağırlıklar değişmez. Model [[train-eval|değerlendirme modunda]] çalıştırılır ve gradyan takibi kapatılır.",
      en: "Applying a trained model to a new input to get a prediction. Only the [[forward-pass|forward pass]] runs and the weights do not change. The model runs in [[train-eval|evaluation mode]] with gradient tracking turned off.",
    },
    model: {
      tr: "Çıkarım sekmesinde kendi çizdiğiniz bir rakamla ya da test görüntüleriyle deneyebilirsiniz.",
      en: "Try it on the Inference tab with a digit you draw or with test images.",
    },
    see: ["checkpoint"],
    link: { tab: "infer" },
  },
  checkpoint: {
    group: "run",
    name: { tr: "Checkpoint (state_dict)", en: "Checkpoint (state_dict)" },
    aka: ["state_dict", "kayıt", "save", "load", "torch.save", ".pt", "model dosyası", "kaydetmek"],
    def: {
      tr: "Modelin öğrendiği [[parameter|ağırlıkların]] dosyaya kaydedilmiş hâli. <code>state_dict</code>, katman adlarını ağırlık tensörlerine eşleyen bir sözlüktür; <code>torch.save</code> ile yazılır, <code>model.load_state_dict</code> ile yüklenir. Böylece model her seferinde yeniden eğitilmez.",
      en: "The model's learned [[parameter|weights]] saved to a file. A <code>state_dict</code> is a dictionary mapping layer names to weight tensors; it is written with <code>torch.save</code> and loaded with <code>model.load_state_dict</code>. That way the model does not have to be trained again every time.",
    },
    model: {
      tr: "<code>SimpleCNN.py</code> modeli kaydetmez; Kodlama sekmesinin ek adımı <code>torch.save</code> ile nasıl kaydedileceğini gösterir.<span data-engine=\"browser\"> CNN bölümünün sayfası, tarayıcıda eğitilen modeli karşılaştırma için eğitim öncesi ağırlıklarla birlikte bu tarayıcıda (IndexedDB) saklar; dosya yazmaz.</span><span data-engine=\"server\"> Python paneli (<code>dashboard.py</code>) eğitilen modeli, karşılaştırma için eğitim öncesi ağırlıklarla birlikte <code>checkpoints/simple_cnn.pt</code> dosyasına kaydeder.</span>",
      en: "<code>SimpleCNN.py</code> does not save the model; the Coding tab's extra step shows how to save it with <code>torch.save</code>.<span data-engine=\"browser\"> The CNN chapter's page keeps a model trained in the browser, together with the weights from before training for comparison, in this browser (IndexedDB); it writes no file.</span><span data-engine=\"server\"> The Python dashboard (<code>dashboard.py</code>) saves the trained model, together with the weights from before training for comparison, to <code>checkpoints/simple_cnn.pt</code>.</span>",
    },
    link: { tab: "code", section: "c-predict" },
  },
  gpu: {
    group: "run",
    name: { tr: "GPU ve CUDA", en: "GPU and CUDA" },
    aka: ["ekran kartı", "graphics card", "cuda", "cpu", "device", "cihaz", "işlemci"],
    def: {
      tr: "GPU, ekran kartının işlemcisidir. Binlerce basit işlemi aynı anda yapabildiği için, çoğu paralel çarpma ve toplamadan oluşan sinir ağı hesaplarını CPU'dan kat kat hızlı yapar. CUDA, PyTorch'un NVIDIA kartlarını kullanmasını sağlayan altyapıdır; model ve veri aynı cihazda olmalıdır (<code>.to(device)</code>).",
      en: "A GPU is the processor of a graphics card. It can do thousands of simple operations at once, so it runs neural network computations, which are mostly parallel multiplications and additions, many times faster than a CPU. CUDA is the platform that lets PyTorch use NVIDIA cards; the model and the data must be on the same device (<code>.to(device)</code>).",
    },
    model: {
      tr: "Eğitimin hangi cihazda çalıştığı üst çubukta yazar. <code>SimpleCNN.py</code> ekran kartını CUDA ile kullanır<span data-engine=\"browser\">; CNN bölümünün sayfasında ise TensorFlow.js, tarayıcının WebGPU (yoksa WebGL) arayüzüyle kullanır</span><span data-engine=\"server\">; Python paneli (<code>dashboard.py</code>) de öyle</span>.",
      en: "The top bar shows which device training runs on. <code>SimpleCNN.py</code> uses the graphics card through CUDA<span data-engine=\"browser\">; on the CNN chapter's page TensorFlow.js uses it through the browser's WebGPU (otherwise WebGL)</span><span data-engine=\"server\">, and so does the Python dashboard (<code>dashboard.py</code>)</span>.",
    },
    link: { tab: "code", section: "c-setup-train" },
  },
  // ------------------------------------------------------------------ detection and segmentation (chapter 4)
  "object-detection": {
    group: "detect",
    name: { tr: "Nesne tespiti", en: "Object detection" },
    aka: ["detection", "tespit", "yolo", "algılama"],
    def: {
      tr: "Bir görüntüdeki her nesneyi bir [[bounding-box|kutu]], bir sınıf ve bir [[confidence-score|güven skoruyla]] bulma görevi. Sınıflandırmadan farkı, nesnelerin kaç tane ve nerede olduğunu da söylemesidir. YOLO bunu görüntüye tek bir kez bakarak yapar.",
      en: "The task of finding every object in an image with a [[bounding-box|box]], a class and a [[confidence-score|confidence score]]. Unlike classification, it also says how many objects there are and where. YOLO does it by looking at the image once.",
    },
    model: {
      tr: "YOLO26n-seg binaları ve spor sahalarını bulur; her birine bir kutu ve bir maske verir.",
      en: "YOLO26n-seg finds buildings and sports fields and gives each a box and a mask.",
    },
    modelLabel: { tr: "Bu bölümün modelinde", en: "In this chapter's model" },
    see: ["instance-segmentation", "semantic-segmentation"],
    link: { chapter: "yolo", tab: "theory", section: "y-tasks" },
  },
  "bounding-box": {
    group: "detect",
    name: { tr: "Sınırlayıcı kutu", en: "Bounding box" },
    aka: ["kutu", "box", "bbox", "xyxy", "xywh"],
    def: {
      tr: "Bir nesneyi içine alan, kenarları görüntünün kenarlarına paralel dikdörtgen. Dört sayıdır: köşelerle \\((x_1, y_1, x_2, y_2)\\) ya da merkez ve boyutla \\((c_x, c_y, w, h)\\). Tahmin edilen bir kutunun doğruluğu [[iou|IoU]] ile ölçülür.",
      en: "A rectangle with sides parallel to the image's that encloses an object. It is four numbers: by its corners \\((x_1, y_1, x_2, y_2)\\) or by its centre and size \\((c_x, c_y, w, h)\\). How right a predicted box is, is measured with [[iou|IoU]].",
    },
    link: [{ chapter: "yolo", tab: "theory", section: "y-iou" }, { chapter: "yolo", tab: "math", section: "ym-boxes" }],
  },
  iou: {
    group: "detect",
    name: { tr: "IoU (kesişim bölü birleşim)", en: "IoU (intersection over union)" },
    aka: ["jaccard", "kesişim", "örtüşme", "overlap", "maske iou", "mask iou"],
    def: {
      tr: "İki kutunun (ya da iki maskenin) ortak alanının, birlikte kapladıkları alana oranı: \\(|A \\cap B| / |A \\cup B|\\). Tam örtüşmede 1, hiç örtüşmeyince 0'dır. Nesnenin boyutundan bağımsız olduğu için bir tahminin doğru sayılıp sayılmayacağına karar verir (genellikle IoU ≥ 0,5); [[nms|NMS]] de kopyaları IoU ile bulur.",
      en: "The area two boxes (or two masks) share, divided by the area they cover together: \\(|A \\cap B| / |A \\cup B|\\). It is 1 for an exact match and 0 when they do not overlap. It does not depend on the object's size, so it decides whether a prediction counts as correct (usually IoU ≥ 0.5); [[nms|NMS]] finds copies with IoU too.",
    },
    see: ["precision", "average-precision"],
    link: [{ chapter: "yolo", tab: "theory", section: "y-iou" }, { chapter: "yolo", tab: "math", section: "ym-iou" }],
  },
  "confidence-score": {
    group: "detect",
    name: { tr: "Güven skoru ve eşiği", en: "Confidence score and threshold" },
    aka: ["skor", "score", "conf", "güven eşiği", "confidence threshold"],
    def: {
      tr: "Bir adayın bir sınıftan bir nesne olduğuna modelin verdiği değer (0 ile 1 arası, [[sigmoid|sigmoid]] çıkışı). Güven eşiğinin altındaki adaylar atılır. Eşik düşerse daha çok nesne bulunur ([[recall|duyarlılık]] artar) ama yanlışlar da çoğalır ([[precision|kesinlik]] düşer).",
      en: "The value the model gives a candidate for being an object of a class (between 0 and 1, a [[sigmoid|sigmoid]] output). Candidates below the confidence threshold are dropped. A lower threshold finds more objects ([[recall|recall]] goes up) but more mistakes too ([[precision|precision]] goes down).",
    },
    model: { tr: "Varsayılan eşik 0,25. Şanghay karesinde 8400 adaydan 107'si bu eşiği geçer.", en: "The default threshold is 0.25. On the Shanghai tile 107 of the 8400 candidates pass it." },
    modelLabel: { tr: "Bu bölümün modelinde", en: "In this chapter's model" },
    link: [{ chapter: "yolo", tab: "theory", section: "y-nms" }, { chapter: "yolo", tab: "math", section: "ym-cls" }],
  },
  nms: {
    group: "detect",
    name: { tr: "NMS (en büyük olmayanları bastırma)", en: "NMS (non-maximum suppression)" },
    aka: ["non-maximum suppression", "bastırma", "suppression"],
    def: {
      tr: "Aynı nesneyi bulan birden fazla adaydan yalnızca en iyisini bırakan adım: adaylar skora göre sıralanır, en yüksek skorlu aday tutulur, aynı sınıftan olup onunla [[iou|IoU]]'su eşiği aşan adaylar silinir ve bu, kalanlarla tekrarlanır. YOLOv10 ve YOLO26 bire bir bir baş eğiterek NMS'e gerek bırakmaz.",
      en: "The step that keeps only the best of several candidates that found the same object: candidates are sorted by score, the highest is kept, those of the same class overlapping it by an [[iou|IoU]] above the threshold are removed, and this repeats with the rest. YOLOv10 and YOLO26 train a one-to-one head so that NMS is not needed.",
    },
    link: [{ chapter: "yolo", tab: "theory", section: "y-nms" }, { chapter: "yolo", tab: "math", section: "ym-nms" }],
  },
  "semantic-segmentation": {
    group: "detect",
    name: { tr: "Anlamsal segmentasyon", en: "Semantic segmentation" },
    aka: ["semantic", "piksel sınıflandırma", "arazi örtüsü", "land cover"],
    def: {
      tr: "Görüntünün her pikseline bir sınıf verme görevi. Yol, su, yeşil alan gibi sayılamayan yüzeyler için uygundur; ama aynı sınıftan bitişik iki nesneyi ayırmaz (bunun için [[instance-segmentation|örnek segmentasyonu]]). Ölçütü sınıf başına IoU ve onların ortalaması mIoU'dur.",
      en: "The task of giving every pixel of an image a class. It suits surfaces you cannot count, like roads, water or grass; but it does not separate two touching objects of the same class (that is [[instance-segmentation|instance segmentation]]). It is measured by the IoU of every class and their mean, mIoU.",
    },
    model: { tr: "YOLO26n-sem altı sınıf boyar; doğrulamada mIoU 0,46.", en: "YOLO26n-sem paints six classes; on the validation set its mIoU is 0.46." },
    modelLabel: { tr: "Bu bölümün modelinde", en: "In this chapter's model" },
    link: { chapter: "yolo", tab: "theory", section: "y-tasks" },
  },
  "instance-segmentation": {
    group: "detect",
    name: { tr: "Örnek segmentasyonu", en: "Instance segmentation" },
    aka: ["instance", "maske", "mask", "segmentation"],
    def: {
      tr: "Her nesneyi ayrı bir maskeyle bulma görevi: [[object-detection|nesne tespiti]] gibi nesneleri sayar, [[semantic-segmentation|anlamsal segmentasyon]] gibi piksel piksel sınır çizer. Bitişik iki bina ayrı iki maske olur. YOLO'da maskeler [[mask-prototype|prototiplerden]] kurulur.",
      en: "The task of finding every object with its own mask: it counts objects like [[object-detection|object detection]] and draws outlines pixel by pixel like [[semantic-segmentation|semantic segmentation]]. Two buildings that touch become two masks. In YOLO the masks are built from [[mask-prototype|prototypes]].",
    },
    link: { chapter: "yolo", tab: "theory", section: "y-tasks" },
  },
  "mask-prototype": {
    group: "detect",
    name: { tr: "Maske prototipleri ve katsayıları", en: "Mask prototypes and coefficients" },
    aka: ["prototip", "prototype", "katsayı", "coefficient", "yolact", "proto"],
    def: {
      tr: "YOLACT'tan gelen fikir: ağ her görüntü için ortak birkaç maske haritası (prototip) üretir; her nesne bu prototipler için birer katsayı verir. Nesnenin maskesi \\(\\sigma(\\sum_k c_k P_k)\\) ile hesaplanır ve kutusuyla kırpılır. Böylece her nesne için ayrı bir maske ağı gerekmez.",
      en: "An idea from YOLACT: for every image the network makes a few shared mask maps (prototypes); every object gives a coefficient for each of them. The object's mask is \\(\\sigma(\\sum_k c_k P_k)\\), cropped by its box. So no separate mask network is needed per object.",
    },
    model: { tr: "32 prototip, her biri 160 × 160; bir prototip pikseli bu karelerde 0,8–1,2 m.", en: "32 prototypes, each 160 × 160; one prototype pixel is 0.8–1.2 m on these tiles." },
    modelLabel: { tr: "Bu bölümün modelinde", en: "In this chapter's model" },
    link: [{ chapter: "yolo", tab: "theory", section: "y-masks" }, { chapter: "yolo", tab: "math", section: "ym-mask" }],
  },
  "label-assignment": {
    group: "detect",
    name: { tr: "Etiket ataması", en: "Label assignment" },
    aka: ["assignment", "atama", "tal", "task aligned", "bire bir", "one-to-one", "one-to-many"],
    def: {
      tr: "Eğitimde hangi adayın hangi gerçek nesneden sorumlu olacağına karar verme. Atanan adayların kutusu, sınıfı ve maskesi o nesneye göre düzeltilir; ötekiler \"nesne yok\" demeyi öğrenir. YOLO26 adayın sınıf skoruna ve kutusunun [[iou|IoU]]'suna birlikte bakar; bire-çok başta bir nesneye birkaç, bire bir başta tek aday atar.",
      en: "Deciding in training which candidate is responsible for which true object. The assigned candidates' box, class and mask are corrected towards that object; the others learn to say \"no object\". YOLO26 looks at a candidate's class score and its box's [[iou|IoU]] together; it assigns several candidates per object in the one-to-many head and a single one in the one-to-one head.",
    },
    link: { chapter: "yolo", tab: "theory", section: "y-train" },
  },
  "transfer-learning": {
    group: "detect",
    name: { tr: "Aktarım öğrenmesi (önceden eğitilmiş ağırlıklar)", en: "Transfer learning (pretrained weights)" },
    aka: ["pretrained", "önceden eğitilmiş", "fine-tuning", "ince ayar", "coco"],
    def: {
      tr: "Başka, genellikle büyük bir veri setinde eğitilmiş ağırlıklardan başlayıp ağı kendi verisiyle eğitmeye devam etmek (ince ayar). İlk katmanların öğrendiği kenar, köşe ve doku bilgisi yeni görevde de işe yarar; az veriyle bile daha hızlı ve daha iyi öğrenilir.",
      en: "Starting from weights trained on another, usually large dataset and going on training the network on your own data (fine-tuning). The edges, corners and textures the first layers learned help in the new task too; even with little data it learns faster and better.",
    },
    model: { tr: "COCO'dan başlayan model maske mAP50 0,50'ye ulaştı, sıfırdan başlayan 0,32'de kaldı (aynı veri, 100 epoch).", en: "The model that started from COCO reached a mask mAP50 of 0.50, the one from scratch stayed at 0.32 (same data, 100 epochs)." },
    modelLabel: { tr: "Bu bölümün modelinde", en: "In this chapter's model" },
    link: [{ chapter: "yolo", tab: "theory", section: "y-train" }, { chapter: "yolo", tab: "train" }],
  },
  "data-augmentation": {
    group: "detect",
    name: { tr: "Veri artırma", en: "Data augmentation" },
    aka: ["augmentation", "mozaik", "mosaic", "çevirme", "flip", "döndürme", "rotation"],
    def: {
      tr: "Eğitim görüntülerini her epoch'ta rastgele değiştirmek (döndürme, çevirme, ölçekleme, renk oynatma, birkaç görüntüyü birleştirme), böylece ağ aynı görüntüyü hiç iki kez aynı görmez. Etiketler de aynı dönüşümle değiştirilir. [[overfitting|Aşırı uyumu]] azaltır; dönüşümler görev için anlamlı olmalıdır.",
      en: "Changing the training images at random in every epoch (rotating, flipping, scaling, changing colours, joining several images), so the network never sees the same image the same way twice. The labels are transformed the same way. It reduces [[overfitting|overfitting]]; the transformations must make sense for the task.",
    },
    model: { tr: "Kuş bakışında \"yukarı\" yok: kareler ±90° döndürüldü, yatay ve dikey çevrildi, dördü bir mozaikte birleştirildi.", en: "A view from above has no \"up\": tiles were rotated by up to ±90°, flipped both ways and joined four at a time in a mosaic." },
    modelLabel: { tr: "Bu bölümün modelinde", en: "In this chapter's model" },
    link: { chapter: "yolo", tab: "theory", section: "y-train" },
  },
  precision: {
    group: "detect",
    name: { tr: "Kesinlik", en: "Precision" },
    aka: ["precision", "isabetlilik"],
    def: {
      tr: "Modelin tespitlerinden ne kadarının doğru olduğu: \\(P = TP / (TP + FP)\\). TP doğru tespit (aynı sınıftan bir gerçek nesneyle yeterli [[iou|IoU]]), FP yanlış tespittir. [[recall|Duyarlılıkla]] birlikte okunur.",
      en: "How many of the model's detections are right: \\(P = TP / (TP + FP)\\). TP is a correct detection (enough [[iou|IoU]] with a true object of its class), FP a wrong one. It is read together with [[recall|recall]].",
    },
    link: [{ chapter: "yolo", tab: "theory", section: "y-metrics" }, { chapter: "yolo", tab: "math", section: "ym-ap" }],
  },
  recall: {
    group: "detect",
    name: { tr: "Duyarlılık", en: "Recall" },
    aka: ["recall", "hassasiyet", "sensitivity"],
    def: {
      tr: "Gerçek nesnelerden ne kadarının bulunduğu: \\(R = TP / (TP + FN)\\). FN, hiçbir tespitin eşleşmediği (kaçırılan) gerçek nesnedir. [[confidence-score|Güven eşiği]] düştükçe artar; [[precision|kesinlik]] ise genellikle düşer.",
      en: "How many of the true objects were found: \\(R = TP / (TP + FN)\\). An FN is a true object no detection matched (missed). It rises as the [[confidence-score|confidence threshold]] falls, while [[precision|precision]] usually falls.",
    },
    link: [{ chapter: "yolo", tab: "theory", section: "y-metrics" }, { chapter: "yolo", tab: "math", section: "ym-ap" }],
  },
  "average-precision": {
    group: "detect",
    name: { tr: "AP ve mAP", en: "AP and mAP" },
    aka: ["map", "map50", "map50-95", "ortalama kesinlik", "average precision", "pr eğrisi", "pr curve"],
    def: {
      tr: "AP, bir sınıfın kesinlik-duyarlılık (PR) eğrisinin altındaki alandır: güven eşiği bütün değerlerden geçerken [[precision|kesinliğin]] ve [[recall|duyarlılığın]] birlikte ne kadar iyi kaldığını tek sayıyla özetler. mAP sınıfların AP ortalamasıdır; mAP50'de [[iou|IoU]] eşiği 0,5'tir, mAP50-95'te 0,50'den 0,95'e on eşiğin ortalamasıdır.",
      en: "AP is the area under a class's precision-recall (PR) curve: it sums up in one number how well [[precision|precision]] and [[recall|recall]] hold together as the confidence threshold sweeps all values. mAP is the mean AP over the classes; mAP50 uses an [[iou|IoU]] threshold of 0.5, mAP50-95 the mean over ten thresholds from 0.50 to 0.95.",
    },
    model: { tr: "Doğrulamada maske AP50: bina 0,69, spor sahası 0,30; mAP50 0,50, mAP50-95 0,25.", en: "Mask AP50 on the validation set: building 0.69, sports field 0.30; mAP50 0.50, mAP50-95 0.25." },
    modelLabel: { tr: "Bu bölümün modelinde", en: "In this chapter's model" },
    link: [{ chapter: "yolo", tab: "theory", section: "y-metrics" }, { chapter: "yolo", tab: "outputs", section: "yr-ap" }, { chapter: "yolo", tab: "math", section: "ym-ap" }],
  },
  "confusion-matrix": {
    group: "detect",
    name: { tr: "Karışıklık matrisi", en: "Confusion matrix" },
    aka: ["confusion matrix", "hata matrisi", "karmaşıklık matrisi", "confusion_matrix.png"],
    def: {
      tr: "Tahminleri gerçekle karşılaştıran tablo: her hücre, \"gerçekte şu sınıf, model bu sınıf dedi\" durumlarının sayısıdır. Köşegen doğruları, köşegen dışı karışmaları gösterir. İki sınıfta dört hücresi vardır: [[true-positive|TP]], [[false-positive|FP]], [[false-negative|FN]], [[true-negative|TN]]. Ultralytics satırlara tahmini, sütunlara gerçeği yazar; tespitte bir \"arka plan\" satırı ve sütunu ekler.",
      en: "A table that compares the predictions with the truth: each cell counts the cases \"really this class, the model said that class\". The diagonal shows the correct ones, the rest the confusions. With two classes it has four cells: [[true-positive|TP]], [[false-positive|FP]], [[false-negative|FN]], [[true-negative|TN]]. Ultralytics puts the prediction in the rows and the truth in the columns; in detection it adds a \"background\" row and column.",
    },
    model: { tr: "confusion_matrix.png güven eşiği 0,001 ile çizilir: arka plan sütunundaki binlerce FP bundandır; eşik 0,25'te çok azalır.", en: "confusion_matrix.png is drawn at confidence threshold 0.001: that is where the thousands of FPs in the background column come from; at 0.25 they shrink a lot." },
    modelLabel: { tr: "Bu bölümün modelinde", en: "In this chapter's model" },
    link: [{ chapter: "yolo", tab: "outputs", section: "yr-cm-intro" }, { chapter: "yolo", tab: "outputs", section: "yr-cm-multi" }, { chapter: "yolo", tab: "outputs", section: "yr-cm-real" }],
  },
  "true-positive": {
    group: "detect",
    name: { tr: "Doğru pozitif (TP)", en: "True positive (TP)" },
    aka: ["tp", "true positive", "doğru tespit", "isabet"],
    def: {
      tr: "Modelin \"var\" dediği ve gerçekten olan durum. Tespitte: aynı sınıftan bir gerçek nesneyle [[iou|IoU]]'su eşiği (genellikle 0,5) geçen kutu. [[confusion-matrix|Karışıklık matrisinin]] köşegenindedir.",
      en: "A case the model calls \"yes\" that really is. In detection: a box whose [[iou|IoU]] with a true object of its class passes the threshold (usually 0.5). It sits on the diagonal of the [[confusion-matrix|confusion matrix]].",
    },
    link: { chapter: "yolo", tab: "outputs", section: "yr-cm-intro" },
  },
  "false-positive": {
    group: "detect",
    name: { tr: "Yanlış pozitif (FP)", en: "False positive (FP)" },
    aka: ["fp", "false positive", "yanlış alarm", "false alarm", "yanlış tespit", "tip i hata", "type i error"],
    def: {
      tr: "Modelin \"var\" dediği ama olmayan durum: yanlış alarm. Tespitte: hiçbir gerçek nesneyle yeterince örtüşmeyen kutu, aynı nesneye konmuş ikinci kutu ya da yanlış sınıflı kutu. Çoğalırsa [[precision|kesinlik]] düşer.",
      en: "A case the model calls \"yes\" that is not: a false alarm. In detection: a box that overlaps no true object enough, a second box on the same object, or a box with the wrong class. More of them lower [[precision|precision]].",
    },
    link: [{ chapter: "yolo", tab: "outputs", section: "yr-cm-intro" }, { chapter: "yolo", tab: "outputs", section: "yr-cm-toy" }],
  },
  "false-negative": {
    group: "detect",
    name: { tr: "Yanlış negatif (FN)", en: "False negative (FN)" },
    aka: ["fn", "false negative", "kaçan", "kaçırılan", "missed", "tip ii hata", "type ii error"],
    def: {
      tr: "Gerçekte olan ama modelin bulamadığı durum: gözden kaçan nesne. Çoğalırsa [[recall|duyarlılık]] düşer. Hasar taraması gibi işlerde en pahalı hatadır; güven eşiğini düşürmek azaltır.",
      en: "A case that really is but the model misses: an object overlooked. More of them lower [[recall|recall]]. In jobs like damage surveys it is the most expensive mistake; lowering the confidence threshold reduces it.",
    },
    link: [{ chapter: "yolo", tab: "outputs", section: "yr-cm-intro" }, { chapter: "yolo", tab: "outputs", section: "yr-cm-toy" }],
  },
  "true-negative": {
    group: "detect",
    name: { tr: "Doğru negatif (TN)", en: "True negative (TN)" },
    aka: ["tn", "true negative"],
    def: {
      tr: "Modelin \"yok\" dediği ve gerçekten olmayan durum. Sınıflandırmada ve piksel piksel segmentasyonda sayılır; nesne tespitinde sayılamaz (bir görüntüde kaç \"boş yer\" vardır?), bu yüzden tespitte [[accuracy|doğruluk]] ve [[specificity|özgüllük]] kullanılmaz.",
      en: "A case the model calls \"no\" that really is not. It is counted in classification and in pixel-by-pixel segmentation; in object detection it cannot be counted (how many \"empty places\" does an image have?), so detection does not use [[accuracy|accuracy]] or [[specificity|specificity]].",
    },
    link: [{ chapter: "yolo", tab: "outputs", section: "yr-cm-toy" }, { chapter: "yolo", tab: "outputs", section: "yr-cm-sem" }],
  },
  "f1-score": {
    group: "detect",
    name: { tr: "F1 skoru", en: "F1 score" },
    aka: ["f1", "f-score", "f ölçüsü", "harmonik ortalama", "harmonic mean"],
    def: {
      tr: "[[precision|Kesinlik]] ile [[recall|duyarlılığın]] harmonik ortalaması: \\(F_1 = 2PR/(P + R)\\). Biri küçükse F1 de küçük kalır (P = 0,9, R = 0,1 ise F1 = 0,18). Ultralytics, konsolda ve tabloda P ile R'yi ortalama F1'in en yüksek olduğu güven eşiğinde yazar.",
      en: "The harmonic mean of [[precision|precision]] and [[recall|recall]]: \\(F_1 = 2PR/(P + R)\\). If either is small, F1 stays small (P = 0.9, R = 0.1 gives F1 = 0.18). Ultralytics writes P and R in the console and the table at the confidence threshold where the mean F1 is highest.",
    },
    link: [{ chapter: "yolo", tab: "outputs", section: "yr-curves" }, { chapter: "yolo", tab: "outputs", section: "yr-cm-intro" }],
  },
  specificity: {
    group: "detect",
    name: { tr: "Özgüllük", en: "Specificity" },
    aka: ["specificity", "true negative rate", "tnr", "seçicilik"],
    def: {
      tr: "Gerçek negatiflerin ne kadarının \"yok\" dendiği: \\(TN / (TN + FP)\\). Tıpta ve hasar taramasında [[recall|duyarlılıkla]] birlikte verilir. [[true-negative|TN]] gerektirdiği için nesne tespitinde hesaplanamaz.",
      en: "How many of the true negatives were called \"no\": \\(TN / (TN + FP)\\). In medicine and damage surveys it is given together with [[recall|recall]]. Since it needs [[true-negative|TN]], it cannot be computed in object detection.",
    },
    link: { chapter: "yolo", tab: "outputs", section: "yr-cm-intro" },
  },
  "class-imbalance": {
    group: "detect",
    name: { tr: "Sınıf dengesizliği", en: "Class imbalance" },
    aka: ["imbalance", "dengesiz veri", "imbalanced data", "azınlık sınıfı", "minority class", "cls_pw"],
    def: {
      tr: "Bir sınıfın örneklerinin ötekilerden çok daha az olması. Model az gördüğü sınıfı kötü öğrenir ve [[accuracy|doğruluk]] yanıltıcı olur. Çareler: daha çok örnek toplamak, azınlık sınıfını içeren örnekleri çoğaltmak, kayıpta ona ağırlık vermek.",
      en: "One class having far fewer examples than the others. The model learns the rare class badly and [[accuracy|accuracy]] misleads. Remedies: collect more examples, repeat the examples with the rare class, weight it in the loss.",
    },
    model: { tr: "Eğitimde 10.974 bina, 529 spor sahası (yaklaşık 21'e 1); saha AP50'si 0,30.", en: "Training has 10,974 buildings and 529 sports fields (about 21 to 1); the field's AP50 is 0.30." },
    modelLabel: { tr: "Bu bölümün modelinde", en: "In this chapter's model" },
    link: { chapter: "yolo", tab: "outputs", section: "yr-labels" },
  },
  "macro-average": {
    group: "detect",
    name: { tr: "Makro, mikro ve ağırlıklı ortalama", en: "Macro, micro and weighted average" },
    aka: ["macro average", "micro average", "weighted average", "makro ortalama", "mikro ortalama", "ağırlıklı ortalama", "macro f1"],
    def: {
      tr: "Çok sınıflı bir modelin sınıf başına ölçülerini tek sayıya indirmenin üç yolu. Makro: her sınıfın [[precision|kesinliği]], [[recall|duyarlılığı]] ya da [[f1-score|F1]]'i ayrı hesaplanır, sonra eşit ağırlıkla ortalanır; az örnekli sınıfın kötü sonucu görünür kalır. Ağırlıklı: sınıflar örnek sayılarıyla ağırlıklanır. Mikro: bütün sınıfların TP, FP ve FN'si toplanıp bir kez hesaplanır; her örneğin tek etiketi varsa [[accuracy|doğruluğa]] eşittir. mAP bir makro ortalamadır.",
      en: "Three ways to bring a multi-class model's per-class scores down to one number. Macro: each class's [[precision|precision]], [[recall|recall]] or [[f1-score|F1]] is computed separately, then averaged with equal weight; a rare class's poor result stays visible. Weighted: the classes are weighted by their number of examples. Micro: every class's TP, FP and FN are added up and computed once; when every example has one label it equals the [[accuracy|accuracy]]. mAP is a macro average.",
    },
    link: { chapter: "yolo", tab: "outputs", section: "yr-cm-multi" },
  },
};

// =====================================================================
// Shared helpers
// =====================================================================
export const otherLang = () => (I18N.lang === "en" ? "tr" : "en");

/** [[key|text]] → a term button. A term does not link to itself. */
export function termify(html, self = null) {
  return html.replace(/\[\[([\w-]+)\|([^\]]+)\]\]/g, (_, key, text) =>
    GLOSSARY[key] && key !== self ? `<button type="button" class="gl-term" data-term="${key}">${text}</button>` : text);
}

/** The Math tab section that derives each term (mathtab.js). */
export const TERM_MATH = {
  parameter: "m-notation", tensor: "m-notation",
  convolution: "m-conv", kernel: "m-conv", stride: "m-conv", padding: "m-conv", channel: "m-conv", "feature-map": "m-conv",
  batchnorm: "m-bn", "train-eval": "m-bn",
  activation: "m-act", relu: "m-act",
  pooling: "m-pool", "receptive-field": "m-pool",
  dropout: "m-drop",
  flatten: "m-fc", "fully-connected": "m-fc",
  softmax: "m-softmax", "cross-entropy": "m-softmax", logit: "m-softmax", loss: "m-softmax",
  backprop: "m-backprop", gradient: "m-backprop", "forward-pass": "m-backprop", step: "m-backprop",
  "gradient-descent": "m-opt", sgd: "m-opt", momentum: "m-opt", adam: "m-opt", optimizer: "m-opt", "learning-rate": "m-opt",
};

/** Section names for links whose section has no heading of its own. */
export const SECTION_NAMES = { "t-schematic": { tr: "Mimari şeması", en: "Architecture diagram" } };

/** "Theory → Convolution" on this page, "CNN · Theory → …" for a place in another chapter. */
export function placeLabel(link) {
  const chapter = link.chapter || "cnn";
  const here = chapter === CHAPTER;
  const tabName = L(TAB_NAMES[link.tab]);
  const prefix = here ? "" : `${L(chapterOf(chapter).short)} · `;
  let name = link.section && (link.title || PLACE_TITLES[link.section] || SECTION_NAMES[link.section]);   // title: a place on another page
  name = name && L(name);
  if (!name && here && link.section) {
    if (link.tab === "code") name = L(hooks.codeSteps?.find((s) => s.id === link.section)?.title);
    if (!name && link.tab === "math") name = L(window.MATH_SECTIONS?.find((s) => s.id === link.section)?.title);
    if (!name) name = document.querySelector(`#${link.section} h2`)?.textContent.trim();
  }
  return prefix + tabName + (name ? ` → ${name}` : "");
}

/** A term's places: this page's first, then the other chapters'. */
export function termPlaces(key) {
  const entry = GLOSSARY[key];
  const places = [entry.link || []].flat();
  if (TERM_MATH[key]) places.push({ tab: "math", section: TERM_MATH[key] });
  const here = (p) => (p.chapter || "cnn") === CHAPTER;
  return [...places.filter(here), ...places.filter((p) => !here(p))];
}

/** Related terms that the definition does not already link to. */
export function relatedTerms(key) {
  const entry = GLOSSARY[key];
  const linked = L(entry.def);
  return (entry.see || []).filter((k) => GLOSSARY[k] && !linked.includes(`[[${k}|`));
}

