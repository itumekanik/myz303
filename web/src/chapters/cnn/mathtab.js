/*
 * Math tab: the mathematics behind every building block of SimpleCNN, with forward
 * formulas, step-by-step derivations of the gradients and small interactive examples.
 *
 * Texts are { tr, en } pairs of HTML; formulas are LaTeX between \( and \) (typeset by
 * math.js), [[key|text]] links a glossary term and <a data-go="tab:section"> links another
 * tab (add data-chapter="mlp" for another chapter's page). Blocks: p, list, h, eq, derive, widget,
 * model, callout, table, links ([tab, section, chapter?]). Uses the helpers from i18n.js, app.js and
 * glossary.js (h, $, L, navigate, termify, placeLabel, fmtNum, texNum).
 */
import { I18N, L } from "../../shared/i18n.js";
import { texNum } from "../../shared/math.js";
import { $, fmtNum, h, locale, navigate, svg, svgText } from "../../shared/core.js";
import { placeLabel, termify } from "../../shared/glossary/terms.js";

(() => {
const R = String.raw;

// =====================================================================
// Content
// =====================================================================
const SECTIONS = [
  // ------------------------------------------------------------------ 1
  {
    id: "m-notation",
    title: { tr: "Gösterim, boyutlar ve parametreler", en: "Notation, shapes and parameters" },
    blocks: [
      { p: {
        tr: R`Teori sekmesinde her yapı taşının ne yaptığını görsel olarak gördük. Burada aynı yapı taşlarını formüllerle ele alıyoruz: her kısımda önce <b>ileri yayılım</b> (girdiden çıktıyı hesaplamak), sonra eğitim için gereken <b>geri yayılım</b> (türevler) var. Türetmeleri <b>Sonraki adım</b> düğmesiyle satır satır açabilir, etkileşimli örneklerde sayıları değiştirip formüllerin nasıl tepki verdiğini görebilirsiniz.`,
        en: R`The Theory tab showed what each building block does. Here we take the same blocks with formulas: every section first gives the <b>forward pass</b> (computing the output from the input), then the <b>backward pass</b> (the derivatives) that training needs. Open derivations line by line with the <b>Next step</b> button, and change the numbers in the interactive examples to see how the formulas react.`,
      } },
      { p: {
        tr: R`Bu sekme, <a href="#" data-go="math:mm-forward" data-chapter="mlp">MLP bölümünün Matematik sekmesinin</a> devamıdır. Orada ileri hesabı, aktivasyon türevlerini, çapraz entropinin gradyanı \(\hat y - y\)'yi, geri yayılımın dört denklemini ve başlangıç ağırlıklarını türettiniz. Bunları burada yeniden türetmiyoruz, kısaca hatırlatıp kullanıyoruz. Yeni olan, CNN'e özgü katmanlardır. Bu sekmenin sonunda şunları yapabileceksiniz:`,
        en: R`This tab continues <a href="#" data-go="math:mm-forward" data-chapter="mlp">chapter 2's Math tab</a>. There you derived the forward pass, the activation derivatives, the cross-entropy gradient \(\hat y - y\), the four equations of backpropagation and the initial weights. We do not derive them again here; we recall them briefly and use them. What is new are the layers specific to a CNN. By the end of this tab you will be able to:`,
      } },
      { list: [
        { tr: R`bir evrişim katmanının çıktı boyutunu, parametre sayısını ve işlem sayısını hesaplamak (kısım 2);`, en: R`compute a convolution layer's output size, number of parameters and number of operations (section 2);` },
        { tr: R`evrişim, BatchNorm, havuzlama ve dropout için \(\delta_y\)'den \(\delta_x\)'e giden geri kuralı yazmak (kısım 2, 3, 5, 6);`, en: R`write the backward rule from \(\delta_y\) to \(\delta_x\) for convolution, BatchNorm, pooling and dropout (sections 2, 3, 5, 6);` },
        { tr: R`MLP bölümündeki \(\hat y - y\) gradyanının 10 sınıfta \(\mathbf p - \mathbf y\) olduğunu göstermek (kısım 8);`, en: R`show that chapter 2's gradient \(\hat y - y\) becomes \(\mathbf p - \mathbf y\) for 10 classes (section 8);` },
        { tr: R`SimpleCNN'in bütün geri yayılımını bu kurallardan kurmak ve Adam'ın güncelleme kuralını okumak (kısım 9, 10).`, en: R`assemble SimpleCNN's whole backward pass from these rules and read Adam's update rule (sections 9, 10).` },
      ] },
      { h: { tr: "Semboller", en: "Symbols" } },
      { table: {
        head: [{ tr: "Sembol", en: "Symbol" }, { tr: "Anlamı", en: "Meaning" }],
        rows: [
          [R`\(x,\ y\)`, { tr: R`bir katmanın girdisi ve çıktısı (8. kısımda \(\mathbf y\), MLP bölümündeki gibi doğru etiketi gösterir)`, en: R`a layer's input and output (in section 8, \(\mathbf y\) is the correct label, as in chapter 2)` }],
          [R`\(z,\ a = f(z)\)`, { tr: "aktivasyondan önceki ve sonraki değer (MLP bölümündeki gibi)", en: "the value before and after the activation (as in chapter 2)" }],
          [R`\(w,\ \mathbf{W},\ b\)`, { tr: "ağırlıklar (tek tek ve matris olarak) ve bias", en: "weights (one by one and as a matrix) and bias" }],
          [R`\(N,\ C,\ H\times W\)`, { tr: "batch'teki görüntü sayısı, kanal sayısı, yükseklik × genişlik", en: "images in the batch, number of channels, height × width" }],
          [R`\(L\)`, { tr: "kayıp: bütün batch için tek bir sayı", en: "the loss: a single number for the whole batch" }],
          [R`\(\delta_y = \dfrac{\partial L}{\partial y}\)`, { tr: "kaybın bir değere göre türevi; geri yayılımda katmandan katmana bu taşınır", en: "the derivative of the loss with respect to a value; this is what backpropagation passes from layer to layer" }],
          [R`\(\odot\)`, { tr: "eleman eleman çarpım", en: "element-wise product" }],
          [R`\(\theta,\ \eta\)`, { tr: "modelin bütün parametreleri ve öğrenme oranı", en: "all parameters of the model and the learning rate" }],
          [R`\(\mathbb{1}[\cdot]\)`, { tr: "gösterge: koşul doğruysa 1, değilse 0", en: "indicator: 1 if the condition holds, 0 otherwise" }],
        ],
      } },
      { callout: "info", title: { tr: "Neden hep δ?", en: "Why δ everywhere?" }, text: {
        tr: R`Geri yayılımda her katman, çıktısına göre türevi \(\delta_y\) alır, girdisine göre türevi \(\delta_x\) hesaplar ve bir önceki katmana verir. Her kısımda bu kuralı, yani \(\delta_y\)'den \(\delta_x\)'e nasıl gidileceğini yazacağız. MLP bölümünde \(\boldsymbol\delta^{(l)} = \partial L/\partial\mathbf z^{(l)}\) idi: tam bağlı katmanın çıktısına göre türev. Burada her işlemi (evrişim, BatchNorm, ReLU, havuzlama…) ayrı bir katman sayıyoruz ve \(\delta_y\) o işlemin çıktısına göre türevdir; tam bağlı bir katman için ikisi aynı şeydir.`,
        en: R`In backpropagation every layer receives the derivative with respect to its output, \(\delta_y\), computes the derivative with respect to its input, \(\delta_x\), and passes it to the layer before. Each section writes down this rule: how to get from \(\delta_y\) to \(\delta_x\). In chapter 2, \(\boldsymbol\delta^{(l)} = \partial L/\partial\mathbf z^{(l)}\): the derivative with respect to a fully connected layer's output. Here every operation (convolution, BatchNorm, ReLU, pooling…) counts as a layer of its own, and \(\delta_y\) is the derivative with respect to that operation's output; for a fully connected layer the two are the same thing.`,
      } },
      { h: { tr: "SimpleCNN'in boyutları", en: "SimpleCNN's shapes" } },
      { table: {
        head: [{ tr: "Katman", en: "Layer" }, { tr: "İşlem", en: "Operation" }, { tr: "Çıktı şekli", en: "Output shape" }, { tr: "Parametre", en: "Parameters" }],
        rows: [
          [{ tr: "Giriş", en: "Input" }, "—", "1×28×28", "0"],
          ["Conv1", { tr: "32 filtre, 5×5", en: "32 filters, 5×5" }, "32×24×24", { tr: R`\(32\cdot(1\cdot 25+1)\) = 832`, en: R`\(32\cdot(1\cdot 25+1)\) = 832` }],
          ["BatchNorm1", "γ, β", "32×24×24", R`\(2\cdot 32\) = 64`],
          ["ReLU, MaxPool1", { tr: "2×2, stride 2", en: "2×2, stride 2" }, "32×12×12", "0"],
          ["Dropout1", { tr: "p = 0,45", en: "p = 0.45" }, "32×12×12", "0"],
          ["Conv2", { tr: "64 filtre, 5×5", en: "64 filters, 5×5" }, "64×8×8", { tr: R`\(64\cdot(32\cdot 25+1)\) = 51.264`, en: R`\(64\cdot(32\cdot 25+1)\) = 51,264` }],
          ["BatchNorm2", "γ, β", "64×8×8", R`\(2\cdot 64\) = 128`],
          ["ReLU, MaxPool2, Dropout2", { tr: "p = 0,35", en: "p = 0.35" }, "64×4×4", "0"],
          [{ tr: "Düzleştirme", en: "Flatten" }, "64·4·4", "1024", "0"],
          ["FC1 + ReLU", "1024 → 512", "512", { tr: R`\(512\cdot(1024+1)\) = 524.800`, en: R`\(512\cdot(1024+1)\) = 524,800` }],
          ["FC2", "512 → 10", "10", { tr: R`\(10\cdot(512+1)\) = 5.130`, en: R`\(10\cdot(512+1)\) = 5,130` }],
          [{ tr: "<b>Toplam</b>", en: "<b>Total</b>" }, "", "", { tr: "<b>582.218</b>", en: "<b>582,218</b>" }],
        ],
      } },
      { h: { tr: "Başlangıç ağırlıkları", en: "Initial weights" } },
      { p: {
        tr: R`Eğitim rastgele ağırlıklardan başlar. PyTorch, Conv ve Linear katmanlarında her ağırlığı ve bias'ı şu aralıktan düzgün dağılımla çeker:`,
        en: R`Training starts from random weights. In Conv and Linear layers, PyTorch draws every weight and bias uniformly from this range:`,
      } },
      { eq: R`w \sim \mathcal{U}\!\left(-\frac{1}{\sqrt{n_{\text{in}}}},\ \frac{1}{\sqrt{n_{\text{in}}}}\right)` },
      { p: {
        tr: R`\(n_{\text{in}}\), bir çıkışın bağlı olduğu giriş sayısıdır (fan-in): Conv1'de \(1\cdot5\cdot5 = 25\), yani ağırlıklar ±0,2 aralığında; FC1'de 1024, yani ±0,031. Bu dağılımın varyansı \(\operatorname{Var}(w) = \frac{1}{3n_{\text{in}}}\)'tir. <a href="#" data-go="math:mm-train" data-chapter="mlp">MLP bölümündeki</a> hesapla, \(n_{\text{in}}\) terimli \(\sum_i w_i x_i\) toplamının varyansı \(n_{\text{in}}\cdot\frac{1}{3n_{\text{in}}}\,\mathbb{E}[x^2] = \frac{1}{3}\,\mathbb{E}[x^2]\) olur: giriş sayısından bağımsızdır, ama oradaki LeCun kuralının (\(\operatorname{Var}(w) = 1/n_{\text{in}}\)) üçte biri, ReLU için önerilen He kuralının (\(2/n_{\text{in}}\)) altıda biridir. Yani bu başlangıçla sinyal, normalleştirme olmasa ReLU'lu her katmanda varyans olarak yaklaşık 6 kat küçülürdü. SimpleCNN'de bu bir sorun değildir: her evrişimden sonra gelen BatchNorm (3. kısım) ölçeği yeniden ayarlar, sonda da yalnızca iki tam bağlı katman vardır. BatchNorm'da başlangıçta \(\gamma = 1,\ \beta = 0\)'dır. Bu sayfadaki <b>eğitim öncesi</b> model de bu dağılımdan çekilmiş ağırlıklardır.`,
        en: R`\(n_{\text{in}}\) is the number of inputs an output is connected to (fan-in): 25 for Conv1 (\(1\cdot5\cdot5\)), so the weights are within ±0.2; 1024 for FC1, so ±0.031. This distribution has variance \(\operatorname{Var}(w) = \frac{1}{3n_{\text{in}}}\). With the calculation from <a href="#" data-go="math:mm-train" data-chapter="mlp">chapter 2</a>, the variance of the sum \(\sum_i w_i x_i\) of \(n_{\text{in}}\) terms is \(n_{\text{in}}\cdot\frac{1}{3n_{\text{in}}}\,\mathbb{E}[x^2] = \frac{1}{3}\,\mathbb{E}[x^2]\): independent of the number of inputs, but a third of what the LeCun rule there gives (\(\operatorname{Var}(w) = 1/n_{\text{in}}\)) and a sixth of the He rule recommended for ReLU (\(2/n_{\text{in}}\)). So with this start, and without normalization, the signal's variance would shrink about 6-fold in every layer with a ReLU. In SimpleCNN this is not a problem: the BatchNorm after every convolution (section 3) resets the scale, and at the end there are only two fully connected layers. BatchNorm starts with \(\gamma = 1,\ \beta = 0\). This page's <b>before-training</b> model holds weights drawn from exactly this distribution.`,
      } },
      { links: [["theory", "t-schematic"], ["code", "c-model"], ["math", "mm-train", "mlp"]] },
    ],
  },

  // ------------------------------------------------------------------ 2
  {
    id: "m-conv",
    title: { tr: "Evrişim", en: "Convolution" },
    blocks: [
      { p: {
        tr: R`Bir [[convolution|evrişim]] katmanının çıktısındaki her sayı, girdinin küçük bir bölgesiyle [[kernel|filtrenin]] <b>ağırlıklı toplamıdır</b>. Önce en basit hâli: tek kanal, stride 1, dolgu yok, \(k\times k\) filtre.`,
        en: R`Every number in the output of a [[convolution|convolution]] layer is a <b>weighted sum</b> of a small region of the input and the [[kernel|filter]]. First the simplest case: one channel, stride 1, no padding, a \(k\times k\) filter.`,
      } },
      { eq: R`y_{i,j} = b + \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} w_{m,n}\, x_{i+m,\; j+n}` },
      { p: {
        tr: R`Aşağıda 5×5 bir girdi ve 3×3 bir dikey kenar filtresi var (\(b = 0\)). Çıktıdaki bir hücreye tıklayın ya da üzerine gelin: o hücrenin hesabındaki dokuz çarpım tek tek yazılır.`,
        en: R`Below is a 5×5 input and a 3×3 vertical-edge filter (\(b = 0\)). Click or hover an output cell: the nine products of its calculation are written out.`,
      } },
      { widget: "conv" },
      { callout: "info", title: { tr: "Aslında çapraz korelasyon", en: "Actually a cross-correlation" }, text: {
        tr: R`Matematikteki evrişim filtreyi 180° çevirerek uygular: \((x * w)_{i,j} = \sum_{m,n} w_{m,n}\, x_{i-m,\,j-n}\). Derin öğrenme kütüphaneleri filtreyi çevirmez (çapraz korelasyon), ama işlemin adı yine evrişimdir. Filtreler öğrenildiği için fark etmez: gerekirse ağ çevrilmiş filtreyi öğrenir.`,
        en: R`Convolution in mathematics flips the filter by 180°: \((x * w)_{i,j} = \sum_{m,n} w_{m,n}\, x_{i-m,\,j-n}\). Deep learning libraries do not flip it (a cross-correlation), but still call it convolution. Since the filters are learned it makes no difference: the network learns the flipped filter if it needs to.`,
      } },
      { h: { tr: "Çok kanallı genel formül", en: "The general multi-channel formula" } },
      { p: {
        tr: R`Gerçek katmanda girdinin \(C_{\text{in}}\) [[channel|kanalı]] vardır ve her çıkış kanalı \(o\), kendi \(C_{\text{in}}\times k\times k\) ağırlığıyla bütün giriş kanallarına birlikte bakar. [[stride|Stride]] \(s\) ve [[padding|dolgu]] \(p\) ile (dolgunun dışına düşen yerlerde \(x = 0\)):`,
        en: R`In a real layer the input has \(C_{\text{in}}\) [[channel|channels]], and every output channel \(o\) looks at all input channels together with its own \(C_{\text{in}}\times k\times k\) weights. With [[stride|stride]] \(s\) and [[padding|padding]] \(p\) (\(x = 0\) outside the image):`,
      } },
      { eq: R`y_{o,i,j} = b_o + \sum_{c=0}^{C_{\text{in}}-1}\sum_{m=0}^{k-1}\sum_{n=0}^{k-1} w_{o,c,m,n}\; x_{c,\; s i+m-p,\; s j+n-p}` },
      { p: {
        tr: R`Bu formülden üç sayı çıkar: çıktı boyutu, parametre sayısı ve çarpma-toplama işlemi sayısı (MAC):`,
        en: R`Three numbers follow from this formula: the output size, the number of parameters and the number of multiply-accumulate operations (MACs):`,
      } },
      { eq: R`H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1 \qquad P = C_{\text{out}}\,(C_{\text{in}}\,k^2 + 1) \qquad \text{MAC} = C_{\text{out}}\,H_{\text{out}}\,W_{\text{out}}\,C_{\text{in}}\,k^2` },
      { p: {
        tr: R`Conv1: \(\lfloor(28-5)/1\rfloor+1 = 24\), \(P = 32\cdot26 = 832\), \(32\cdot24\cdot24\cdot25 = 460.800\) MAC. Conv2: \(P = 64\cdot801 = 51.264\) ama \(64\cdot8\cdot8\cdot800 = 3.276.800\) MAC. FC1'in 524.800 parametresine karşılık yalnızca 524.288 MAC'i var. Yani <b>parametrelerin çoğu tam bağlı katmanda, hesabın çoğu evrişimdedir</b>: evrişimde aynı ağırlıklar her konumda yeniden kullanılır.`,
        en: R`Conv1: \(\lfloor(28-5)/1\rfloor+1 = 24\), \(P = 32\cdot26 = 832\), \(32\cdot24\cdot24\cdot25 = 460{,}800\) MACs. Conv2: \(P = 64\cdot801 = 51{,}264\) but \(64\cdot8\cdot8\cdot800 = 3{,}276{,}800\) MACs. FC1 has 524,800 parameters but only 524,288 MACs. So <b>most parameters are in the fully connected layer, most computation is in the convolutions</b>: a convolution reuses the same weights at every position.`,
      } },
      { h: { tr: "Geri yayılım", en: "Backward pass" } },
      { p: {
        tr: R`Eğitim için iki türev gerekir: filtre ağırlıklarına göre (onları güncellemek için) ve girdiye göre (bir önceki katmana iletmek için). Tek kanal ve stride 1 için türetelim; \(\delta_{i,j} = \partial L/\partial y_{i,j}\), çıktıdan gelen gradyandır.`,
        en: R`Training needs two derivatives: with respect to the filter weights (to update them) and with respect to the input (to pass on to the previous layer). Let us derive them for one channel and stride 1; \(\delta_{i,j} = \partial L/\partial y_{i,j}\) is the gradient coming from the output.`,
      } },
      { derive: { title: { tr: "Ağırlık ve girdi gradyanı", en: "Weight and input gradients" }, steps: [
        { tex: R`\frac{\partial L}{\partial w_{m,n}} = \sum_{i,j} \frac{\partial L}{\partial y_{i,j}}\;\frac{\partial y_{i,j}}{\partial w_{m,n}}`, why: {
          tr: "Zincir kuralı. Aynı ağırlık her konumda kullanıldığı için (ağırlık paylaşımı) bütün çıktı konumlarının katkıları toplanır.",
          en: "The chain rule. The same weight is used at every position (weight sharing), so the contributions of all output positions add up." } },
        { tex: R`\frac{\partial y_{i,j}}{\partial w_{m,n}} = x_{i+m,\;j+n}`, why: {
          tr: R`\(y_{i,j}\)'nin toplamında \(w_{m,n}\) yalnızca bir terimde geçer ve o terimin çarpanı \(x_{i+m,\,j+n}\)'dir.`,
          en: R`\(w_{m,n}\) appears in only one term of the sum for \(y_{i,j}\), multiplied by \(x_{i+m,\,j+n}\).` } },
        { tex: R`\frac{\partial L}{\partial w_{m,n}} = \sum_{i,j} \delta_{i,j}\; x_{i+m,\;j+n} \qquad \frac{\partial L}{\partial b} = \sum_{i,j}\delta_{i,j}`, why: {
          tr: R`Sonuç yine bir korelasyondur: girdi bu kez filtreyle değil, çıktı gradyanı \(\delta\) ile taranır. Bu, MLP bölümündeki (3) denkleminin (gradyan = girdi × hata) ağırlık paylaşımlı hâlidir: ağırlığın kullanıldığı bütün konumların "girdi × hata" katkıları toplanır.`,
          en: R`The result is again a correlation: this time the input is scanned with the output gradient \(\delta\) instead of the filter. This is chapter 2's equation (3) (gradient = input × error) with weight sharing: the "input × error" contributions of every position where the weight is used add up.` } },
        { tex: R`\frac{\partial L}{\partial x_{u,v}} = \sum_{m,n} \delta_{u-m,\;v-n}\; w_{m,n}`, why: {
          tr: R`Girdi pikseli \(x_{u,v}\), filtrenin \((m,n)\) hücresi üzerine geldiği her çıktıyı, yani \(y_{u-m,\,v-n}\)'yi etkiler. Görüntünün dışına düşen \(\delta\) değerleri sıfır sayılır.`,
          en: R`The input pixel \(x_{u,v}\) affects every output where the filter's cell \((m,n)\) lands on it, i.e. \(y_{u-m,\,v-n}\). \(\delta\) values outside the image count as zero.` } },
        { tex: R`\delta_x = \operatorname{pad}_{k-1}(\delta_y) \star \operatorname{rot}_{180}(w)`, why: {
          tr: R`Bu, kenarlarına \(k-1\) sıra sıfır eklenmiş \(\delta\)'nın, 180° çevrilmiş filtreyle korelasyonudur (\(\star\): ileri yayılımdaki işlem). İleri yayılımda çevirmediğimiz filtre, geri yayılımda çevrilmiş olarak karşımıza çıkar.`,
          en: R`This is a correlation (\(\star\), the forward operation) of \(\delta\), padded with \(k-1\) rows of zeros, with the filter rotated by 180°. The filter we did not flip in the forward pass shows up flipped in the backward pass.` } },
      ] } },
      { p: {
        tr: R`Çok kanallı katmanda aynı kurallar her (çıkış kanalı \(o\), giriş kanalı \(c\)) çifti için geçerlidir: \(\dfrac{\partial L}{\partial w_{o,c,m,n}} = \sum_{i,j}\delta_{o,i,j}\,x_{c,\,i+m,\,j+n}\) ve \(\delta_{x,c} = \sum_{o}\operatorname{pad}_{k-1}(\delta_{y,o}) \star \operatorname{rot}_{180}(w_{o,c})\). Her giriş kanalı, kendisine bakan bütün çıkış kanallarının gradyanlarını toplar; tam bağlı katmandaki \(\mathbf W^\top\boldsymbol\delta\)'nın evrişimdeki karşılığı budur.`,
        en: R`In a multi-channel layer the same rules hold for every pair (output channel \(o\), input channel \(c\)): \(\dfrac{\partial L}{\partial w_{o,c,m,n}} = \sum_{i,j}\delta_{o,i,j}\,x_{c,\,i+m,\,j+n}\) and \(\delta_{x,c} = \sum_{o}\operatorname{pad}_{k-1}(\delta_{y,o}) \star \operatorname{rot}_{180}(w_{o,c})\). Every input channel collects the gradients of all the output channels that look at it; this is the convolutional counterpart of the fully connected layer's \(\mathbf W^\top\boldsymbol\delta\).`,
      } },
      { model: {
        tr: R`<b>SimpleCNN'de:</b> Conv1'in ağırlık gradyanı, batch'teki 256 görüntünün her birinde 24×24 konumun katkılarının toplamıdır (kayıp batch ortalaması olduğu için \(1/256\) ile ölçeklenir). Conv1'in girdi gradyanı ise hiç hesaplanmaz: giriş görüntüsü öğrenilen bir şey değildir.`,
        en: R`<b>In SimpleCNN:</b> Conv1's weight gradient sums the contributions of 24×24 positions in each of the 256 images of the batch (scaled by \(1/256\), since the loss is the batch mean). Conv1's input gradient is not computed at all: the input image is not learned.`,
      } },
      { links: [["theory", "t-conv"], ["code", "c-model"]] },
    ],
  },

  // ------------------------------------------------------------------ 3
  {
    id: "m-bn",
    title: { tr: "Batch Normalization", en: "Batch normalization" },
    blocks: [
      { p: {
        tr: R`MLP bölümünde sinyalin ölçeğini başlangıç ağırlıklarını dikkatle seçerek korumaya çalıştık; [[batchnorm|BatchNorm]] bunu eğitimin her adımında, ölçeği veriden ölçerek yapar. Her kanal için ayrı çalışır. Bir kanalın batch'teki bütün değerleri \(x_1,\dots,x_m\) olsun; Conv1'den sonra \(m = N\cdot H\cdot W = 256\cdot24\cdot24 = 147.456\). Eğitimde dört adım uygulanır:`,
        en: R`In chapter 2 we tried to keep the signal's scale by choosing the initial weights carefully; [[batchnorm|BatchNorm]] does it at every training step, by measuring the scale from the data. It works on each channel separately. Let \(x_1,\dots,x_m\) be all values of one channel in the batch; after Conv1, \(m = N\cdot H\cdot W = 256\cdot24\cdot24 = 147{,}456\). Training applies four steps:`,
      } },
      { eq: R`\mu = \frac{1}{m}\sum_{i=1}^{m} x_i \qquad \sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu)^2 \qquad \hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}} \qquad y_i = \gamma\,\hat{x}_i + \beta` },
      { p: {
        tr: R`Aşağıda dört sayıyla küçük bir örnek var. Değerleri değiştirip her ara sonucu görün: normalleştirilmiş değerlerin ortalaması her zaman 0, karelerinin ortalaması (\(\varepsilon\) yüzünden neredeyse) 1 çıkar.`,
        en: R`Below is a small example with four numbers. Change them and watch every intermediate result: the normalized values always have mean 0 and a mean square of (almost, because of \(\varepsilon\)) 1.`,
      } },
      { widget: "bn" },
      { h: { tr: "Çıkarımda: sabit bir doğrusal dönüşüm", en: "At inference: a fixed linear map" } },
      { p: {
        tr: R`Eğitimin her adımında [[moving-average|hareketli ortalamalar]] güncellenir (\(\alpha = 0{,}1\)); varyans için yansız tahmin \(\frac{m}{m-1}\sigma^2\) kullanılır:`,
        en: R`At every training step the [[moving-average|running averages]] are updated (\(\alpha = 0.1\)); for the variance the unbiased estimate \(\frac{m}{m-1}\sigma^2\) is used:`,
      } },
      { eq: R`\hat\mu \leftarrow (1-\alpha)\,\hat\mu + \alpha\,\mu \qquad \hat\sigma^2 \leftarrow (1-\alpha)\,\hat\sigma^2 + \alpha\,\frac{m}{m-1}\,\sigma^2` },
      { p: {
        tr: R`Çıkarımda bu sabit değerler kullanıldığından BatchNorm her kanalda yalnızca bir ölçekleme ve kaydırmadır:`,
        en: R`At inference these fixed values are used, so BatchNorm is just a scale and a shift per channel:`,
      } },
      { eq: R`y = a\,x + c \qquad a = \frac{\gamma}{\sqrt{\hat\sigma^2+\varepsilon}} \qquad c = \beta - a\,\hat\mu` },
      { p: {
        tr: R`Bu yüzden çıkarımda BatchNorm, önündeki evrişimin ağırlıklarına katlanarak (\(w' = a\,w\), \(b' = a\,b + c\)) tamamen ortadan kaldırılabilir; modelleri hızlandırmak için sık yapılan bir işlemdir.`,
        en: R`So at inference BatchNorm can be folded into the weights of the convolution before it (\(w' = a\,w\), \(b' = a\,b + c\)) and disappear completely; a common trick to speed models up.`,
      } },
      { h: { tr: "Geri yayılım", en: "Backward pass" } },
      { p: {
        tr: R`BatchNorm'un türevi ilginçtir, çünkü bir değer \(x_i\) çıktıyı üç yoldan etkiler: kendi \(\hat x_i\)'si üzerinden ve bütün değerlerin paylaştığı \(\mu\) ile \(\sigma^2\) üzerinden. Kısaltmalar: \(s = \sqrt{\sigma^2+\varepsilon}\), \(\delta_i = \partial L/\partial y_i\).`,
        en: R`BatchNorm's derivative is interesting because a value \(x_i\) affects the output in three ways: through its own \(\hat x_i\), and through \(\mu\) and \(\sigma^2\), which all values share. Shorthands: \(s = \sqrt{\sigma^2+\varepsilon}\), \(\delta_i = \partial L/\partial y_i\).`,
      } },
      { derive: { title: { tr: "BatchNorm'un gradyanları", en: "BatchNorm's gradients" }, steps: [
        { tex: R`\frac{\partial L}{\partial \beta} = \sum_i \delta_i \qquad \frac{\partial L}{\partial \gamma} = \sum_i \delta_i\,\hat{x}_i`, why: {
          tr: R`\(y_i = \gamma\hat x_i + \beta\) olduğundan \(\partial y_i/\partial\beta = 1\) ve \(\partial y_i/\partial\gamma = \hat x_i\); bütün \(i\)'ler toplanır.`,
          en: R`Since \(y_i = \gamma\hat x_i + \beta\): \(\partial y_i/\partial\beta = 1\) and \(\partial y_i/\partial\gamma = \hat x_i\); summed over all \(i\).` } },
        { tex: R`\frac{\partial L}{\partial \hat{x}_i} = \gamma\,\delta_i`, why: {
          tr: "Normalleştirilmiş değere kadar geri gelmek, γ ile çarpmaktır.",
          en: "Going back to the normalized value means multiplying by γ." } },
        { tex: R`\frac{\partial L}{\partial \sigma^2} = \sum_i \gamma\,\delta_i\,(x_i-\mu)\left(-\tfrac{1}{2}\right)s^{-3} = -\frac{\gamma}{2s^2}\sum_i \delta_i\,\hat{x}_i`, why: {
          tr: R`\(\hat x_i = (x_i-\mu)\,(\sigma^2+\varepsilon)^{-1/2}\)'nin \(\sigma^2\)'ye göre türevi; son eşitlikte \((x_i-\mu)/s = \hat x_i\) kullanıldı.`,
          en: R`The derivative of \(\hat x_i = (x_i-\mu)\,(\sigma^2+\varepsilon)^{-1/2}\) with respect to \(\sigma^2\); the last equality uses \((x_i-\mu)/s = \hat x_i\).` } },
        { tex: R`\frac{\partial L}{\partial \mu} = -\frac{\gamma}{s}\sum_i \delta_i`, why: {
          tr: R`\(\mu\), her \(\hat x_i\)'de \(-1/s\) çarpanıyla geçer. \(\sigma^2\) üzerinden gelen katkı sıfırdır, çünkü \(\sum_i (x_i-\mu) = 0\).`,
          en: R`\(\mu\) appears in every \(\hat x_i\) with the factor \(-1/s\). The contribution through \(\sigma^2\) is zero, because \(\sum_i (x_i-\mu) = 0\).` } },
        { tex: R`\frac{\partial L}{\partial x_i} = \frac{\gamma\,\delta_i}{s} + \frac{\partial L}{\partial \sigma^2}\,\frac{2(x_i-\mu)}{m} + \frac{\partial L}{\partial \mu}\,\frac{1}{m}`, why: {
          tr: R`Üç yolun toplamı: doğrudan, varyans üzerinden (\(\partial\sigma^2/\partial x_i = 2(x_i-\mu)/m\)) ve ortalama üzerinden (\(\partial\mu/\partial x_i = 1/m\)).`,
          en: R`The sum of the three paths: direct, through the variance (\(\partial\sigma^2/\partial x_i = 2(x_i-\mu)/m\)) and through the mean (\(\partial\mu/\partial x_i = 1/m\)).` } },
        { tex: R`\frac{\partial L}{\partial x_i} = \frac{\gamma}{m\,s}\left(m\,\delta_i - \sum_j \delta_j - \hat{x}_i \sum_j \delta_j\,\hat{x}_j\right)`, why: {
          tr: R`3. ve 4. adımı yerine koyup sadeleştirince. Yorum: gelen gradyandan ortalaması ve \(\hat x\) yönündeki bileşeni çıkarılır; bu yüzden \(\sum_i \partial L/\partial x_i = 0\). Kanalın bütün değerlerini birlikte kaydırmak normalleştirmenin sonucunu değiştirmez, gradyan da bu yönde sıfırdır.`,
          en: R`Substituting steps 3 and 4 and simplifying. Reading: the mean of the incoming gradient and its component along \(\hat x\) are removed, so \(\sum_i \partial L/\partial x_i = 0\). Shifting all values of a channel together does not change the normalized result, so the gradient in that direction is zero.` } },
      ] } },
      { p: {
        tr: R`Son satırın bir sonucu: BatchNorm'dan hemen önceki evrişimin bias'ı \(b_o\), kanalın bütün değerlerine aynı miktarda eklenir, yani gradyanı \(\partial L/\partial b_o = \sum_i \partial L/\partial x_i = 0\)'dır (yuvarlama hataları dışında). Ortalama çıkarılınca bias zaten yok olur; bu yüzden BatchNorm'dan önceki evrişimlerde genellikle <code>bias=False</code> yazılır. SimpleCNN'de Conv1 ve Conv2'nin 32 + 64 = 96 bias'ı bu nedenle gereksizdir (bu sekmenin 1. kısmındaki tabloda sayıldılar).`,
        en: R`A consequence of the last line: the bias \(b_o\) of the convolution right before BatchNorm is added equally to every value of the channel, so its gradient is \(\partial L/\partial b_o = \sum_i \partial L/\partial x_i = 0\) (up to rounding errors). Subtracting the mean removes the bias anyway; that is why convolutions before BatchNorm are usually written with <code>bias=False</code>. In SimpleCNN the 32 + 64 = 96 biases of Conv1 and Conv2 are therefore redundant (the table in section 1 counts them).`,
      } },
      { links: [["theory", "t-bn"]] },
    ],
  },

  // ------------------------------------------------------------------ 4
  {
    id: "m-act",
    title: { tr: "Aktivasyon fonksiyonları ve türevleri", en: "Activation functions and their derivatives" },
    blocks: [
      { p: {
        tr: R`Bu bölüm büyük ölçüde <a href="#" data-go="math:mm-act" data-chapter="mlp">MLP bölümünün</a> kısa bir hatırlatmasıdır. [[activation|Aktivasyon]] her değere ayrı uygulanır: \(a_i = f(z_i)\). Bu yüzden geri yayılımı da eleman eleman ve çok basittir; MLP bölümündeki (2) denkleminin \(\odot f'(\mathbf z^{(l)})\) çarpanı tam olarak budur:`,
        en: R`This section is mostly a short recap of <a href="#" data-go="math:mm-act" data-chapter="mlp">chapter 2</a>. An [[activation|activation]] is applied to every value separately: \(a_i = f(z_i)\). Its backward pass is therefore element-wise and very simple; it is exactly the factor \(\odot f'(\mathbf z^{(l)})\) in chapter 2's equation (2):`,
      } },
      { eq: R`\frac{\partial L}{\partial z_i} = f'(z_i)\;\frac{\partial L}{\partial a_i} \qquad\Longleftrightarrow\qquad \delta_z = f'(z)\odot\delta_a` },
      { p: {
        tr: R`Yani bir aktivasyonun eğitime etkisini <b>türevi</b> belirler. Üç fonksiyonu ve türevlerini karşılaştırın; ℓ kaydırıcısıyla katman sayısını artırıp sigmoid'le ReLU arasında gradyanın kaç kat küçüldüğüne bakın:`,
        en: R`So an activation's effect on training is set by its <b>derivative</b>. Compare three functions and their derivatives; raise the number of layers with the ℓ slider and see how many times smaller the gradient gets with the sigmoid compared with ReLU:`,
      } },
      { widget: "act" },
      { h: { tr: "ReLU", en: "ReLU" } },
      { eq: R`f(z) = \max(0,z) \qquad f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}` },
      { p: {
        tr: R`[[relu|ReLU]] geri yayılımda bir kapıdır: pozitif girişlerde gradyanı olduğu gibi geçirir, negatiflerde durdurur (\(z = 0\)'da türev tanımsızdır, PyTorch 0 alır). Hiçbir girişte pozitif olmayan bir nöron hiç gradyan almaz ve bir daha öğrenemez; buna "ölü ReLU" denir.`,
        en: R`[[relu|ReLU]] is a gate in the backward pass: it lets the gradient through unchanged for positive inputs and blocks it for negative ones (the derivative is undefined at \(z = 0\); PyTorch uses 0). A neuron that is never positive gets no gradient and can never learn again: a "dead ReLU".`,
      } },
      { h: { tr: "Sigmoid ve kaybolan gradyan", en: "Sigmoid and the vanishing gradient" } },
      { derive: { title: { tr: "Hatırlatma: sigmoid'in türevi (MLP bölümü)", en: "Recap: the derivative of the sigmoid (chapter 2)" }, steps: [
        { tex: R`\sigma(z) = \frac{1}{1+e^{-z}} = \left(1+e^{-z}\right)^{-1}`, why: { tr: "Tanım.", en: "The definition." } },
        { tex: R`\sigma'(z) = -\left(1+e^{-z}\right)^{-2}\cdot\left(-e^{-z}\right) = \frac{e^{-z}}{\left(1+e^{-z}\right)^2}`, why: {
          tr: R`Zincir kuralı: dış fonksiyon \(u^{-1}\), iç fonksiyon \(u = 1+e^{-z}\).`,
          en: R`The chain rule: outer function \(u^{-1}\), inner function \(u = 1+e^{-z}\).` } },
        { tex: R`\sigma'(z) = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)\,\bigl(1-\sigma(z)\bigr)`, why: {
          tr: R`Çünkü \(1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}}\). Türev, fonksiyonun kendi değerinden hesaplanır; ileri yayılımda bulunan \(\sigma(z)\) saklanıp geri yayılımda kullanılır.`,
          en: R`Because \(1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}}\). The derivative is computed from the function's own value; the \(\sigma(z)\) found in the forward pass is kept for the backward pass.` } },
        { tex: R`\max_z \sigma'(z) = \sigma'(0) = \tfrac{1}{2}\cdot\tfrac{1}{2} = \tfrac{1}{4}`, why: {
          tr: R`\(u(1-u)\) en büyük değerini \(u = 1/2\)'de alır.`,
          en: R`\(u(1-u)\) is largest at \(u = 1/2\).` } },
      ] } },
      { p: {
        tr: R`Gradyan \(\ell\) katman geriye giderken her katmanda bir \(f'\) ile çarpılır. Sigmoid'de bu çarpanların her biri en fazla \(1/4\) olduğu için (ağırlıkların etkisi bir yana) gradyan \((1/4)^\ell\) hızıyla küçülür: 10 katmanda yaklaşık milyonda bire iner ([[vanishing-gradient|kaybolan gradyan]]). ReLU'nun pozitif taraftaki türevi 1 olduğu için bu sorun yaşanmaz; derin ağlarda ReLU'nun tercih edilmesinin ana nedeni budur. Tanh, sigmoid'in ölçeklenmiş hâlidir: \(\tanh(z) = 2\sigma(2z)-1\), \(\tanh'(z) = 1-\tanh^2(z)\). Türevi en fazla 1'dir ama \(|z|\) büyüdükçe o da sıfıra yaklaşır (doyma).`,
        en: R`Going back through \(\ell\) layers, the gradient is multiplied by an \(f'\) in every layer. For the sigmoid each factor is at most \(1/4\), so (leaving the weights aside) the gradient shrinks like \((1/4)^\ell\): to about one millionth after 10 layers (the [[vanishing-gradient|vanishing gradient]]). ReLU's derivative is 1 on the positive side, so it does not have this problem; that is the main reason deep networks prefer it. Tanh is a rescaled sigmoid: \(\tanh(z) = 2\sigma(2z)-1\), \(\tanh'(z) = 1-\tanh^2(z)\). Its derivative is at most 1, but it also approaches zero as \(|z|\) grows (saturation).`,
      } },
      { links: [["theory", "t-act"], ["math", "mm-act", "mlp"]] },
    ],
  },

  // ------------------------------------------------------------------ 5
  {
    id: "m-pool",
    title: { tr: "Havuzlama", en: "Pooling" },
    blocks: [
      { p: {
        tr: R`2×2 [[pooling|max pooling]], stride 2: her pencereden en büyük değer seçilir. Öğrenilecek parametre yoktur.`,
        en: R`2×2 [[pooling|max pooling]] with stride 2: the largest value of each window is kept. There is nothing to learn.`,
      } },
      { eq: R`y_{i,j} = \max_{m,n\,\in\,\{0,1\}} x_{2i+m,\;2j+n}` },
      { h: { tr: "Geri yayılım: gradyan kazanana gider", en: "Backward pass: the gradient goes to the winner" } },
      { p: {
        tr: R`Bir penceredeki dört değerden yalnızca en büyüğü çıktıyı etkiler; diğerleri biraz değişse çıktı değişmez. Türev bu yüzden bir göstergedir:`,
        en: R`Only the largest of the four values in a window affects the output; changing the others a little does not change it. The derivative is therefore an indicator:`,
      } },
      { eq: R`\frac{\partial L}{\partial x_{u,v}} = \delta_{i,j}\cdot\mathbb{1}\!\left[(u,v) = \arg\max_{(i,j)\text{ penceresi}} x\right]`, eqEn: R`\frac{\partial L}{\partial x_{u,v}} = \delta_{i,j}\cdot\mathbb{1}\!\left[(u,v) = \arg\max_{\text{window }(i,j)} x\right]` },
      { p: {
        tr: R`PyTorch ileri yayılımda kazananın konumunu saklar ve geri yayılımda gradyanı oraya yazar; diğer üç konum sıfır alır. Average pooling'de ise gradyan dört konuma eşit paylaştırılır: \(\delta_{i,j}/4\). Aşağıda deneyin: max ile average arasında geçiş yapın ve <b>Yeni sayılar</b>'a basın. Max pooling'de 16 girdi konumundan kaçı gradyan alıyor? Average pooling'de kaçı?`,
        en: R`PyTorch remembers the winner's position in the forward pass and writes the gradient there in the backward pass; the other three positions get zero. With average pooling the gradient is split equally over the four positions: \(\delta_{i,j}/4\). Try it below: switch between max and average and press <b>New numbers</b>. With max pooling, how many of the 16 input positions get a gradient? With average pooling?`,
      } },
      { widget: "pool" },
      { h: { tr: "Algı alanı nasıl büyür?", en: "How the receptive field grows" } },
      { p: {
        tr: R`Bir katmandaki tek bir değerin giriş görüntüsünde gördüğü bölgenin ([[receptive-field|algı alanı]]) kenarı \(r\), her katmanda şöyle büyür; \(j\), o katmana kadarki stride'ların çarpımıdır:`,
        en: R`The side \(r\) of the region a single value sees in the input image (its [[receptive-field|receptive field]]) grows like this in every layer; \(j\) is the product of the strides so far:`,
      } },
      { eq: R`r_\ell = r_{\ell-1} + (k_\ell - 1)\,j_{\ell-1} \qquad j_\ell = j_{\ell-1}\,s_\ell \qquad r_0 = j_0 = 1` },
      { table: {
        head: [{ tr: "Katman", en: "Layer" }, "k", "s", "j", "r"],
        rows: [
          [{ tr: "Giriş", en: "Input" }, "—", "—", "1", "1"],
          ["Conv1", "5", "1", "1", "5"],
          ["MaxPool1", "2", "2", "2", "6"],
          ["Conv2", "5", "1", "2", "14"],
          ["MaxPool2", "2", "2", "4", "16"],
        ],
      } },
      { p: {
        tr: R`Son havuzlamadan sonra her değer görüntünün 16×16 pikselini görür: 28×28'lik bir rakamın büyük bölümünü. Havuzlama stride'ı ikiye katladığı için sonraki evrişimin her adımı iki kat geniş bir alanı kapsar.`,
        en: R`After the last pooling every value sees 16×16 pixels of the image: most of a 28×28 digit. Because pooling doubles the stride, every step of the next convolution covers twice as wide an area.`,
      } },
      { links: [["theory", "t-pool"]] },
    ],
  },

  // ------------------------------------------------------------------ 6
  {
    id: "m-drop",
    title: { tr: "Dropout", en: "Dropout" },
    blocks: [
      { eq: R`r_i \sim \operatorname{Bernoulli}(1-p) \qquad \tilde{h}_i = \frac{r_i\,h_i}{1-p}` },
      { p: {
        tr: R`Eğitimde her değer \(p\) olasılıkla sıfırlanır (\(r_i = 0\)), kalanlar \(1/(1-p)\) ile büyütülür. Çıkarımda \(\tilde h = h\): hiçbir şey yapılmaz. Bu büyütmenin nedeni beklenen değerdir:`,
        en: R`During training every value is zeroed with probability \(p\) (\(r_i = 0\)) and the rest are scaled up by \(1/(1-p)\). At inference \(\tilde h = h\): nothing happens. The reason for the scaling is the expected value:`,
      } },
      { derive: { title: { tr: "Beklenen değer ve varyans", en: "Expected value and variance" }, steps: [
        { tex: R`\mathbb{E}[\tilde h_i] = \mathbb{E}[r_i]\,\frac{h_i}{1-p} = (1-p)\,\frac{h_i}{1-p} = h_i`, why: {
          tr: R`\(\mathbb{E}[r_i] = P(r_i = 1) = 1-p\). Ortalamada çıktı Dropout yokmuş gibidir; bu yüzden çıkarımda ölçekleme gerekmez.`,
          en: R`\(\mathbb{E}[r_i] = P(r_i = 1) = 1-p\). On average the output is as if there were no Dropout, which is why inference needs no scaling.` } },
        { tex: R`\mathbb{E}[\tilde h_i^2] = (1-p)\,\frac{h_i^2}{(1-p)^2} = \frac{h_i^2}{1-p}`, why: {
          tr: R`\(r_i\) yalnızca 0 ya da 1 olduğu için \(r_i^2 = r_i\).`,
          en: R`\(r_i\) is only ever 0 or 1, so \(r_i^2 = r_i\).` } },
        { tex: R`\operatorname{Var}[\tilde h_i] = \frac{h_i^2}{1-p} - h_i^2 = \frac{p}{1-p}\,h_i^2`, why: {
          tr: R`Dropout'un eklediği gürültü \(p\) ile hızla artar: \(p = 0{,}5\)'te standart sapma \(|h_i|\) kadardır.`,
          en: R`The noise Dropout adds grows quickly with \(p\): at \(p = 0.5\) the standard deviation equals \(|h_i|\).` } },
      ] } },
      { p: {
        tr: R`Geri yayılımda aynı maske kullanılır: sıfırlanan değerler gradyan almaz, kalanların gradyanı da aynı oranda büyütülür:`,
        en: R`The backward pass uses the same mask: dropped values get no gradient, and the gradient of the others is scaled up by the same factor:`,
      } },
      { eq: R`\frac{\partial L}{\partial h_i} = \frac{r_i}{1-p}\,\frac{\partial L}{\partial \tilde h_i}` },
      { p: {
        tr: R`Aşağıda \(p\)'yi değiştirin ve <b>Yeni maske</b>'ye birkaç kez basın: tek bir maskenin çıktısı her seferinde değişir, ama 10.000 maskenin ortalaması \(h\)'de kalır. \(p\)'yi 0,9'a doğru götürünce ölçülen varyansın \(\frac{p}{1-p}h^2\) formülüne uyarak ne kadar hızlı büyüdüğüne bakın.`,
        en: R`Below, change \(p\) and press <b>New mask</b> a few times: the output of a single mask changes every time, but the mean of 10,000 masks stays at \(h\). Move \(p\) towards 0.9 and watch how fast the measured variance grows, following the formula \(\frac{p}{1-p}h^2\).`,
      } },
      { widget: "drop" },
      { model: {
        tr: R`<b>SimpleCNN'de:</b> \(p = 0{,}45\) → kalanlar \(1/0{,}55 \approx 1{,}82\) ile; \(p = 0{,}35\) → \(1/0{,}65 \approx 1{,}54\) ile büyütülür.`,
        en: R`<b>In SimpleCNN:</b> \(p = 0.45\) → the remaining values are scaled by \(1/0.55 \approx 1.82\); \(p = 0.35\) → by \(1/0.65 \approx 1.54\).`,
      } },
      { links: [["theory", "t-drop"]] },
    ],
  },

  // ------------------------------------------------------------------ 7
  {
    id: "m-fc",
    title: { tr: "Düzleştirme ve tam bağlı katman", en: "Flattening and the fully connected layer" },
    blocks: [
      { p: {
        tr: R`Son havuzlamanın çıktısı \(64\times4\times4\)'lük bir tensördür. <code>torch.flatten</code> onu satır satır okuyarak 1024 uzunluğunda bir vektöre çevirir ([[flatten|düzleştirme]]):`,
        en: R`The output of the last pooling is a \(64\times4\times4\) tensor. <code>torch.flatten</code> reads it row by row into a vector of length 1024 ([[flatten|flattening]]):`,
      } },
      { eq: R`v_{16c+4i+j} = x_{c,i,j} \qquad c\in\{0,\dots,63\},\quad i,j\in\{0,\dots,3\}` },
      { p: {
        tr: R`[[fully-connected|Tam bağlı katman]] bu vektörü bir matrisle çarpıp bias ekler. FC1'de \(\mathbf{W}\in\mathbb{R}^{512\times1024}\):`,
        en: R`The [[fully-connected|fully connected layer]] multiplies this vector by a matrix and adds a bias. In FC1, \(\mathbf{W}\in\mathbb{R}^{512\times1024}\):`,
      } },
      { eq: R`\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b} \qquad y_j = \sum_{i} W_{j i}\, x_i + b_j` },
      { p: {
        tr: R`Bu, <a href="#" data-go="math:mm-forward" data-chapter="mlp">MLP bölümündeki</a> \(\mathbf z^{(l)} = \mathbf W^{(l)}\mathbf a^{(l-1)} + \mathbf b^{(l)}\) satırının ta kendisidir: FC1 ve FC2, SimpleCNN'in sonunda 1024 → 512 → 10 bir MLP oluşturur. Aşağıdaki gradyanlar da MLP bölümünün (3) ve (4) denklemleri ile (2)'nin \(\mathbf W^\top\boldsymbol\delta\) kısmıdır; burada kısaca tekrar ediyoruz.`,
        en: R`This is exactly the line \(\mathbf z^{(l)} = \mathbf W^{(l)}\mathbf a^{(l-1)} + \mathbf b^{(l)}\) of <a href="#" data-go="math:mm-forward" data-chapter="mlp">chapter 2</a>: FC1 and FC2 form a 1024 → 512 → 10 MLP at the end of SimpleCNN. The gradients below are chapter 2's equations (3) and (4) and the \(\mathbf W^\top\boldsymbol\delta\) part of (2); we repeat them briefly here.`,
      } },
      { h: { tr: "Geri yayılım", en: "Backward pass" } },
      { derive: { title: { tr: "Matris, bias ve girdi gradyanı", en: "Matrix, bias and input gradients" }, steps: [
        { tex: R`\frac{\partial y_j}{\partial W_{ji}} = x_i \;\;\Rightarrow\;\; \frac{\partial L}{\partial W_{ji}} = \delta_j\, x_i`, why: {
          tr: R`\(W_{ji}\) yalnızca \(y_j\)'de ve \(x_i\) çarpanıyla geçer.`,
          en: R`\(W_{ji}\) appears only in \(y_j\), multiplied by \(x_i\).` } },
        { tex: R`\frac{\partial L}{\partial \mathbf{W}} = \boldsymbol{\delta}\,\mathbf{x}^{\top}`, why: {
          tr: R`Bütün \((j,i)\)'ler birlikte bir dış çarpımdır: \(512\times1\) çarpı \(1\times1024\) = \(512\times1024\), yani \(\mathbf{W}\) ile aynı şekil. Gradyan her zaman parametreyle aynı şekildedir; hesabı kontrol etmenin kolay bir yolu.`,
          en: R`All \((j,i)\) together form an outer product: \(512\times1\) times \(1\times1024\) = \(512\times1024\), the shape of \(\mathbf{W}\). A gradient always has the shape of its parameter; an easy way to check a calculation.` } },
        { tex: R`\frac{\partial L}{\partial \mathbf{b}} = \boldsymbol{\delta}`, why: { tr: R`Çünkü \(\partial y_j/\partial b_j = 1\).`, en: R`Because \(\partial y_j/\partial b_j = 1\).` } },
        { tex: R`\frac{\partial L}{\partial x_i} = \sum_j W_{ji}\,\delta_j \;\;\Rightarrow\;\; \frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^{\top}\boldsymbol{\delta}`, why: {
          tr: R`\(x_i\) bütün \(y_j\)'lere katkı verir ve katkılar toplanır. İleri yayılımda \(\mathbf{W}\), geri yayılımda transpozu \(\mathbf{W}^\top\) kullanılır.`,
          en: R`\(x_i\) contributes to every \(y_j\), and the contributions add up. The forward pass uses \(\mathbf{W}\), the backward pass its transpose \(\mathbf{W}^\top\).` } },
        { tex: R`\frac{\partial L}{\partial \mathbf{W}} = \sum_{n=1}^{N} \boldsymbol{\delta}^{(n)}\,\mathbf{x}^{(n)\top} = \Delta^{\top} X`, why: {
          tr: R`Bir batch'te her görüntünün dış çarpımı toplanır. Satırları \(\mathbf{x}^{(n)}\) olan \(X\in\mathbb{R}^{N\times1024}\) ve satırları \(\boldsymbol\delta^{(n)}\) olan \(\Delta\in\mathbb{R}^{N\times512}\) ile bu tek bir matris çarpımıdır; [[gpu|GPU]]'lar tam olarak bu işlemde hızlıdır.`,
          en: R`In a batch the outer products of all images add up. With \(X\in\mathbb{R}^{N\times1024}\) (rows \(\mathbf{x}^{(n)}\)) and \(\Delta\in\mathbb{R}^{N\times512}\) (rows \(\boldsymbol\delta^{(n)}\)) this is a single matrix product, exactly the operation [[gpu|GPUs]] are fast at.` } },
      ] } },
      { h: { tr: "Sayılarla küçük bir örnek", en: "A small example with numbers" } },
      { eq: {
        tr: R`\mathbf{W} = \begin{pmatrix} 1 & -1 & 2 \\ 0 & 3 & 1 \end{pmatrix},\;\; \mathbf{x} = \begin{pmatrix} 2 \\ 1 \\ -1 \end{pmatrix},\;\; \mathbf{b} = \begin{pmatrix} 0{,}5 \\ -1 \end{pmatrix} \;\;\Rightarrow\;\; \mathbf{y} = \begin{pmatrix} 2 - 1 - 2 + 0{,}5 \\ 0 + 3 - 1 - 1 \end{pmatrix} = \begin{pmatrix} -0{,}5 \\ 1 \end{pmatrix}`,
        en: R`\mathbf{W} = \begin{pmatrix} 1 & -1 & 2 \\ 0 & 3 & 1 \end{pmatrix},\;\; \mathbf{x} = \begin{pmatrix} 2 \\ 1 \\ -1 \end{pmatrix},\;\; \mathbf{b} = \begin{pmatrix} 0.5 \\ -1 \end{pmatrix} \;\;\Rightarrow\;\; \mathbf{y} = \begin{pmatrix} 2 - 1 - 2 + 0.5 \\ 0 + 3 - 1 - 1 \end{pmatrix} = \begin{pmatrix} -0.5 \\ 1 \end{pmatrix}`,
      } },
      { p: { tr: R`Çıktıdan gelen gradyan \(\boldsymbol\delta\) ise:`, en: R`If the gradient coming from the output is \(\boldsymbol\delta\):` } },
      { eq: R`\boldsymbol{\delta} = \begin{pmatrix} 1 \\ -2 \end{pmatrix} \;\;\Rightarrow\;\; \frac{\partial L}{\partial \mathbf{W}} = \boldsymbol{\delta}\,\mathbf{x}^{\top} = \begin{pmatrix} 2 & 1 & -1 \\ -4 & -2 & 2 \end{pmatrix},\qquad \frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^{\top}\boldsymbol{\delta} = \begin{pmatrix} 1 \\ -7 \\ 0 \end{pmatrix}` },
      { links: [["theory", "t-softmax"], ["code", "c-forward"], ["math", "mm-backprop", "mlp"]] },
    ],
  },

  // ------------------------------------------------------------------ 8
  {
    id: "m-softmax",
    title: { tr: "Softmax ve çapraz entropi", en: "Softmax and cross-entropy" },
    blocks: [
      { eq: R`p_k = \operatorname{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=0}^{K-1} e^{z_j}} \qquad L = -\sum_{k=0}^{K-1} y_k \log p_k = -\log p_c` },
      { p: {
        tr: R`\(K = 10\): sınıf sayısı (sınıflar 0–9); \(\mathbf{z}\): FC2'nin 10 [[logit|logiti]]; \(c\): doğru sınıf; \(\mathbf{y}\): doğru sınıfta 1, diğerlerinde 0 olan "one-hot" vektör; \(\log\): MLP bölümündeki \(\ln\), yani doğal logaritma. Bütün logitler eşitken \(p_k = 1/10\) ve \(L = \log 10 \approx 2{,}303\). Eğitimin başında logitler küçük ve birbirine yakın olduğu için kayıp bu değere yakın çıkar; çok farklıysa (örneğin 5) başlangıçta bir şeyler yanlıştır.`,
        en: R`\(K = 10\): the number of classes (classes 0–9); \(\mathbf{z}\): the 10 [[logit|logits]] of FC2; \(c\): the correct class; \(\mathbf{y}\): the "one-hot" vector, 1 at the correct class and 0 elsewhere; \(\log\): chapter 2's \(\ln\), the natural logarithm. With all logits equal, \(p_k = 1/10\) and \(L = \log 10 \approx 2.303\). At the start of training the logits are small and close to each other, so the loss comes out close to this value; if it is very different (say 5), something is wrong with the setup.`,
      } },
      { callout: "info", title: { tr: "MLP bölümüyle bağlantı: iki sınıfta softmax = sigmoid", en: "The link to chapter 2: with two classes, softmax = sigmoid" }, text: {
        tr: R`\(K = 2\) iken \(p_1 = \dfrac{e^{z_1}}{e^{z_0}+e^{z_1}} = \dfrac{1}{1+e^{-(z_1-z_0)}} = \sigma(z_1 - z_0)\): softmax sigmoid'e, çok sınıflı çapraz entropi de <a href="#" data-go="math:mm-loss" data-chapter="mlp">MLP bölümündeki</a> ikili çapraz entropiye iner. Aşağıda bulacağımız \(p_k - y_k\) gradyanı, oradaki \(\hat y - y\)'nin 10 sınıflı hâlidir.`,
        en: R`With \(K = 2\), \(p_1 = \dfrac{e^{z_1}}{e^{z_0}+e^{z_1}} = \dfrac{1}{1+e^{-(z_1-z_0)}} = \sigma(z_1 - z_0)\): softmax reduces to the sigmoid, and the multiclass cross-entropy to the binary cross-entropy of <a href="#" data-go="math:mm-loss" data-chapter="mlp">chapter 2</a>. The gradient \(p_k - y_k\) we find below is the 10-class version of \(\hat y - y\) there.`,
      } },
      { h: { tr: "Sayısal kararlılık", en: "Numerical stability" } },
      { p: {
        tr: R`\(e^{z}\), \(z = 1000\) gibi değerlerde bilgisayarın sayı aralığını aşar. Neyse ki softmax, bütün logitlerden aynı sabit çıkarılınca değişmez:`,
        en: R`\(e^{z}\) overflows the computer's number range for values such as \(z = 1000\). Fortunately softmax does not change when the same constant is subtracted from all logits:`,
      } },
      { eq: R`\frac{e^{z_k - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c}\,e^{z_k}}{e^{-c}\sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}` },
      { p: {
        tr: R`Bu yüzden hesapta \(c = \max_j z_j\) çıkarılır: en büyük üs 0 olur ve taşma imkânsızlaşır. PyTorch kaybı doğrudan şu "log-sum-exp" biçiminde hesaplar; modelde ayrı bir softmax katmanı olmamasının nedeni budur:`,
        en: R`So the computation subtracts \(c = \max_j z_j\): the largest exponent becomes 0 and overflow is impossible. PyTorch computes the loss directly in this "log-sum-exp" form, which is why the model has no separate softmax layer:`,
      } },
      { eq: R`L = -z_c + \log\sum_{j} e^{z_j}` },
      { h: { tr: "Gradyan: tahmin eksi hedef", en: "The gradient: prediction minus target" } },
      { derive: { title: { tr: "Çapraz entropinin logitlere göre türevi", en: "The derivative of cross-entropy with respect to the logits" }, steps: [
        { tex: R`L = -z_c + \log\sum_j e^{z_j}`, why: {
          tr: R`Log-sum-exp biçimi: \(-\log p_c = -\log e^{z_c} + \log\sum_j e^{z_j}\).`,
          en: R`The log-sum-exp form: \(-\log p_c = -\log e^{z_c} + \log\sum_j e^{z_j}\).` } },
        { tex: R`\frac{\partial}{\partial z_k}\,\log\sum_j e^{z_j} = \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k`, why: {
          tr: R`\(\log u\)'nun türevi \(u'/u\)'dur; toplamda \(z_k\)'ye bağlı tek terim \(e^{z_k}\)'dir.`,
          en: R`The derivative of \(\log u\) is \(u'/u\); the only term of the sum that depends on \(z_k\) is \(e^{z_k}\).` } },
        { tex: R`\frac{\partial(-z_c)}{\partial z_k} = -\mathbb{1}[k = c] = -y_k`, why: { tr: "Yalnızca doğru sınıfın logiti için −1.", en: "−1 only for the logit of the correct class." } },
        { tex: R`\frac{\partial L}{\partial z_k} = p_k - y_k`, why: {
          tr: R`İki parçanın toplamı. Gradyan <b>tahmin edilen olasılık eksi hedeftir</b>: doğru sınıf için \(p_c - 1 \le 0\) (logit büyütülür), diğerleri için \(p_k \ge 0\) (küçültülür). Değerler hep \([-1, 1]\) aralığındadır; bu da eğitimi kararlı yapar.`,
          en: R`The sum of the two parts. The gradient is the <b>predicted probability minus the target</b>: \(p_c - 1 \le 0\) for the correct class (its logit is pushed up), \(p_k \ge 0\) for the others (pushed down). The values always lie in \([-1, 1]\), which keeps training stable.` } },
        { tex: R`\frac{\partial \bar L}{\partial \mathbf{z}^{(n)}} = \frac{1}{N}\left(\mathbf{p}^{(n)} - \mathbf{y}^{(n)}\right)`, why: {
          tr: R`Batch kaybı ortalama olduğu için, \(\bar L = \frac1N\sum_n L^{(n)}\), her görüntünün gradyanı \(1/N\) ile ölçeklenir.`,
          en: R`The batch loss is a mean, \(\bar L = \frac1N\sum_n L^{(n)}\), so every image's gradient is scaled by \(1/N\).` } },
      ] } },
      { p: {
        tr: R`Aşağıda logitleri ve doğru sınıfı değiştirin. <b>Analitik</b> sütun \(p_k - y_k\) formülüdür; yanında, kaybı her logit için çok az değiştirerek ölçülen <b>sayısal gradyan</b> var. İkisinin aynı çıkması türetmenin doğru olduğunu gösterir (gradyan kontrolü).`,
        en: R`Change the logits and the correct class below. The <b>analytic</b> column is the formula \(p_k - y_k\); next to it is the <b>numerical gradient</b>, measured by changing the loss very slightly for each logit. The two agreeing shows the derivation is right (a gradient check).`,
      } },
      { widget: "softmax" },
      { callout: "info", title: { tr: "Softmax'ın Jacobian'ı", en: "The softmax Jacobian" }, text: {
        tr: R`Softmax tek başına türevlenirse \(\dfrac{\partial p_k}{\partial z_j} = p_k\,\bigl(\mathbb{1}[k=j] - p_j\bigr)\), yani \(K\times K\) bir matris çıkar. Çapraz entropiyle birleştirince bu matris \(\mathbf{p} - \mathbf{y}\)'ye sadeleşir; kütüphanelerin ikisini tek bir işlem olarak uygulamasının bir nedeni de budur.`,
        en: R`Differentiating softmax on its own gives \(\dfrac{\partial p_k}{\partial z_j} = p_k\,\bigl(\mathbb{1}[k=j] - p_j\bigr)\), a \(K\times K\) matrix. Combined with cross-entropy it simplifies to \(\mathbf{p} - \mathbf{y}\); one reason libraries implement the two as a single operation.`,
      } },
      { links: [["theory", "t-softmax"], ["code", "c-setup-train"]] },
    ],
  },

  // ------------------------------------------------------------------ 9
  {
    id: "m-backprop",
    title: { tr: "Geri yayılım: hepsini birleştirmek", en: "Backpropagation: putting it together" },
    blocks: [
      { p: {
        tr: R`Model, katmanların art arda uygulanmasıdır: \(L = \ell\bigl(f_K(\cdots f_2(f_1(x;\theta_1);\theta_2)\cdots;\theta_K)\bigr)\); burada \(K\), 8. kısımdaki gibi sınıf sayısı değil, katman sayısıdır. Zincir kuralı, kaybın her parametreye göre türevini sondan başa doğru, katman katman hesaplamayı sağlar ([[backprop|geri yayılım]]). \(\delta_k\), \(k\). katmanın çıktısına göre gradyan olsun:`,
        en: R`The model is a chain of layers: \(L = \ell\bigl(f_K(\cdots f_2(f_1(x;\theta_1);\theta_2)\cdots;\theta_K)\bigr)\); here \(K\) is the number of layers, not the number of classes as in section 8. The chain rule computes the derivative of the loss with respect to every parameter from the end to the start, layer by layer ([[backprop|backpropagation]]). Let \(\delta_k\) be the gradient with respect to the output of layer \(k\):`,
      } },
      { eq: R`\delta_{k-1} = \left(\frac{\partial f_k}{\partial x_k}\right)^{\!\top}\delta_k \qquad \frac{\partial L}{\partial \theta_k} = \left(\frac{\partial f_k}{\partial \theta_k}\right)^{\!\top}\delta_k` },
      { p: {
        tr: R`Her katman yalnızca kendi yerel türevini bilir: gelen \(\delta\)'yı alır, girdisine göre gradyanı bir önceki katmana verir ve varsa kendi parametrelerinin gradyanını hesaplar. Ağın tamamını hiçbir katmanın bilmesi gerekmez.`,
        en: R`Every layer only knows its own local derivative: it takes the incoming \(\delta\), passes the gradient with respect to its input to the layer before, and computes the gradient of its own parameters, if any. No layer needs to know the whole network.`,
      } },
      { p: {
        tr: R`<a href="#" data-go="math:mm-backprop" data-chapter="mlp">MLP bölümünün dört denklemi</a> bu kuralın özel hâlidir. (2) numaralı denklem, aşağıdaki tablonun iki satırının art arda uygulanmasıdır: tam bağlı katman \(\mathbf W^\top\boldsymbol\delta\) verir, aktivasyon onu \(\odot f'(\mathbf z)\) ile çarpar. (3) ve (4) tam bağlı katmanın parametre gradyanlarıdır, (1) de softmax + çapraz entropi satırının iki sınıflı hâli. CNN'de yeni olan yalnızca katman türleridir: her yeni katman için gereken tek şey, kendi geri kuralıdır. Bütün döngüyü, MLP bölümündeki gibi, tek bir nöronla sayılarla hatırlayalım (bu kez kayıp kare hata):`,
        en: R`<a href="#" data-go="math:mm-backprop" data-chapter="mlp">Chapter 2's four equations</a> are special cases of this rule. Equation (2) is two rows of the table below applied one after the other: the fully connected layer gives \(\mathbf W^\top\boldsymbol\delta\), and the activation multiplies it by \(\odot f'(\mathbf z)\). (3) and (4) are the fully connected layer's parameter gradients, and (1) is the two-class version of the softmax + cross-entropy row. What is new in a CNN are only the kinds of layers: each new layer needs nothing but its own backward rule. Let us recall the whole loop with numbers for a single neuron, as in chapter 2 (this time with the squared error as the loss):`,
      } },
      { derive: { title: { tr: "Tek bir nöronla bir eğitim adımı", en: "One training step with a single neuron" }, steps: [
        { tex: {
          tr: R`z = w x + b = 0{,}5\cdot 2 + (-0{,}5) = 0{,}5,\quad a = \max(0, z) = 0{,}5,\quad L = (a-t)^2 = (0{,}5-2)^2 = 2{,}25`,
          en: R`z = w x + b = 0.5\cdot 2 + (-0.5) = 0.5,\quad a = \max(0, z) = 0.5,\quad L = (a-t)^2 = (0.5-2)^2 = 2.25` }, why: {
          tr: R`İleri yayılım: \(w = 0{,}5,\ b = -0{,}5,\ x = 2\), hedef \(t = 2\). Ara sonuçlar (\(z\), \(a\)) geri yayılım için saklanır.`,
          en: R`Forward pass: \(w = 0.5,\ b = -0.5,\ x = 2\), target \(t = 2\). The intermediate results (\(z\), \(a\)) are kept for the backward pass.` } },
        { tex: { tr: R`\frac{\partial L}{\partial a} = 2(a - t) = 2\,(0{,}5 - 2) = -3`, en: R`\frac{\partial L}{\partial a} = 2(a - t) = 2\,(0.5 - 2) = -3` }, why: {
          tr: "Kaybın çıktıya göre türevi: geri yayılım buradan başlar.", en: "The derivative of the loss with respect to the output: backpropagation starts here." } },
        { tex: R`\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a}\cdot \mathbb{1}[z>0] = -3\cdot 1 = -3`, why: {
          tr: R`ReLU kapısı: \(z > 0\) olduğu için gradyan aynen geçer.`, en: R`The ReLU gate: \(z > 0\), so the gradient passes unchanged.` } },
        { tex: R`\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z}\cdot x = -3\cdot 2 = -6,\qquad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}\cdot 1 = -3`, why: {
          tr: R`\(z = wx + b\) olduğundan \(\partial z/\partial w = x\) ve \(\partial z/\partial b = 1\).`, en: R`Since \(z = wx + b\): \(\partial z/\partial w = x\) and \(\partial z/\partial b = 1\).` } },
        { tex: {
          tr: R`w \leftarrow 0{,}5 - 0{,}05\cdot(-6) = 0{,}8,\qquad b \leftarrow -0{,}5 - 0{,}05\cdot(-3) = -0{,}35`,
          en: R`w \leftarrow 0.5 - 0.05\cdot(-6) = 0.8,\qquad b \leftarrow -0.5 - 0.05\cdot(-3) = -0.35` }, why: {
          tr: R`Gradyan inişi, \(\eta = 0{,}05\). Gradyanlar negatif olduğu için iki parametre de artar.`,
          en: R`Gradient descent with \(\eta = 0.05\). The gradients are negative, so both parameters grow.` } },
        { tex: {
          tr: R`z = 0{,}8\cdot 2 - 0{,}35 = 1{,}25,\qquad L = (1{,}25 - 2)^2 = 0{,}5625`,
          en: R`z = 0.8\cdot 2 - 0.35 = 1.25,\qquad L = (1.25 - 2)^2 = 0.5625` }, why: {
          tr: "Yeni ileri yayılım: tek adımda kayıp 2,25'ten 0,5625'e indi. Eğitim, bu döngünün yüz binlerce parametre için binlerce kez tekrarıdır.",
          en: "A new forward pass: in one step the loss fell from 2.25 to 0.5625. Training is this loop, repeated thousands of times for hundreds of thousands of parameters." } },
      ] } },
      { h: { tr: "SimpleCNN'in geri yayılım tablosu", en: "SimpleCNN's backward table" } },
      { table: {
        head: [{ tr: "Katman", en: "Layer" }, { tr: "İleri", en: "Forward" }, R`\({\delta_x}\)`, { tr: "Parametre gradyanı", en: "Parameter gradient" }],
        rows: [
          ["Softmax + CE", R`\(L = -\log p_c\)`, R`\(\mathbf{p}-\mathbf{y}\)`, "—"],
          ["FC2, FC1", R`\(\mathbf{W}\mathbf{x}+\mathbf{b}\)`, R`\(\mathbf{W}^\top\boldsymbol\delta\)`, R`\(\boldsymbol\delta\mathbf{x}^\top,\ \boldsymbol\delta\)`],
          ["ReLU", R`\(\max(0,z)\)`, R`\(\delta\odot\mathbb{1}[z>0]\)`, "—"],
          [{ tr: "Düzleştirme", en: "Flatten" }, { tr: "yeniden şekillendirme", en: "reshape" }, { tr: "geri şekillendirme", en: "reshape back" }, "—"],
          ["Dropout", R`\(r\odot h/(1-p)\)`, R`\(r\odot\delta/(1-p)\)`, "—"],
          ["MaxPool", { tr: "pencere maksimumu", en: "window maximum" }, { tr: R`\(\delta\) kazanana`, en: R`\(\delta\) to the winner` }, "—"],
          ["BatchNorm", R`\(\gamma\hat x+\beta\)`, { tr: "3. kısım, 6. adım", en: "section 3, step 6" }, R`\(\sum\delta\hat x,\ \sum\delta\)`],
          ["Conv", R`\(w\star x+b\)`, R`\(\operatorname{pad}(\delta)\star\operatorname{rot}_{180}(w)\)`, R`\(x\star\delta,\ \sum\delta\)`],
        ],
      } },
      { p: {
        tr: R`Tablo yukarıdan aşağı, yani geri yayılımın sırasıyla okunur. PyTorch'ta <code>loss.backward()</code> tam olarak bunu yapar: ileri yayılım sırasında hangi işlemin hangi tensörlerle yapıldığını bir hesap grafında kaydeder ve geri giderken her işlemin geri kuralını uygular (otomatik türev).`,
        en: R`The table is read top to bottom, in the order of the backward pass. In PyTorch, <code>loss.backward()</code> does exactly this: during the forward pass it records which operation was applied to which tensors in a computation graph, and on the way back applies each operation's backward rule (automatic differentiation).`,
      } },
      { callout: "warn", title: { tr: "Gradyanlar birikir", en: "Gradients accumulate" }, text: {
        tr: R`<code>backward()</code> her parametrenin <code>.grad</code> alanına yazmaz, <b>ekler</b>: \(g \leftarrow g + \partial L/\partial\theta\). Aynı parametrenin birden fazla yerde kullanıldığı durumlar için bu gereklidir. Bu yüzden her adımda önce <code>optimizer.zero_grad()</code> çağrılır.`,
        en: R`<code>backward()</code> does not write into each parameter's <code>.grad</code>, it <b>adds</b> to it: \(g \leftarrow g + \partial L/\partial\theta\). That is needed when the same parameter is used in several places. So every step first calls <code>optimizer.zero_grad()</code>.`,
      } },
      { callout: "tip", title: { tr: "Sayısal gradyan kontrolü", en: "Numerical gradient check" }, text: {
        tr: R`Bir türetmeyi sınamanın en kolay yolu sonlu farktır: \(\dfrac{\partial L}{\partial\theta} \approx \dfrac{L(\theta+h) - L(\theta-h)}{2h}\); hatası \(h^2\) mertebesindedir. Softmax kısmındaki örnek bu kontrolü canlı yapıyor.`,
        en: R`The easiest way to test a derivation is a finite difference: \(\dfrac{\partial L}{\partial\theta} \approx \dfrac{L(\theta+h) - L(\theta-h)}{2h}\), with an error of order \(h^2\). The example in the softmax section runs this check live.`,
      } },
      { links: [["code", "c-train-epoch"], ["theory", "t-opt"]] },
    ],
  },

  // ------------------------------------------------------------------ 10
  {
    id: "m-opt",
    title: { tr: "Optimizasyon: SGD, Momentum, Adam", en: "Optimization: SGD, Momentum, Adam" },
    blocks: [
      { p: {
        tr: R`Gradyan, kaybın en hızlı arttığı yönü gösterir; [[gradient-descent|gradyan inişi]] parametreleri ters yönde küçük bir adım kaydırır. Ama gerçek gradyan bütün eğitim verisi üzerinden tanımlıdır:`,
        en: R`The gradient points where the loss grows fastest; [[gradient-descent|gradient descent]] moves the parameters a small step the other way. But the true gradient is defined over the whole training set:`,
      } },
      { eq: R`\nabla L(\theta) = \frac{1}{M}\sum_{n=1}^{M}\nabla \ell_n(\theta) \qquad \hat{\mathbf{g}} = \frac{1}{N}\sum_{n\in B}\nabla \ell_n(\theta)` },
      { p: {
        tr: R`\(M = 48.000\) görüntünün hepsiyle tek adım atmak yerine [[sgd|SGD]], rastgele bir batch \(B\) ile gradyanı tahmin eder. Batch rastgele seçildiği için tahmin yansızdır, \(\mathbb{E}[\hat{\mathbf g}] = \nabla L\); varyansı ise \(1/N\) ile azalır. Büyük batch, daha az gürültülü ama daha pahalı adımlar demektir. Bu, <a href="#" data-go="math:mm-train" data-chapter="mlp">MLP bölümündeki</a> mini-batch gradyan inişidir (orada batch büyüklüğü \(B\) idi, burada \(N = 256\)); yeni olan, aşağıdaki iki iyileştirmedir.`,
        en: R`Instead of one step with all \(M = 48{,}000\) images, [[sgd|SGD]] estimates the gradient from a random batch \(B\). Since the batch is random the estimate is unbiased, \(\mathbb{E}[\hat{\mathbf g}] = \nabla L\), and its variance falls as \(1/N\). A large batch means less noisy but more expensive steps. This is the mini-batch gradient descent of <a href="#" data-go="math:mm-train" data-chapter="mlp">chapter 2</a> (there the batch size was \(B\), here \(N = 256\)); what is new are the two improvements below.`,
      } },
      { eq: R`\text{SGD:}\qquad \theta_t = \theta_{t-1} - \eta\,\mathbf{g}_t` },
      { h: { tr: "Momentum", en: "Momentum" } },
      { eq: R`\mathbf{v}_t = \mu\,\mathbf{v}_{t-1} + \mathbf{g}_t \qquad \theta_t = \theta_{t-1} - \eta\,\mathbf{v}_t` },
      { p: {
        tr: R`Hız \(\mathbf v\), geçmiş gradyanların üstel ağırlıklı toplamıdır: \(\mathbf v_t = \sum_{i\le t}\mu^{t-i}\mathbf g_i\). Gradyan hep aynıysa bu geometrik seri \(\mathbf g/(1-\mu)\)'ya yakınsar: \(\mu = 0{,}9\) ile tutarlı yönlerde adım 10 kat büyür, yönü sürekli değişen bileşenler ise birbirini götürür ([[momentum|momentum]]). Buradaki \(\mu\) momentum katsayısıdır; BatchNorm'daki ortalamayla karıştırmayın.`,
        en: R`The velocity \(\mathbf v\) is an exponentially weighted sum of past gradients: \(\mathbf v_t = \sum_{i\le t}\mu^{t-i}\mathbf g_i\). If the gradient stays the same, this geometric series converges to \(\mathbf g/(1-\mu)\): with \(\mu = 0.9\) the step grows tenfold in consistent directions, while components whose sign keeps changing cancel out ([[momentum|momentum]]). Here \(\mu\) is the momentum coefficient; do not confuse it with BatchNorm's mean.`,
      } },
      { h: { tr: "Adam", en: "Adam" } },
      { eq: R`\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t \qquad \mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2` },
      { eq: R`\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t} \qquad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t} \qquad \theta_t = \theta_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}` },
      { p: {
        tr: R`[[adam|Adam]]'da \(\mathbf m\) gradyanın, \(\mathbf v\) karesinin hareketli ortalamasıdır; bütün işlemler parametre başınadır (\(\mathbf g_t^2\) eleman eleman kare). Buradaki \(\mathbf v\), momentumdaki hızla aynı şey değildir; ikisi de literatürde bu harfle yazılır. \(\mathbf m\) ve \(\mathbf v\) sıfırdan başladığı için ilk adımlarda sıfıra doğru çekiktir; bias düzeltmesi bunu giderir:`,
        en: R`In [[adam|Adam]], \(\mathbf m\) is a moving average of the gradient and \(\mathbf v\) of its square; everything is per parameter (\(\mathbf g_t^2\) is an element-wise square). This \(\mathbf v\) is not the velocity of momentum; the literature writes both with this letter. Since \(\mathbf m\) and \(\mathbf v\) start at zero, they are biased towards zero in the first steps; the bias correction removes that:`,
      } },
      { derive: { title: { tr: "Bias düzeltmesi neden 1 − βᵗ?", en: "Why is the bias correction 1 − βᵗ?" }, steps: [
        { tex: R`\mathbf{m}_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{\,t-i}\,\mathbf{g}_i`, why: {
          tr: R`Özyinelemeyi \(\mathbf m_0 = 0\)'dan başlayarak açınca.`, en: R`Unrolling the recursion from \(\mathbf m_0 = 0\).` } },
        { tex: R`\mathbb{E}[\mathbf{m}_t] = \mathbb{E}[\mathbf g]\;(1-\beta_1)\sum_{i=1}^{t}\beta_1^{\,t-i}`, why: {
          tr: "Gradyanların beklenen değeri adımlar boyunca aynıysa.", en: "If the expected gradient stays the same over the steps." } },
        { tex: R`(1-\beta_1)\sum_{i=1}^{t}\beta_1^{\,t-i} = 1-\beta_1^{\,t}`, why: { tr: "Geometrik seri toplamı.", en: "The sum of a geometric series." } },
        { tex: R`\mathbb{E}\!\left[\frac{\mathbf{m}_t}{1-\beta_1^t}\right] = \mathbb{E}[\mathbf g]`, why: {
          tr: R`Bölünce tahmin yansız olur. Örneğin \(t = 1\)'de \(\mathbf m_1 = 0{,}1\,\mathbf g_1\) iken \(\hat{\mathbf m}_1 = \mathbf g_1\)'dir. \(\mathbf v\) için de aynısı \(\beta_2\) ile geçerlidir.`,
          en: R`Dividing makes the estimate unbiased. For example at \(t = 1\), \(\mathbf m_1 = 0.1\,\mathbf g_1\) but \(\hat{\mathbf m}_1 = \mathbf g_1\). The same holds for \(\mathbf v\) with \(\beta_2\).` } },
      ] } },
      { p: {
        tr: R`Gradyan sabitse \(\hat{\mathbf m} = \mathbf g\), \(\hat{\mathbf v} = \mathbf g^2\) ve adım \(\eta\,\mathbf g/|\mathbf g| = \eta\,\operatorname{sign}(\mathbf g)\) olur: gradyanın büyüklüğü ne olursa olsun her parametre adım başına yaklaşık \(\eta\) kadar hareket eder. Adam'ın [[learning-rate|öğrenme oranı]] seçimini kolaylaştırmasının nedeni budur. Aşağıda ilk adımları hesaplayın: önce sabit 0,5 ile sabit 0,001 gradyanını karşılaştırın (adım boyu değişiyor mu?), sonra işaret değiştiren gradyanı seçin, en sonda da bias düzeltmesini kapatıp ilk adımlara bakın.`,
        en: R`If the gradient is constant, \(\hat{\mathbf m} = \mathbf g\), \(\hat{\mathbf v} = \mathbf g^2\) and the step is \(\eta\,\mathbf g/|\mathbf g| = \eta\,\operatorname{sign}(\mathbf g)\): whatever the size of the gradient, every parameter moves about \(\eta\) per step. That is why Adam makes the [[learning-rate|learning rate]] easy to choose. Compute the first steps below: first compare the constant gradients 0.5 and 0.001 (does the step size change?), then pick the alternating gradient, and finally turn the bias correction off and look at the first steps.`,
      } },
      { widget: "adam" },
      { model: {
        tr: R`<b>SimpleCNN'de:</b> Adam, \(\eta = 0{,}001\), \(\beta_1 = 0{,}9\), \(\beta_2 = 0{,}999\), \(\epsilon = 10^{-8}\); 10 epoch × 188 batch = 1880 adım.`,
        en: R`<b>In SimpleCNN:</b> Adam with \(\eta = 0.001\), \(\beta_1 = 0.9\), \(\beta_2 = 0.999\), \(\epsilon = 10^{-8}\); 10 epochs × 188 batches = 1880 steps.`,
      } },
      { h: { tr: "Özet: bu sekmede ne öğrendik?", en: "Recap: what did this tab add?" } },
      { list: [
        { tr: R`Evrişim, yalnızca yerel bağlantıları olan ve ağırlıklarını paylaşan bir tam bağlı katman gibidir: ağırlık gradyanı "girdi × hata"nın bütün konumlar üzerinden toplamı, girdi gradyanı da 180° çevrilmiş filtreyle bir korelasyondur.`, en: R`A convolution is like a fully connected layer with only local connections and shared weights: the weight gradient is "input × error" summed over all positions, and the input gradient is a correlation with the filter rotated by 180°.` },
        { tr: R`BatchNorm her kanalı ölçeklendirir; gradyanından ortalama ve \(\hat x\) bileşeni çıkarılır. Havuzlama gradyanı kazanana, dropout aynı maskeyle kalanlara iletir.`, en: R`BatchNorm rescales every channel; the mean and the \(\hat x\) component are removed from its gradient. Pooling sends the gradient to the winner, dropout to the survivors of the same mask.` },
        { tr: R`Softmax + çapraz entropinin gradyanı \(\mathbf p - \mathbf y\): MLP bölümündeki \(\hat y - y\)'nin 10 sınıflı hâli.`, en: R`The gradient of softmax + cross-entropy is \(\mathbf p - \mathbf y\): the 10-class version of chapter 2's \(\hat y - y\).` },
        { tr: R`Geri yayılım, her katmanın yerel kuralının sondan başa zincirlenmesidir; <code>loss.backward()</code> bunu otomatik yapar. Adam, gradyanın yönü tutarlıyken her parametrenin adımını, gradyanın büyüklüğünden bağımsız olarak yaklaşık \(\eta\)'da tutar.`, en: R`Backpropagation chains every layer's local rule from the end to the start; <code>loss.backward()</code> does it automatically. While the gradient's direction is consistent, Adam keeps every parameter's step near \(\eta\), whatever the size of its gradient.` },
      ] },
      { links: [["theory", "t-opt"], ["code", "c-setup-train"], ["math", "mm-train", "mlp"]] },
    ],
  },
];
window.MATH_SECTIONS = SECTIONS;

// =====================================================================
// Small helpers
// =====================================================================
const mt = { started: false, rendered: false, observer: null, derive: {}, state: {} };
const T = (value) => (value == null || typeof value === "string" ? value || "" : L(value));
const nm = (value, digits = 2) => texNum(fmtNum(value, digits));            // a number inside TeX
/** A number inside TeX with at most `digits` decimals and no trailing zeros (5, 2,5, 0,125). */
const nmt = (value, digits = 3) => texNum(new Intl.NumberFormat(locale(), { maximumFractionDigits: digits }).format(value));
const paren = (value, digits = 2) => (value < 0 ? `(${nmt(value, digits)})` : nmt(value, digits));
const inline = (tex) => `\\(${tex}\\)`;
const display = (tex) => `\\(\\displaystyle ${tex}\\)`;
/** Static, author-written HTML with glossary links. */
const rich = (html) => termify(T(html));
function htmlNode(tag, attrs, html) {
  const node = h(tag, attrs);
  node.innerHTML = rich(html);
  return node;
}
/** A number in scientific notation for TeX, e.g. 9,5 · 10⁻⁷. */
function sci(value, digits = 1) {
  if (value === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(value)));
  if (exp >= -2 && exp <= 3) return nm(value, Math.max(0, digits + 1 - Math.max(exp, 0)));
  return `${nm(value / 10 ** exp, digits)}\\cdot 10^{${exp}}`;
}
const eqBox = (tex, cls = "") => h("div", { class: `m-eq ${cls}`.trim(), text: display(tex) });

/** A small matrix of numbers as a grid of cells. */
function grid(values, rows, cols, { cls = "", fmt = (v) => fmtNum(v, 0), cellClass = () => "", attrs = () => ({}) } = {}) {
  const box = h("div", { class: `m-grid ${cls}`.trim(), style: `grid-template-columns: repeat(${cols}, auto)` });
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = values[r * cols + c];
      box.append(h("span", { class: `m-cell ${cellClass(r, c, v)}`.trim(), ...attrs(r, c), text: fmt(v) }));
    }
  }
  return box;
}
const figure = (caption, node) => h("figure", { class: "m-fig" }, node, h("figcaption", { text: caption }));
const seg = (options, current, onPick) => {
  const box = h("div", { class: "seg" });
  for (const [value, label] of options) {
    const b = h("button", { type: "button", "aria-pressed": String(value === current), text: label });
    b.addEventListener("click", () => onPick(value));
    box.append(b);
  }
  return box;
};

// =====================================================================
// Interactive examples
// =====================================================================
const WIDGETS = {
  /** A 5×5 input, a 3×3 kernel, and the nine products behind any output cell. */
  conv() {
    const X = [1, 2, 0, 1, 3, 0, 1, 3, 1, 0, 2, 1, 0, 2, 1, 1, 0, 1, 3, 2, 0, 2, 1, 0, 1];
    const K = [1, 0, -1, 1, 0, -1, 1, 0, -1];
    const Y = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let m = 0; m < 3; m++) for (let n = 0; n < 3; n++) s += K[m * 3 + n] * X[(i + m) * 5 + j + n];
        Y.push(s);
      }
    }
    const st = (mt.state.conv ??= { i: 0, j: 0 });
    const box = h("div", { class: "m-widget" });
    const draw = () => {
      const { i, j } = st;
      const inX = (r, c) => r >= i && r < i + 3 && c >= j && c < j + 3;
      const yGrid = grid(Y, 3, 3, {
        cls: "pick",
        cellClass: (r, c) => (r === i && c === j ? "sel" : ""),
        attrs: (r, c) => ({ tabindex: "0", role: "button", "data-r": r, "data-c": c, "aria-label": `y(${r},${c})` }),
      });
      const terms = [];
      for (let m = 0; m < 3; m++) for (let n = 0; n < 3; n++) terms.push(`${X[(i + m) * 5 + j + n]}\\cdot${paren(K[m * 3 + n], 0)}`);
      box.replaceChildren(
        h("div", { class: "m-row" },
          figure(L({ tr: "Girdi x (5×5)", en: "Input x (5×5)" }), grid(X, 5, 5, { cellClass: (r, c) => (inX(r, c) ? "win" : "") })),
          h("span", { class: "m-op", text: "⋆" }),
          figure(L({ tr: "Filtre w (3×3)", en: "Filter w (3×3)" }), grid(K, 3, 3, { cls: "kernel", cellClass: (r, c, v) => (v > 0 ? "pos" : v < 0 ? "neg" : "") })),
          h("span", { class: "m-op", text: "=" }),
          figure(L({ tr: "Çıktı y (3×3) · bir hücre seçin", en: "Output y (3×3) · pick a cell" }), yGrid)),
        h("div", { class: "m-calc", text: display(`y_{${i},${j}} = ${terms.slice(0, 3).join(" + ")} + ${terms.slice(3, 6).join(" + ")} + ${terms.slice(6).join(" + ")} = \\mathbf{${nm(Y[i * 3 + j], 0)}}`) }),
        h("p", { class: "m-hint", text: L({
          tr: "Filtre, sol sütunu artı, sağ sütunu eksi sayar: soldan sağa parlaklığın azaldığı yerlerde büyük pozitif, arttığı yerlerde negatif sonuç verir.",
          en: "The filter counts the left column positively and the right one negatively: a large positive result where brightness drops from left to right, a negative one where it rises.",
        }) }));
      const pick = (event) => {
        const cell = event.target.closest?.("[data-r]");
        if (!cell) return;
        const r = Number(cell.dataset.r);
        const c = Number(cell.dataset.c);
        if (r === st.i && c === st.j) return;
        st.i = r;
        st.j = c;
        draw();
        box.querySelector(`[data-r="${r}"][data-c="${c}"]`)?.focus({ preventScroll: true });
      };
      yGrid.addEventListener("pointerover", pick);
      yGrid.addEventListener("click", pick);
      yGrid.addEventListener("focusin", pick);
    };
    draw();
    return box;
  },

  /** BatchNorm on four editable numbers, every intermediate value written out. */
  bn() {
    const st = (mt.state.bn ??= { xs: [1, 2, 4, 9], gamma: 2, beta: 0.5 });
    const eps = 1e-5;
    const out = h("div", { class: "m-out" });
    const input = (value, label, onChange) => {
      const el = h("input", { type: "number", step: "0.1", value: String(value), "aria-label": label });
      el.addEventListener("input", () => {
        const v = Number(el.value);
        if (Number.isFinite(v) && el.value !== "") { onChange(v); update(); }
      });
      return h("label", { class: "m-input" }, h("span", { text: label }), el);
    };
    const update = () => {
      const { xs, gamma, beta } = st;
      const m = xs.length;
      const mu = xs.reduce((a, b) => a + b, 0) / m;
      const varr = xs.reduce((a, b) => a + (b - mu) ** 2, 0) / m;
      const s = Math.sqrt(varr + eps);
      const xh = xs.map((x) => (x - mu) / s);
      const ys = xh.map((v) => gamma * v + beta);
      const rows = xs.map((x, i) => h("tr", {},
        h("td", { text: String(i + 1) }), h("td", { text: fmtNum(x, 2) }), h("td", { text: fmtNum(x - mu, 3) }),
        h("td", { text: fmtNum(xh[i], 3) }), h("td", { text: fmtNum(ys[i], 3) })));
      out.replaceChildren(
        h("div", { class: "m-calc", text: display(`\\mu = \\frac{${xs.map((x) => paren(x, 3)).join(" + ")}}{${m}} = ${nmt(mu, 3)}`) }),
        h("div", { class: "m-calc", text: display(`\\sigma^2 = \\frac{${xs.map((x) => `${paren(x - mu, 3)}^2`).join(" + ")}}{${m}} = ${nmt(varr, 3)} \\qquad s = \\sqrt{\\sigma^2 + \\varepsilon} = ${nm(s, 4)}`) }),
        h("div", { class: "table-wrap" }, h("table", { class: "data-table compact m-table" },
          h("thead", {}, h("tr", {}, ...["i", "x_i", "x_i - \\mu", "\\hat x_i = (x_i-\\mu)/s", "y_i = \\gamma\\hat x_i + \\beta"].map((c) => h("th", { text: inline(c) })))),
          h("tbody", {}, ...rows))),
        h("div", { class: "m-calc", text: display(`\\sum_i \\hat x_i = ${nm(xh.reduce((a, b) => a + b, 0) + 0, 3)} \\qquad \\frac{1}{m}\\sum_i \\hat x_i^2 = ${nm(xh.reduce((a, b) => a + b * b, 0) / m, 6)}`) }));
    };
    const inputs = h("div", { class: "m-inputs" },
      ...st.xs.map((x, i) => input(x, `x${"₁₂₃₄"[i]}`, (v) => { st.xs[i] = v; })),
      input(st.gamma, "γ", (v) => { st.gamma = v; }),
      input(st.beta, "β", (v) => { st.beta = v; }));
    update();
    return h("div", { class: "m-widget" }, inputs, out);
  },

  /** f and f′ for ReLU, sigmoid and tanh, and what ℓ layers of f′ do to a gradient. */
  act() {
    const FN = {
      relu: { label: "ReLU", f: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : 0), maxd: 1 },
      sigmoid: { label: "Sigmoid", f: (z) => 1 / (1 + Math.exp(-z)), d: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); }, maxd: 0.25 },
      tanh: { label: "Tanh", f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2, maxd: 1 },
    };
    const st = (mt.state.act ??= { fn: "sigmoid", depth: 10 });
    const box = h("div", { class: "m-widget" });
    const W = 560, H = 230, x0 = -6, x1 = 6, y0 = -1.2, y1 = 2.2;
    const X = (x) => 34 + ((x - x0) / (x1 - x0)) * (W - 44);
    const Y = (y) => 12 + ((y1 - y) / (y1 - y0)) * (H - 36);
    const path = (g) => {
      let d = "";
      for (let k = 0; k <= 240; k++) {
        const x = x0 + ((x1 - x0) * k) / 240;
        const y = Math.max(y0 - 1, Math.min(y1 + 1, g(x)));
        d += `${k ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
      }
      return d;
    };
    // The controls are built once (rebuilding a slider while it is dragged would stop the drag)
    const pickRow = h("div", { class: "t-controls" });
    const plotHost = h("div");
    const depthOut = h("output", { text: String(st.depth) });
    const depth = h("input", { type: "range", min: "1", max: "30", step: "1", value: String(st.depth) });
    const calc = h("div", { class: "m-calc" });
    depth.addEventListener("input", () => { st.depth = Number(depth.value); depthOut.textContent = depth.value; drawCalc(); });
    const drawCalc = () => {
      const fn = FN[st.fn];
      const factor = fn.d(1);
      calc.textContent = display(
        `\\bigl(f'(1)\\bigr)^{\\ell} = ${nm(factor, 3)}^{${st.depth}} \\approx ${sci(factor ** st.depth)}` +
        `\\qquad \\bigl(\\max f'\\bigr)^{\\ell} = ${nm(fn.maxd, 2)}^{${st.depth}} \\approx ${sci(fn.maxd ** st.depth)}`);
    };
    box.append(pickRow, plotHost,
      h("div", { class: "t-controls" }, h("label", {}, L({ tr: "Katman sayısı ℓ", en: "Number of layers ℓ" }), depth, depthOut)),
      calc,
      h("p", { class: "m-hint", text: L({
        tr: "Geri yayılımda gradyan her katmanda bir f′ ile çarpılır. Sigmoid'de en iyi durumda bile (z = 0) çarpan 1/4'tür; ReLU'da pozitif girişler için tam 1.",
        en: "In the backward pass the gradient is multiplied by an f′ in every layer. For the sigmoid the factor is 1/4 even in the best case (z = 0); for ReLU it is exactly 1 for positive inputs.",
      }) }));
    const draw = () => {
      const fn = FN[st.fn];
      const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "m-plot", role: "img", "aria-label": `${fn.label}: f, f′` });
      const clipId = "m-act-clip";
      const defs = svg("defs");
      const clip = svg("clipPath", { id: clipId });
      clip.append(svg("rect", { x: X(x0), y: Y(y1), width: X(x1) - X(x0), height: Y(y0) - Y(y1) }));
      defs.append(clip);
      root.append(defs);
      for (const y of [-1, 0, 1, 2]) {
        root.append(svg("line", { x1: X(x0), x2: X(x1), y1: Y(y), y2: Y(y), class: y === 0 ? "baseline" : "gridline" }));
        root.append(svgText(X(x0) - 6, Y(y) + 4, fmtNum(y, 0), "tick", "end"));
      }
      for (const x of [-6, -4, -2, 0, 2, 4, 6]) {
        root.append(svg("line", { x1: X(x), x2: X(x), y1: Y(y1), y2: Y(y0), class: x === 0 ? "baseline" : "gridline" }));
        root.append(svgText(X(x), H - 6, fmtNum(x, 0), "tick", "middle"));
      }
      const g = svg("g", { "clip-path": `url(#${clipId})` });
      g.append(svg("path", { d: path(fn.f), class: "m-curve f" }));
      g.append(svg("path", { d: path(fn.d), class: "m-curve d" }));
      root.append(g);
      pickRow.replaceChildren(
        seg(Object.entries(FN).map(([k, v]) => [k, v.label]), st.fn, (k) => { st.fn = k; draw(); }),
        h("span", { class: "m-legend" }, h("i", { class: "key f" }), "f(z)", h("i", { class: "key d" }), "f′(z)"));
      plotHost.replaceChildren(root);
      drawCalc();
    };
    draw();
    return box;
  },

  /** Max or average pooling forward, and where the incoming gradient lands. */
  pool() {
    const st = (mt.state.pool ??= { mode: "max", x: [1, 3, 2, 0, 4, 2, 1, 5, 0, 1, 3, 2, 2, 6, 1, 1] });
    const D = [0.5, -1, 2, 0.25];
    const box = h("div", { class: "m-widget" });
    const draw = () => {
      const { x, mode } = st;
      const win = (r, c) => Math.floor(r / 2) * 2 + Math.floor(c / 2);
      const arg = [0, 1, 2, 3].map((w) => {
        const cells = [0, 1, 2, 3].map((k) => (Math.floor(w / 2) * 2 + Math.floor(k / 2)) * 4 + (w % 2) * 2 + (k % 2));
        return cells.reduce((best, k) => (x[k] > x[best] ? k : best), cells[0]);
      });
      const y = [0, 1, 2, 3].map((w) => {
        const cells = [0, 1, 2, 3].map((k) => (Math.floor(w / 2) * 2 + Math.floor(k / 2)) * 4 + (w % 2) * 2 + (k % 2));
        return mode === "max" ? x[arg[w]] : cells.reduce((s, k) => s + x[k], 0) / 4;
      });
      const dx = x.map((_, k) => {
        const w = win(Math.floor(k / 4), k % 4);
        return mode === "max" ? (arg[w] === k ? D[w] : 0) : D[w] / 4;
      });
      const winClass = (r, c) => `w${win(r, c)}`;
      const dec = mode === "max" ? 2 : 3;
      box.replaceChildren(
        h("div", { class: "t-controls" },
          seg([["max", "Max pooling"], ["avg", "Average pooling"]], mode, (m) => { st.mode = m; draw(); }),
          (() => {
            const b = h("button", { type: "button", class: "btn small", text: L({ tr: "Yeni sayılar", en: "New numbers" }) });
            b.addEventListener("click", () => { st.x = Array.from({ length: 16 }, () => Math.floor(Math.random() * 10)); draw(); });
            return b;
          })()),
        h("div", { class: "m-row" },
          figure(L({ tr: "Girdi x · kazananlar kalın", en: "Input x · winners in bold" }),
            grid(x, 4, 4, { cellClass: (r, c) => `${winClass(r, c)}${mode === "max" && arg[win(r, c)] === r * 4 + c ? " win-max" : ""}` })),
          h("span", { class: "m-op", text: "→" }),
          figure(L({ tr: "Çıktı y", en: "Output y" }), grid(y, 2, 2, { fmt: (v) => fmtNum(v, mode === "max" ? 0 : 2), cellClass: (r, c) => `w${r * 2 + c}` })),
          h("span", { class: "m-op wide", text: L({ tr: "geri:", en: "back:" }) }),
          figure(L({ tr: "Gelen gradyan δ", en: "Incoming gradient δ" }), grid(D, 2, 2, { fmt: (v) => fmtNum(v, 2), cellClass: (r, c) => `w${r * 2 + c}` })),
          h("span", { class: "m-op", text: "→" }),
          figure("∂L/∂x", grid(dx, 4, 4, { fmt: (v) => (v === 0 ? "0" : fmtNum(v, dec)), cellClass: (r, c) => `${winClass(r, c)}${dx[r * 4 + c] !== 0 && mode === "max" ? " win-max" : ""}` }))),
        h("p", { class: "m-hint", text: mode === "max"
          ? L({ tr: "Her penceredeki gradyan yalnızca kazanan konuma gider; diğer 12 konum sıfır gradyan alır, yani o adımda hiç öğrenmez.", en: "Each window's gradient goes only to the winning position; the other 12 positions get zero gradient, so they learn nothing in that step." })
          : L({ tr: "Average pooling'de pencerenin dört konumu da gradyanın dörtte birini alır.", en: "With average pooling all four positions of a window get a quarter of the gradient." }) }));
    };
    draw();
    return box;
  },

  /** One random mask, and the average of many masks: Dropout keeps the expected value. */
  drop() {
    const st = (mt.state.drop ??= { p: 0.45, seed: 1 });
    const hs = [0.8, 1.5, 0.3, 2.0, 1.1, 0.6];
    const pOut = h("output", { text: fmtNum(st.p, 2) });
    const slider = h("input", { type: "range", min: "0", max: "0.9", step: "0.05", value: String(st.p) });
    slider.addEventListener("input", () => { st.p = Number(slider.value); pOut.textContent = fmtNum(st.p, 2); draw(); });
    const again = h("button", { type: "button", class: "btn small", text: L({ tr: "Yeni maske", en: "New mask" }) });
    again.addEventListener("click", () => { st.seed++; draw(); });
    const dyn = h("div", { class: "m-out" });
    const box = h("div", { class: "m-widget" }, h("div", { class: "t-controls" }, h("label", {}, "p", slider, pOut), again), dyn);
    const draw = () => {
      const { p } = st;
      let seed = st.seed * 9301 + 49297;
      const rand = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
      const one = hs.map((v) => (rand() >= p ? v / (1 - p) : 0));
      const trials = 10000;
      const sum = hs.map(() => 0);
      const sq = hs.map(() => 0);
      for (let t = 0; t < trials; t++) {
        hs.forEach((v, i) => {
          const o = Math.random() >= p ? v / (1 - p) : 0;
          sum[i] += o;
          sq[i] += o * o;
        });
      }
      const rows = hs.map((v, i) => {
        const mean = sum[i] / trials;
        const variance = sq[i] / trials - mean * mean;
        return h("tr", {},
          h("td", { text: fmtNum(v, 2) }),
          h("td", { class: one[i] === 0 ? "m-dropped" : "", text: fmtNum(one[i], 3) }),
          h("td", { text: fmtNum(mean, 3) }),
          h("td", { text: `${fmtNum(variance, 3)} / ${fmtNum((p / (1 - p)) * v * v, 3)}` }));
      });
      dyn.replaceChildren(
        h("div", { class: "table-wrap" }, h("table", { class: "data-table compact m-table" },
          h("thead", {}, h("tr", {},
            h("th", { text: inline("h_i") }),
            h("th", { text: `${inline("\\tilde h_i")} · ${L({ tr: "bu maske", en: "this mask" })}` }),
            h("th", { text: L({ tr: "10.000 maskenin ortalaması", en: "mean of 10,000 masks" }) }),
            h("th", { text: `${L({ tr: "varyans: ölçülen / formül", en: "variance: measured / formula" })}` }))),
          h("tbody", {}, ...rows))),
        h("p", { class: "m-hint", text: L({
          tr: `Bu maskede ${one.filter((v) => v === 0).length}/${hs.length} değer sıfırlandı, kalanlar 1/(1 − p) = ${fmtNum(1 / (1 - p), 2)} ile büyütüldü. Ortalama sütunu her zaman h'ye yakındır.`,
          en: `This mask dropped ${one.filter((v) => v === 0).length}/${hs.length} values and scaled the rest by 1/(1 − p) = ${fmtNum(1 / (1 - p), 2)}. The mean column always stays close to h.`,
        }) }));
    };
    draw();
    return box;
  },

  /** Softmax, cross-entropy and its gradient, checked against finite differences. */
  softmax() {
    const st = (mt.state.softmax ??= { z: [2, 1, -1], c: 0 });
    const out = h("div", { class: "m-out" });
    const loss = (z, c) => {
      const mx = Math.max(...z);
      const lse = mx + Math.log(z.reduce((s, v) => s + Math.exp(v - mx), 0));
      return lse - z[c];
    };
    const update = () => {
      const { z, c } = st;
      const mx = Math.max(...z);
      const e = z.map((v) => Math.exp(v - mx));
      const sum = e.reduce((a, b) => a + b, 0);
      const p = e.map((v) => v / sum);
      const hstep = 1e-5;
      const numeric = z.map((_, k) => {
        const zp = z.slice(); zp[k] += hstep;
        const zm = z.slice(); zm[k] -= hstep;
        return (loss(zp, c) - loss(zm, c)) / (2 * hstep);
      });
      const diff = Math.max(...z.map((_, k) => Math.abs(numeric[k] - (p[k] - (k === c ? 1 : 0)))));
      out.replaceChildren(
        h("div", { class: "table-wrap" }, h("table", { class: "data-table compact m-table" },
          h("thead", {}, h("tr", {}, ...["k", "z_k", "e^{z_k - \\max z}", "p_k", "y_k"].map((t) => h("th", { text: inline(t) })),
            h("th", { text: `${L({ tr: "analitik", en: "analytic" })} ${inline("p_k - y_k")}` }),
            h("th", { text: L({ tr: "sayısal", en: "numerical" }) }))),
          h("tbody", {}, ...z.map((v, k) => h("tr", { class: k === c ? "best" : "" },
            h("td", { text: String(k) }), h("td", { text: fmtNum(v, 2) }), h("td", { text: fmtNum(e[k], 4) }),
            h("td", { text: fmtNum(p[k], 4) }), h("td", { text: k === c ? "1" : "0" }),
            h("td", { text: fmtNum(p[k] - (k === c ? 1 : 0), 6) }), h("td", { text: fmtNum(numeric[k], 6) })))))),
        h("div", { class: "m-calc", text: display(`L = -\\log p_{${c}} = -\\log ${nm(p[c], 4)} = ${nm(loss(z, c), 4)}`) }),
        h("p", { class: "m-hint", text: L({
          tr: `Analitik ve sayısal gradyan arasındaki en büyük fark: ${inline(sci(diff))} (${inline("h = 10^{-5}")} ile sonlu fark).`,
          en: `Largest difference between the analytic and numerical gradient: ${inline(sci(diff))} (finite difference with ${inline("h = 10^{-5}")}).`,
        }) }));
    };
    const inputs = h("div", { class: "m-inputs" },
      ...st.z.map((v, k) => {
        const el = h("input", { type: "number", step: "0.5", value: String(v), "aria-label": `z${k}` });
        el.addEventListener("input", () => { const n = Number(el.value); if (el.value !== "" && Number.isFinite(n)) { st.z[k] = n; update(); } });
        return h("label", { class: "m-input" }, h("span", { text: inline(`z_${k}`) }), el);
      }),
      h("label", { class: "m-input" }, h("span", { text: L({ tr: "doğru sınıf c", en: "correct class c" }) }),
        (() => {
          const sel = h("select", {}, ...st.z.map((_, k) => h("option", { value: String(k), text: String(k) })));
          sel.value = String(st.c);
          sel.addEventListener("change", () => { st.c = Number(sel.value); update(); });
          return sel;
        })()));
    update();
    return h("div", { class: "m-widget" }, inputs, out);
  },

  /** Adam's first steps for a chosen gradient sequence, with or without bias correction. */
  adam() {
    const st = (mt.state.adam ??= { pattern: "const", correct: true });
    const box = h("div", { class: "m-widget" });
    const PATTERNS = {
      const: { label: { tr: "sabit g = 0,5", en: "constant g = 0.5" }, g: () => 0.5 },
      tiny: { label: { tr: "sabit g = 0,001", en: "constant g = 0.001" }, g: () => 0.001 },
      noisy: { label: { tr: "gürültülü (ort. 0,2)", en: "noisy (mean 0.2)" }, g: (t) => 0.2 + 0.6 * Math.sin(t * 2.39996) },
      flip: { label: { tr: "işaret değiştiren ±1", en: "alternating ±1" }, g: (t) => (t % 2 ? 1 : -1) },
    };
    const draw = () => {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8, eta = 0.001;
      let m = 0, v = 0, total = 0;
      const rows = [];
      for (let t = 1; t <= 10; t++) {
        const g = PATTERNS[st.pattern].g(t);
        m = b1 * m + (1 - b1) * g;
        v = b2 * v + (1 - b2) * g * g;
        const mh = st.correct ? m / (1 - b1 ** t) : m;
        const vh = st.correct ? v / (1 - b2 ** t) : v;
        const step = -eta * mh / (Math.sqrt(vh) + eps);
        total += step;
        rows.push(h("tr", {}, h("td", { text: String(t) }), h("td", { text: fmtNum(g, 3) }),
          h("td", { text: fmtNum(m, 4) }), h("td", { text: v.toExponential(2) }),
          h("td", { text: fmtNum(mh, 4) }), h("td", { text: vh.toExponential(2) }),
          h("td", { class: "m-strong", text: fmtNum(step / eta, 3) })));
      }
      box.replaceChildren(
        h("div", { class: "t-controls" },
          h("label", {}, L({ tr: "Gradyanlar", en: "Gradients" }),
            (() => {
              const sel = h("select", {}, ...Object.entries(PATTERNS).map(([k, p]) => h("option", { value: k, text: L(p.label) })));
              sel.value = st.pattern;
              sel.addEventListener("change", () => { st.pattern = sel.value; draw(); });
              return sel;
            })()),
          h("label", {},
            (() => {
              const cb = h("input", { type: "checkbox" });
              cb.checked = st.correct;
              cb.addEventListener("change", () => { st.correct = cb.checked; draw(); });
              return cb;
            })(),
            L({ tr: "Bias düzeltmesi", en: "Bias correction" }))),
        h("div", { class: "table-wrap" }, h("table", { class: "data-table compact m-table" },
          h("thead", {}, h("tr", {}, ...["t", "g_t", "m_t", "v_t", "\\hat m_t", "\\hat v_t", "\\Delta\\theta_t / \\eta"].map((c) => h("th", { text: inline(c) })))),
          h("tbody", {}, ...rows))),
        h("p", { class: "m-hint", text: L({
          tr: `10 adımda toplam hareket: ${fmtNum(total / eta, 2)} η. ${st.correct
            ? "Düzeltmeyle sabit gradyanda her adım ≈ −1 η'dır, gradyan 0,5 de olsa 0,001 de olsa. Düzeltmeyi kapatın: ilk adımlar η'nın üç katından büyük olur, çünkü v (β₂ = 0,999) m'den (β₁ = 0,9) çok daha yavaş dolar."
            : "Düzeltme kapalı: v, m'den çok daha yavaş dolduğu için ilk adımlar çok büyük (t = 1'de ≈ 3,2 η). Bu, eğitimin başında ağırlıkları gereğinden fazla sarsar."}`,
          en: `Total movement in 10 steps: ${fmtNum(total / eta, 2)} η. ${st.correct
            ? "With the correction each step is ≈ −1 η for a constant gradient, whether it is 0.5 or 0.001. Turn the correction off: the first steps get more than three times η, because v (β₂ = 0.999) fills up much more slowly than m (β₁ = 0.9)."
            : "Correction off: v fills up much more slowly than m, so the first steps are too large (≈ 3.2 η at t = 1). That shakes the weights more than needed at the start of training."}`,
        }) }));
    };
    draw();
    return box;
  },
};

// =====================================================================
// Rendering
// =====================================================================
/** A derivation revealed one line at a time. */
function derivation(block, key) {
  const steps = block.steps;
  const box = h("div", { class: "m-derive" });
  const counter = h("span", { class: "m-derive-count" });
  const list = h("ol", { class: "m-derive-steps" });
  for (const step of steps) {
    list.append(h("li", {}, eqBox(T(step.tex)), htmlNode("p", { class: "m-why" }, step.why)));
  }
  const next = h("button", { type: "button", class: "btn small primary", text: L({ tr: "Sonraki adım ↓", en: "Next step ↓" }) });
  const all = h("button", { type: "button", class: "btn small", text: L({ tr: "Hepsini göster", en: "Show all" }) });
  const reset = h("button", { type: "button", class: "btn small", text: L({ tr: "Baştan", en: "Start over" }) });
  const show = (n, animate) => {
    mt.derive[key] = n;
    [...list.children].forEach((li, k) => {
      const visible = k < n;
      if (visible && li.hidden && animate) li.classList.add("m-reveal");
      li.hidden = !visible;
    });
    counter.textContent = L({ tr: `${n} / ${steps.length} adım`, en: `step ${n} of ${steps.length}` });
    next.disabled = n >= steps.length;
    all.disabled = n >= steps.length;
    reset.disabled = n <= 1;
  };
  next.addEventListener("click", () => {
    show(Math.min(steps.length, (mt.derive[key] ?? 1) + 1), true);
    list.children[mt.derive[key] - 1]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
  all.addEventListener("click", () => show(steps.length, true));
  reset.addEventListener("click", () => show(1, false));
  box.append(
    h("div", { class: "m-derive-head" }, h("span", { class: "m-derive-kicker", text: L({ tr: "Türetme", en: "Derivation" }) }),
      h("strong", { text: L(block.title) }), counter),
    list,
    h("div", { class: "m-derive-ctl" }, next, all, reset));
  show(mt.derive[key] ?? 1, false);
  return box;
}

function renderBlock(block, key) {
  if (block.p) return htmlNode("p", {}, block.p);
  if (block.list) {
    const list = h(block.ordered ? "ol" : "ul", { class: "tut-list" });
    for (const item of block.list) list.append(htmlNode("li", {}, item));
    return list;
  }
  if (block.h) return h("h3", { class: "m-h3", text: L(block.h) });
  if (block.eq) return eqBox(I18N.lang === "en" && block.eqEn ? block.eqEn : T(block.eq));
  if (block.derive) return derivation(block.derive, key);
  if (block.widget) return WIDGETS[block.widget]();
  if (block.model) return htmlNode("div", { class: "in-model m-model" }, block.model);
  if (block.callout) {
    const icons = { tip: "💡", warn: "⚠️", info: "ℹ️" };
    return h("div", { role: "note", class: `tut-callout ${block.callout}` },
      h("span", { class: "tut-callout-icon", text: icons[block.callout], "aria-hidden": "true" }),
      h("div", {}, h("strong", { text: L(block.title) }), htmlNode("p", {}, block.text)));
  }
  if (block.table) {
    return h("div", { class: "table-wrap" }, h("table", { class: "data-table tut-table m-table" },
      h("thead", {}, h("tr", {}, ...block.table.head.map((c) => htmlNode("th", {}, c)))),
      h("tbody", {}, ...block.table.rows.map((row) => h("tr", {}, ...row.map((c) => htmlNode("td", {}, c)))))));
  }
  if (block.links) {
    return h("div", { class: "m-links" }, h("span", { text: L({ tr: "Ayrıca bakın:", en: "See also:" }) }),
      ...block.links.map(([tab, section, chapter]) => h("a", { href: "#", "data-go": `${tab}:${section}`, "data-chapter": chapter,
        text: `${placeLabel({ chapter, tab, section })} →` })));
  }
  return null;
}

const goToSection = (id) => navigate("math", id);

function renderMath() {
  const main = $("#mt-main");
  const toc = $("#mt-toc");
  main.replaceChildren();
  toc.replaceChildren();
  SECTIONS.forEach((section, k) => {
    const title = L(section.title);
    const article = h("article", { class: "card tut-step m-step", id: section.id },
      h("header", { class: "tut-step-head" },
        h("span", { class: "tut-num", text: String(k + 1) }),
        h("div", {}, h("span", { class: "tut-kicker", text: L({ tr: `Kısım ${k + 1} / ${SECTIONS.length}`, en: `Section ${k + 1} of ${SECTIONS.length}` }) }),
          h("h2", { text: title }))));
    section.blocks.forEach((block, b) => {
      const node = renderBlock(block, `${section.id}:${b}`);
      if (node) article.append(node);
    });
    const nav = h("footer", { class: "tut-nav" });
    if (k > 0) {
      const prev = h("button", { type: "button", class: "btn small", text: `← ${L(SECTIONS[k - 1].title)}` });
      prev.addEventListener("click", () => goToSection(SECTIONS[k - 1].id));
      nav.append(prev);
    }
    if (k < SECTIONS.length - 1) {
      const next = h("button", { type: "button", class: "btn small primary", text: `${L(SECTIONS[k + 1].title)} →` });
      next.addEventListener("click", () => goToSection(SECTIONS[k + 1].id));
      nav.append(next);
    }
    article.append(nav);
    main.append(article);
    const link = h("button", { type: "button", "data-step": section.id },
      h("span", { class: "toc-num", text: String(k + 1) }), h("span", { text: title }));
    link.addEventListener("click", () => goToSection(section.id));
    toc.append(h("li", {}, link));
  });

  // Highlight the section being read in the table of contents
  mt.observer?.disconnect();
  mt.observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const b of toc.querySelectorAll("button")) b.classList.toggle("active", b.dataset.step === entry.target.id);
    }
  }, { rootMargin: "-25% 0px -65% 0px" });
  for (const article of main.querySelectorAll(".m-step")) mt.observer.observe(article);
  mt.rendered = true;
}

/** The section nearest the top of the viewport and its offset (null if the tab is hidden). */
function readingPosition() {
  const steps = [...document.querySelectorAll("#mt-main .m-step")];
  if (!steps.length || !steps[0].getClientRects().length) return null;
  const topbar = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--topbar-h")) || 59;
  let anchor = steps[0];
  for (const step of steps) {
    if (step.getBoundingClientRect().top <= topbar + 40) anchor = step;
    else break;
  }
  return { id: anchor.id, top: anchor.getBoundingClientRect().top };
}

// =====================================================================
// Links between the tabs
// =====================================================================
/** Theory sections get a link to their mathematics. */
const THEORY_TO_MATH = { "t-conv": "m-conv", "t-bn": "m-bn", "t-act": "m-act", "t-pool": "m-pool", "t-drop": "m-drop", "t-softmax": "m-softmax", "t-opt": "m-opt" };
function addTheoryLinks() {
  for (const [theory, math] of Object.entries(THEORY_TO_MATH)) {
    const head = document.querySelector(`#${theory} .t-head`);
    if (!head) continue;
    const link = head.querySelector(".m-theory-link") || head.appendChild(h("a", { href: "#", class: "m-theory-link", "data-go": `math:${math}` }));
    link.textContent = L({ tr: "∑ Matematiği →", en: "∑ The math →" });
  }
}

document.addEventListener("tab-shown", (event) => {
  if (event.detail !== "math" || mt.started) return;
  mt.started = true;
  renderMath();
});

document.addEventListener("lang-change", () => {
  addTheoryLinks();
  if (!mt.rendered) return;
  const position = readingPosition();
  renderMath();
  if (!position) return;
  const anchor = document.getElementById(position.id);
  if (!anchor) return;
  if (position.top >= 0) window.scrollBy({ top: anchor.getBoundingClientRect().top - position.top, behavior: "instant" });
  else anchor.scrollIntoView({ behavior: "instant", block: "start" });
});

addTheoryLinks();
})();
