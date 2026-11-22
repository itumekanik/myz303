/*
 * Chapter 2, Math and Coding tabs: the content, rendered by lesson.js (createLesson).
 * Texts are { tr, en } pairs of HTML with \( TeX \) and [[key|text]] glossary links.
 * Uses net.js (NET) and kit.js (KIT) for the widgets.
 */
import MLP_PY from "../../../../MLP.py";   // bundled as text (scripts/build.mjs)
import { L } from "../../shared/i18n.js";
import { PLACE_TITLES, fmtNum, h } from "../../shared/core.js";
import { createLesson } from "../../shared/lesson.js";
import { NET } from "../../engines/net.js";
import { KIT } from "../../shared/kit.js";

(() => {
const R = String.raw;
const { nf, tx, slider, createGraph, COLORS, redrawers, seeded } = KIT;
const sigma = NET.sigmoid;
const eq = (tex) => h("div", { class: "m-eq", text: `\\(\\displaystyle ${tex}\\)` });

// =====================================================================
// Math widgets
// =====================================================================
const WIDGETS = {
  /** σ(z), σ'(z) = σ(1 − σ) and the same slope measured numerically. */
  sigmoid(state) {
    const s = (state.sigmoid ??= { z: 1.5 });
    const box = h("div", { class: "m-widget" });
    const out = h("div", { class: "m-widget-out" });
    const draw = () => {
      const p = sigma(s.z);
      const e = 1e-4;
      const numeric = (sigma(s.z + e) - sigma(s.z - e)) / (2 * e);
      out.replaceChildren(
        eq(R`\sigma(${tx(s.z, 1)}) = ${tx(p, 4)} \qquad \sigma(z)\big(1 - \sigma(z)\big) = ${tx(p, 4)} \cdot ${tx(1 - p, 4)} = ${tx(p * (1 - p), 4)}`),
        eq(R`\frac{\sigma(z + \varepsilon) - \sigma(z - \varepsilon)}{2\varepsilon}\bigg|_{\varepsilon = 10^{-4}} = ${tx(numeric, 4)}`),
        h("p", { class: "m-why", text: Math.abs(s.z) > 4
          ? L({ tr: `|z| büyük: türev ${nf(p * (1 - p), 4)}, neredeyse 0. Nöron doymuş; bu girdiden öğrenme çok yavaş.`, en: `|z| is large: the derivative is ${nf(p * (1 - p), 4)}, almost 0. The neuron is saturated; it learns very slowly from this input.` })
          : L({ tr: "Formül ile sayısal türev aynı sonucu veriyor.", en: "The formula and the numerical derivative agree." }) }));
    };
    box.append(h("div", { class: "t-controls" }, slider("z", { min: -8, max: 8, step: 0.1, value: s.z, onInput: (v) => { s.z = v; draw(); } })), out);
    draw();
    return box;
  },

  /** Backpropagation against numerical derivatives on a random 2-3-1 network. */
  gradcheck(state) {
    const s = (state.gradcheck ??= { seed: 4 });
    const box = h("div", { class: "m-widget" });
    const out = h("div", { class: "m-widget-out" });
    const again = h("button", { type: "button", class: "btn small", text: L({ tr: "Yeni rastgele ağ", en: "A new random network" }) });
    again.addEventListener("click", () => { s.seed += 1; draw(); });
    const draw = () => {
      const rand = seeded(s.seed);
      const net = NET.create([2, 3, 1], { act: "tanh", out: "sigmoid", rand });
      net.layers.forEach((L_) => { L_.b = L_.b.map(() => rand() - 0.5); });
      const batch = [{ x: [rand() * 2 - 1, rand() * 2 - 1], y: 1 }, { x: [rand() * 2 - 1, rand() * 2 - 1], y: 0 }];
      const g = NET.gradient(net, batch);
      const n = NET.numericGradient(net, batch);
      const rows = [];
      [["W^{(1)}_{11}", 0, "W", 0, 0], ["W^{(1)}_{32}", 0, "W", 2, 1], ["b^{(1)}_{2}", 0, "b", 1], ["W^{(2)}_{12}", 1, "W", 0, 1], ["b^{(2)}", 1, "b", 0]]
        .forEach(([name, l, kind, j, i]) => {
          const a = kind === "W" ? g[l].W[j][i] : g[l].b[j];
          const b = kind === "W" ? n[l].W[j][i] : n[l].b[j];
          rows.push(h("tr", {}, h("td", { text: `\\(${name}\\)` }), h("td", { class: "num", text: a.toFixed(8) }), h("td", { class: "num", text: b.toFixed(8) }),
            h("td", { class: "num", text: Math.abs(a - b).toExponential(1) })));
        });
      out.replaceChildren(h("div", { class: "table-wrap" }, h("table", { class: "pc-table" },
        h("thead", {}, h("tr", {}, ...[L({ tr: "Parametre", en: "Parameter" }), L({ tr: "Geri yayılım", en: "Backpropagation" }), L({ tr: "Sayısal", en: "Numerical" }), L({ tr: "Fark", en: "Difference" })].map((t) => h("th", { text: t })))),
        h("tbody", {}, ...rows))),
      h("p", { class: "m-why", text: L({ tr: "Fark 10⁻¹⁰'dan küçük: geri yayılım doğru. Sayısal türev her parametre için iki ileri hesap ister; geri yayılım hepsini bir geri hesapta bulur.", en: "The difference is below 10⁻¹⁰: backpropagation is right. The numerical derivative takes two forward passes per parameter; backpropagation finds them all in one backward pass." }) }));
    };
    box.append(h("div", { class: "t-controls" }, again), out);
    draw();
    return box;
  },

  /** Two sigmoids make a bump: σ(k(x − a)) − σ(k(x − b)). */
  bump(state) {
    const s = (state.bump ??= { a: -0.3, b: 0.4, k: 12 });
    const box = h("div", { class: "m-widget" });
    const stage = h("div", { class: "pc-stage" });
    const graph = createGraph(stage, { x: [-1, 1], y: [-0.2, 1.2], xTicks: [-1, -0.5, 0, 0.5, 1], yTicks: [0, 0.5, 1], xName: "x", ratio: 2.2,
      label: L({ tr: "İki sigmoidin farkı bir tepe yapar", en: "The difference of two sigmoids makes a bump" }) });
    const draw = () => {
      const lo = Math.min(s.a, s.b);
      const hi = Math.max(s.a, s.b);
      graph.draw({ curves: [
        { f: (x) => sigma(s.k * (x - lo)), color: COLORS.muted, width: 1.4, dash: true },
        { f: (x) => sigma(s.k * (x - hi)), color: COLORS.line, width: 1.4, dash: true },
        { f: (x) => sigma(s.k * (x - lo)) - sigma(s.k * (x - hi)), color: COLORS.lineB, width: 3, fill: "rgba(42,120,214,0.12)" },
      ] });
    };
    box.append(h("div", { class: "t-controls" },
      slider("a", { min: -0.9, max: 0.9, step: 0.05, value: s.a, onInput: (v) => { s.a = v; draw(); } }),
      slider("b", { min: -0.9, max: 0.9, step: 0.05, value: s.b, onInput: (v) => { s.b = v; draw(); } }),
      slider("k", { min: 1, max: 60, step: 1, value: s.k, fmt: (v) => fmtNum(v, 0), onInput: (v) => { s.k = v; draw(); } })), stage);
    redrawers.add(draw);
    draw();
    return box;
  },
};

// =====================================================================
// Math tab
// =====================================================================
const MATH = [
  {
    id: "mm-forward",
    title: PLACE_TITLES["mm-forward"],
    blocks: [
      { p: {
        tr: R`Ağın \(L\) katmanı olsun; \(l\). katmanda \(n_l\) nöron var, \(n_0\) girdi sayısı. Her katmanın bir ağırlık matrisi ve bir bias vektörü vardır. \(\mathbf W^{(l)}\)'nin \(j\). satırı, \(l\). katmandaki \(j\). nöronun ağırlıklarıdır: her nöron, 1. bölümdeki perceptron gibi, kendi satırıyla bir iç çarpım yapar.`,
        en: R`Let the network have \(L\) layers; layer \(l\) has \(n_l\) neurons, and \(n_0\) is the number of inputs. Every layer has a weight matrix and a bias vector. Row \(j\) of \(\mathbf W^{(l)}\) holds the weights of neuron \(j\) in layer \(l\): every neuron takes a dot product with its own row, like the perceptron of chapter 1.`,
      } },
      { table: {
        head: [{ tr: "Sembol", en: "Symbol" }, { tr: "Boyut", en: "Shape" }, { tr: "Anlamı", en: "Meaning" }],
        rows: [
          [R`\(\mathbf a^{(0)} = \mathbf x\)`, R`\(n_0\)`, { tr: "girdi", en: "the input" }],
          [R`\(\mathbf W^{(l)}\)`, R`\(n_l \times n_{l-1}\)`, { tr: "\\(l\\). katmanın [[weight|ağırlıkları]]", en: "the [[weight|weights]] of layer \\(l\\)" }],
          [R`\(\mathbf b^{(l)}\)`, R`\(n_l\)`, { tr: "\\(l\\). katmanın [[bias|biasları]]", en: "the [[bias|biases]] of layer \\(l\\)" }],
          [R`\(\mathbf z^{(l)}\)`, R`\(n_l\)`, { tr: "ağırlıklı toplamlar (aktivasyondan önce)", en: "the weighted sums (before the activation)" }],
          [R`\(\mathbf a^{(l)} = f(\mathbf z^{(l)})\)`, R`\(n_l\)`, { tr: "katmanın çıktısı; \\(f\\) her elemana ayrı uygulanır", en: "the layer's output; \\(f\\) is applied to each element" }],
          [R`\(\hat y = a^{(L)}\)`, R`\(1\)`, { tr: "ağın tahmini (sigmoid çıktı)", en: "the network's prediction (a sigmoid output)" }],
        ],
      } },
      { eq: R`\mathbf z^{(l)} = \mathbf W^{(l)}\,\mathbf a^{(l-1)} + \mathbf b^{(l)}, \qquad \mathbf a^{(l)} = f\big(\mathbf z^{(l)}\big), \qquad l = 1, \dots, L` },
      { p: {
        tr: R`<b>Dikkat:</b> \(L\) harfi hem katman sayısı hem kayıp için kullanılır; üst indiste \((L)\) son katman, \(\partial L\)'de kayıp demektir. Teori sekmesindeki 2-2-1 ağda gizli ağırlıklar \(w_{ji} = W^{(1)}_{ji}\), çıktı ağırlıkları \(v_j = W^{(2)}_{1j}\), çıktı biası \(c = b^{(2)}\), çıktının toplamı \(z_o = z^{(2)}\) ve hatası \(\delta_o = \delta^{(2)}\)'dir (\(\delta\), 4. bölümde tanımlanır).`,
        en: R`<b>Note:</b> the letter \(L\) stands both for the number of layers and for the loss; as a superscript \((L)\) it is the last layer, in \(\partial L\) it is the loss. In the 2-2-1 network of the Theory tab the hidden weights are \(w_{ji} = W^{(1)}_{ji}\), the output weights \(v_j = W^{(2)}_{1j}\), the output bias \(c = b^{(2)}\), the output's sum \(z_o = z^{(2)}\) and its error \(\delta_o = \delta^{(2)}\) (\(\delta\) is defined in section 4).`,
      } },
      { p: {
        tr: R`[[forward-pass|İleri hesap]] bu iki satırın katman katman tekrarıdır. [[parameter|Parametre]] sayısı her katman için ağırlıklar artı biaslardır:`,
        en: R`The [[forward-pass|forward pass]] repeats these two lines layer by layer. The number of [[parameter|parameters]] is the weights plus the biases of every layer:`,
      } },
      { eq: R`P = \sum_{l=1}^{L} n_l\,(n_{l-1} + 1)` },
      { p: {
        tr: R`Teori 2'deki 2-2-1 ağ: \(2\cdot 3 + 1\cdot 3 = 9\) parametre. Deneme alanında 4'er nöronlu iki gizli katmanla kurulan 2-4-4-1 ağ: \(4\cdot 3 + 4\cdot 5 + 1\cdot 5 = 37\). SimpleCNN'in sonundaki MLP (1024 → 512 → 10): \(512\cdot 1025 + 10\cdot 513 = 529\,930\).`,
        en: R`The 2-2-1 network of Theory 2: \(2\cdot 3 + 1\cdot 3 = 9\) parameters. The 2-4-4-1 network you get in the Playground with two hidden layers of 4:\(4\cdot 3 + 4\cdot 5 + 1\cdot 5 = 37\). The MLP at the end of SimpleCNN (1024 → 512 → 10): \(512\cdot 1025 + 10\cdot 513 = 529{,}930\).`,
      } },
      { derive: {
        title: { tr: "Neden doğrusal olmayan bir aktivasyon gerekir?", en: "Why a nonlinear activation is needed" },
        steps: [
          { tex: R`\mathbf a^{(2)} = \mathbf W^{(2)}\big(\mathbf W^{(1)}\mathbf x + \mathbf b^{(1)}\big) + \mathbf b^{(2)}`,
            why: { tr: R`Aktivasyon olmasaydı (\(f(z) = z\)) iki katmanlı bir ağ böyle hesaplardı.`, en: R`Without an activation (\(f(z) = z\)) a two-layer network would compute this.` } },
          { tex: R`= \big(\mathbf W^{(2)}\mathbf W^{(1)}\big)\,\mathbf x + \big(\mathbf W^{(2)}\mathbf b^{(1)} + \mathbf b^{(2)}\big)`,
            why: { tr: "Matris çarpımı dağılır ve birleşir.", en: "Matrix multiplication distributes and associates." } },
          { tex: R`= \mathbf W'\mathbf x + \mathbf b'`,
            why: { tr: R`Sonuç yine tek bir doğrusal katman: kaç katman eklersek ekleyelim ağ bir doğrudan (hiper düzlemden) başka sınır çizemez ve XOR'u çözemez. Katmanları anlamlı kılan, aradaki doğrusal olmayan \(f\)'dir.`, en: R`The result is again a single linear layer: however many layers we add, the network cannot draw anything but a line (a hyperplane) and cannot solve XOR. What makes the layers worth having is the nonlinear \(f\) between them.` } },
        ],
      } },
      { links: [{ tab: "theory", section: "ml-hidden" }, { tab: "play" }] },
    ],
  },
  {
    id: "mm-act",
    title: PLACE_TITLES["mm-act"],
    blocks: [
      { p: {
        tr: R`Geri yayılımda her nöronun aktivasyon türevi \(f'(z)\) bir çarpan olarak gelir. Bu yüzden türevlerin hem hesaplanması kolay olmalı hem de çok küçük olmamalı. [[sigmoid|Sigmoid]] ile başlayalım.`,
        en: R`In backpropagation every neuron's activation derivative \(f'(z)\) appears as a factor. So the derivatives should be both easy to compute and not too small. Let us start with the [[sigmoid|sigmoid]].`,
      } },
      { derive: {
        title: { tr: "σ'(z) = σ(z)(1 − σ(z))", en: "σ'(z) = σ(z)(1 − σ(z))" },
        steps: [
          { tex: R`\sigma(z) = \big(1 + e^{-z}\big)^{-1}`, why: { tr: "Tanım.", en: "The definition." } },
          { tex: R`\sigma'(z) = -\big(1 + e^{-z}\big)^{-2}\cdot\big(-e^{-z}\big) = \frac{e^{-z}}{\big(1 + e^{-z}\big)^{2}}`,
            why: { tr: "Zincir kuralı: dışarıdaki kuvvet ve içerideki üstel.", en: "The chain rule: the outer power and the inner exponential." } },
          { tex: R`= \frac{1}{1 + e^{-z}}\cdot\frac{e^{-z}}{1 + e^{-z}}`, why: { tr: "Kesri ikiye ayırın.", en: "Split the fraction in two." } },
          { tex: R`\frac{e^{-z}}{1 + e^{-z}} = \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = 1 - \sigma(z)`, why: { tr: "İkinci çarpan 1 − σ'dır.", en: "The second factor is 1 − σ." } },
          { tex: R`\sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big) \le \tfrac14`,
            why: { tr: R`İleri hesapta zaten bulunan \(\sigma(z)\)'den türev bedavaya gelir. \(p(1-p)\) en büyük değerini \(p = 1/2\)'de (\(z = 0\)) alır: \(1/4\).`, en: R`The derivative comes for free from \(\sigma(z)\), which the forward pass already computed. \(p(1-p)\) is largest at \(p = 1/2\) (\(z = 0\)): \(1/4\).` } },
        ],
      } },
      { widget: "sigmoid" },
      { table: {
        head: [{ tr: "Aktivasyon", en: "Activation" }, R`\(f(z)\)`, R`\(f'(z)\)`, { tr: "Not", en: "Note" }],
        rows: [
          [{ tr: "[[sigmoid|Sigmoid]]", en: "[[sigmoid|Sigmoid]]" }, R`\(\frac{1}{1+e^{-z}}\)`, R`\(\sigma(1-\sigma) \le \frac14\)`, { tr: "çıktı 0..1: olasılık", en: "output 0..1: a probability" }],
          [{ tr: "[[tanh|tanh]]", en: "[[tanh|tanh]]" }, R`\(2\sigma(2z) - 1\)`, R`\(1 - \tanh^2 z \le 1\)`, { tr: "sıfır merkezli", en: "centred on zero" }],
          [{ tr: "[[relu|ReLU]]", en: "[[relu|ReLU]]" }, R`\(\max(0, z)\)`, R`\(1\ (z > 0),\ 0\ (z < 0)\)`, { tr: "doymaz (pozitif tarafta)", en: "does not saturate (on the positive side)" }],
        ],
      } },
      { h: { tr: "Kaybolan gradyan", en: "The vanishing gradient" } },
      { p: {
        tr: R`4. bölümde göreceğimiz gibi, bir hata sinyali her katmandan geçerken \(f'(z)\) ve bir ağırlıkla çarpılır. Sigmoid'de bu çarpan en fazla \(1/4\)'tür; ağırlıklar büyük değilse sinyal her katmanda en az 4 kat küçülür: 10 katmandan sonra türev çarpanlarının çarpımı en fazla \((1/4)^{10} \approx 10^{-6}\). İlk katmanlar neredeyse hiç öğrenmez: [[vanishing-gradient|kaybolan gradyan]]. ReLU'nun pozitif taraftaki eğimi 1 olduğu için bu sorunu büyük ölçüde ortadan kaldırır; SimpleCNN de ReLU kullanır.`,
        en: R`As we will see in section 4, an error signal is multiplied by \(f'(z)\) and a weight at every layer it passes. With the sigmoid that factor is at most \(1/4\); unless the weights are large, the signal shrinks at least 4-fold per layer: after 10 layers the product of the derivative factors is at most \((1/4)^{10} \approx 10^{-6}\). The first layers barely learn: the [[vanishing-gradient|vanishing gradient]]. ReLU's slope is 1 on the positive side, which largely removes the problem; SimpleCNN uses ReLU too.`,
      } },
      { links: [{ tab: "theory", section: "ml-act" }, { chapter: "cnn", tab: "math", section: "m-act", title: { tr: "Aktivasyon fonksiyonları ve türevleri", en: "Activation functions and their derivatives" } }] },
    ],
  },
  {
    id: "mm-loss",
    title: PLACE_TITLES["mm-loss"],
    blocks: [
      { p: {
        tr: R`Sigmoid çıktıyı "\(y = 1\) olasılığı" olarak okuyalım: \(P(y = 1 \mid \mathbf x) = \hat y\). Tek bir örneğin olasılığı \(y\) 0 ya da 1 iken tek bir ifadeyle yazılabilir; en iyi ağırlıklar, gözlenen etiketlere en yüksek olasılığı verenlerdir (en çok olabilirlik).`,
        en: R`Read the sigmoid output as "the probability that \(y = 1\)": \(P(y = 1 \mid \mathbf x) = \hat y\). The probability of one example, with \(y\) either 0 or 1, can be written as a single expression; the best weights are those that give the observed labels the highest probability (maximum likelihood).`,
      } },
      { eq: R`P(y \mid \mathbf x) = \hat y^{\,y}\,(1 - \hat y)^{1 - y} \quad\Longrightarrow\quad L = -\ln P = -\big[\,y \ln \hat y + (1 - y)\ln(1 - \hat y)\,\big]` },
      { p: {
        tr: R`Olasılığı büyütmek, eksi logaritmasını küçültmekle aynı şeydir: [[cross-entropy|ikili çapraz entropi]] buradan gelir. Örnekler bağımsızsa çarpımların logaritması toplam olur ve eğitimdeki kayıp örneklerin ortalamasıdır. Şimdi çıktının toplamına göre türevini alalım.`,
        en: R`Making the probability larger is the same as making its negative logarithm smaller: that is where the [[cross-entropy|binary cross-entropy]] comes from. If the examples are independent, the log of the product becomes a sum, and the training loss is the average over the examples. Now take its derivative with respect to the output's sum.`,
      } },
      { derive: {
        title: { tr: "∂L/∂z = ŷ − y", en: "∂L/∂z = ŷ − y" },
        steps: [
          { tex: R`\frac{\partial L}{\partial \hat y} = -\frac{y}{\hat y} + \frac{1 - y}{1 - \hat y}`, why: { tr: R`\(\ln\)'nin türevi \(1/u\).`, en: R`The derivative of \(\ln u\) is \(1/u\).` } },
          { tex: R`\frac{\partial \hat y}{\partial z} = \hat y\,(1 - \hat y)`, why: { tr: "Önceki kısımdan: sigmoid'in türevi.", en: "From the previous section: the sigmoid's derivative." } },
          { tex: R`\frac{\partial L}{\partial z} = -y\,(1 - \hat y) + (1 - y)\,\hat y`, why: { tr: "Zincir kuralı: iki türevi çarpın; paydalar sadeleşir.", en: "The chain rule: multiply the two derivatives; the denominators cancel." } },
          { tex: R`= \hat y - y`, why: { tr: R`Açıp toplayınca \(-y + y\hat y + \hat y - y\hat y\) kalır. Çıktıdaki hata sinyali, tahmin ile etiketin farkıdır; ne kadar yanlışsa o kadar büyük.`, en: R`Expanding gives \(-y + y\hat y + \hat y - y\hat y\). The error signal at the output is the difference between the prediction and the label; the more wrong, the larger.` } },
        ],
      } },
      { h: { tr: "MSE ile karşılaştırma", en: "Compared with MSE" } },
      { eq: R`L_{\text{MSE}} = \tfrac12(\hat y - y)^2 \quad\Longrightarrow\quad \frac{\partial L_{\text{MSE}}}{\partial z} = (\hat y - y)\,\hat y\,(1 - \hat y)` },
      { p: {
        tr: R`Fazladan bir \(\hat y(1 - \hat y)\) çarpanı var. Model emin ve yanlışsa (\(y = 1\), \(\hat y \approx 0\)) bu çarpan neredeyse 0'dır: en çok düzeltilmesi gereken anda gradyan en küçüktür. Çapraz entropide gradyan bu durumda \(\approx -1\), yani tam güçtedir. Teori 3'teki MSE eğrisinin solundaki düzlük budur.`,
        en: R`There is an extra factor \(\hat y(1 - \hat y)\). If the model is confident and wrong (\(y = 1\), \(\hat y \approx 0\)), this factor is almost 0: the gradient is smallest exactly when the most correction is needed. With cross-entropy the gradient is then \(\approx -1\), at full strength. That is the flat stretch on the left of the MSE curve in Theory 3.`,
      } },
      { callout: "info", title: { tr: "Çok sınıflı durum", en: "Many classes" },
        text: { tr: R`SimpleCNN'in 10 sınıfı var: sigmoid yerine [[softmax|softmax]], ikili yerine çok sınıflı çapraz entropi. Türev aynı biçime iner: \(\partial L/\partial z_k = p_k - y_k\).`,
          en: R`SimpleCNN has 10 classes: [[softmax|softmax]] instead of the sigmoid, and the multiclass cross-entropy. The derivative comes out in the same form: \(\partial L/\partial z_k = p_k - y_k\).` } },
      { links: [{ tab: "theory", section: "ml-loss" }, { chapter: "cnn", tab: "math", section: "m-softmax", title: { tr: "Softmax ve çapraz entropi", en: "Softmax and cross-entropy" } }] },
    ],
  },
  {
    id: "mm-backprop",
    title: PLACE_TITLES["mm-backprop"],
    blocks: [
      { p: {
        tr: R`Her katman için hata sinyalini tanımlayalım: \(\boldsymbol\delta^{(l)} = \partial L / \partial \mathbf z^{(l)}\). Bütün gradyanlar bu \(\delta\)'lardan çıkar ve \(\delta\)'lar sondan başa doğru, birbirinden hesaplanır. Aşağıda \(\odot\) eleman eleman çarpımdır.`,
        en: R`Define the error signal of every layer: \(\boldsymbol\delta^{(l)} = \partial L / \partial \mathbf z^{(l)}\). All the gradients follow from these \(\delta\)s, and the \(\delta\)s are computed from one another, from the end to the start. Below, \(\odot\) is element-wise multiplication.`,
      } },
      { derive: {
        title: { tr: "Dört denklem", en: "The four equations" },
        steps: [
          { tex: R`\textbf{(1)}\quad \boldsymbol\delta^{(L)} = \hat y - y`,
            why: { tr: "Çıktı katmanı: önceki bölümün sonucu (sigmoid + çapraz entropi).", en: "The output layer: the result of the previous section (sigmoid + cross-entropy)." } },
          { tex: R`z^{(l+1)}_k = \sum_j W^{(l+1)}_{kj}\, f\big(z^{(l)}_j\big) + b^{(l+1)}_k`,
            why: { tr: R`\(z^{(l)}_j\) kaybı yalnızca sonraki katmanın toplamları \(z^{(l+1)}_k\) üzerinden etkiler; her birine \(W_{kj} f'(z_j)\) hızıyla.`, en: R`\(z^{(l)}_j\) affects the loss only through the next layer's sums \(z^{(l+1)}_k\), each at the rate \(W_{kj} f'(z_j)\).` } },
          { tex: R`\textbf{(2)}\quad \delta^{(l)}_j = \sum_k \delta^{(l+1)}_k\, W^{(l+1)}_{kj}\, f'\big(z^{(l)}_j\big) \quad\Longleftrightarrow\quad \boldsymbol\delta^{(l)} = \big(\mathbf W^{(l+1)}\big)^{\!\top}\boldsymbol\delta^{(l+1)} \odot f'\big(\mathbf z^{(l)}\big)`,
            why: { tr: R`Çok değişkenli [[chain-rule|zincir kuralı]]: bütün yollar üzerinden toplam. Hata, ileri hesaptaki ağırlıkların transpozuyla geriye taşınır ve her nöronda eğimle çarpılır.`, en: R`The multivariable [[chain-rule|chain rule]]: a sum over all the paths. The error is carried back by the transpose of the forward weights and multiplied by the slope at every neuron.` } },
          { tex: R`\textbf{(3)}\quad \frac{\partial L}{\partial W^{(l)}_{ji}} = \delta^{(l)}_j\, a^{(l-1)}_i \quad\Longleftrightarrow\quad \frac{\partial L}{\partial \mathbf W^{(l)}} = \boldsymbol\delta^{(l)}\big(\mathbf a^{(l-1)}\big)^{\!\top}`,
            why: { tr: R`\(z^{(l)}_j\)'nin \(W_{ji}\)'ye göre türevi \(a^{(l-1)}_i\)'dir: ağırlığın gradyanı = girdisi × hedef nöronun hatası.`, en: R`The derivative of \(z^{(l)}_j\) with respect to \(W_{ji}\) is \(a^{(l-1)}_i\): a weight's gradient = its input × the error of the neuron it feeds.` } },
          { tex: R`\textbf{(4)}\quad \frac{\partial L}{\partial \mathbf b^{(l)}} = \boldsymbol\delta^{(l)}`,
            why: { tr: R`Bias, girdisi hep 1 olan bir ağırlıktır.`, en: R`A bias is a weight whose input is always 1.` } },
        ],
      } },
      { list: [
        { tr: R`<b>İleri:</b> \(l = 1, \dots, L\) için \(\mathbf z^{(l)}\) ve \(\mathbf a^{(l)}\)'yi hesapla ve sakla.`, en: R`<b>Forward:</b> for \(l = 1, \dots, L\), compute and keep \(\mathbf z^{(l)}\) and \(\mathbf a^{(l)}\).` },
        { tr: R`<b>Çıktı:</b> \(\boldsymbol\delta^{(L)} = \hat y - y\) (1).`, en: R`<b>Output:</b> \(\boldsymbol\delta^{(L)} = \hat y - y\) (1).` },
        { tr: R`<b>Geri:</b> \(l = L-1, \dots, 1\) için \(\boldsymbol\delta^{(l)}\)'yi (2) ile bul.`, en: R`<b>Backward:</b> for \(l = L-1, \dots, 1\), find \(\boldsymbol\delta^{(l)}\) with (2).` },
        { tr: R`<b>Gradyanlar:</b> (3) ve (4); sonra her parametreyi \(\theta \leftarrow \theta - \eta\,\partial L/\partial\theta\) ile güncelle.`, en: R`<b>Gradients:</b> (3) and (4); then update every parameter with \(\theta \leftarrow \theta - \eta\,\partial L/\partial\theta\).` },
      ], ordered: true },
      { p: {
        tr: R`<b>Maliyet:</b> ileri hesap her ağırlık için bir çarpma yapar; geri hesap her ağırlık için iki (biri (2)'de, biri (3)'te). Yani geri hesap, parametre sayısı ne olursa olsun, yaklaşık iki ileri hesap tutarındadır ve bütün gradyanları birden verir. Sayısal türevle her parametre için ayrı iki ileri hesap gerekirdi. Aşağıda ikisini karşılaştırın; [[backprop|geri yayılımı]] kodlarken bu kontrolü yapmak iyi bir alışkanlıktır.`,
        en: R`<b>Cost:</b> the forward pass does one multiplication per weight; the backward pass does two per weight (one in (2), one in (3)). So the backward pass costs about two forward passes, whatever the number of parameters, and gives all the gradients at once. With numerical derivatives it would take two forward passes for each parameter. Compare the two below; checking this is a good habit when coding [[backprop|backpropagation]].`,
      } },
      { widget: "gradcheck" },
      { links: [{ tab: "theory", section: "ml-backprop" }, { tab: "code", section: "mc-backward" },
        { chapter: "cnn", tab: "math", section: "m-backprop", title: { tr: "Geri yayılım: hepsini birleştirmek", en: "Backpropagation: putting it together" } }] },
    ],
  },
  {
    id: "mm-train",
    title: PLACE_TITLES["mm-train"],
    blocks: [
      { p: {
        tr: R`Eğitim kaybı bütün örneklerin ortalamasıdır, gradyanı da örnek gradyanlarının ortalaması. Her adımda bütün veriyi kullanmak (tam [[batch|batch]]) yavaştır; rastgele seçilmiş küçük bir grup, bir mini-batch, gerçek gradyanın gürültülü ama yansız bir tahminini verir:`,
        en: R`The training loss is the average over all the examples, and its gradient is the average of the examples' gradients. Using all the data for every step (a full [[batch|batch]]) is slow; a small randomly chosen group, a mini-batch, gives a noisy but unbiased estimate of the true gradient:`,
      } },
      { eq: R`\theta \leftarrow \theta - \eta\,\frac{1}{B}\sum_{i \in \text{batch}} \nabla_\theta L_i` },
      { p: {
        tr: R`Bütün veri bir kez geçilince bir [[epoch|epoch]] tamamlanır. Bu [[sgd|stokastik gradyan inişidir]]; SimpleCNN aynı fikri Adam ile ve 256'lık batch'lerle kullanır.`,
        en: R`When all the data has been used once, an [[epoch|epoch]] is complete. This is [[sgd|stochastic gradient descent]]; SimpleCNN uses the same idea with Adam and batches of 256.`,
      } },
      { h: { tr: "Neden ağırlıklar sıfırdan başlamaz?", en: "Why the weights do not start at zero" } },
      { derive: {
        title: { tr: "Simetri hiç bozulmaz", en: "The symmetry never breaks" },
        steps: [
          { tex: R`\mathbf W^{(1)} = \mathbf 0,\ \ \mathbf v = (v, v, \dots, v) \;\Rightarrow\; h_1 = h_2 = \dots = h_H`,
            why: { tr: "Bütün gizli nöronlar aynı ağırlıklarla başlarsa her girdide aynı çıktıyı verir.", en: "If every hidden neuron starts with the same weights, it gives the same output for every input." } },
          { tex: R`\delta_j = \delta_o\, v\, f'(z_j) \ \text{ hepsi aynı} \;\Rightarrow\; \frac{\partial L}{\partial \mathbf w_j} \ \text{ hepsi aynı}`,
            why: { tr: R`Denklem (2) ve (3): aynı çıktı, aynı çıktı ağırlığı, aynı hata, aynı gradyan.`, en: R`Equations (2) and (3): the same output, the same output weight, the same error, the same gradient.` } },
          { tex: R`\mathbf w_1 = \mathbf w_2 = \dots = \mathbf w_H \ \text{ her adımdan sonra}`,
            why: { tr: R`Aynı başlayıp aynı güncellenen nöronlar sonsuza kadar kopya kalır: \(H\) nöronlu ağ tek nöronlu gibi davranır. Rastgele başlangıç bu simetriyi bozar.`, en: R`Neurons that start equal and are updated equally stay copies forever: the \(H\)-neuron network acts like a one-neuron network. A random start breaks the symmetry.` } },
        ],
      } },
      { p: {
        tr: R`Rastgele ama hangi ölçekte? \(z = \sum_i w_i x_i\)'nin varyansı, bağımsız ve sıfır ortalamalı terimler için \(n_{\text{in}}\,\mathrm{Var}(w)\,\mathrm{Var}(x)\)'tir. \(\mathrm{Var}(w) = 1/n_{\text{in}}\) seçmek (LeCun başlangıcı; Xavier/Glorot başlangıcı bunun ileri ve geri yönü dengeleyen \(2/(n_{\text{in}} + n_{\text{out}})\) biçimidir) sinyali katmandan katmana aynı büyüklükte tutar; ReLU yarısını sıfırladığı için orada \(2/n_{\text{in}}\) kullanılır (He başlangıcı). Çok büyük ağırlıklar sigmoid ve tanh'ı doyurur (gradyan kaybolur), çok küçükler sinyali söndürür. Bu bölümdeki bütün kodlar bu kuralı kullanır.`,
        en: R`Random, but at what scale? For independent zero-mean terms, the variance of \(z = \sum_i w_i x_i\) is \(n_{\text{in}}\,\mathrm{Var}(w)\,\mathrm{Var}(x)\). Choosing \(\mathrm{Var}(w) = 1/n_{\text{in}}\) (LeCun initialization; Xavier/Glorot initialization is its variant \(2/(n_{\text{in}} + n_{\text{out}})\), which balances the forward and backward directions) keeps the signal the same size from layer to layer; ReLU zeroes half of it, so there \(2/n_{\text{in}}\) is used (He initialization). Weights that are too large saturate the sigmoid and tanh (the gradient vanishes); weights that are too small fade the signal. All the code in this chapter uses this rule.`,
      } },
      { links: [{ tab: "play" }, { tab: "code", section: "mc-class" }, { chapter: "cnn", tab: "math", section: "m-opt", title: { tr: "Optimizasyon: SGD, Momentum, Adam", en: "Optimization: SGD, Momentum, Adam" } }] },
    ],
  },
  {
    id: "mm-universal",
    title: PLACE_TITLES["mm-universal"],
    blocks: [
      { callout: "info", title: { tr: "Teorem (Cybenko 1989, Hornik 1991)", en: "Theorem (Cybenko 1989, Hornik 1991)" },
        text: { tr: R`\(f\), \([0,1]^n\) üzerinde sürekli bir fonksiyon ve \(\varepsilon > 0\) olsun. Öyle bir \(H\) ve \(v_j, \mathbf w_j, b_j\) vardır ki \(F(\mathbf x) = \sum_{j=1}^{H} v_j\,\sigma(\mathbf w_j\cdot\mathbf x + b_j)\) her \(\mathbf x\) için \(|F(\mathbf x) - f(\mathbf x)| < \varepsilon\) sağlar. Cybenko sigmoid için ispatladı; Hornik sürekli, sınırlı ve sabit olmayan her aktivasyon için, Leshno ve arkadaşları (1993) polinom olmayan her aktivasyon için (ReLU dahil) gösterdi.`,
          en: R`Let \(f\) be a continuous function on \([0,1]^n\) and \(\varepsilon > 0\). There are \(H\) and \(v_j, \mathbf w_j, b_j\) such that \(F(\mathbf x) = \sum_{j=1}^{H} v_j\,\sigma(\mathbf w_j\cdot\mathbf x + b_j)\) satisfies \(|F(\mathbf x) - f(\mathbf x)| < \varepsilon\) for every \(\mathbf x\). Cybenko proved it for the sigmoid; Hornik showed it for every continuous, bounded, non-constant activation, and Leshno and colleagues (1993) for every non-polynomial activation (ReLU included).` } },
      { p: {
        tr: R`Bir boyutta neden doğru olduğunu görmek kolay. Ana fikir: iki sigmoid bir "tepe" yapar, tepelerle de istediğimiz şekli kurarız.`,
        en: R`In one dimension it is easy to see why it is true. The main idea: two sigmoids make a "bump", and with bumps we can build any shape.`,
      } },
      { derive: {
        title: { tr: "Tepelerle yaklaşmak", en: "Approximating with bumps" },
        steps: [
          { tex: R`\sigma\big(k(x - a)\big) \xrightarrow{\;k \to \infty\;} \begin{cases} 1 & x > a \\ 0 & x < a \end{cases}`,
            why: { tr: R`Dikliği büyüyen bir sigmoid, \(a\)'da bir basamağa yaklaşır (Teori 1).`, en: R`A sigmoid of growing steepness approaches a step at \(a\) (Theory 1).` } },
          { tex: { tr: R`\sigma\big(k(x - a)\big) - \sigma\big(k(x - b)\big) \approx \begin{cases} 1 & a < x < b \\ 0 & \text{başka yerde} \end{cases}`,
            en: R`\sigma\big(k(x - a)\big) - \sigma\big(k(x - b)\big) \approx \begin{cases} 1 & a < x < b \\ 0 & \text{elsewhere} \end{cases}` },
            why: { tr: R`İki basamağın farkı \([a, b]\) aralığında 1 olan bir tepedir: iki gizli nöron.`, en: R`The difference of two steps is a bump that is 1 on \([a, b]\): two hidden neurons.` } },
          { tex: R`F(x) = \sum_{i=1}^{N} f(c_i)\,\Big[\sigma\big(k(x - a_{i})\big) - \sigma\big(k(x - a_{i+1})\big)\Big]`,
            why: { tr: R`\([0, 1]\) aralığını \(N\) eşit parçaya \([a_i, a_{i+1}]\) bölün ve her parçaya, ortasındaki değer \(f(c_i)\) yüksekliğinde bir tepe koyun: \(2N\) nöronluk bir merdiven.`, en: R`Split the interval \([0, 1]\) into \(N\) equal pieces \([a_i, a_{i+1}]\) and put a bump of height \(f(c_i)\), the value at its middle, on each piece: a staircase of \(2N\) neurons.` } },
          { tex: R`|F(x) - f(x)| \le \max_{|x - x'| \le 1/N} |f(x) - f(x')| \xrightarrow{\;N \to \infty\;} 0`,
            why: { tr: R`\(f\) sürekli olduğu için (kapalı aralıkta düzgün sürekli) parçalar inceldikçe merdiven \(f\)'ye istediğimiz kadar yaklaşır (\(k\) yeterince büyük seçilirse).`, en: R`Since \(f\) is continuous (uniformly, on a closed interval), the staircase gets as close to \(f\) as we like as the pieces get thinner (with \(k\) large enough).` } },
        ],
      } },
      { widget: "bump" },
      { p: {
        tr: R`<b>Teorem ne söylemez?</b> Gereken \(H\)'yi söylemez; bu yapı boyut sayısıyla üstel olarak büyür. Gradyan inişinin o ağırlıkları bulacağını da söylemez, eğitim noktalarına uymanın yeni noktalarda iyi sonuç vereceğini de. Derinliğin faydası ayrıca kanıtlanmıştır: bazı fonksiyonları sığ bir ağ ancak üstel sayıda nöronla yapabilirken derin bir ağ az nöronla yapar (Telgarsky 2016).`,
        en: R`<b>What the theorem does not say:</b> it does not give the \(H\) needed; this construction grows exponentially with the number of dimensions. Nor does it say that gradient descent will find those weights, or that fitting the training points will give good results on new points. The benefit of depth has been proven separately: some functions take a shallow network exponentially many neurons, while a deep network needs few (Telgarsky 2016).`,
      } },
      { links: [{ tab: "theory", section: "ml-capacity" }] },
    ],
  },
];

// =====================================================================
// Coding tab
// =====================================================================
const TORCH_CODE = `import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

model = nn.Sequential(
    nn.Linear(2, 8),    # W1, b1
    nn.Tanh(),
    nn.Linear(8, 1),    # W2, b2
)
loss_fn = nn.BCEWithLogitsLoss()    # sigmoid + binary cross-entropy in one step
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(1, 2001):
    logits = model(X)               # forward pass
    loss = loss_fn(logits, y)
    optimizer.zero_grad()           # clear the old gradients
    loss.backward()                 # backpropagation: fills p.grad for every parameter
    optimizer.step()                # p = p - lr * p.grad
    if epoch % 500 == 0:
        print(f"epoch {epoch}   loss {loss.item():.4f}")

print((torch.sigmoid(model(X)) > 0.5).int().flatten().tolist())`;

const RUN_OUTPUT = [
  "$ .venv\\Scripts\\python.exe MLP.py",
  "Gradient check: largest difference 9.1e-12",
  "",
  "XOR",
  "  epoch   500   loss 0.0112   accuracy 100%",
  "  epoch  1000   loss 0.0042   accuracy 100%",
  "  epoch  1500   loss 0.0025   accuracy 100%",
  "  epoch  2000   loss 0.0017   accuracy 100%",
  "  predictions: [0 1 1 0]   labels: [0 1 1 0]",
  "",
  "Two moons",
  "  epoch   100   loss 0.1223   accuracy 95%",
  "  epoch   200   loss 0.1215   accuracy 96%",
  "  epoch   300   loss 0.1092   accuracy 96%",
  "  test accuracy: 90.5%",
];

const CODE = [
  {
    id: "mc-plan",
    title: PLACE_TITLES["mc-plan"],
    blocks: [
      { p: {
        tr: R`Bu bölümde, depo kökündeki <code>MLP.py</code>'yi adım adım okuyacağız: tek gizli katmanlı (tanh) ve sigmoid çıktılı bir ağ, çapraz entropi kaybı, mini-batch gradyan inişi ve elle yazılmış geri yayılım. Hepsi NumPy ile, yaklaşık 150 satır. Program üç şey yapar: geri yayılımı sayısal türevle kontrol eder, XOR'u öğrenir ve "iki hilal" verisinde eğri bir sınır öğrenir.`,
        en: R`In this part we read <code>MLP.py</code> at the root of the repository step by step: a network with one hidden layer (tanh) and a sigmoid output, the cross-entropy loss, mini-batch gradient descent and hand-written backpropagation. All in NumPy, about 150 lines. The program does three things: it checks backpropagation against numerical derivatives, learns XOR and learns a curved boundary on the "two moons" data.`,
      } },
      { table: {
        head: [{ tr: "Kodda", en: "In the code" }, { tr: "Şekil", en: "Shape" }, { tr: "Matematikte", en: "In the math" }],
        rows: [
          [R`<code>X</code>`, R`(n, 2)`, { tr: "satır satır \\(n\\) örnek", en: "\\(n\\) examples, one per row" }],
          [R`<code>W1</code>, <code>b1</code>`, R`(2, 8), (8,)`, R`\((\mathbf W^{(1)})^\top,\ \mathbf b^{(1)}\)`],
          [R`<code>H</code>`, R`(n, 8)`, { tr: "her örneğin gizli katmanı \\(\\mathbf a^{(1)}\\)", en: "every example's hidden layer \\(\\mathbf a^{(1)}\\)" }],
          [R`<code>W2</code>, <code>b2</code>`, R`(8, 1), (1,)`, R`\((\mathbf W^{(2)})^\top,\ \mathbf b^{(2)}\)`],
          [R`<code>Y_hat</code>`, R`(n, 1)`, R`\(\hat y\)`],
        ],
      } },
      { callout: "tip", title: { tr: "Neden transpoze?", en: "Why transposed?" },
        text: { tr: R`Matematikte bir örnek bir sütun vektörüdür ve \(\mathbf W\mathbf x\) yazarız. Kodda örnekler <code>X</code>'in satırlarıdır, böylece bütün batch tek çarpımla hesaplanır: <code>X @ W1</code> = \((\mathbf W^{(1)}\mathbf X^\top)^\top\). PyTorch'un <code>nn.Linear</code>'ı da girdileri satır olarak alır.`,
          en: R`In the math an example is a column vector and we write \(\mathbf W\mathbf x\). In the code the examples are the rows of <code>X</code>, so a whole batch is one product: <code>X @ W1</code> = \((\mathbf W^{(1)}\mathbf X^\top)^\top\). PyTorch's <code>nn.Linear</code> also takes its inputs as rows.` } },
    ],
  },
  {
    id: "mc-data",
    title: PLACE_TITLES["mc-data"],
    blocks: [
      { p: {
        tr: R`Ayarlar en üstte. XOR'un dört noktası sabit; "iki hilal" ise rastgele üretilir: iki iç içe yarım daire, üzerine gürültü. Tek bir doğru bu iki sınıfı ayıramaz.`,
        en: R`The settings are at the top. XOR's four points are fixed; "two moons" is generated at random: two interleaving half circles with noise on top. No single line can separate the two classes.`,
      } },
      { file: { from: "#--------------- Hyperparameters", to: "return X, labels.astype(float)", notes: [
        { at: "hidden_units = 8", text: { tr: "Gizli katmandaki nöron sayısı. 2 de XOR için yeter (Teori 2), ama bazen takılır; 8 daha güvenli.", en: "The number of neurons in the hidden layer. 2 is enough for XOR (Theory 2) but sometimes gets stuck; 8 is safer." } },
        { at: "learning_rate = 0.5", text: { tr: "Öğrenme oranı η. Bu küçük ağ için büyük bir değer işe yarıyor.", en: "The learning rate η. A large value works for this small network." } },
        { at: "rng = np.random.default_rng(seed)", text: { tr: "Bütün rastgelelik (ağırlıklar, veri, karıştırma) tek bir üreticiden: aynı seed, aynı sonuç.", en: "All randomness (weights, data, shuffling) comes from one generator: the same seed, the same result." } },
        { at: "def make_moons(n, noise=0.25):", text: { tr: "Gürültü 0,25: hilaller biraz iç içe geçer; hiçbir sınır yeni noktalarda %100 doğru olamaz.", en: "Noise 0.25: the moons overlap a little; no boundary can be 100% right on new points." } },
      ] } },
    ],
  },
  {
    id: "mc-act",
    title: PLACE_TITLES["mc-act"],
    blocks: [
      { p: {
        tr: R`Çıktı için sigmoid, gizli katman için tanh. Geri yayılımda tanh'ın türevi gerekir; onu \(z\) ile değil, ileri hesapta zaten bulunan çıktı \(a = \tanh z\) ile yazıyoruz: \(1 - a^2\).`,
        en: R`The sigmoid for the output, tanh for the hidden layer. Backpropagation needs tanh's derivative; we write it not with \(z\) but with the output \(a = \tanh z\), which the forward pass has already computed: \(1 - a^2\).`,
      } },
      { file: { from: "def sigmoid(z):", to: "return 1 - a ** 2" } },
      { links: [{ tab: "math", section: "mm-act" }] },
    ],
  },
  {
    id: "mc-class",
    title: PLACE_TITLES["mc-class"],
    blocks: [
      { p: {
        tr: R`Ağırlıklar rastgele ve \(1/\sqrt{n_{\text{in}}}\) ölçeğinde başlar, biaslar sıfırdan (Matematik, 5. kısım). <code>forward</code> bir batch'i iki satırda hesaplar ve geri yayılım için ara sonuçları (<code>X</code>, <code>H</code>, <code>Y_hat</code>) saklar.`,
        en: R`The weights start random, at the scale \(1/\sqrt{n_{\text{in}}}\), and the biases at zero (Math, section 5). <code>forward</code> computes a batch in two lines and keeps the intermediate results (<code>X</code>, <code>H</code>, <code>Y_hat</code>) for backpropagation.`,
      } },
      { file: { from: "class MLP:", to: "return self.Y_hat[:, 0]", notes: [
        { at: "self.W1 = rng.normal", text: { tr: "Sıfırdan başlasaydı bütün gizli nöronlar kopya kalırdı (simetri).", en: "Starting at zero, all hidden neurons would stay copies (symmetry)." } },
        { at: "self.Z1 = X @ self.W1 + self.b1", text: { tr: "\\(\\mathbf z^{(1)} = \\mathbf W^{(1)}\\mathbf x + \\mathbf b^{(1)}\\), bütün örnekler için birden. `b1` her satıra eklenir (broadcasting).", en: "\\(\\mathbf z^{(1)} = \\mathbf W^{(1)}\\mathbf x + \\mathbf b^{(1)}\\), for all examples at once. `b1` is added to every row (broadcasting)." } },
        { at: "self.Y_hat = sigmoid(self.Z2)", text: { tr: "Her örnek için sınıf 1 olasılığı.", en: "The probability of class 1 for every example." } },
      ] } },
      { links: [{ tab: "math", section: "mm-forward" }] },
    ],
  },
  {
    id: "mc-backward",
    title: PLACE_TITLES["mc-backward"],
    blocks: [
      { p: {
        tr: R`Kayıp, ortalama çapraz entropi. <code>backward</code> ise Matematik 4. kısımdaki dört denklemin birebir kodudur; tek fark, her şeyin bir batch için matris olarak yazılması ve ortalama alındığı için \(1/n\) çarpanı.`,
        en: R`The loss is the mean cross-entropy. <code>backward</code> is the four equations of Math section 4, line for line; the only differences are that everything is written as matrices for a batch, and the factor \(1/n\) because we take the mean.`,
      } },
      { file: { from: "    def loss(self, y_hat, y):", to: "return dW1, db1, dW2, db2", notes: [
        { at: "delta2 = (self.Y_hat - y[:, None]) / n", text: { tr: "Denklem (1): \\(\\boldsymbol\\delta^{(2)} = \\hat y - y\\), ortalama için \\(n\\)'ye bölünmüş.", en: "Equation (1): \\(\\boldsymbol\\delta^{(2)} = \\hat y - y\\), divided by \\(n\\) for the mean." } },
        { at: "dW2 = self.H.T @ delta2", text: { tr: "Denklem (3): gradyan = girdi × hata. `H.T @ delta2` bütün örneklerin katkılarını toplar.", en: "Equation (3): gradient = input × error. `H.T @ delta2` adds up the contributions of all the examples." } },
        { at: "db2 = delta2.sum(axis=0)", text: { tr: "Denklem (4): biasın gradyanı hatanın kendisi.", en: "Equation (4): the bias's gradient is the error itself." } },
        { at: "delta1 = (delta2 @ self.W2.T) * tanh_derivative(self.H)", text: { tr: "Denklem (2): hata ağırlıkların transpozuyla geri taşınır ve tanh'ın eğimiyle çarpılır.", en: "Equation (2): the error is carried back through the transposed weights and multiplied by tanh's slope." } },
      ] } },
      { links: [{ tab: "math", section: "mm-backprop" }, { tab: "theory", section: "ml-backprop" }] },
    ],
  },
  {
    id: "mc-fit",
    title: PLACE_TITLES["mc-fit"],
    blocks: [
      { p: {
        tr: R`<code>step</code> gradyan inişinin tek adımıdır. <code>fit</code> her epoch'ta örnekleri karıştırır, mini-batch'lere böler ve her batch için ileri hesap, geri yayılım ve güncelleme yapar. Bu üçlü, SimpleCNN'in eğitim döngüsünün de iskeletidir.`,
        en: R`<code>step</code> is one step of gradient descent. <code>fit</code> shuffles the examples every epoch, splits them into mini-batches and, for every batch, does a forward pass, backpropagation and an update. That trio is also the skeleton of SimpleCNN's training loop.`,
      } },
      { file: { from: "def step(self, grads, lr):", to: "return (self.forward(X) > 0.5).astype(int)", notes: [
        { at: "self.W1 -= lr * dW1", text: { tr: "\\(\\theta \\leftarrow \\theta - \\eta\\,\\partial L/\\partial\\theta\\). `-=` diziyi yerinde değiştirir.", en: "\\(\\theta \\leftarrow \\theta - \\eta\\,\\partial L/\\partial\\theta\\). `-=` changes the array in place." } },
        { at: "order = rng.permutation(len(X))", text: { tr: "Her epoch'ta farklı sıra: batch'ler her seferinde farklı karışımlar olur.", en: "A different order every epoch: the batches are different mixes each time." } },
        { at: "self.step(self.backward(y[batch]), lr)", text: { tr: "İleri, geri, güncelle. PyTorch'ta: `loss.backward()` ve `optimizer.step()`.", en: "Forward, backward, update. In PyTorch: `loss.backward()` and `optimizer.step()`." } },
      ] } },
    ],
  },
  {
    id: "mc-check",
    title: PLACE_TITLES["mc-check"],
    blocks: [
      { p: {
        tr: R`Geri yayılımı elle yazınca küçük bir işaret ya da transpoze hatası ağı sessizce bozar: kod çalışır ama öğrenmez. Çare, her parametreyi \(\pm\varepsilon\) oynatıp kaybın değişimini ölçmek ve <code>backward</code>'ın sonucuyla karşılaştırmaktır (merkezî fark):`,
        en: R`When backpropagation is written by hand, a small sign or transpose mistake breaks the network silently: the code runs but does not learn. The cure is to nudge every parameter by \(\pm\varepsilon\), measure the change of the loss and compare it with what <code>backward</code> gives (a central difference):`,
      } },
      { eq: R`\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}` },
      { file: { from: "def gradient_check(", to: "return worst", notes: [
        { at: "for idx in np.ndindex(param.shape):", text: { tr: "Her parametre için iki ileri hesap: yavaş, ama yalnızca kontrol için.", en: "Two forward passes per parameter: slow, but only for checking." } },
        { at: "param[idx] = old", text: { tr: "Parametreyi eski değerine geri koymayı unutmayın.", en: "Do not forget to put the parameter back." } },
      ] } },
      { links: [{ tab: "math", section: "mm-backprop" }] },
    ],
  },
  {
    id: "mc-run",
    title: PLACE_TITLES["mc-run"],
    blocks: [
      { file: { from: "if __name__ == \"__main__\":", to: "test accuracy:" } },
      { p: {
        tr: R`Depo kökünde çalıştırın (sanal ortamdaki Python ile). Bir saniyeden kısa sürer:`,
        en: R`Run it at the root of the repository (with the Python of the virtual environment). It takes less than a second:`,
      } },
      { term: "PowerShell", lines: RUN_OUTPUT },
      { list: [
        { tr: R`<b>Gradyan kontrolü:</b> fark \(10^{-11}\) civarında: geri yayılım doğru.`, en: R`<b>Gradient check:</b> the difference is around \(10^{-11}\): backpropagation is right.` },
        { tr: R`<b>XOR:</b> 500 epoch'ta dört örnek de doğru; kayıp sonra da düşmeye devam eder, çünkü ağ cevaplarından giderek daha emin olur.`, en: R`<b>XOR:</b> all four examples are right by epoch 500; the loss keeps falling after that, because the network becomes ever more confident.` },
        { tr: R`<b>İki hilal:</b> eğitim verisinde %96, hiç görmediği test verisinde %90,5. Aradaki fark [[generalization|genellemenin]] ölçüsüdür; gürültü yüzünden bazı noktalar diğer sınıfın içinde kalır.`, en: R`<b>Two moons:</b> 96% on the training data, 90.5% on test data it has never seen. The gap is a measure of [[generalization|generalization]]; because of the noise, some points lie inside the other class.` },
      ] },
    ],
  },
  {
    id: "mc-torch",
    title: PLACE_TITLES["mc-torch"],
    blocks: [
      { p: {
        tr: R`PyTorch'la aynı ağ: <code>nn.Linear</code> bir \(\mathbf W, \mathbf b\) çiftidir, <code>loss.backward()</code> bizim <code>backward</code>'ımızı otomatik yapar, <code>optimizer.step()</code> da <code>step</code>'imizi.`,
        en: R`The same network in PyTorch: <code>nn.Linear</code> is a \(\mathbf W, \mathbf b\) pair, <code>loss.backward()</code> does our <code>backward</code> automatically, and <code>optimizer.step()</code> our <code>step</code>.`,
      } },
      { code: TORCH_CODE, name: "mlp_torch.py", notes: [
        { at: "nn.Linear(2, 8),", text: { tr: "W1: (8, 2), b1: (8,). PyTorch ağırlığı matematikteki gibi (çıktı, girdi) şeklinde saklar.", en: "W1: (8, 2), b1: (8,). PyTorch stores the weight as in the math: (outputs, inputs)." } },
        { at: "loss_fn = nn.BCEWithLogitsLoss()", text: { tr: "Sigmoid'i kayıpla birleştirir; sayısal olarak daha kararlı. Bu yüzden modelin son katmanında sigmoid yok.", en: "Combines the sigmoid with the loss; numerically more stable. That is why the model has no sigmoid at the end." } },
        { at: "loss.backward()", text: { tr: "Otomatik türev: PyTorch ileri hesabı kaydetti, şimdi zincir kuralını geriye uygular.", en: "Automatic differentiation: PyTorch recorded the forward pass and now applies the chain rule backwards." } },
        { at: "optimizer.zero_grad()", text: { tr: "PyTorch gradyanları toplar; her adımda sıfırlamak gerekir.", en: "PyTorch accumulates gradients; they must be reset every step." } },
      ] },
      { term: "PowerShell", lines: ["$ .venv\\Scripts\\python.exe mlp_torch.py", "epoch 500   loss 0.0087", "epoch 1000   loss 0.0032", "epoch 1500   loss 0.0019", "epoch 2000   loss 0.0013", "[0, 1, 1, 0]"] },
      { p: {
        tr: R`SimpleCNN'in eğitim döngüsü de aynı dört satırdır: ileri hesap, kayıp, <code>backward</code>, <code>step</code>. Değişen yalnızca model (evrişim katmanları), kayıp (10 sınıflı çapraz entropi) ve optimizer (Adam).`,
        en: R`SimpleCNN's training loop is the same four lines: forward pass, loss, <code>backward</code>, <code>step</code>. Only the model (convolution layers), the loss (10-class cross-entropy) and the optimizer (Adam) change.`,
      } },
      { links: [
        { chapter: "cnn", tab: "code", section: "c-model", title: { tr: "Modeli tanımlamak: katmanlar", en: "Defining the model: layers" } },
        { chapter: "cnn", tab: "code", section: "c-train-epoch", title: { tr: "Bir epoch eğitim", en: "Training for one epoch" } }] },
    ],
  },
  {
    id: "mc-exercises",
    title: PLACE_TITLES["mc-exercises"],
    blocks: [
      { list: [
        { tr: R`<b>Gizli nöron sayısı:</b> <code>hidden_units</code>'i 2'ye indirin ve <code>seed</code>'i 0'dan 9'a değiştirin. XOR kaç seed'de takılıyor? 8 nöronla neden takılmıyor? <i>İpucu:</i> takılan bir çalıştırmada kayıp 0,35 civarında kalır (\(\ln 2 / 2\): iki nokta \(\hat y \approx 0{,}5\) alır) ve doğruluk %50'dir; bizim denememizde 10 seed'den 4'ü takıldı. Teori 2'deki "Bazen takılır" notu sebebini anlatır.`, en: R`<b>Hidden neurons:</b> lower <code>hidden_units</code> to 2 and change <code>seed</code> from 0 to 9. For how many seeds does XOR get stuck? Why not with 8 neurons? <i>Hint:</i> in a stuck run the loss stays near 0.35 (\(\ln 2 / 2\): two points get \(\hat y \approx 0.5\)) and the accuracy is 50%; in our run 4 of the 10 seeds got stuck. The "Sometimes it gets stuck" note in Theory 2 explains why.` },
        { tr: R`<b>Bir hata ekleyin:</b> <code>backward</code>'da <code>* tanh_derivative(self.H)</code>'ı silin. Gradyan kontrolü ne diyor? XOR yine öğreniliyor mu? İki hilal? <i>İpucu:</i> farkın \(10^{-11}\)'den kaç kat büyüdüğüne bakın. Yanlış bir gradyan çoğu zaman yine kabaca doğru yönü gösterir; bu yüzden böyle bir hatayı yalnızca sonuçlara bakarak fark etmek zordur.`, en: R`<b>Add a bug:</b> in <code>backward</code>, delete <code>* tanh_derivative(self.H)</code>. What does the gradient check say? Is XOR still learned? And two moons? <i>Hint:</i> look at how many times larger than \(10^{-11}\) the difference gets. A wrong gradient often still points roughly the right way, which is why such a bug is hard to notice from the results alone.` },
        { tr: R`<b>Sıfırdan başlatın:</b> <code>W1</code> ve <code>W2</code>'yi <code>np.zeros</code> ile başlatın. Ne oluyor? (Matematik, 5. kısım.) <i>İpucu:</i> tanh ile <code>H = tanh(0) = 0</code> olur; <code>dW2 = H.T @ delta2</code> ve <code>delta1</code>'in içindeki <code>W2</code> neye eşit? Hangi parametre değişebilen tek parametre olarak kalır?`, en: R`<b>Start from zero:</b> initialize <code>W1</code> and <code>W2</code> with <code>np.zeros</code>. What happens? (Math, section 5.) <i>Hint:</i> with tanh, <code>H = tanh(0) = 0</code>; what are <code>dW2 = H.T @ delta2</code> and the <code>W2</code> inside <code>delta1</code> equal to? Which parameter is the only one left that can change?` },
        { tr: R`<b>Aşırı uyum:</b> iki hilalde <code>hidden_units = 64</code> ve <code>epochs_moons = 3000</code> yapın, eğitim verisini 40 örneğe indirin (<code>make_moons(400)</code> yerine <code>make_moons(40)</code>). Eğitim ve test doğruluğu nasıl değişiyor? <i>Beklenen:</i> eğitimde %100, testte ise öncekinden (%90,5) düşük.`, en: R`<b>Overfitting:</b> on two moons set <code>hidden_units = 64</code> and <code>epochs_moons = 3000</code>, and cut the training data to 40 examples (<code>make_moons(40)</code> instead of <code>make_moons(400)</code>). How do the training and test accuracy change? <i>Expected:</i> 100% in training, and lower than before (90.5%) on the test data.` },
        { tr: R`<b>ReLU:</b> gizli katmanda tanh yerine ReLU kullanın: <code>forward</code>'da <code>np.maximum(0, self.Z1)</code>, <code>backward</code>'da türev olarak <code>(self.Z1 &gt; 0)</code>. <code>W1</code>'in başlangıç ölçeğini \(\sqrt{2/n_{\text{in}}}\) yapın. <i>Dikkat:</i> gradyan kontrolü şimdi büyük bir fark (\(\sim 10^{-2}\)) gösterebilir, kodunuz doğru olsa bile: XOR'un \((0, 0)\) girdisinde biaslar sıfır olduğu için <code>Z1</code> tam 0'dır, yani ReLU'nun kırık noktasında; orada sayısal türev iki yandaki eğimlerin ortalamasını (\(1/2\)) verir. Kontrolü <code>Z1</code>'in hiçbir yerde tam 0 olmadığı girdilerle, örneğin iki hilal verisiyle yapın.`, en: R`<b>ReLU:</b> use ReLU instead of tanh in the hidden layer: <code>np.maximum(0, self.Z1)</code> in <code>forward</code> and <code>(self.Z1 &gt; 0)</code> as the derivative in <code>backward</code>. Make the starting scale of <code>W1</code> \(\sqrt{2/n_{\text{in}}}\). <i>Careful:</i> the gradient check may now show a large difference (\(\sim 10^{-2}\)) even if your code is right: for XOR's input \((0, 0)\), with zero biases, <code>Z1</code> is exactly 0, at ReLU's kink, where the numerical derivative gives the average of the slopes on the two sides (\(1/2\)). Run the check on inputs where no <code>Z1</code> is exactly 0, for example the two-moons data.` },
      ], ordered: true },
      { callout: "tip", title: { tr: "Deneme alanında da deneyin", en: "Try it in the Playground too" },
        text: { tr: R`4. alıştırmanın görsel hâli: Deneme alanında az nokta ve çok nöronla sınırın noktaların etrafında nasıl kıvrıldığına bakın.`,
          en: R`The visual version of exercise 4: in the Playground, with few points and many neurons, watch how the boundary curls around the points.` } },
    ],
  },
];

createLesson({ tab: "math", main: "#mm-main", toc: "#mm-toc", sections: MATH, widgets: WIDGETS, kicker: { tr: "Kısım", en: "Section" } });
createLesson({ tab: "code", main: "#mc-main", toc: "#mc-toc", sections: CODE, kicker: { tr: "Adım", en: "Step" },
  source: { text: MLP_PY, name: "MLP.py" } });
})();
