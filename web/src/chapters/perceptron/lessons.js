/*
 * Chapter 1, Math and Coding tabs: the content, rendered by lesson.js (createLesson).
 * Texts are { tr, en } pairs of HTML with \( TeX \) and [[key|text]] glossary links.
 */
import PERCEPTRON_PY from "../../../../Perceptron.py";   // bundled as text (scripts/build.mjs)
import { L } from "../../shared/i18n.js";
import { PLACE_TITLES, fmtNum, h } from "../../shared/core.js";
import { texNum } from "../../shared/math.js";
import { createLesson } from "../../shared/lesson.js";

(() => {
const R = String.raw;
const nm = (v, d = 2) => texNum(fmtNum(Math.abs(v) < 0.5 * 10 ** -d ? 0 : v, d));
const np = (v, d = 2) => (v < 0 ? `(${nm(v, d)})` : nm(v, d));
const eq = (tex) => h("div", { class: "m-eq", text: `\\(\\displaystyle ${tex}\\)` });

/** A labeled range input with its value. */
function range(label, { min, max, step, value, onInput }) {
  const input = h("input", { type: "range", min, max, step, value });
  const out = h("output", { text: fmtNum(value, 2) });
  input.addEventListener("input", () => {
    out.textContent = fmtNum(Number(input.value), 2);
    onInput(Number(input.value));
  });
  const node = h("label", {}, h("span", { class: "pc-slider-name" }), input, out);
  node.firstChild.innerHTML = label;
  return node;
}
function choice(options, current, onPick) {
  const box = h("div", { class: "seg" });
  for (const [value, label] of options) {
    const b = h("button", { type: "button", "aria-pressed": String(value === current), text: label });
    b.addEventListener("click", () => {
      for (const other of box.children) other.setAttribute("aria-pressed", String(other === b));
      onPick(value);
    });
    box.append(b);
  }
  return box;
}

// =====================================================================
// Math widgets
// =====================================================================
const WIDGETS = {
  /** z = w·x + b with its numbers, and the same sum as one dot product with the bias folded in. */
  forward(state) {
    const s = (state.forward ??= { x: [1, 0], w: [0.8, -0.4], b: -0.3 });
    const box = h("div", { class: "m-widget" });
    const out = h("div", { class: "m-widget-out" });
    const draw = () => {
      const z = s.w[0] * s.x[0] + s.w[1] * s.x[1] + s.b;
      out.replaceChildren(
        eq(R`z = w_1 x_1 + w_2 x_2 + b = ${np(s.w[0])}\cdot ${np(s.x[0], 1)} + ${np(s.w[1])}\cdot ${np(s.x[1], 1)} + ${np(s.b)} = ${nm(z)}`),
        eq(R`\tilde{\mathbf w}\cdot\tilde{\mathbf x} = (${nm(s.w[0])},\ ${nm(s.w[1])},\ ${nm(s.b)})\cdot(${nm(s.x[0], 1)},\ ${nm(s.x[1], 1)},\ 1) = ${nm(z)}`),
        eq(R`\hat y = H(${nm(z)}) = ${z > 0 ? 1 : 0}`));
    };
    box.append(
      h("div", { class: "t-controls" },
        range("x<sub>1</sub>", { min: -1, max: 2, step: 0.1, value: s.x[0], onInput: (v) => { s.x[0] = v; draw(); } }),
        range("x<sub>2</sub>", { min: -1, max: 2, step: 0.1, value: s.x[1], onInput: (v) => { s.x[1] = v; draw(); } })),
      h("div", { class: "t-controls" },
        range("w<sub>1</sub>", { min: -2, max: 2, step: 0.1, value: s.w[0], onInput: (v) => { s.w[0] = v; draw(); } }),
        range("w<sub>2</sub>", { min: -2, max: 2, step: 0.1, value: s.w[1], onInput: (v) => { s.w[1] = v; draw(); } }),
        range("b", { min: -2, max: 2, step: 0.1, value: s.b, onInput: (v) => { s.b = v; draw(); } })),
      out);
    draw();
    return box;
  },

  /** One correction with its numbers, and the check z' = z + η e (‖x‖² + 1). */
  update(state) {
    const s = (state.update ??= { x: [1, 1], y: 1, w: [0.2, -0.6], b: 0.1, eta: 0.5 });
    const box = h("div", { class: "m-widget" });
    const out = h("div", { class: "m-widget-out" });
    const draw = () => {
      const z = s.w[0] * s.x[0] + s.w[1] * s.x[1] + s.b;
      const yHat = z > 0 ? 1 : 0;
      const e = s.y - yHat;
      const w2 = [s.w[0] + s.eta * e * s.x[0], s.w[1] + s.eta * e * s.x[1]];
      const b2 = s.b + s.eta * e;
      const z2 = w2[0] * s.x[0] + w2[1] * s.x[1] + b2;
      const lines = [
        eq(R`z = ${np(s.w[0])}\cdot ${s.x[0]} + ${np(s.w[1])}\cdot ${s.x[1]} + ${np(s.b)} = ${nm(z)} \;\Rightarrow\; \hat y = ${yHat},\quad e = y - \hat y = ${e}`),
      ];
      if (!e) {
        lines.push(h("p", { class: "m-why", text: L({ tr: "Cevap doğru: e = 0, hiçbir şey değişmez. Başka bir y ya da x seçin.", en: "The answer is right: e = 0, nothing changes. Pick another y or x." }) }));
      } else {
        lines.push(
          eq(R`\mathbf w' = (${nm(s.w[0])},\ ${nm(s.w[1])}) + ${nm(s.eta)}\cdot(${e})\cdot(${s.x[0]},\ ${s.x[1]}) = (${nm(w2[0])},\ ${nm(w2[1])}),\quad b' = ${nm(s.b)} + ${nm(s.eta)}\cdot(${e}) = ${nm(b2)}`),
          eq(R`z' = ${nm(z2)} = ${nm(z)} + ${nm(s.eta)}\cdot(${e})\cdot(${s.x[0] ** 2 + s.x[1] ** 2} + 1)`),
          h("p", { class: "m-why", text: (e > 0 ? (z2 > 0) : (z2 <= 0))
            ? L({ tr: "Tek düzeltme bu örneği doğru tarafa geçirdi.", en: "One correction moved this example to the right side." })
            : L({ tr: "z doğru yönde değişti ama örnek hâlâ yanlış tarafta: birkaç düzeltme daha gerekir (ya da daha büyük bir η).", en: "z moved the right way but the example is still on the wrong side: it takes a few more corrections (or a larger η)." }) }));
      }
      out.replaceChildren(...lines);
    };
    box.append(
      h("div", { class: "t-controls" },
        h("span", { class: "mini-title pc-inline", text: "x" }),
        choice([["0,0", "(0, 0)"], ["0,1", "(0, 1)"], ["1,0", "(1, 0)"], ["1,1", "(1, 1)"]], s.x.join(","), (v) => { s.x = v.split(",").map(Number); draw(); }),
        h("span", { class: "mini-title pc-inline", text: "y" }),
        choice([["0", "0"], ["1", "1"]], String(s.y), (v) => { s.y = Number(v); draw(); })),
      h("div", { class: "t-controls" },
        range("w<sub>1</sub>", { min: -2, max: 2, step: 0.1, value: s.w[0], onInput: (v) => { s.w[0] = v; draw(); } }),
        range("w<sub>2</sub>", { min: -2, max: 2, step: 0.1, value: s.w[1], onInput: (v) => { s.w[1] = v; draw(); } }),
        range("b", { min: -2, max: 2, step: 0.1, value: s.b, onInput: (v) => { s.b = v; draw(); } }),
        range("η", { min: 0.05, max: 1, step: 0.05, value: s.eta, onInput: (v) => { s.eta = v; draw(); } })),
      out);
    draw();
    return box;
  },
};

// =====================================================================
// Math tab
// =====================================================================
const MATH = [
  {
    id: "pm-notation",
    title: PLACE_TITLES["pm-notation"],
    blocks: [
      { p: {
        tr: R`Teori sekmesinde perceptronun ne yaptığını resimlerle gördük. Burada aynı şeyi formüllerle yazıyor ve kuralların <b>neden</b> işe yaradığını ispatlıyoruz. Türetmeleri <b>Sonraki adım</b> düğmesiyle satır satır açın; etkileşimli örneklerde sayıları değiştirip formüllerin nasıl tepki verdiğine bakın.`,
        en: R`The Theory tab showed what the perceptron does, with pictures. Here we write the same with formulas and prove <b>why</b> the rules work. Open the derivations line by line with the <b>Next step</b> button, and change the numbers in the interactive examples to see how the formulas react.`,
      } },
      { h: { tr: "Semboller", en: "Symbols" } },
      { table: {
        head: [{ tr: "Sembol", en: "Symbol" }, { tr: "Anlamı", en: "Meaning" }],
        rows: [
          [R`\(\mathbf x = (x_1, \dots, x_n)\)`, { tr: "girdi vektörü (bu bölümde n = 2)", en: "the input vector (n = 2 in this chapter)" }],
          [R`\(\mathbf w = (w_1, \dots, w_n),\ b\)`, { tr: "[[weight|ağırlıklar]] ve [[bias|bias]]: öğrenilen [[parameter|parametreler]]", en: "the [[weight|weights]] and the [[bias|bias]]: the learned [[parameter|parameters]]" }],
          [R`\(z\)`, { tr: "ağırlıklı toplam (aktivasyon öncesi değer)", en: "the weighted sum (the value before the activation)" }],
          [R`\(H(z)\)`, { tr: "[[step-function|basamak (Heaviside) fonksiyonu]]", en: "the [[step-function|step (Heaviside) function]]" }],
          [R`\(\hat y,\ y\)`, { tr: "perceptronun çıktısı ve doğru cevap (0 ya da 1)", en: "the perceptron's output and the right answer (0 or 1)" }],
          [R`\(t = 2y - 1\)`, { tr: "aynı etiket −1 / +1 olarak (ispatlarda işimizi kolaylaştırır)", en: "the same label as −1 / +1 (makes the proofs simpler)" }],
          [R`\(\eta\)`, { tr: "[[learning-rate|öğrenme oranı]]", en: "the [[learning-rate|learning rate]]" }],
          [R`\((\mathbf x_i, y_i),\ i = 1, \dots, N\)`, { tr: "eğitim örnekleri", en: "the training examples" }],
        ],
      } },
      { h: { tr: "İleri hesap", en: "The forward computation" } },
      { eq: R`z = \mathbf w \cdot \mathbf x + b = \sum_{j=1}^{n} w_j x_j + b, \qquad \hat y = H(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}` },
      { h: { tr: "Bias'ı ağırlıklara katmak", en: "Folding the bias into the weights" } },
      { p: {
        tr: R`Her girdi vektörüne değeri hep 1 olan bir bileşen eklenirse, bias o girdinin ağırlığı olur: \(\tilde{\mathbf x} = (x_1, \dots, x_n, 1)\) ve \(\tilde{\mathbf w} = (w_1, \dots, w_n, b)\) ile \(z = \tilde{\mathbf w}\cdot\tilde{\mathbf x}\). Formüller tek bir vektörle kısalır; ispatlarda bu biçimi kullanacağız. Aşağıda iki biçimin aynı sayıyı verdiğini görün.`,
        en: R`If every input vector gets an extra component that is always 1, the bias becomes the weight of that input: with \(\tilde{\mathbf x} = (x_1, \dots, x_n, 1)\) and \(\tilde{\mathbf w} = (w_1, \dots, w_n, b)\), \(z = \tilde{\mathbf w}\cdot\tilde{\mathbf x}\). The formulas get shorter with a single vector; the proofs use this form. Below, see that both forms give the same number.`,
      } },
      { widget: "forward" },
      { model: {
        tr: R`<b>Parametre sayısı:</b> Bu bölümün perceptronunda \(n = 2\): iki ağırlık ve bir bias, toplam 3 parametre. SimpleCNN'in son katmanı (FC2) 512 girdili 10 nörondan oluşur: \(10 \cdot 512 + 10 = 5130\) parametre; her nöron aynı \(\mathbf w\cdot\mathbf x + b\) hesabını yapar.`,
        en: R`<b>Number of parameters:</b> the perceptron of this chapter has \(n = 2\): two weights and a bias, 3 parameters in all. The last layer of SimpleCNN (FC2) has 10 neurons with 512 inputs each: \(10 \cdot 512 + 10 = 5130\) parameters; every neuron computes the same \(\mathbf w\cdot\mathbf x + b\).`,
      } },
      { links: [{ tab: "theory", section: "p-neuron" }, { tab: "code", section: "pc-class" }] },
    ],
  },
  {
    id: "pm-geometry",
    title: PLACE_TITLES["pm-geometry"],
    blocks: [
      { p: {
        tr: R`[[decision-boundary|Karar sınırı]], \(z\)'nin sıfır olduğu noktaların kümesidir: \(\{\mathbf x : \mathbf w\cdot\mathbf x + b = 0\}\). İki boyutta bir doğru, \(n\) boyutta bir [[hyperplane|hiper düzlemdir]]. İki temel özelliğini ispatlayalım.`,
        en: R`The [[decision-boundary|decision boundary]] is the set of points where \(z\) is zero: \(\{\mathbf x : \mathbf w\cdot\mathbf x + b = 0\}\). In two dimensions it is a line, in \(n\) dimensions a [[hyperplane|hyperplane]]. Let us prove its two basic properties.`,
      } },
      { derive: {
        title: { tr: "w sınıra diktir", en: "w is perpendicular to the boundary" },
        steps: [
          { tex: R`\mathbf w\cdot\mathbf p + b = 0, \qquad \mathbf w\cdot\mathbf q + b = 0`,
            why: { tr: R`Sınırın üzerinde iki nokta \(\mathbf p\) ve \(\mathbf q\) alalım; ikisi de denklemi sağlar.`, en: R`Take two points \(\mathbf p\) and \(\mathbf q\) on the boundary; both satisfy the equation.` } },
          { tex: R`\mathbf w\cdot(\mathbf p - \mathbf q) = 0`,
            why: { tr: R`İki denklemi birbirinden çıkarınca \(b\) gider.`, en: R`Subtracting the two equations removes \(b\).` } },
          { tex: R`\mathbf w \perp (\mathbf p - \mathbf q)`,
            why: { tr: R`\(\mathbf p - \mathbf q\) sınır boyunca uzanan bir vektördür ve \(\mathbf w\) ile iç çarpımı sıfırdır: \(\mathbf w\) sınırdaki her yöne diktir. \(\mathbf w\) yönünde ilerledikçe \(z\) büyüdüğü için \(\mathbf w\) 1 tarafını gösterir.`, en: R`\(\mathbf p - \mathbf q\) is a vector along the boundary and its dot product with \(\mathbf w\) is zero: \(\mathbf w\) is perpendicular to every direction in the boundary. Since \(z\) grows as you move along \(\mathbf w\), \(\mathbf w\) points to the 1 side.` } },
        ],
      } },
      { derive: {
        title: { tr: "Bir noktanın sınıra uzaklığı", en: "The distance of a point from the boundary" },
        steps: [
          { tex: R`\mathbf x = \mathbf x_\perp + r\,\frac{\mathbf w}{\lVert\mathbf w\rVert}`,
            why: { tr: R`\(\mathbf x\)'i sınır üzerindeki izdüşümü \(\mathbf x_\perp\) ile sınıra dik bir adımın toplamı olarak yazalım; \(r\) işaretli uzaklıktır.`, en: R`Write \(\mathbf x\) as its projection \(\mathbf x_\perp\) onto the boundary plus a step perpendicular to it; \(r\) is the signed distance.` } },
          { tex: R`z = \mathbf w\cdot\mathbf x_\perp + b + r\,\frac{\mathbf w\cdot\mathbf w}{\lVert\mathbf w\rVert} = 0 + r\,\lVert\mathbf w\rVert`,
            why: { tr: R`\(\mathbf x_\perp\) sınırda olduğu için ilk iki terimin toplamı 0; \(\mathbf w\cdot\mathbf w = \lVert\mathbf w\rVert^2\).`, en: R`\(\mathbf x_\perp\) is on the boundary, so the first two terms add up to 0; \(\mathbf w\cdot\mathbf w = \lVert\mathbf w\rVert^2\).` } },
          { tex: R`r = \frac{z}{\lVert\mathbf w\rVert}, \qquad r(\mathbf 0) = \frac{b}{\lVert\mathbf w\rVert}`,
            why: { tr: R`\(z\), sınıra olan uzaklığın \(\lVert\mathbf w\rVert\) katıdır: işareti tarafı, büyüklüğü uzaklığı söyler. Orijinde \(z = b\) olduğundan sınırın orijine uzaklığı \(|b|/\lVert\mathbf w\rVert\)'dir.`, en: R`\(z\) is the distance from the boundary times \(\lVert\mathbf w\rVert\): its sign tells the side, its size the distance. At the origin \(z = b\), so the boundary's distance from the origin is \(|b|/\lVert\mathbf w\rVert\).` } },
        ],
      } },
      { p: {
        tr: R`<b>Ölçek:</b> \(\mathbf w\) ve \(b\) aynı \(c > 0\) sayısıyla çarpılırsa denklemin çözümleri değişmez, sınır yerinde kalır; yalnızca her noktanın \(z\)'si \(c\) katına çıkar. \(c < 0\) ise sınır aynı kalır ama iki taraf yer değiştirir. Örnek: AND için \(\mathbf w = (1, 1)\), \(b = -1{,}5\) seçersek sınırın orijine uzaklığı \(1{,}5/\sqrt 2 \approx 1{,}06\), \((1, 1)\) noktasında \(z = 0{,}5\) ve uzaklık \(0{,}5/\sqrt 2 \approx 0{,}35\) olur.`,
        en: R`<b>Scale:</b> if \(\mathbf w\) and \(b\) are multiplied by the same number \(c > 0\), the solutions of the equation do not change and the boundary stays put; only the \(z\) of every point is multiplied by \(c\). With \(c < 0\) the boundary stays but the two sides swap. Example: choosing \(\mathbf w = (1, 1)\), \(b = -1.5\) for AND, the boundary's distance from the origin is \(1.5/\sqrt 2 \approx 1.06\); at the point \((1, 1)\), \(z = 0.5\) and the distance is \(0.5/\sqrt 2 \approx 0.35\).`,
      } },
      { links: [{ tab: "theory", section: "p-boundary" }] },
    ],
  },
  {
    id: "pm-rule",
    title: PLACE_TITLES["pm-rule"],
    blocks: [
      { p: {
        tr: R`Örnekler tek tek gösterilir. Hata \(e = y - \hat y\) üç değer alabilir ve [[perceptron-rule|perceptron kuralı]] her örnekten sonra şu düzeltmeyi yapar:`,
        en: R`The examples are shown one at a time. The error \(e = y - \hat y\) can take three values, and the [[perceptron-rule|perceptron rule]] makes this correction after every example:`,
      } },
      { eq: R`\mathbf w \leftarrow \mathbf w + \eta\, e\, \mathbf x, \qquad b \leftarrow b + \eta\, e \qquad\Longleftrightarrow\qquad \tilde{\mathbf w} \leftarrow \tilde{\mathbf w} + \eta\, e\, \tilde{\mathbf x}` },
      { table: {
        head: [R`\(y\)`, R`\(\hat y\)`, R`\(e\)`, { tr: "Düzeltme", en: "Correction" }],
        rows: [
          ["1", "0", "+1", { tr: R`\(\mathbf w\)'ye \(\eta\mathbf x\), \(b\)'ye \(\eta\) eklenir`, en: R`add \(\eta\mathbf x\) to \(\mathbf w\) and \(\eta\) to \(b\)` }],
          ["0", "1", "−1", { tr: R`\(\mathbf w\)'den \(\eta\mathbf x\), \(b\)'den \(\eta\) çıkarılır`, en: R`subtract \(\eta\mathbf x\) from \(\mathbf w\) and \(\eta\) from \(b\)` }],
          [{ tr: "aynı", en: "equal" }, { tr: "aynı", en: "equal" }, "0", { tr: "yok", en: "none" }],
        ],
      } },
      { derive: {
        title: { tr: "Bir düzeltme z'yi doğru yöne iter", en: "A correction pushes z the right way" },
        steps: [
          { tex: R`\mathbf w' = \mathbf w + \eta e\,\mathbf x, \qquad b' = b + \eta e`,
            why: { tr: R`Yanlış sınıflanan bir örnek (\(e \ne 0\)) için kuralı uygulayalım.`, en: R`Apply the rule to a misclassified example (\(e \ne 0\)).` } },
          { tex: R`z' = \mathbf w'\cdot\mathbf x + b' = \mathbf w\cdot\mathbf x + b + \eta e\,(\mathbf x\cdot\mathbf x + 1)`,
            why: { tr: R`Aynı örneğin yeni \(z\)'sini hesaplayıp terimleri dağıtalım.`, en: R`Compute the same example's new \(z\) and expand the terms.` } },
          { tex: R`z' = z + \eta e\,\big(\lVert\mathbf x\rVert^2 + 1\big)`,
            why: { tr: R`\(\mathbf x\cdot\mathbf x = \lVert\mathbf x\rVert^2\). Parantez içi her zaman pozitiftir.`, en: R`\(\mathbf x\cdot\mathbf x = \lVert\mathbf x\rVert^2\). The bracket is always positive.` } },
          { tex: R`e = +1 \;\Rightarrow\; z' > z, \qquad e = -1 \;\Rightarrow\; z' < z`,
            why: { tr: R`1 olması gereken örneğin \(z\)'si büyür, 0 olması gerekeninki küçülür. Tek düzeltme örneği her zaman doğru tarafa geçirmeyebilir ama <b>hep doğru yöne</b> iter. Bias'ın "1" girdisi sayesinde, \(\mathbf x = \mathbf 0\) olan bir örnekte bile \(z\) değişir.`, en: R`The \(z\) of an example that should be 1 grows, that of an example that should be 0 shrinks. One correction does not always move the example to the right side, but it <b>always pushes the right way</b>. Thanks to the bias's input of "1", \(z\) changes even for an example with \(\mathbf x = \mathbf 0\).` } },
        ],
      } },
      { p: {
        tr: R`Aşağıda bir düzeltmeyi kendiniz hesaplayın ve son satırdaki \(z' = z + \eta e(\lVert\mathbf x\rVert^2 + 1)\) eşitliğini kontrol edin.`,
        en: R`Compute a correction yourself below and check the equality \(z' = z + \eta e(\lVert\mathbf x\rVert^2 + 1)\) on the last line.`,
      } },
      { widget: "update" },
      { callout: "info", title: { tr: "Bir düzeltme ötekileri bozabilir", en: "A correction can spoil the others" },
        text: { tr: R`Kural yalnızca o anki örneğe bakar; onu düzeltirken daha önce doğru olan bir örneği yanlış tarafa geçirebilir. Yine de veri doğrusal ayrılabilirse bu gidip gelme sonsuza kadar sürmez: yakınsama teoremi (bu sekmenin 5. kısmı) bunu ispatlar.`,
          en: R`The rule only looks at the current example; while fixing it, it can move an example that was right before to the wrong side. Still, if the data is linearly separable, this back and forth cannot go on forever: the convergence theorem (section 5 of this tab) proves it.` } },
      { links: [{ tab: "theory", section: "p-rule" }, { tab: "code", section: "pc-fit" }] },
    ],
  },
  {
    id: "pm-gd",
    title: PLACE_TITLES["pm-gd"],
    blocks: [
      { p: {
        tr: R`Perceptron kuralı bir sezgi gibi görünüyor ama aslında bir kaybı küçülten [[gradient-descent|gradyan inişidir]]. Etiketleri \(t = 2y - 1 \in \{-1, +1\}\) olarak yazalım. Bir örnek yanlış taraftaysa \(t\,z \le 0\) olur. <b>Perceptron ölçütü</b> denen [[loss|kayıp]]:`,
        en: R`The perceptron rule looks like a heuristic, but it is in fact a [[gradient-descent|gradient descent]] that reduces a loss. Write the labels as \(t = 2y - 1 \in \{-1, +1\}\). An example on the wrong side has \(t\,z \le 0\). The [[loss|loss]] called the <b>perceptron criterion</b>:`,
      } },
      { eq: R`\ell(\mathbf w, b;\ \mathbf x, t) = \max\big(0,\ -t\,z\big), \qquad z = \mathbf w\cdot\mathbf x + b` },
      { p: {
        tr: R`Doğru taraftaki bir örneğin kaybı 0'dır. Yanlış taraftakinin kaybı \(-t\,z = |z|\), yani sınıra uzaklığının \(\lVert\mathbf w\rVert\) katıdır: sınırdan ne kadar uzakta yanlışsa ceza o kadar büyük.`,
        en: R`An example on the right side has zero loss. One on the wrong side has loss \(-t\,z = |z|\), its distance from the boundary times \(\lVert\mathbf w\rVert\): the farther on the wrong side, the bigger the penalty.`,
      } },
      { derive: {
        title: { tr: "Kaybın gradyanı ve bir SGD adımı", en: "The gradient of the loss and one SGD step" },
        steps: [
          { tex: R`t\,z > 0:\quad \ell = 0 \;\Rightarrow\; \frac{\partial \ell}{\partial \mathbf w} = \mathbf 0,\quad \frac{\partial \ell}{\partial b} = 0`,
            why: { tr: R`Doğru sınıflanan örnek kaybı değiştirmez; gradyanı sıfırdır.`, en: R`A correctly classified example does not change the loss; its gradient is zero.` } },
          { tex: R`t\,z < 0:\quad \ell = -t\,(\mathbf w\cdot\mathbf x + b) \;\Rightarrow\; \frac{\partial \ell}{\partial \mathbf w} = -t\,\mathbf x,\quad \frac{\partial \ell}{\partial b} = -t`,
            why: { tr: R`Yanlış sınıflanan örnekte kayıp \(\mathbf w\) ve \(b\)'nin doğrusal bir fonksiyonudur.`, en: R`For a misclassified example the loss is a linear function of \(\mathbf w\) and \(b\).` } },
          { tex: R`\mathbf w \leftarrow \mathbf w - \eta\,\frac{\partial \ell}{\partial \mathbf w} = \mathbf w + \eta\,t\,\mathbf x, \qquad b \leftarrow b + \eta\,t`,
            why: { tr: R`Tek örnekle bir [[sgd|SGD]] adımı: gradyanın tersi yönünde \(\eta\) kadar.`, en: R`One [[sgd|SGD]] step on one example: \(\eta\) times the opposite of the gradient.` } },
          { tex: R`e = y - \hat y = t \quad\Rightarrow\quad \mathbf w \leftarrow \mathbf w + \eta\,e\,\mathbf x`,
            why: { tr: R`Yanlış sınıflanan bir örnekte \(y = 1, \hat y = 0\) ise \(e = 1 = t\); \(y = 0, \hat y = 1\) ise \(e = -1 = t\). Bu tam olarak perceptron kuralı. (\(z = 0\)'da kaybın bir kırığı vardır ve gradyan tek değildir. \(H(0) = 0\) olduğu için kural orada \(y = 1\) ise adım atar, \(y = 0\) ise atmaz; ikisi de geçerli bir seçimdir.)`, en: R`For a misclassified example, \(y = 1, \hat y = 0\) gives \(e = 1 = t\) and \(y = 0, \hat y = 1\) gives \(e = -1 = t\). This is exactly the perceptron rule. (At \(z = 0\) the loss has a kink and the gradient is not unique. Since \(H(0) = 0\), the rule takes the step there if \(y = 1\) and does nothing if \(y = 0\); both are valid choices.)` } },
        ],
      } },
      { h: { tr: "Neden basamak fonksiyonunun türevini kullanmıyoruz?", en: "Why not use the derivative of the step function?" } },
      { p: {
        tr: R`Daha doğal görünen kayıp, çıktının hatasının karesidir: \((y - H(z))^2\). Ama \(H'(z) = 0\) (\(z \neq 0\) için), yani bu kaybın \(\mathbf w\)'ye göre gradyanı neredeyse her yerde sıfırdır: hiçbir sinyal yoktur. Perceptron ölçütü \(H\)'nin türevini hiç almadan \(z\)'ye doğrudan bakarak bu sorunu atlatır. Çok katmanlı ağlarda ise bu hile işe yaramaz: gizli bir nöron için hedef yoktur, çıktıdaki hatanın sinyali ona ancak sonraki nöronların aktivasyonlarından geriye geçerek ulaşabilir. Bu yüzden orada \(H\) yerine türevi sıfır olmayan bir [[activation|aktivasyon]] kullanılır, örneğin sigmoid:`,
        en: R`The more natural-looking loss is the squared error of the output: \((y - H(z))^2\). But \(H'(z) = 0\) (for \(z \neq 0\)), so the gradient of this loss with respect to \(\mathbf w\) is zero almost everywhere: there is no signal at all. The perceptron criterion sidesteps this by looking at \(z\) directly, never differentiating \(H\). In multilayer networks this trick does not work: a hidden neuron has no target, and the error signal from the output can only reach it by passing back through the activations of the neurons after it. So there \(H\) is replaced by an [[activation|activation]] with a nonzero derivative, for example the sigmoid:`,
      } },
      { eq: R`\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big) > 0` },
      { callout: "tip", title: { tr: "CNN bölümüyle bağlantı", en: "The link to the CNN chapter" },
        text: { tr: R`SimpleCNN'in eğitimi de aynı kalıpla çalışır: bir kayıp, onun gradyanı ve gradyanın tersi yönünde bir adım (en sade biçimi \(\theta \leftarrow \theta - \eta\,\nabla L\); SimpleCNN bunun geliştirilmiş bir biçimi olan Adam'ı kullanır). Değişen, kaybın [[cross-entropy|çapraz entropi]] olması ve gradyanın katmanlar boyunca [[backprop|geri yayılımla]] hesaplanmasıdır.`,
          en: R`SimpleCNN is trained with the same pattern: a loss, its gradient and a step against the gradient (in its plainest form \(\theta \leftarrow \theta - \eta\,\nabla L\); SimpleCNN uses Adam, a refined version of it). What changes is that the loss is [[cross-entropy|cross-entropy]] and the gradient is computed through the layers with [[backprop|backpropagation]].` } },
      { links: [
        { chapter: "cnn", tab: "math", section: "m-opt", title: { tr: "Optimizasyon: SGD, Momentum, Adam", en: "Optimization: SGD, Momentum, Adam" } },
        { chapter: "cnn", tab: "theory", section: "t-opt", title: { tr: "Geri yayılım ve optimizasyon", en: "Backpropagation and optimization" } }] },
    ],
  },
  {
    id: "pm-converge",
    title: PLACE_TITLES["pm-converge"],
    blocks: [
      { p: {
        tr: R`Bias'ı katılmış vektörleri (\(\tilde{\mathbf x}_i\), \(\tilde{\mathbf w}\)) ve \(\pm 1\) etiketlerini (\(t_i\)) kullanalım; başlangıç \(\tilde{\mathbf w}_0 = \mathbf 0\). İki varsayım:`,
        en: R`Use the vectors with the bias folded in (\(\tilde{\mathbf x}_i\), \(\tilde{\mathbf w}\)) and the \(\pm 1\) labels (\(t_i\)); start from \(\tilde{\mathbf w}_0 = \mathbf 0\). Two assumptions:`,
      } },
      { list: [
        { tr: R`<b>Sınırlı veri:</b> her örnek için \(\lVert\tilde{\mathbf x}_i\rVert \le R\).`, en: R`<b>Bounded data:</b> \(\lVert\tilde{\mathbf x}_i\rVert \le R\) for every example.` },
        { tr: R`<b>Marjla ayrılabilirlik:</b> \(\lVert\mathbf u\rVert = 1\) olan bir \(\mathbf u\) vektörü vardır ki her örnek için \(t_i\,\mathbf u\cdot\tilde{\mathbf x}_i \ge \gamma > 0\). Yani bütün örnekleri en az \(\gamma\) [[margin|marjla]] ayıran bir sınır vardır.`, en: R`<b>Separable with a margin:</b> there is a vector \(\mathbf u\) with \(\lVert\mathbf u\rVert = 1\) such that \(t_i\,\mathbf u\cdot\tilde{\mathbf x}_i \ge \gamma > 0\) for every example. That is, some boundary separates all the examples with a [[margin|margin]] of at least \(\gamma\).` },
      ] },
      { eq: { tr: R`\textbf{Teorem (Novikoff, 1962):}\quad k \;\le\; \left(\frac{R}{\gamma}\right)^{2}`, en: R`\textbf{Theorem (Novikoff, 1962):}\quad k \;\le\; \left(\frac{R}{\gamma}\right)^{2}` } },
      { p: {
        tr: R`Burada \(k\), perceptron kuralının yaptığı toplam düzeltme sayısıdır. Örnekler hangi sırayla gelirse gelsin, en fazla bu kadar hata yapılır; sonra perceptron her örneği doğru sınıflandırır.`,
        en: R`Here \(k\) is the total number of corrections the perceptron rule makes. Whatever the order of the examples, at most this many mistakes are made; after that the perceptron classifies every example correctly.`,
      } },
      { derive: {
        title: { tr: "İspat", en: "Proof" },
        steps: [
          { tex: R`\tilde{\mathbf w}_k = \tilde{\mathbf w}_{k-1} + \eta\,t\,\tilde{\mathbf x}, \qquad t\,\tilde{\mathbf w}_{k-1}\cdot\tilde{\mathbf x} \le 0`,
            why: { tr: R`\(k\). düzeltme, yanlış sınıflanan bir \((\tilde{\mathbf x}, t)\) örneğinde yapılır; yanlış olduğu için \(t\,\tilde{\mathbf w}_{k-1}\cdot\tilde{\mathbf x} \le 0\).`, en: R`The \(k\)-th correction happens on a misclassified example \((\tilde{\mathbf x}, t)\); being wrong means \(t\,\tilde{\mathbf w}_{k-1}\cdot\tilde{\mathbf x} \le 0\).` } },
          { tex: R`\mathbf u\cdot\tilde{\mathbf w}_k = \mathbf u\cdot\tilde{\mathbf w}_{k-1} + \eta\,t\,\mathbf u\cdot\tilde{\mathbf x} \;\ge\; \mathbf u\cdot\tilde{\mathbf w}_{k-1} + \eta\gamma \quad\Rightarrow\quad \mathbf u\cdot\tilde{\mathbf w}_k \ge k\,\eta\gamma`,
            why: { tr: R`<b>Doğru yönde büyüme:</b> marj varsayımıyla her düzeltme \(\tilde{\mathbf w}\)'nin \(\mathbf u\) yönündeki bileşenini en az \(\eta\gamma\) büyütür. \(\mathbf 0\)'dan başlayıp bunu \(k\) düzeltme boyunca toplayınca sağdaki sonuç çıkar.`, en: R`<b>Growth in the right direction:</b> by the margin assumption, every correction grows the component of \(\tilde{\mathbf w}\) along \(\mathbf u\) by at least \(\eta\gamma\). Starting from \(\mathbf 0\) and adding this up over \(k\) corrections gives the result on the right.` } },
          { tex: R`\lVert\tilde{\mathbf w}_k\rVert^2 = \lVert\tilde{\mathbf w}_{k-1}\rVert^2 + 2\eta\,t\,\tilde{\mathbf w}_{k-1}\cdot\tilde{\mathbf x} + \eta^2\lVert\tilde{\mathbf x}\rVert^2 \;\le\; \lVert\tilde{\mathbf w}_{k-1}\rVert^2 + \eta^2 R^2 \quad\Rightarrow\quad \lVert\tilde{\mathbf w}_k\rVert^2 \le k\,\eta^2 R^2`,
            why: { tr: R`<b>Uzunluk yavaş büyür:</b> ortadaki terim hata olduğu için \(\le 0\), son terim \(\le \eta^2 R^2\). Uzunluğun karesi her düzeltmede en fazla \(\eta^2R^2\) artar.`, en: R`<b>The length grows slowly:</b> the middle term is \(\le 0\) because it was a mistake, the last term is \(\le \eta^2 R^2\). The squared length grows by at most \(\eta^2R^2\) per correction.` } },
          { tex: R`k\,\eta\gamma \;\le\; \mathbf u\cdot\tilde{\mathbf w}_k \;\le\; \lVert\mathbf u\rVert\,\lVert\tilde{\mathbf w}_k\rVert \;\le\; \sqrt{k}\,\eta R`,
            why: { tr: R`Cauchy–Schwarz eşitsizliği (\(\lVert\mathbf u\rVert = 1\)) iki sonucu birleştirir: \(\mathbf u\) yönündeki bileşen \(k\) ile, uzunluk ise yalnızca \(\sqrt k\) ile büyüyor. Bu sonsuza kadar süremez.`, en: R`The Cauchy–Schwarz inequality (\(\lVert\mathbf u\rVert = 1\)) joins the two results: the component along \(\mathbf u\) grows like \(k\), the length only like \(\sqrt k\). That cannot go on forever.` } },
          { tex: R`\sqrt{k} \le \frac{R}{\gamma} \quad\Rightarrow\quad k \le \left(\frac{R}{\gamma}\right)^{2} \qquad\blacksquare`,
            why: { tr: R`Her iki yanı \(\sqrt k\,\eta\gamma\)'ya bölüp karesini alınca. \(\eta\) sadeleşti: sıfırdan başlayan bir perceptronda öğrenme oranı düzeltme sayısını değiştirmez, çünkü \(\tilde{\mathbf w}\)'yi yalnızca ölçekler.`, en: R`Divide both sides by \(\sqrt k\,\eta\gamma\) and square. \(\eta\) cancelled: for a perceptron that starts from zero, the learning rate does not change the number of corrections, because it only scales \(\tilde{\mathbf w}\).` } },
        ],
      } },
      { p: {
        tr: R`<b>Ne anlatıyor?</b> Marj küçüldükçe sınır büyür: birbirine yakın sınıfları ayırmak daha çok düzeltme ister (Teori'deki deney). Veri ayrılabilir değilse \(\mathbf u\) yoktur, teorem bir şey söylemez ve kural gerçekten de sonsuza kadar döner.`,
        en: R`<b>What does it say?</b> The smaller the margin, the bigger the bound: separating classes that are close together takes more corrections (the experiment in Theory). If the data is not separable there is no \(\mathbf u\), the theorem says nothing, and the rule really does cycle forever.`,
      } },
      { links: [{ tab: "theory", section: "p-converge" }] },
    ],
  },
  {
    id: "pm-xor",
    title: PLACE_TITLES["pm-xor"],
    blocks: [
      { p: {
        tr: R`Bir perceptronun XOR'u hesapladığını varsayalım. \(\hat y = H(w_1x_1 + w_2x_2 + b)\) dört girdi için dört koşul verir:`,
        en: R`Suppose a perceptron computes XOR. \(\hat y = H(w_1x_1 + w_2x_2 + b)\) gives four conditions for the four inputs:`,
      } },
      { table: {
        head: [R`\((x_1, x_2)\)`, "XOR", { tr: "Koşul", en: "Condition" }],
        rows: [
          ["(0, 0)", "0", R`\(b \le 0\)`],
          ["(0, 1)", "1", R`\(w_2 + b > 0\)`],
          ["(1, 0)", "1", R`\(w_1 + b > 0\)`],
          ["(1, 1)", "0", R`\(w_1 + w_2 + b \le 0\)`],
        ],
      } },
      { derive: {
        title: { tr: "Çelişki", en: "The contradiction" },
        steps: [
          { tex: R`(w_2 + b) + (w_1 + b) > 0 \;\Rightarrow\; w_1 + w_2 + 2b > 0`,
            why: { tr: R`Ortadaki iki koşulu toplayalım.`, en: R`Add the two middle conditions.` } },
          { tex: R`w_1 + w_2 + b > -b`,
            why: { tr: R`İki yandan \(b\) çıkaralım.`, en: R`Subtract \(b\) from both sides.` } },
          { tex: R`b \le 0 \;\Rightarrow\; -b \ge 0 \;\Rightarrow\; w_1 + w_2 + b > 0`,
            why: { tr: R`İlk koşul \(-b\)'nin negatif olmadığını söyler.`, en: R`The first condition says \(-b\) is not negative.` } },
          { tex: { tr: R`w_1 + w_2 + b > 0 \quad\text{ve}\quad w_1 + w_2 + b \le 0 \qquad\blacksquare`, en: R`w_1 + w_2 + b > 0 \quad\text{and}\quad w_1 + w_2 + b \le 0 \qquad\blacksquare` },
            why: { tr: R`Bu, son koşulla çelişir. Demek ki dört koşulu birden sağlayan \(w_1, w_2, b\) yoktur: XOR [[linear-separability|doğrusal ayrılabilir]] değildir ve hiçbir perceptron onu hesaplayamaz.`, en: R`This contradicts the last condition. So no \(w_1, w_2, b\) satisfy all four conditions: XOR is not [[linear-separability|linearly separable]], and no perceptron can compute it.` } },
        ],
      } },
      { h: { tr: "İki katmanla çözüm", en: "The solution with two layers" } },
      { eq: { tr: R`h_1 = H(x_1 + x_2 - 0{,}5)\ \ (\text{OR}), \qquad h_2 = H(-x_1 - x_2 + 1{,}5)\ \ (\text{NAND}), \qquad \hat y = H(h_1 + h_2 - 1{,}5)\ \ (\text{AND})`,
        en: R`h_1 = H(x_1 + x_2 - 0.5)\ \ (\text{OR}), \qquad h_2 = H(-x_1 - x_2 + 1.5)\ \ (\text{NAND}), \qquad \hat y = H(h_1 + h_2 - 1.5)\ \ (\text{AND})` } },
      { table: {
        head: [R`\((x_1, x_2)\)`, R`\((h_1, h_2)\)`, R`\(\hat y\)`, "XOR"],
        rows: [["(0, 0)", "(0, 1)", "0", "0"], ["(0, 1)", "(1, 1)", "1", "1"], ["(1, 0)", "(1, 1)", "1", "1"], ["(1, 1)", "(1, 0)", "0", "0"]],
      } },
      { p: {
        tr: R`İşin sırrı tablonun ortasındaki sütunda: [[hidden-layer|gizli katman]] dört noktayı yeni bir uzaya, \((h_1, h_2)\) uzayına taşır. Orada XOR'un 1 olduğu iki nokta aynı yere, \((1, 1)\)'e düşer; ötekiler \((0, 1)\) ve \((1, 0)\)'dadır ve onları tek bir doğru ayırır. Yani gizli katman veriyi <b>doğrusal ayrılabilir bir gösterime</b> dönüştürür. Derin öğrenmenin ana fikri budur: katmanlar, son katmanın işini kolaylaştıran gösterimler öğrenir. SimpleCNN'in evrişim katmanları da görüntüyü, son katmanın rakamları ayırabileceği bir gösterime çevirir.`,
        en: R`The secret is in the middle column of the table: the [[hidden-layer|hidden layer]] moves the four points into a new space, the \((h_1, h_2)\) space. There the two points where XOR is 1 land on the same spot, \((1, 1)\); the others are at \((0, 1)\) and \((1, 0)\), and one straight line separates them. So the hidden layer turns the data into a <b>linearly separable representation</b>. This is the main idea of deep learning: layers learn representations that make the last layer's job easy. The convolution layers of SimpleCNN, too, turn an image into a representation in which the last layer can tell the digits apart.`,
      } },
      { model: {
        tr: R`<b>Sıradaki bölüm (MLP):</b> Buradaki ağırlıkları biz seçtik. Gizli nöronlar için doğru cevap bilinmediğinden perceptron kuralı onları öğrenemez; bunu sigmoid gibi türevlenebilir aktivasyonlar ve geri yayılım (Rumelhart, Hinton, Williams, 1986) çözer.`,
        en: R`<b>Next chapter (MLP):</b> we chose the weights here. Since nobody knows the right answer for the hidden neurons, the perceptron rule cannot learn them; differentiable activations such as the sigmoid and backpropagation (Rumelhart, Hinton, Williams, 1986) solve this.`,
      } },
      { links: [{ tab: "theory", section: "p-xor" }, { tab: "code", section: "pc-exercises" }] },
    ],
  },
];

// =====================================================================
// Coding tab
// =====================================================================
const RUN_OUTPUT = [
  "$ .venv\\Scripts\\python.exe Perceptron.py",
  "--- AND ---",
  "Mistakes per epoch: [2, 2, 3, 2, 2, 3, 2, 1, 0]",
  "Boundary: 0.26*x1 +0.13*x2 -0.30 = 0",
  "Predictions: [0 0 0 1]  targets: [0 0 0 1]  accuracy: 100%",
  "Converged after 9 epochs.",
  "",
  "--- OR ---",
  "Mistakes per epoch: [1, 1, 0]",
  "Boundary: 0.06*x1 +0.03*x2 +0.00 = 0",
  "Predictions: [0 1 1 1]  targets: [0 1 1 1]  accuracy: 100%",
  "Converged after 3 epochs.",
  "",
  "--- NAND ---",
  "Mistakes per epoch: [2, 3, 1, 0]",
  "Boundary: -0.14*x1 -0.07*x2 +0.20 = 0",
  "Predictions: [1 1 1 0]  targets: [1 1 1 0]  accuracy: 100%",
  "Converged after 4 epochs.",
  "",
  "--- XOR ---",
  "Mistakes per epoch: [2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]",
  "Boundary: -0.14*x1 -0.07*x2 +0.10 = 0",
  "Predictions: [1 1 0 0]  targets: [0 1 1 0]  accuracy: 50%",
  "No convergence in 20 epochs: no straight line separates XOR.",
];

const TORCH_CODE = `import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0., 0., 0., 1.])           # AND

layer = nn.Linear(2, 1)                      # z = x @ W.T + b: the same weighted sum
lr = 0.1

with torch.no_grad():                        # the perceptron rule needs no gradients
    for epoch in range(20):
        mistakes = 0
        for x_i, y_i in zip(X, y):
            y_hat = (layer(x_i) > 0).float()     # step function
            error = y_i - y_hat                  # +1, -1 or 0
            if error != 0:
                layer.weight += lr * error * x_i
                layer.bias += lr * error
                mistakes += 1
        if mistakes == 0:
            break

print(f"epochs: {epoch + 1}")
print(f"w = {layer.weight.squeeze().tolist()}, b = {layer.bias.item():.2f}")
print((layer(X) > 0).int().squeeze().tolist())`;

const CODE = [
  {
    id: "pc-plan",
    title: PLACE_TITLES["pc-plan"],
    blocks: [
      { p: {
        tr: R`Bu sekmede Teori'de gördüğümüz perceptronu sıfırdan, yaklaşık 90 satırlık bir Python dosyası olarak yazıyoruz: <code>Perceptron.py</code>. Dosya projenin kök klasöründe hazır; adımları okurken kendi editörünüzde de açabilirsiniz. Gösterilen satır numaraları dosyadakilerle aynıdır.`,
        en: R`In this tab we write the perceptron from the Theory tab from scratch, as a Python file of about 90 lines: <code>Perceptron.py</code>. The file is ready in the project's root folder; you can open it in your own editor while reading. The line numbers shown are the file's.`,
      } },
      { list: [
        { tr: R`<b>Veri:</b> dört mantık kapısının (AND, OR, NAND, XOR) dört örneği.`, en: R`<b>Data:</b> the four examples of four logic gates (AND, OR, NAND, XOR).` },
        { tr: R`<b>Basamak fonksiyonu:</b> \(z > 0\) ise 1, değilse 0.`, en: R`<b>Step function:</b> 1 if \(z > 0\), otherwise 0.` },
        { tr: R`<b>Perceptron sınıfı:</b> ağırlıkları tutar, tahmin eder (<code>predict</code>) ve öğrenir (<code>fit</code>).`, en: R`<b>The Perceptron class:</b> holds the weights, predicts (<code>predict</code>) and learns (<code>fit</code>).` },
        { tr: R`<b>Ana program:</b> her kapı için yeni bir perceptron eğitir ve sonucu yazdırır.`, en: R`<b>Main program:</b> trains a new perceptron for each gate and prints the result.` },
      ] },
      { p: {
        tr: R`Tek gereken kütüphane NumPy; PyTorch ile birlikte <code>.venv</code>'e zaten kuruldu. Dosyayı proje klasöründe şöyle çalıştırırsınız:`,
        en: R`The only library needed is NumPy; it is already installed in <code>.venv</code> together with PyTorch. Run the file in the project folder like this:`,
      } },
      { term: "PowerShell", lines: ["$ .venv\\Scripts\\python.exe Perceptron.py"] },
      { callout: "tip", title: { tr: "Neden PyTorch değil?", en: "Why not PyTorch?" },
        text: { tr: R`Perceptron kuralı gradyan istemez; tek ihtiyacımız vektörlerle toplama ve çarpma. Her satırın ne yaptığını görmek için düz NumPy en açığı. 7. adımda aynı şeyi PyTorch ile yazıp CNN koduna bağlayacağız.`,
          en: R`The perceptron rule needs no gradients; all we need is adding and multiplying vectors. Plain NumPy shows most clearly what every line does. In step 7 we write the same with PyTorch and connect it to the CNN code.` } },
    ],
  },
  {
    id: "pc-data",
    title: PLACE_TITLES["pc-data"],
    blocks: [
      { p: {
        tr: R`Önce [[hyperparameter|hiperparametreler]], sonra veri. Her mantık kapısının yalnızca dört olası girdisi var; bunları bir kez <code>X</code> matrisine yazıp her kapının cevaplarını ayrı bir vektörde tutuyoruz.`,
        en: R`First the [[hyperparameter|hyperparameters]], then the data. Every logic gate has only four possible inputs; we write them once into the matrix <code>X</code> and keep each gate's answers in a separate vector.`,
      } },
      { file: { from: "#--------------- Hyperparameters", to: "}", notes: [
        { at: "learning_rate = 0.1", text: { tr: "η: bir düzeltmenin büyüklüğü (Teori, 3. kısım).", en: "η: the size of one correction (Theory, section 3)." } },
        { at: "seed = 0", text: { tr: "Başlangıç ağırlıkları rastgele; aynı seed aynı başlangıcı ve aynı eğitimi verir.", en: "The starting weights are random; the same seed gives the same start and the same training." } },
        { at: "X = np.array([[0, 0],", text: { tr: "Her satır bir örnek: `X`'in biçimi (4, 2). `dtype=float`, ağırlıklarla çarpılacağı için.", en: "Every row is one example: `X` has shape (4, 2). `dtype=float` because it is multiplied by the weights." } },
        { at: '"XOR":', text: { tr: "XOR'u bilerek ekliyoruz: öğrenilemeyeceğini göreceğiz.", en: "We add XOR on purpose: we will see that it cannot be learned." } },
      ] } },
    ],
  },
  {
    id: "pc-step",
    title: PLACE_TITLES["pc-step"],
    blocks: [
      { file: { from: "def step(z):", to: "return (z > 0).astype(int)" } },
      { p: {
        tr: R`<code>z &gt; 0</code> bir doğru/yanlış değeri (dizi için doğru/yanlış dizisi) verir; <code>astype(int)</code> bunu 1/0'a çevirir. Fonksiyon hem tek bir sayıyla hem de bütün bir diziyle çalışır: NumPy işlemi her elemana ayrı ayrı uygular. Eşitsizliğin kesin olduğuna dikkat edin: \(z = 0\) ise sonuç 0'dır, Teori'deki gibi.`,
        en: R`<code>z &gt; 0</code> gives a true/false value (for an array, an array of them); <code>astype(int)</code> turns it into 1/0. The function works on a single number and on a whole array alike: NumPy applies the operation to every element. Note that the inequality is strict: \(z = 0\) gives 0, as in Theory.`,
      } },
    ],
  },
  {
    id: "pc-class",
    title: PLACE_TITLES["pc-class"],
    blocks: [
      { file: { from: "class Perceptron:", to: "return step(z)", notes: [
        { at: "rng = np.random.default_rng(seed)", text: { tr: "Kendi rastgele sayı üretecimiz: aynı seed, aynı sayılar.", en: "Our own random number generator: same seed, same numbers." } },
        { at: "self.w = rng.normal(0.0, 0.5, n_inputs)", text: { tr: "Küçük rastgele ağırlıklar (ortalama 0, std. sapma 0,5); başlangıç sınırı rastgele bir doğrudur.", en: "Small random weights (mean 0, std. dev. 0.5); the starting boundary is a random line." } },
        { at: "z = X @ self.w + self.b", text: { tr: "`@` matris çarpımıdır: (4, 2) @ (2,) → (4,). Dört örneğin z'si tek satırda; `b` her birine eklenir.", en: "`@` is matrix multiplication: (4, 2) @ (2,) → (4,). The z of all four examples in one line; `b` is added to each." } },
      ] } },
      { p: {
        tr: R`<code>predict</code> Teori'deki formülün aynısı: \(\hat y = H(\mathbf w\cdot\mathbf x + b)\). Tek fark, dört örneği tek seferde hesaplaması: <code>X @ self.w</code>, her satırın [[weight|ağırlıklarla]] iç çarpımını verir.`,
        en: R`<code>predict</code> is the very formula from Theory: \(\hat y = H(\mathbf w\cdot\mathbf x + b)\). The only difference is that it computes all four examples at once: <code>X @ self.w</code> gives the dot product of every row with the [[weight|weights]].`,
      } },
    ],
  },
  {
    id: "pc-fit",
    title: PLACE_TITLES["pc-fit"],
    blocks: [
      { file: { from: "def fit(self, X, y, max_epochs=20):", to: "return history", notes: [
        { at: "for x_i, y_i in zip(X, y):", text: { tr: "Örnekler tek tek: bir epoch dört örnek.", en: "The examples one by one: an epoch is four examples." } },
        { at: "error = y_i - y_hat", text: { tr: "e = y − ŷ: +1, −1 ya da 0.", en: "e = y − ŷ: +1, −1 or 0." } },
        { at: "self.w += self.lr * error * x_i", text: { tr: "w ← w + η e x: kuralın kendisi.", en: "w ← w + η e x: the rule itself." } },
        { at: "if mistakes == 0:", text: { tr: "Hatasız bir epoch: yakınsadı, durabiliriz.", en: "An epoch without a mistake: converged, we can stop." } },
      ] } },
      { p: {
        tr: R`Formülle yan yana koyun: \(\mathbf w \leftarrow \mathbf w + \eta\,(y - \hat y)\,\mathbf x\) ve \(b \leftarrow b + \eta\,(y - \hat y)\). <code>if error != 0</code> koşulu güncelleme için şart değildir (\(e = 0\) iken güncelleme zaten hiçbir şey değiştirmezdi); asıl işi hataları saymaktır. <code>history</code> her epoch'taki hata sayısını tutar; Deneme alanındaki grafik de bunu çizer.`,
        en: R`Put it next to the formula: \(\mathbf w \leftarrow \mathbf w + \eta\,(y - \hat y)\,\mathbf x\) and \(b \leftarrow b + \eta\,(y - \hat y)\). The <code>if error != 0</code> test is not needed for the update (with \(e = 0\) the update would change nothing anyway); its real job is to count the mistakes. <code>history</code> keeps the number of mistakes in each epoch; the chart in the Playground draws the same thing.`,
      } },
      { callout: "info", title: { tr: "Örnek örnek mi, batch batch mi?", en: "Example by example, or batch by batch?" },
        text: { tr: R`Perceptron kuralı <i>çevrimiçi</i> bir kuraldır: her örnekten hemen sonra günceller. SimpleCNN ise 256 görüntülük [[batch|batch]]'lerin ortalama gradyanıyla (Adam algoritmasıyla) günceller. İkisi de aynı ailenin üyesidir: [[sgd|stokastik gradyan inişi]] ve onun türevleri.`,
          en: R`The perceptron rule is an <i>online</i> rule: it updates right after every example. SimpleCNN updates with the average gradient of [[batch|batches]] of 256 images (with the Adam optimizer). Both belong to the same family: [[sgd|stochastic gradient descent]] and its variants.` } },
    ],
  },
  {
    id: "pc-run",
    title: PLACE_TITLES["pc-run"],
    blocks: [
      { file: { from: 'if __name__ == "__main__":', to: 'print(f"No convergence' } },
      { term: "PowerShell", lines: RUN_OUTPUT },
      { list: [
        { tr: R`<b>AND</b> 9, <b>OR</b> 3, <b>NAND</b> 4 epoch'ta yakınsadı: son epoch'taki hata sayısı 0. Bulunan sınırlar Teori'deki çözümlerden farklı ama aynı işi görüyor; ayıran sınır tek değildir.`, en: R`<b>AND</b> converged in 9 epochs, <b>OR</b> in 3, <b>NAND</b> in 4: the last epoch has 0 mistakes. The boundaries found differ from the solutions in Theory but do the same job; the separating boundary is not unique.` },
        { tr: R`<b>OR</b>'da \(b = 0{,}00\): \((0, 0)\) noktası tam sınırın üzerinde ve \(z = 0\) olduğu için 0 sınıfına düşüyor. Kesin eşitsizliğin (\(z > 0\)) önemi burada.`, en: R`For <b>OR</b>, \(b = 0.00\): the point \((0, 0)\) lies right on the boundary and falls into class 0 because \(z = 0\). This is where the strict inequality (\(z > 0\)) matters.` },
        { tr: R`<b>XOR</b> hiç yakınsamadı: hata sayısı 4'te takılı kaldı ve doğruluk %50. 20 yerine 20.000 epoch verseniz de sonuç değişmez (Matematik, 6. kısım).`, en: R`<b>XOR</b> never converged: the mistakes got stuck at 4 and the accuracy is 50%. Giving it 20,000 epochs instead of 20 would not change that (Math, section 6).` },
        { tr: R`Epoch başına 4 hata ama doğruluk %50 nasıl olur? Hatalar epoch <i>sırasında</i>, ağırlıklar her örnekte değişirken sayılır; doğruluk ise eğitim bittikten sonra son ağırlıklarla ölçülür. 4. epoch'tan sonra sınır her epoch'ta dört örneğin her birinde yer değiştirir ve epoch sonunda hep aynı yere döner: kural bir döngüye girmiştir.`, en: R`How can there be 4 mistakes per epoch but 50% accuracy? The mistakes are counted <i>during</i> the epoch, while the weights change at every example; the accuracy is measured after training, with the final weights. From the 4th epoch on, the boundary moves at each of the four examples and is back in the same place at the end of every epoch: the rule is stuck in a cycle.` },
      ] },
    ],
  },
  {
    id: "pc-torch",
    title: PLACE_TITLES["pc-torch"],
    blocks: [
      { p: {
        tr: R`CNN bölümündeki (3. bölüm) SimpleCNN PyTorch ile yazıldı. Aynı perceptronu PyTorch'un yapı taşlarıyla yazınca bağlantı netleşir: <code>nn.Linear(2, 1)</code> tam olarak \(\mathbf w\cdot\mathbf x + b\) hesaplayan bir katmandır.`,
        en: R`SimpleCNN in chapter 3 is written with PyTorch. Writing the same perceptron with PyTorch's building blocks makes the link clear: <code>nn.Linear(2, 1)</code> is a layer that computes exactly \(\mathbf w\cdot\mathbf x + b\).`,
      } },
      { code: TORCH_CODE, name: "perceptron_torch.py", notes: [
        { at: "layer = nn.Linear(2, 1)", text: { tr: "2 girdi, 1 çıktı: ağırlık matrisi (1, 2), bias (1,). SimpleCNN'in `fc1` ve `fc2` katmanları da `nn.Linear`.", en: "2 inputs, 1 output: weight matrix (1, 2), bias (1,). SimpleCNN's `fc1` and `fc2` layers are `nn.Linear` too." } },
        { at: "with torch.no_grad():", text: { tr: "Ağırlıkları elle değiştiriyoruz; PyTorch'un gradyan takibine gerek yok.", en: "We change the weights by hand; no need for PyTorch's gradient tracking." } },
        { at: "y_hat = (layer(x_i) > 0).float()", text: { tr: "Basamak fonksiyonu. SimpleCNN'in `fc1` katmanında bunun yerine ReLU var; `fc2`'de ise hiç aktivasyon yok, ham skorlar (logit) doğrudan kayba gider.", en: "The step function. SimpleCNN's `fc1` has ReLU here instead; `fc2` has no activation at all, its raw scores (logits) go straight to the loss." } },
      ] },
      { term: "PowerShell", lines: ["$ .venv\\Scripts\\python.exe perceptron_torch.py", "epochs: 2", "w = [0.09470603615045547, 0.4793229401111603], b = -0.48", "[0, 0, 0, 1]"] },
      { p: {
        tr: R`SimpleCNN'deki bir tam bağlı katman, yan yana duran yüzlerce perceptron gibidir; farkları üç tane: basamak yerine [[relu|ReLU]] (son katman <code>fc2</code>'de hiç aktivasyon yok), perceptron kuralı yerine <code>loss.backward()</code> ile hesaplanan [[gradient|gradyanlar]] ve tek tek örnek yerine [[batch|batch]]'ler.`,
        en: R`A fully connected layer in SimpleCNN is like hundreds of perceptrons side by side, with three differences: [[relu|ReLU]] instead of the step (the last layer, <code>fc2</code>, has no activation at all), [[gradient|gradients]] computed by <code>loss.backward()</code> instead of the perceptron rule, and [[batch|batches]] instead of single examples.`,
      } },
      { links: [
        { chapter: "cnn", tab: "code", section: "c-model", title: { tr: "Modeli tanımlamak: katmanlar", en: "Defining the model: layers" } },
        { chapter: "cnn", tab: "code", section: "c-train-epoch", title: { tr: "Bir epoch eğitim", en: "Training for one epoch" } }] },
    ],
  },
  {
    id: "pc-exercises",
    title: PLACE_TITLES["pc-exercises"],
    blocks: [
      { p: {
        tr: R`Alıştırmalar kolaydan zora sıralı. Her birinde önce sonucu tahmin edin, sonra <code>Perceptron.py</code>'nin bir kopyasında deneyin.`,
        en: R`The exercises go from easy to hard. For each one, predict the result first, then try it on a copy of <code>Perceptron.py</code>.`,
      } },
      { list: [
        { tr: R`<b>Seed:</b> <code>seed</code>'i değiştirin (örneğin 1, 2, 3). Bulunan sınır ve epoch sayısı neden değişiyor? (İpucu: seed başlangıç ağırlıklarını belirler; ayıran pek çok doğru vardır ve kural ulaştığı ilk doğruda durur.)`, en: R`<b>Seed:</b> change <code>seed</code> (to 1, 2, 3, for example). Why do the boundary found and the number of epochs change? (Hint: the seed sets the starting weights; many lines separate the data, and the rule stops at the first one it reaches.)` },
        { tr: R`<b>Yeni kapı:</b> <code>GATES</code>'e NOR'u (<code>[1, 0, 0, 0]</code>) ekleyin. Çalıştırmadan önce: yakınsamasını bekler misiniz? (İpucu: NOR, OR'un tersidir; Teori'nin 1. kısmında onu hesaplayan \(w_1, w_2, b\)'yi elle bulabiliyor musunuz?)`, en: R`<b>A new gate:</b> add NOR (<code>[1, 0, 0, 0]</code>) to <code>GATES</code>. Before running: do you expect it to converge? (Hint: NOR is the opposite of OR; can you find \(w_1, w_2, b\) that compute it by hand, in section 1 of Theory?)` },
        { tr: R`<b>Öğrenme oranı:</b> <code>learning_rate</code>'i önce 0,01, sonra 1,0 yapın. AND kaç epoch'ta yakınsıyor? Sonra <code>__init__</code>'te ağırlıkları sıfırla başlatın (<code>np.zeros(n_inputs)</code>) ve aynı öğrenme oranlarını yeniden deneyin. Artık AND her öğrenme oranında aynı hataları yapıp aynı epoch'ta yakınsıyor; neden? (İpucu: Matematik sekmesi, 5. kısmın son adımı: sıfırdan başlayınca \(\eta\) ağırlıkları yalnızca ölçekler.)`,
          en: R`<b>Learning rate:</b> set <code>learning_rate</code> to 0.01, then to 1.0. In how many epochs does AND converge? Then start the weights at zero in <code>__init__</code> (<code>np.zeros(n_inputs)</code>) and try the same learning rates again. Now AND makes the same mistakes and converges in the same epoch for every learning rate; why? (Hint: Math tab, the last step of section 5: starting from zero, \(\eta\) only scales the weights.)` },
        { tr: R`<b>Karıştırma:</b> <code>fit</code>'te her epoch'un başında örneklerin sırasını karıştırın. (İpucu: üreteci <code>__init__</code>'te <code>self.rng</code> olarak saklayın; sonra <code>order = self.rng.permutation(len(X))</code> ve <code>zip(X[order], y[order])</code>.) AND yine yakınsıyor mu? XOR'u kurtarıyor mu? Yakınsama teoremi örneklerin sırası hakkında ne diyordu?`, en: R`<b>Shuffling:</b> in <code>fit</code>, shuffle the order of the examples at the start of every epoch. (Hint: keep the generator as <code>self.rng</code> in <code>__init__</code>; then <code>order = self.rng.permutation(len(X))</code> and <code>zip(X[order], y[order])</code>.) Does AND still converge? Does it rescue XOR? What did the convergence theorem say about the order of the examples?` },
        { tr: R`<b>XOR'u kurtarın:</b> girdilere üçüncü bir özellik ekleyin: \(x_3 = x_1 x_2\) (<code>X</code>'e üçüncü sütun, <code>n_inputs=3</code>; ana programdaki <code>w1, w2 = model.w</code> satırını da üç ağırlığa göre düzeltin). Perceptron artık XOR'u öğreniyor mu? Neden?`, en: R`<b>Rescue XOR:</b> add a third feature to the inputs: \(x_3 = x_1 x_2\) (a third column in <code>X</code>, <code>n_inputs=3</code>; also fix the line <code>w1, w2 = model.w</code> in the main program for three weights). Does the perceptron learn XOR now? Why?` },
      ], ordered: true },
      { callout: "tip", title: { tr: "Son alıştırma için ipucu", en: "Hint for the last exercise" },
        text: { tr: R`Üç boyutta dört nokta \((0,0,0), (0,1,0), (1,0,0), (1,1,1)\) olur ve artık bir düzlemle ayrılabilir: örneğin \(\mathbf w = (1, 1, -2)\), \(b = -0{,}5\). Doğru özelliği elle eklemek işe yarar; MLP'nin gizli katmanı ise böyle özellikleri kendisi öğrenir.`,
          en: R`In three dimensions the four points become \((0,0,0), (0,1,0), (1,0,0), (1,1,1)\) and a plane can now separate them: for example \(\mathbf w = (1, 1, -2)\), \(b = -0.5\). Adding the right feature by hand works; the hidden layer of an MLP learns such features by itself.` } },
    ],
  },
];

createLesson({ tab: "math", main: "#pm-main", toc: "#pm-toc", sections: MATH, widgets: WIDGETS, kicker: { tr: "Kısım", en: "Section" } });
createLesson({ tab: "code", main: "#pc-main", toc: "#pc-toc", sections: CODE, kicker: { tr: "Adım", en: "Step" },
  source: { text: PERCEPTRON_PY, name: "Perceptron.py" } });
})();
