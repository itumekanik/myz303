/*
 * Chapter 1: the perceptron. Everything runs in the browser.
 *
 * Theory tab: five sections, each with a live demo (the neuron, the decision boundary, the
 * learning rule, linear separability, XOR). Playground tab: place points and watch a
 * perceptron learn to separate them.
 *
 * Texts are { tr, en } pairs of HTML; \( TeX \) is typeset by math.js and [[key|text]] links a
 * glossary term. The panels are rebuilt on every language change; the demos keep their state.
 * Uses kit.js (the plane, sliders, number formatting), i18n.js (L), core.js (h, svg, $,
 * fmtNum, lineChart, showTip, …), math.js (texNum) and glossary.js (termify).
 */

/** This page's tabs (core.js navigates between them). */
import { TABS } from "../../shared/page.ts";
import { L } from "../../shared/i18n.js";
import { $, PLACE_TITLES, clamp, fmtNum, fmtPct, h, hideTip, lineChart, linearTicks, lutColor, niceStep, showTip, svg, svgText } from "../../shared/core.js";
import { KIT } from "../../shared/kit.js";
import { ruleStep, stepFn, zOf } from "./model.ts";

Object.assign(TABS, {
  theory: { tab: "#tab-theory", panel: "#panel-theory", hash: "#teori" },
  play: { tab: "#tab-play", panel: "#panel-play", hash: "#deneme" },
  math: { tab: "#tab-math", panel: "#panel-math", hash: "#matematik" },
  code: { tab: "#tab-code", panel: "#panel-code", hash: "#kodlama" },
  practice: { tab: "#tab-practice", panel: "#panel-practice", hash: "#alistirma" },
});

(() => {
const { R, seeded, gauss, nf, tx, txs, txp, html, formula, COLORS, readColors, redrawers, observers,
  createPlane, slider, segmented, chartHost } = KIT;

// =====================================================================
// The perceptron and its data
// =====================================================================
const INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]];
const GATES = {
  and: { name: "AND", y: [0, 0, 0, 1], solution: [[1, 1], -1.5], hint: { tr: "ikisi de 1 mi?", en: "are both 1?" } },
  or: { name: "OR", y: [0, 1, 1, 1], solution: [[1, 1], -0.5], hint: { tr: "en az biri 1 mi?", en: "is at least one 1?" } },
  nand: { name: "NAND", y: [1, 1, 1, 0], solution: [[-1, -1], 1.5], hint: { tr: "ikisi birden 1 değil mi?", en: "not both 1?" } },
  xor: { name: "XOR", y: [0, 1, 1, 0], solution: null, hint: { tr: "yalnızca biri mi 1?", en: "is exactly one 1?" } },
};
const gatePoints = (gate) => INPUTS.map((x, i) => ({ x, y: GATES[gate].y[i] }));


// =====================================================================
// Pieces of this chapter's demos (the plane, sliders and colors are in kit.js)
// =====================================================================
/** Shows z and the class under the pointer. */
function planeTooltip(plane, model) {
  plane.canvas.addEventListener("pointermove", (event) => {
    if (event.pointerType === "touch") return;
    const box = plane.canvas.getBoundingClientRect();
    const [x, y] = plane.toData(event.clientX - box.left, event.clientY - box.top);
    const m = model();
    if (!m) return hideTip();
    const z = zOf(m.w, m.b, [x, y]);
    showTip(event.clientX, event.clientY, `x₁ = ${nf(x)}, x₂ = ${nf(y)}`, [
      { value: nf(z), label: "z = w·x + b" },
      { value: String(stepFn(z)), label: L({ tr: "çıktı ŷ", en: "output ŷ" }), color: stepFn(z) ? COLORS.one : COLORS.zero },
    ]);
  });
  plane.canvas.addEventListener("pointerleave", hideTip);
}

const gateSeg = (current, onPick, gates = ["and", "or", "nand", "xor"]) =>
  segmented(gates.map((g) => [g, GATES[g].name]), current, onPick, L({ tr: "Mantık kapısı", en: "Logic gate" }));

/** Mistakes per epoch as a line chart. */
function mistakesChart(host, history, { max = null } = {}) {
  const n = history.length;
  const top = Math.max(max ?? 0, 1, ...history);
  const yStep = niceStep(top, 4);
  const xMax = Math.max(n, 5);
  lineChart(host, {
    label: L({ tr: "Epoch başına hata sayısı", en: "Mistakes per epoch" }),
    empty: L({ tr: "Henüz epoch yok", en: "No epochs yet" }),
    x: { min: 1, max: xMax, ticks: linearTicks(1, xMax, Math.max(1, niceStep(xMax, 6)), (v) => fmtNum(v, 0)) },
    y: { min: 0, max: Math.ceil(top / yStep) * yStep, ticks: linearTicks(0, top, yStep, (v) => fmtNum(v, 0)) },
    series: [{ points: history.map((m, i) => [i + 1, m]), color: "var(--series-2)", markers: n <= 40 }],
    tooltip: (i) => ({ title: `Epoch ${i + 1}`, rows: [{ value: fmtNum(history[i], 0), label: L({ tr: "hata", en: "mistakes" }), color: "var(--series-2)" }] }),
  });
}

/** The boundary equation in TeX: 0,5 x₁ + 0,5 x₂ − 0,7 = 0. */
const boundaryTex = (w, b) => `${tx(w[0])}\\,x_1 ${txs(w[1])}\\,x_2 ${txs(b)} = 0`;

// =====================================================================
// Texts
// =====================================================================
const TEXT = {
  intro: {
    h: PLACE_TITLES["p-intro"],
    p: [
      { tr: R`1958'de psikolog Frank Rosenblatt, beyindeki sinir hücresinden esinlenen bir makine tanıttı: [[perceptron|perceptron]]. Birkaç girdiyi [[weight|ağırlıklarla]] çarpıp toplar; toplam bir eşiği aşarsa 1, aşmazsa 0 der. Asıl yeniliği, ağırlıkların elle ayarlanmayıp <b>örneklerden öğrenilmesiydi</b>: perceptron yanlış cevap verdiğinde ağırlıklarını hatayı azaltacak yönde biraz değiştirir.`,
        en: R`In 1958 the psychologist Frank Rosenblatt introduced a machine inspired by the nerve cells of the brain: the [[perceptron|perceptron]]. It multiplies a few inputs by [[weight|weights]] and adds them up; if the sum exceeds a threshold it says 1, otherwise 0. The real novelty was that the weights were not set by hand but <b>learned from examples</b>: when the perceptron gives a wrong answer, it changes its weights a little in the direction that reduces the error.` },
      { tr: R`Bugünkü derin ağların her nöronu hâlâ aynı hesabı yapar: ağırlıklı toplam, [[bias|bias]] ve bir [[activation|aktivasyon fonksiyonu]]. Bu bölümde tek bir nöronun neler yapabildiğini ve en az onun kadar öğretici olan, neleri <b>yapamadığını</b> göreceğiz. Bölümler sırayla okunmak üzere yazıldı; her birinin sağındaki örnekle oynayın.`,
        en: R`Every neuron of today's deep networks still does the same computation: a weighted sum, a [[bias|bias]] and an [[activation|activation function]]. In this chapter we will see what a single neuron can do and, just as instructive, what it <b>cannot</b> do. The sections are meant to be read in order; play with the example on the right of each.` },
      { tr: R`<b>Bu bölümün sonunda</b> şunları yapabileceksiniz: bir perceptronun çıktısını elle hesaplamak; ağırlıkların ve bias'ın karar sınırını nasıl döndürüp kaydırdığını açıklamak; öğrenme kuralını adım adım uygulamak; öğrenmenin neden ancak veri doğrusal ayrılabilirse bittiğini ve tek bir perceptronun XOR'u neden öğrenemediğini açıklamak.`,
        en: R`<b>By the end of this chapter</b> you will be able to: compute a perceptron's output by hand; explain how the weights and the bias turn and shift the decision boundary; apply the learning rule step by step; explain why learning ends only when the data is linearly separable, and why a single perceptron cannot learn XOR.` },
    ],
    bio: {
      title: { tr: "Biyolojik nörondan yapay nörona", en: "From the biological neuron to the artificial one" },
      rows: [
        [{ tr: "Dendritler sinyal toplar", en: "Dendrites collect signals" }, { tr: R`girdiler \(x_1, x_2, \dots\)`, en: R`inputs \(x_1, x_2, \dots\)` }],
        [{ tr: "Sinapsların gücü", en: "The strength of the synapses" }, { tr: R`ağırlıklar \(w_1, w_2, \dots\)`, en: R`weights \(w_1, w_2, \dots\)` }],
        [{ tr: "Hücre gövdesi sinyalleri toplar", en: "The cell body adds the signals up" }, { tr: R`ağırlıklı toplam \(z\)`, en: R`weighted sum \(z\)` }],
        [{ tr: "Eşik aşılırsa nöron ateşlenir", en: "Past a threshold the neuron fires" }, { tr: R`bias \(b\) ve basamak fonksiyonu`, en: R`bias \(b\) and the step function` }],
        [{ tr: "Akson sinyali iletir", en: "The axon passes the signal on" }, { tr: R`çıktı \(\hat y\) (0 ya da 1)`, en: R`output \(\hat y\) (0 or 1)` }],
      ],
    },
    fact: {
      title: { tr: "Tarihten", en: "From history" },
      text: { tr: R`Rosenblatt'ın makalesi <i>"The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"</i> (Psychological Review, 1958). Perceptron önce bir IBM 704 bilgisayarında benzetimle gösterildi; donanım hâli <b>Mark I Perceptron</b> (1960), 20×20 = 400 ışık algılayıcısından oluşan bir "göze" bağlıydı ve ağırlıkları elektrik motorlarının çevirdiği potansiyometrelerdi. Basit şekilleri ayırt etmeyi öğrenebiliyordu.`,
        en: R`Rosenblatt's paper: <i>"The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"</i> (Psychological Review, 1958). The perceptron was first shown as a simulation on an IBM 704 computer; its hardware version, the <b>Mark I Perceptron</b> (1960), was wired to an "eye" of 20×20 = 400 light sensors, and its weights were potentiometers turned by electric motors. It could learn to tell simple shapes apart.` },
    },
  },
  neuron: {
    h: PLACE_TITLES["p-neuron"],
    p1: { tr: R`Perceptron her girdiyi kendi ağırlığıyla çarpar, çarpımları toplar ve [[bias|bias]] ekler. Bu sayıya \(z\) diyelim:`,
      en: R`The perceptron multiplies each input by its own weight, adds the products up and adds a [[bias|bias]]. Call this number \(z\):` },
    p2: { tr: R`Sonra bir [[step-function|basamak fonksiyonu]] karar verir: \(z\) pozitifse çıktı 1, değilse 0.`,
      en: R`Then a [[step-function|step function]] decides: if \(z\) is positive the output is 1, otherwise 0.` },
    p3: { tr: R`[[weight|Ağırlık]] bir girdinin ne kadar önemli olduğunu ve hangi yöne ittiğini söyler: büyük pozitif ağırlık girdiyi "evet" tarafına, negatif ağırlık "hayır" tarafına iter. Bias eşiği belirler: \(b = -0{,}7\) ise ağırlıklı toplamın 0,7'yi aşması gerekir. Yani bias, nöronun ne kadar kolay ateşlendiğini ayarlar.`,
      en: R`A [[weight|weight]] says how much an input matters and which way it pushes: a large positive weight pushes the input towards "yes", a negative one towards "no". The bias sets the threshold: with \(b = -0.7\) the weighted sum has to exceed 0.7. So the bias sets how easily the neuron fires.` },
    p4: { tr: R`<b>Deneyin:</b> Sağdaki perceptronun girdilerine tıklayarak 0 ya da 1 yapın, kaydırıcılarla ağırlıkları ve bias'ı değiştirin. Hedef, tablonun dört satırını da doğru yapan değerleri bulmak, yani perceptronu bir [[logic-gate|mantık kapısına]] dönüştürmek. Başlangıçtaki ağırlıklar AND'in 2 satırını yanlış yapıyor. İpucu: ağırlıklara dokunmadan yalnızca \(b\)'yi azaltmak yeter; \(b\) hangi aralıkta olmalı? AND'den sonra OR ve NAND'ı deneyin.`,
      en: R`<b>Try it:</b> click the inputs of the perceptron on the right to make them 0 or 1, and change the weights and the bias with the sliders. The goal is to find values that make all four rows of the table right, that is, to turn the perceptron into a [[logic-gate|logic gate]]. The starting weights get 2 rows of AND wrong. Hint: leave the weights alone and only lower \(b\); in what range must \(b\) be? After AND, try OR and NAND.` },
    model: { tr: R`<b>Bugünkü ağlarda:</b> Rosenblatt'ın perceptronunda girdiler bir görüntüden, 400 ışık algılayıcısından geliyordu; burada anlaşılır olsun diye iki girdi kullanıyoruz. CNN bölümündeki (3. bölüm) SimpleCNN'in son katmanındaki her nöron da aynı ağırlıklı toplamı 512 girdiyle hesaplar.`,
      en: R`<b>In today's networks:</b> the inputs of Rosenblatt's perceptron came from an image, from 400 light sensors; here we use two inputs to keep things clear. Every neuron in the last layer of SimpleCNN (the CNN chapter, chapter 3) computes the same weighted sum, over 512 inputs.` },
  },
  boundary: {
    h: PLACE_TITLES["p-boundary"],
    p1: { tr: R`İki girdiyi düzlemdeki bir \(\mathbf x = (x_1, x_2)\) noktasının koordinatları gibi düşünün. \(z = 0\) olan noktalar bir <b>doğru</b> oluşturur:`,
      en: R`Think of the two inputs as the coordinates of a point \(\mathbf x = (x_1, x_2)\) in the plane. The points where \(z = 0\) form a <b>straight line</b>:` },
    p2: { tr: R`Bu doğru düzlemi ikiye böler: bir yanda \(z > 0\) (çıktı 1, yeşil bölge), öbür yanda \(z < 0\) (çıktı 0). Buna [[decision-boundary|karar sınırı]] denir. Perceptron bu yüzden bir <b>doğrusal sınıflandırıcıdır</b>: ne kadar eğitilirse eğitilsin, sınırı hep düz bir çizgidir.`,
      en: R`The line splits the plane in two: on one side \(z > 0\) (output 1, the green region), on the other \(z < 0\) (output 0). It is called the [[decision-boundary|decision boundary]]. That is why the perceptron is a <b>linear classifier</b>: however long it is trained, its boundary is always a straight line.` },
    p3: { tr: R`Ağırlık vektörü \(\mathbf w = (w_1, w_2)\) sınıra <b>diktir</b> ve 1 tarafını gösterir: \(\mathbf w\)'yi döndürmek sınırı döndürür. Bias ise sınırı kendine paralel kaydırır; sınırın orijine uzaklığı \(|b| / \lVert \mathbf w \rVert\)'dir (\(\lVert \mathbf w \rVert = \sqrt{w_1^2 + w_2^2}\), \(\mathbf w\)'nin uzunluğu). \(\mathbf w\) ile \(b\)'yi aynı pozitif sayıyla çarpmak sınırı yerinden oynatmaz, yalnızca \(z\)'nin büyüklüğünü değiştirir.`,
      en: R`The weight vector \(\mathbf w = (w_1, w_2)\) is <b>perpendicular</b> to the boundary and points to the 1 side: turning \(\mathbf w\) turns the boundary. The bias shifts the boundary parallel to itself; its distance from the origin is \(|b| / \lVert \mathbf w \rVert\) (\(\lVert \mathbf w \rVert = \sqrt{w_1^2 + w_2^2}\), the length of \(\mathbf w\)). Multiplying \(\mathbf w\) and \(b\) by the same positive number does not move the boundary; it only changes the size of \(z\).` },
    p4: { tr: R`<b>Deneyin:</b> Sağda sınırın ortasındaki tutamağı sürükleyerek sınırı kaydırın (\(b\) değişir), \(\mathbf w\) okunun ucunu sürükleyerek döndürün (\(\mathbf w\) değişir). Fareyle düzlemin üzerinde gezinince her noktanın \(z\) değeri görünür. Sonra <b>w ve b'yi 2 ile çarp</b>'a basın: sınır kıpırdamaz, yalnızca \(\lVert\mathbf w\rVert\) ve \(z\) değerleri iki katına çıkar. İkiden fazla girdide fikir aynıdır: 3 girdide sınır bir düzlem, \(n\) girdide bir [[hyperplane|hiper düzlemdir]].`,
      en: R`<b>Try it:</b> on the right, drag the handle in the middle of the boundary to shift it (\(b\) changes) and the tip of the \(\mathbf w\) arrow to turn it (\(\mathbf w\) changes). Hover over the plane to see the \(z\) value of any point. Then press <b>Multiply w and b by 2</b>: the boundary does not move; only \(\lVert\mathbf w\rVert\) and the \(z\) values double. With more than two inputs the idea is the same: with 3 inputs the boundary is a plane, with \(n\) inputs a [[hyperplane|hyperplane]].` },
    model: { tr: R`<b>Aynı perceptron:</b> Bu düzlem yukarıdaki 1. kısmın perceptronunu gösterir. Birinde yaptığınız değişiklik ötekine de yansır; tablodaki dört satır buradaki dört noktadır.`,
      en: R`<b>The same perceptron:</b> this plane shows the perceptron of section 1. A change in one shows in the other; the four rows of the table are the four points here.` },
  },
  rule: {
    h: PLACE_TITLES["p-rule"],
    p1: { tr: R`Ağırlıkları elle bulmak iki girdide kolay ama 400 girdide pratikte imkânsız. Rosenblatt'ın [[perceptron-rule|öğrenme kuralı]] bunu örneklerden kendisi yapar. Örnekler sırayla gösterilir. Perceptron doğru cevap verirse hiçbir şey değişmez; yanlış cevap verirse ağırlıklar hataya, \(e = y - \hat y\)'ye göre düzeltilir (\(y\) doğru cevap, \(\hat y\) perceptronun cevabı, \(\eta\) küçük pozitif bir sayı):`,
      en: R`Finding the weights by hand is easy with two inputs but practically impossible with 400. Rosenblatt's [[perceptron-rule|learning rule]] does it by itself, from examples. The examples are shown one by one. If the perceptron answers right, nothing changes; if it answers wrong, the weights are corrected according to the error \(e = y - \hat y\) (\(y\) is the right answer, \(\hat y\) the perceptron's answer, \(\eta\) a small positive number):` },
    cases: [
      { tr: R`<b>\(y = 1\), \(\hat y = 0\)</b> (1 olmalıydı, \(e = +1\)): \(\mathbf w\)'ye \(\eta\,\mathbf x\), \(b\)'ye \(\eta\) eklenir. Bu örneğin \(z\)'si büyür.`, en: R`<b>\(y = 1\), \(\hat y = 0\)</b> (should have been 1, \(e = +1\)): \(\eta\,\mathbf x\) is added to \(\mathbf w\) and \(\eta\) to \(b\). This example's \(z\) grows.` },
      { tr: R`<b>\(y = 0\), \(\hat y = 1\)</b> (0 olmalıydı, \(e = -1\)): \(\mathbf w\)'den \(\eta\,\mathbf x\), \(b\)'den \(\eta\) çıkarılır. \(z\) küçülür.`, en: R`<b>\(y = 0\), \(\hat y = 1\)</b> (should have been 0, \(e = -1\)): \(\eta\,\mathbf x\) is subtracted from \(\mathbf w\) and \(\eta\) from \(b\). \(z\) shrinks.` },
      { tr: R`<b>Doğru cevap</b> (\(e = 0\)): değişiklik yok.`, en: R`<b>Right answer</b> (\(e = 0\)): no change.` },
    ],
    p2: { tr: R`Değeri 0 olan bir girdinin ağırlığı değişmez, çünkü o girdi hataya katkıda bulunmamıştır. [[learning-rate|Öğrenme oranı]] \(\eta\) her düzeltmenin büyüklüğüdür. Bütün örneklerin bir kez gösterilmesine [[epoch|epoch]] denir; bir epoch boyunca hiç hata olmazsa öğrenme biter.`,
      en: R`The weight of an input that is 0 does not change, because that input did not contribute to the error. The [[learning-rate|learning rate]] \(\eta\) is the size of each correction. Showing every example once is called an [[epoch|epoch]]; when an epoch passes without a mistake, learning is over.` },
    p3: { tr: R`<b>Deneyin:</b> <b>Sonraki örnek</b> ile kuralı adım adım uygulayın. Mavi halkalı nokta şu anki örnektir; kesikli çizgi düzeltmeden önceki sınırdır. Her adımdan önce tahmin edin: bu örnek yanlış mı sınıflanacak, yanlışsa sınır hangi yöne kayacak? \((0, 0)\) örneğinde yalnızca \(b\)'nin değiştiğine dikkat edin. Sonra XOR'u seçip ne olduğuna bakın.`,
      en: R`<b>Try it:</b> apply the rule step by step with <b>Next example</b>. The point with the blue ring is the current example; the dashed line is the boundary before the correction. Before each step, predict: will this example be misclassified, and if so, which way will the boundary move? Notice that for the example \((0, 0)\) only \(b\) changes. Then pick XOR and see what happens.` },
    model: { tr: R`<b>Bugünkü ağlarda:</b> SimpleCNN de aynı fikirle eğitilir: hata, ağırlıkları onu azaltacak yönde küçük adımlarla değiştirir. Fark, yönün [[gradient|gradyanla]] hesaplanması. Matematik sekmesinde perceptron kuralının da bir gradyan inişi olduğunu göreceğiz.`,
      en: R`<b>In today's networks:</b> SimpleCNN is trained with the same idea: the error changes the weights in small steps in the direction that reduces it. The difference is that the direction is computed with the [[gradient|gradient]]. The Math tab shows that the perceptron rule is a gradient descent too.` },
  },
  converge: {
    h: PLACE_TITLES["p-converge"],
    p1: { tr: R`İki sınıfı tek bir doğruyla ayırmak mümkünse veriye [[linear-separability|doğrusal ayrılabilir]] denir. <b>Perceptron yakınsama teoremi</b> (Novikoff, 1962), veri doğrusal ayrılabilirse perceptron kuralının sonlu sayıda düzeltmeden sonra bütün örnekleri doğru sınıflandıracağını söyler; başlangıç ağırlıkları ve örneklerin sırası ne olursa olsun.`,
      en: R`If one straight line can separate the two classes, the data is called [[linear-separability|linearly separable]]. The <b>perceptron convergence theorem</b> (Novikoff, 1962) says that if the data is linearly separable, the perceptron rule classifies every example correctly after a finite number of corrections, whatever the starting weights and the order of the examples.` },
    p2: { tr: R`Düzeltme sayısı sınıflar arasındaki boşluğa, yani [[margin|marja]] bağlıdır: sınıflar birbirine ne kadar yakınsa, sınırı aralarına oturtmak o kadar çok düzeltme ister. Teorem bir üst sınır da verir: düzeltme sayısı en fazla \((R/\gamma)^2\)'dir (\(R\) en uzak noktanın uzaklığı, \(\gamma\) marj; tam tanımları Matematik sekmesinde). Bu, en kötü durum için bir sınırdır; gerçek düzeltme sayısı genellikle çok daha azdır.`,
      en: R`The number of corrections depends on the gap between the classes, the [[margin|margin]]: the closer the classes, the more corrections it takes to fit the boundary between them. The theorem also gives an upper bound: at most \((R/\gamma)^2\) corrections (\(R\) is the distance of the farthest point, \(\gamma\) the margin; the Math tab defines both exactly). It is a worst-case bound; the actual number of corrections is usually far smaller.` },
    p3: { tr: R`<b>Deneyin:</b> Önce tahmin edin: boşluk küçülünce düzeltme sayısı ve teorem sınırı ne olur? Sonra boşluğu büyük (0,4) ve küçük (0,02) yapıp her birinde <b>Eğit</b>'e basın; her deneme tabloya bir satır ekler. Tek bir veri kümesinde sonuç gürültülüdür: <b>Yeni veri</b> ile birkaç kez tekrarlayın. Teorem sınırı boşluk küçüldükçe hızla büyür; düzeltme sayısı da ortalamada artar ama hep sınırın çok altında kalır.`,
      en: R`<b>Try it:</b> first predict: what happens to the number of corrections and to the theorem bound as the gap shrinks? Then set a large gap (0.4) and a small one (0.02) and press <b>Train</b> for each; every run adds a row to the table. On a single dataset the result is noisy: repeat a few times with <b>New data</b>. The theorem bound grows fast as the gap shrinks; the number of corrections grows too on average, but always stays far below the bound.` },
    model: { tr: R`<b>Dikkat:</b> Perceptron örnekleri ayıran <i>ilk</i> sınırda durur, en iyisinde değil. Sınır bir sınıfa çok yakın kalabilir; o yakınlara düşen yeni bir örnek kolayca yanlış sınıflanabilir. Sonraki yöntemler (örneğin destek vektör makineleri) marjı en büyük yapan sınırı arar.`,
      en: R`<b>Note:</b> the perceptron stops at the <i>first</i> boundary that separates the examples, not the best one. The boundary can end up very close to one class, and a new example that falls near it can easily be misclassified. Later methods (support vector machines, for example) look for the boundary with the largest margin.` },
  },
  xor: {
    h: PLACE_TITLES["p-xor"],
    p1: { tr: R`XOR ("ya biri ya öteki") kapısı, girdilerden yalnızca biri 1 olduğunda 1 verir. Dört noktaya bakın: iki yeşil nokta bir köşegende, iki kırmızı çarpı öbür köşegende. Hiçbir doğru yeşilleri bir yana, kırmızıları öbür yana ayıramaz: XOR doğrusal ayrılabilir değildir ve tek bir perceptron onu <b>asla</b> öğrenemez. Kural sonsuza kadar düzeltme yapar, hata hiçbir epoch'ta sıfıra inmez. Buna [[xor-problem|XOR problemi]] denir.`,
      en: R`The XOR ("one or the other") gate gives 1 when exactly one of the inputs is 1. Look at the four points: the two green dots are on one diagonal, the two red crosses on the other. No straight line can put the greens on one side and the reds on the other: XOR is not linearly separable, and a single perceptron can <b>never</b> learn it. The rule keeps correcting forever and the mistakes never reach zero in an epoch. This is called the [[xor-problem|XOR problem]].` },
    try: { tr: R`<b>Deneyin:</b> Önce tahmin edin: epoch başına hata hiç 0'a inecek mi? Sonra sağda <b>Eğit</b>'e basın ve grafiği izleyin.`,
      en: R`<b>Try it:</b> first predict: will the mistakes per epoch ever reach 0? Then press <b>Train</b> on the right and watch the chart.` },
    p2: { tr: R`1969'da Marvin Minsky ve Seymour Papert, <i>Perceptrons</i> kitabında tek katmanlı perceptronların bu tür sınırlarını matematiksel olarak gösterdi. Kitap, sinir ağlarına olan ilginin ve desteğin uzun yıllar azalmasına katkıda bulundu; bu döneme sonradan ilk "yapay zekâ kışı" dendi.`,
      en: R`In 1969 Marvin Minsky and Seymour Papert showed limits like this of single-layer perceptrons mathematically, in their book <i>Perceptrons</i>. The book contributed to a long decline of interest in and funding for neural networks; that period was later called the first "AI winter".` },
    p3: { tr: R`Çıkış yolu perceptronları <b>art arda bağlamak</b>: XOR = AND(OR, NAND). İlk katmandaki iki perceptron iki ayrı doğru çizer, ikinci katmandaki perceptron bu iki cevabı birleştirir. Sağda <b>İki katman</b>'ı seçin. Ancak bu ağırlıkları biz elle seçtik: aradaki [[hidden-layer|gizli katmanın]] ağırlıkları perceptron kuralıyla öğrenilemez, çünkü gizli bir nöron için "doğru cevabı" kimse söylemez. Bunu çözen geri yayılım (1986), bir sonraki bölümün, MLP'nin konusu.`,
      en: R`The way out is to <b>chain perceptrons</b>: XOR = AND(OR, NAND). The two perceptrons of the first layer draw two separate lines; the perceptron of the second layer combines their answers. Pick <b>Two layers</b> on the right. But we chose these weights by hand: the weights of the [[hidden-layer|hidden layer]] in between cannot be learned with the perceptron rule, because nobody tells a hidden neuron what its "right answer" is. Backpropagation (1986), which solves this, is the subject of the next chapter, the MLP.` },
    model: { tr: R`<b>Neden geri yayılım basamak fonksiyonuyla çalışmaz?</b> Basamak fonksiyonunun eğimi her yerde 0'dır (\(z = 0\)'da ise tanımsız); hata sinyali ondan geriye geçemez. Bu yüzden çok katmanlı ağlar sigmoid ya da [[relu|ReLU]] gibi eğimi olan aktivasyonlar kullanır.`,
      en: R`<b>Why doesn't backpropagation work with the step function?</b> Its slope is 0 everywhere (and undefined at \(z = 0\)); no error signal can pass back through it. That is why multilayer networks use activations with a slope, such as the sigmoid or [[relu|ReLU]].` },
    recap: { tr: R`<b>Özet:</b> Perceptron \(z = w_1x_1 + w_2x_2 + b\) toplamını hesaplar ve \(z > 0\) ise 1, değilse 0 der. Karar sınırı \(z = 0\) doğrusudur: \(\mathbf w\) onu döndürür, \(b\) kaydırır. Öğrenme kuralı her hatada \(\mathbf w\)'ye \(\eta\,e\,\mathbf x\), \(b\)'ye \(\eta\,e\) ekler. Veri doğrusal ayrılabilirse kural sonlu sayıda düzeltmeden sonra durur; değilse, XOR'daki gibi, hiç durmaz. Perceptronları katmanlara dizmek XOR'u çözer, ama gizli katmanı eğitmek için basamak yerine türevlenebilir bir aktivasyon ve geri yayılım gerekir: sıradaki bölümün konusu.`,
      en: R`<b>Recap:</b> the perceptron computes \(z = w_1x_1 + w_2x_2 + b\) and says 1 if \(z > 0\), otherwise 0. The decision boundary is the line \(z = 0\): \(\mathbf w\) turns it, \(b\) shifts it. The learning rule adds \(\eta\,e\,\mathbf x\) to \(\mathbf w\) and \(\eta\,e\) to \(b\) at every mistake. If the data is linearly separable the rule stops after a finite number of corrections; if not, as with XOR, it never stops. Stacking perceptrons in layers solves XOR, but training the hidden layer needs a differentiable activation instead of the step, and backpropagation: the subject of the next chapter.` },
  },
};

// =====================================================================
// Theory 1 · the neuron (shared state with section 2)
// =====================================================================
const P = { w: [0.5, 0.5], b: -0.2, gate: "and", x: [1, 0] };
const neuronViews = new Set();   // sections 1 and 2 redraw together
const neuronChanged = () => { for (const view of neuronViews) view(); };

function neuronDiagram() {
  const box = h("div", { class: "pc-neuron" });
  const root = svg("svg", { viewBox: "0 0 640 250", role: "group", "aria-label": L({ tr: "Yapay nöron: girişleri değiştirmek için tıklayın", en: "The artificial neuron: click the inputs to change them" }) });
  box.append(root);
  const g = (cls) => svg("g", { class: cls });
  const edges = g("pc-edges");
  const nodes = g("pc-nodes");
  root.append(edges, nodes);

  const inPos = [[70, 70], [70, 190]];
  const sum = [330, 130];
  const stepBox = [446, 130];
  const out = [572, 130];
  const edgeEls = inPos.map(([x, y], i) => {
    const line = svg("line", { x1: x + 30, y1: y, x2: sum[0] - 34, y2: sum[1] + (i ? 12 : -12), class: "pc-edge" });
    const wLabel = svgText((x + sum[0]) / 2 - 6, (y + sum[1]) / 2 + (i ? 26 : -16), "", "pc-wlabel", "middle");
    const prod = svgText((x + sum[0]) / 2 + 6, (y + sum[1]) / 2 + (i ? -6 : 14), "", "pc-prod", "middle");
    edges.append(line, wLabel, prod);
    return { line, wLabel, prod };
  });
  const biasLine = svg("line", { x1: sum[0], y1: 22, x2: sum[0], y2: sum[1] - 36, class: "pc-edge bias" });
  const biasHead = svg("path", { d: `M${sum[0] - 5},${sum[1] - 44} L${sum[0]},${sum[1] - 35} L${sum[0] + 5},${sum[1] - 44}`, class: "pc-head" });
  const biasLabel = svgText(sum[0] + 10, 30, "", "pc-wlabel", "start");
  edges.append(biasLine, biasHead, biasLabel,
    svg("line", { x1: sum[0] + 34, y1: sum[1], x2: stepBox[0] - 34, y2: stepBox[1], class: "pc-edge" }),
    svg("line", { x1: stepBox[0] + 34, y1: stepBox[1], x2: out[0] - 30, y2: out[1], class: "pc-edge" }));

  const inputs = inPos.map(([x, y], i) => {
    const grp = svg("g", { class: "pc-input", tabindex: 0, role: "button" });
    grp.append(svg("circle", { cx: x, cy: y, r: 28 }), svgText(x, y + 7, "", "pc-in-value", "middle"),
      svgText(x - 40, y + 6, "", "pc-in-name", "end"));
    grp.querySelectorAll("text")[1].innerHTML = `x<tspan baseline-shift="sub" font-size="12">${i + 1}</tspan>`;
    const toggle = () => { P.x[i] = 1 - P.x[i]; neuronChanged(); };
    grp.addEventListener("click", toggle);
    grp.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
    nodes.append(grp);
    return grp;
  });
  const sumNode = svg("g", { class: "pc-sum" });
  sumNode.append(svg("circle", { cx: sum[0], cy: sum[1], r: 34 }), svgText(sum[0], sum[1] + 10, "Σ", "pc-sigma", "middle"));
  const zLabel = svgText(sum[0], sum[1] + 58, "", "pc-zlabel", "middle");
  const stepNode = svg("g", { class: "pc-step" });
  stepNode.append(svg("rect", { x: stepBox[0] - 34, y: stepBox[1] - 30, width: 68, height: 60, rx: 9 }),
    svg("path", { d: `M${stepBox[0] - 22},${stepBox[1] + 14} H${stepBox[0]} V${stepBox[1] - 14} H${stepBox[0] + 22}`, class: "pc-step-glyph" }));
  const stepLabel = svgText(stepBox[0], stepBox[1] + 52, "", "pc-caption", "middle");
  const outNode = svg("g", { class: "pc-out" });
  outNode.append(svg("circle", { cx: out[0], cy: out[1], r: 28 }), svgText(out[0], out[1] + 7, "", "pc-in-value", "middle"));
  const outLabel = svgText(out[0], out[1] + 52, "", "pc-caption", "middle");
  nodes.append(sumNode, zLabel, stepNode, stepLabel, outNode, outLabel);
  const live = h("div", { class: "pc-live" });
  box.append(live);

  const weightColor = (w) => lutColor("div", 127.5 + 127.5 * clamp(w / 2, -1, 1));
  box.update = () => {
    const z = zOf(P.w, P.b, P.x);
    const yHat = stepFn(z);
    root.setAttribute("aria-label", L({ tr: `Perceptron: x₁ = ${P.x[0]}, x₂ = ${P.x[1]}, z = ${nf(z)}, çıktı ${yHat}`, en: `Perceptron: x₁ = ${P.x[0]}, x₂ = ${P.x[1]}, z = ${nf(z)}, output ${yHat}` }));
    edgeEls.forEach((e, i) => {
      e.line.style.stroke = weightColor(P.w[i]);
      e.line.style.strokeWidth = 1.5 + 3.5 * Math.min(1, Math.abs(P.w[i]) / 2);
      e.line.style.opacity = P.x[i] ? 1 : 0.35;
      e.wLabel.innerHTML = `w<tspan baseline-shift="sub" font-size="11">${i + 1}</tspan> = ${nf(P.w[i])}`;
      e.prod.textContent = `${nf(P.w[i])}·${P.x[i]} = ${nf(P.w[i] * P.x[i])}`;
    });
    biasLine.style.stroke = weightColor(P.b);
    biasLabel.textContent = `b = ${nf(P.b)}`;
    inputs.forEach((grp, i) => {
      grp.classList.toggle("on", P.x[i] === 1);
      grp.querySelector("text").textContent = String(P.x[i]);
      grp.setAttribute("aria-label", L({ tr: `Girdi x${i + 1} = ${P.x[i]}; değiştirmek için tıklayın`, en: `Input x${i + 1} = ${P.x[i]}; click to change` }));
    });
    zLabel.textContent = `z = ${nf(z)}`;
    stepLabel.textContent = L({ tr: "z > 0 ?", en: "z > 0 ?" });
    outNode.classList.toggle("on", yHat === 1);
    outNode.querySelector("text").textContent = String(yHat);
    outLabel.innerHTML = L({ tr: "çıktı ŷ", en: "output ŷ" });
    live.textContent = `\\(z = ${txp(P.w[0])}\\cdot ${P.x[0]} + ${txp(P.w[1])}\\cdot ${P.x[1]} ${txs(P.b)} = ${tx(z)} ${z > 0 ? ">" : "\\le"} 0 \\;\\Rightarrow\\; \\hat y = ${yHat}\\)`;
  };
  return box;
}

function buildNeuronDemo() {
  const demo = h("div", { class: "t-demo" });
  const diagram = neuronDiagram();
  const sliders = [
    slider("", { min: -2, max: 2, step: 0.05, value: P.w[0], onInput: (v) => { P.w[0] = v; neuronChanged(); } }),
    slider("", { min: -2, max: 2, step: 0.05, value: P.w[1], onInput: (v) => { P.w[1] = v; neuronChanged(); } }),
    slider("", { min: -2, max: 2, step: 0.05, value: P.b, onInput: (v) => { P.b = v; neuronChanged(); } }),
  ];
  sliders[0].querySelector(".pc-slider-name").innerHTML = "w<sub>1</sub>";
  sliders[1].querySelector(".pc-slider-name").innerHTML = "w<sub>2</sub>";
  sliders[2].querySelector(".pc-slider-name").textContent = "b";
  const seg = gateSeg(P.gate, (g) => { P.gate = g; neuronChanged(); });
  const solve = h("button", { type: "button", class: "btn small", text: L({ tr: "Bir çözüm göster", en: "Show a solution" }) });
  solve.addEventListener("click", () => {
    const sol = GATES[P.gate].solution;
    if (!sol) { msg.dataset.flash = "1"; neuronChanged(); return; }
    P.w = [...sol[0]];
    P.b = sol[1];
    neuronChanged();
  });
  const table = h("table", { class: "data-table compact pc-truth" });
  const msg = h("p", { class: "pc-msg", role: "status" });
  demo.append(
    h("div", { class: "t-controls" }, h("span", { class: "mini-title pc-inline", text: L({ tr: "Hedef kapı", en: "Target gate" }) }), seg, solve),
    diagram,
    h("div", { class: "t-controls" }, ...sliders),
    h("div", { class: "table-wrap" }, table),
    msg);

  const update = () => {
    diagram.update();
    sliders[0].set(P.w[0]);
    sliders[1].set(P.w[1]);
    sliders[2].set(P.b);
    seg.set(P.gate);
    const gate = GATES[P.gate];
    let right = 0;
    const rows = INPUTS.map((x, i) => {
      const z = zOf(P.w, P.b, x);
      const yHat = stepFn(z);
      const ok = yHat === gate.y[i];
      if (ok) right++;
      const current = x[0] === P.x[0] && x[1] === P.x[1];
      const tr = h("tr", { class: `${current ? "current" : ""} ${ok ? "" : "wrong"}`.trim(), tabindex: 0 },
        h("td", { text: String(x[0]) }), h("td", { text: String(x[1]) }), h("td", { text: nf(z) }),
        h("td", { text: String(yHat) }), h("td", { text: String(gate.y[i]) }),
        h("td", { class: ok ? "pc-ok" : "pc-bad", text: ok ? "✓" : "✗" }));
      const pick = () => { P.x = [...x]; neuronChanged(); };
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      return tr;
    });
    table.replaceChildren(
      h("thead", {}, h("tr", {}, ...["x₁", "x₂", "z", "ŷ", L({ tr: `${gate.name} hedefi y`, en: `${gate.name} target y` }), ""].map((c) => h("th", { text: c })))),
      h("tbody", {}, ...rows));
    const flash = msg.dataset.flash === "1";
    delete msg.dataset.flash;
    msg.className = `pc-msg ${right === 4 ? "good" : ""}`;
    if (right === 4) msg.textContent = L({ tr: `4/4 doğru: bu ağırlıklarla perceptron ${gate.name} kapısını hesaplıyor.`, en: `4/4 right: with these weights the perceptron computes the ${gate.name} gate.` });
    else if (P.gate === "xor") {
      msg.replaceChildren(L({ tr: `${right}/4 doğru. XOR için 4/4 yapan ağırlık yok; nedenini `, en: `${right}/4 right. No weights get 4/4 for XOR; see why in ` }),
        h("a", { href: "#", "data-go": "theory:p-xor", text: L({ tr: "5. kısımda görün", en: "section 5" }) }), ".");
      if (flash) msg.classList.add("flash");
    } else msg.textContent = L({ tr: `${right}/4 doğru. Hedef: ${gate.name} (${L(gate.hint)}). Ağırlıkları ve bias'ı değiştirin.`, en: `${right}/4 right. Target: ${gate.name} (${L(gate.hint)}). Change the weights and the bias.` });
  };
  neuronViews.add(update);
  update();
  return demo;
}

// =====================================================================
// Theory 2 · the decision boundary (draggable)
// =====================================================================
function buildBoundaryDemo() {
  const demo = h("div", { class: "t-demo" });
  const seg = gateSeg(P.gate, (g) => { P.gate = g; neuronChanged(); });
  const scale = h("button", { type: "button", class: "btn small", text: L({ tr: "w ve b'yi 2 ile çarp", en: "Multiply w and b by 2" }) });
  const half = h("button", { type: "button", class: "btn small", text: L({ tr: "… ½ ile çarp", en: "… by ½" }) });
  // Doubling must keep every value inside the sliders' range [-2, 2]: clamping one of them would move the boundary
  const canDouble = () => Math.max(Math.abs(P.w[0]), Math.abs(P.w[1]), Math.abs(P.b)) <= 1 + 1e-9;
  scale.addEventListener("click", () => { if (!canDouble()) return; P.w = P.w.map((v) => v * 2); P.b *= 2; neuronChanged(); });
  half.addEventListener("click", () => { P.w = P.w.map((v) => v / 2); P.b /= 2; neuronChanged(); });
  demo.append(h("div", { class: "t-controls" }, seg, scale, half));
  const stage = h("div", { class: "pc-stage" });
  demo.append(stage);
  const plane = createPlane(stage, { label: L({ tr: "Karar sınırı ve dört girdi noktası", en: "The decision boundary and the four input points" }) });
  planeTooltip(plane, () => P);

  // Two handles over the canvas: shift (middle of the boundary) and turn (tip of w)
  const shiftHandle = h("button", { type: "button", class: "pc-handle shift", "aria-label": L({ tr: "Sınırı kaydır (ok tuşları)", en: "Shift the boundary (arrow keys)" }) });
  const turnHandle = h("button", { type: "button", class: "pc-handle turn", "aria-label": L({ tr: "w'yi döndür (ok tuşları)", en: "Turn w (arrow keys)" }) });
  plane.wrap.append(shiftHandle, turnHandle);
  plane.afterDraw = () => {
    for (const [el, key] of [[shiftHandle, "foot"], [turnHandle, "tip"]]) {
      const p = plane.handles?.[key];
      el.hidden = !p;
      if (!p) continue;
      const [px, py] = plane.toPx(...p);
      const inside = px >= 0 && py >= 0 && px <= plane.size && py <= plane.size;
      el.hidden = !inside;
      el.style.left = `${px}px`;
      el.style.top = `${py}px`;
    }
  };
  const norm = () => Math.hypot(P.w[0], P.w[1]);
  const pointer = (event) => {
    const box = plane.canvas.getBoundingClientRect();
    return plane.toData(event.clientX - box.left, event.clientY - box.top);
  };
  const shiftTo = (p) => { P.b = clamp(-(P.w[0] * p[0] + P.w[1] * p[1]), -2, 2); };
  /** Turning keeps ‖w‖, so a component can pass the sliders' ±2; scaling w and b together keeps the same line. */
  const fitSliders = () => {
    const scale = Math.max(Math.abs(P.w[0]), Math.abs(P.w[1])) / 2;
    if (scale > 1) {
      P.w = P.w.map((v) => v / scale);
      P.b /= scale;
    }
    P.b = clamp(P.b, -2, 2);
  };
  const turnTo = (p) => {
    const f = plane.handles?.foot;
    if (!f) return;
    const ang = Math.atan2(p[1] - f[1], p[0] - f[0]);
    const n = norm();
    P.w = [n * Math.cos(ang), n * Math.sin(ang)];
    P.b = -(P.w[0] * f[0] + P.w[1] * f[1]);
    fitSliders();
  };
  for (const [el, move] of [[shiftHandle, shiftTo], [turnHandle, turnTo]]) {
    el.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      el.setPointerCapture(event.pointerId);
      el.classList.add("dragging");
    });
    el.addEventListener("pointermove", (event) => {
      if (!el.hasPointerCapture(event.pointerId)) return;
      move(pointer(event));
      neuronChanged();
    });
    const end = () => el.classList.remove("dragging");
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
  }
  shiftHandle.addEventListener("keydown", (event) => {
    const d = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[event.key];
    if (!d) return;
    event.preventDefault();
    P.b = clamp(P.b + 0.05 * d, -2, 2);
    neuronChanged();
  });
  turnHandle.addEventListener("keydown", (event) => {
    const d = { ArrowUp: 1, ArrowLeft: 1, ArrowDown: -1, ArrowRight: -1 }[event.key];
    if (!d || !plane.handles) return;
    event.preventDefault();
    const f = plane.handles.foot;
    const ang = Math.atan2(P.w[1], P.w[0]) + d * (5 * Math.PI / 180);
    const n = norm();
    P.w = [n * Math.cos(ang), n * Math.sin(ang)];
    P.b = -(P.w[0] * f[0] + P.w[1] * f[1]);
    fitSliders();
    neuronChanged();
  });

  const readout = h("div", { class: "pc-readout" });
  demo.append(readout);
  const update = () => {
    seg.set(P.gate);
    scale.disabled = !canDouble();
    const gate = GATES[P.gate];
    plane.draw({
      classify: (x, y) => stepFn(zOf(P.w, P.b, [x, y])),
      lines: [{ w: P.w, b: P.b }],
      normal: { w: P.w, b: P.b },
      points: INPUTS.map((x, i) => ({ x, y: gate.y[i], wrong: stepFn(zOf(P.w, P.b, x)) !== gate.y[i] })),
      coords: true,
    });
    const n = norm();
    const wrong = INPUTS.filter((x, i) => stepFn(zOf(P.w, P.b, x)) !== gate.y[i]).length;
    readout.replaceChildren(
      h("div", { class: "pc-eq", text: n > 1e-9 ? `\\(${boundaryTex(P.w, P.b)}\\)` : L({ tr: "w = 0: sınır yok, her nokta aynı sınıfta", en: "w = 0: no boundary, every point is in the same class" }) }),
      h("div", { class: "t-stats" },
        h("span", {}, `\\(\\lVert\\mathbf w\\rVert = ${tx(n)}\\)`),
        n > 1e-9 ? h("span", {}, L({ tr: "orijine uzaklık ", en: "distance from the origin " }), `\\(|b|/\\lVert\\mathbf w\\rVert = ${tx(Math.abs(P.b) / n)}\\)`) : null,
        h("span", { class: wrong ? "pc-bad" : "pc-ok" }, wrong
          ? L({ tr: `${gate.name}: ${wrong} nokta yanlış tarafta (kesikli halka)`, en: `${gate.name}: ${wrong} ${wrong === 1 ? "point" : "points"} on the wrong side (dashed ring)` })
          : L({ tr: `${gate.name}: her nokta doğru tarafta ✓`, en: `${gate.name}: every point on the right side ✓` }))));
  };
  neuronViews.add(update);
  update();
  return demo;
}

// =====================================================================
// Theory 3 · the learning rule, one example at a time
// =====================================================================
// Chosen so AND, OR and NAND all end with every point clearly off the boundary
const RULE_START = { w: [-0.4, 0.4], b: 0.1 };
const RS = { gate: "and", eta: 0.2, model: null, i: 0, epoch: 1, mistakes: 0, history: [], log: [], last: null, done: false, timer: 0 };
function resetRule() {
  RS.model = { w: [...RULE_START.w], b: RULE_START.b };
  Object.assign(RS, { i: 0, epoch: 1, mistakes: 0, history: [], log: [], last: null, done: false });
}
resetRule();

function ruleNext() {
  if (RS.done) return;
  const k = RS.i;
  const res = ruleStep(RS.model, INPUTS[k], GATES[RS.gate].y[k], RS.eta);
  RS.last = { ...res, k, epoch: RS.epoch, eta: RS.eta };
  RS.log.push(RS.last);
  if (res.e) RS.mistakes++;
  RS.i++;
  if (RS.i === INPUTS.length) {
    RS.history.push(RS.mistakes);
    if (RS.mistakes === 0) RS.done = true;
    else if (RS.epoch >= 30) RS.done = true;
    else {
      RS.epoch++;
      RS.i = 0;
      RS.mistakes = 0;
    }
  }
}

function buildRuleDemo() {
  const demo = h("div", { class: "t-demo" });
  const seg = gateSeg(RS.gate, (g) => { RS.gate = g; stop(); resetRule(); update(); });
  const eta = slider("η", { min: 0.05, max: 1, step: 0.05, value: RS.eta, onInput: (v) => { RS.eta = v; } });
  const next = h("button", { type: "button", class: "btn small primary", text: L({ tr: "Sonraki örnek ›", en: "Next example ›" }) });
  const epochBtn = h("button", { type: "button", class: "btn small", text: L({ tr: "Epoch'u bitir »", en: "Finish the epoch »" }) });
  const play = h("button", { type: "button", class: "btn small" });
  const reset = h("button", { type: "button", class: "btn small", text: L({ tr: "Baştan", en: "Start over" }) });
  demo.append(h("div", { class: "t-controls" }, seg, eta), h("div", { class: "t-controls" }, next, epochBtn, play, reset));
  const status = h("p", { class: "pc-msg", role: "status" });
  const stage = h("div", { class: "pc-stage" });
  demo.append(status, h("div", { class: "pc-rule-grid" }, stage, h("div", { class: "pc-calc" })));
  const plane = createPlane(stage, { label: L({ tr: "Öğrenme kuralı: sınır her düzeltmede hareket eder", en: "The learning rule: the boundary moves at each correction" }) });
  planeTooltip(plane, () => RS.model);
  const calc = demo.querySelector(".pc-calc");
  const table = h("table", { class: "data-table compact pc-log" });
  demo.append(h("div", { class: "table-wrap pc-log-wrap" }, table));

  function stop() {
    clearInterval(RS.timer);
    RS.timer = 0;
    play.textContent = L({ tr: "▶ Oynat", en: "▶ Play" });
  }
  next.addEventListener("click", () => { stop(); ruleNext(); update(); });
  epochBtn.addEventListener("click", () => {
    stop();
    const epoch = RS.epoch;
    do ruleNext(); while (!RS.done && RS.epoch === epoch && RS.i !== 0);
    update();
  });
  play.addEventListener("click", () => {
    if (RS.timer) { stop(); return; }
    if (RS.done) resetRule();
    play.textContent = L({ tr: "❚❚ Durdur", en: "❚❚ Pause" });
    RS.timer = setInterval(() => {
      ruleNext();
      update();
      if (RS.done || !demo.isConnected) stop();
    }, 650);
  });
  reset.addEventListener("click", () => { stop(); resetRule(); update(); });
  stop();
  RS.stop = stop;

  function update() {
    seg.set(RS.gate);
    eta.set(RS.eta);
    const m = RS.model;
    const gate = GATES[RS.gate];
    const last = RS.last;
    const lines = [];
    if (last?.e) lines.push({ w: last.before.w, b: last.before.b, ghost: true });
    lines.push({ w: m.w, b: m.b });
    plane.draw({
      classify: (x, y) => stepFn(zOf(m.w, m.b, [x, y])),
      lines,
      normal: { w: m.w, b: m.b },
      points: INPUTS.map((x, i) => ({ x, y: gate.y[i], current: last && last.k === i, wrong: stepFn(zOf(m.w, m.b, x)) !== gate.y[i] })),
      coords: true,
    });
    next.disabled = RS.done;
    epochBtn.disabled = RS.done;

    // The calculation for the last example
    if (!last) {
      calc.replaceChildren(h("p", { class: "sub", text: L({
        tr: `Başlangıç: w = (${nf(m.w[0])}; ${nf(m.w[1])}), b = ${nf(m.b)}. İlk örneği görmek için "Sonraki örnek"e basın.`,
        en: `Start: w = (${nf(m.w[0])}, ${nf(m.w[1])}), b = ${nf(m.b)}. Press "Next example" to see the first example.` }) }));
    } else {
      const { x, y, z, yHat, e, before, after } = last;
      const rows = [
        h("div", { class: "pc-calc-head", text: L({ tr: `Epoch ${last.epoch}, örnek ${last.k + 1}: x = (${x[0]}, ${x[1]}), hedef y = ${y}`, en: `Epoch ${last.epoch}, example ${last.k + 1}: x = (${x[0]}, ${x[1]}), target y = ${y}` }) }),
        h("div", { text: `\\(z = ${txp(before.w[0])}\\cdot ${x[0]} + ${txp(before.w[1])}\\cdot ${x[1]} ${txs(before.b)} = ${tx(z)} \\;\\Rightarrow\\; \\hat y = ${yHat}\\)` }),
        h("div", { text: `\\(e = y - \\hat y = ${y} - ${yHat} = ${e}\\)` }),
      ];
      if (!e) rows.push(h("div", { class: "pc-ok", text: L({ tr: "Doğru cevap: ağırlıklar değişmez.", en: "Right answer: the weights stay." }) }));
      else {
        const k = `${tx(last.eta)}\\cdot(${e})`;
        rows.push(
          h("div", { text: `\\(w_1 \\leftarrow ${tx(before.w[0])} + ${k}\\cdot ${x[0]} = ${tx(after.w[0])}\\)` }),
          h("div", { text: `\\(w_2 \\leftarrow ${tx(before.w[1])} + ${k}\\cdot ${x[1]} = ${tx(after.w[1])}\\)` }),
          h("div", { text: `\\(b \\leftarrow ${tx(before.b)} + ${k} = ${tx(after.b)}\\)` }),
          h("div", { class: "pc-bad", text: e > 0
            ? L({ tr: "1 olmalıydı: sınır bu noktayı yeşil tarafa almak için hareket etti.", en: "Should have been 1: the boundary moved to bring this point to the green side." })
            : L({ tr: "0 olmalıydı: sınır bu noktayı kırmızı tarafa bırakmak için hareket etti.", en: "Should have been 0: the boundary moved to leave this point on the red side." }) }));
      }
      calc.replaceChildren(...rows);
    }

    // Status
    const converged = RS.done && RS.history.at(-1) === 0;
    status.className = `pc-msg ${converged ? "good" : RS.done ? "bad" : ""}`;
    if (converged) status.textContent = L({ tr: `Epoch ${RS.epoch} hiç hatasız geçti: perceptron ${gate.name}'i öğrendi ✓ (toplam ${RS.log.filter((r) => r.e).length} düzeltme)`, en: `Epoch ${RS.epoch} passed without a mistake: the perceptron has learned ${gate.name} ✓ (${RS.log.filter((r) => r.e).length} corrections in all)` });
    else if (RS.done) status.textContent = L({ tr: `30 epoch geçti, hata hâlâ sıfır değil. ${gate.name} doğrusal ayrılabilir değil (5. kısım).`, en: `30 epochs and the mistakes are still not zero. ${gate.name} is not linearly separable (section 5).` });
    else status.textContent = L({ tr: `Epoch ${RS.epoch} · bu epoch'ta şu ana kadar ${RS.mistakes} hata${RS.history.length ? ` · önceki epoch'lar: ${RS.history.join(", ")}` : ""}`, en: `Epoch ${RS.epoch} · ${RS.mistakes} ${RS.mistakes === 1 ? "mistake" : "mistakes"} so far in this epoch${RS.history.length ? ` · earlier epochs: ${RS.history.join(", ")}` : ""}` });

    // Log of the steps (latest first)
    const head = ["#", "Epoch", "x", "y", "z", "ŷ", "e", "w₁", "w₂", "b"];
    table.replaceChildren(
      h("thead", {}, h("tr", {}, ...head.map((c) => h("th", { text: c })))),
      h("tbody", {}, ...(RS.log.length ? RS.log.map((r, n) => h("tr", { class: r.e ? "pc-upd" : "" },
        h("td", { text: String(n + 1) }), h("td", { text: String(r.epoch) }), h("td", { text: `(${r.x[0]}, ${r.x[1]})` }),
        h("td", { text: String(r.y) }), h("td", { text: nf(r.z) }), h("td", { text: String(r.yHat) }),
        h("td", { text: r.e > 0 ? "+1" : String(r.e) }), h("td", { text: nf(r.after.w[0]) }), h("td", { text: nf(r.after.w[1]) }),
        h("td", { text: nf(r.after.b) }))).reverse()
        : [h("tr", { class: "empty" }, h("td", { colspan: head.length, text: L({ tr: "Henüz adım yok", en: "No steps yet" }) }))])));
  }
  update();
  return demo;
}

// =====================================================================
// Theory 4 · margin and the number of corrections
// =====================================================================
const CV = { gap: 0.2, seed: 7, runs: [], data: null, model: null, history: [] };

/** n points in the unit square, labeled by a fixed line, with nothing closer to it than gap/2. */
function separableData(gap, seed, n = 40) {
  const rand = seeded(seed);
  const ang = rand() * Math.PI * 2;
  const u = [Math.cos(ang), Math.sin(ang)];
  const c = [0.5, 0.5];
  const pts = [];
  let guard = 0;
  while (pts.length < n && guard++ < 20000) {
    const x = [0.04 + rand() * 0.92, 0.04 + rand() * 0.92];
    const s = u[0] * (x[0] - c[0]) + u[1] * (x[1] - c[1]);
    if (Math.abs(s) < gap / 2) continue;
    pts.push({ x, y: s > 0 ? 1 : 0 });
  }
  // Theorem bound with this separator, in the augmented space (x1, x2, 1)
  const wt = [u[0], u[1], -(u[0] * c[0] + u[1] * c[1])];
  const nt = Math.hypot(...wt);
  let gamma = Infinity;
  let R = 0;
  for (const p of pts) {
    const t = p.y ? 1 : -1;
    gamma = Math.min(gamma, (t * (wt[0] * p.x[0] + wt[1] * p.x[1] + wt[2])) / nt);
    R = Math.max(R, Math.hypot(p.x[0], p.x[1], 1));
  }
  return { pts, true: { w: [u[0], u[1]], b: wt[2] }, bound: Math.floor((R / gamma) ** 2) };
}

function trainToConvergence(pts, eta = 0.5, maxEpochs = 500) {
  const model = { w: [0, 0], b: 0 };
  const history = [];
  let corrections = 0;
  for (let epoch = 0; epoch < maxEpochs; epoch++) {
    let m = 0;
    for (const p of pts) if (ruleStep(model, p.x, p.y, eta).e) m++;
    history.push(m);
    corrections += m;
    if (!m) break;
  }
  return { model, history, corrections, converged: history.at(-1) === 0 };
}

function buildConvergeDemo() {
  const demo = h("div", { class: "t-demo" });
  const gap = slider(L({ tr: "Boşluk", en: "Gap" }), { min: 0.02, max: 0.4, step: 0.02, value: CV.gap, onInput: (v) => { CV.gap = v; CV.data = null; CV.model = null; CV.history = []; update(); } });
  const train = h("button", { type: "button", class: "btn small primary", text: L({ tr: "Eğit", en: "Train" }) });
  const fresh = h("button", { type: "button", class: "btn small", text: L({ tr: "Yeni veri", en: "New data" }) });
  const clear = h("button", { type: "button", class: "btn small", text: L({ tr: "Tabloyu temizle", en: "Clear the table" }) });
  demo.append(h("div", { class: "t-controls" }, gap, train, fresh, clear));
  const grid = h("div", { class: "pc-two" });
  const stage = h("div", { class: "pc-stage" });
  const right = h("div", { class: "pc-side-stack" });
  grid.append(stage, right);
  demo.append(grid);
  const plane = createPlane(stage, { domain: [-0.1, 1.1, -0.1, 1.1], ticks: [0, 0.5, 1], label: L({ tr: "Doğrusal ayrılabilir iki sınıf", en: "Two linearly separable classes" }) });
  planeTooltip(plane, () => CV.model);
  const chart = chartHost(150);
  const table = h("table", { class: "data-table compact" });
  right.append(h("div", { class: "mini-title", text: L({ tr: "Epoch başına hata", en: "Mistakes per epoch" }) }), chart,
    h("div", { class: "table-wrap" }, table));

  train.addEventListener("click", () => {
    CV.data ??= separableData(CV.gap, CV.seed);
    const res = trainToConvergence(CV.data.pts);
    CV.model = res.model;
    CV.history = res.history;
    CV.runs.push({ gap: CV.gap, corrections: res.corrections, epochs: res.history.length, bound: CV.data.bound, converged: res.converged });
    update();
  });
  fresh.addEventListener("click", () => { CV.seed++; CV.data = null; CV.model = null; CV.history = []; update(); });
  clear.addEventListener("click", () => { CV.runs = []; update(); });

  function update() {
    gap.set(CV.gap);
    CV.data ??= separableData(CV.gap, CV.seed);
    const m = CV.model;
    plane.draw({
      classify: m ? (x, y) => stepFn(zOf(m.w, m.b, [x, y])) : null,
      lines: m ? [{ w: m.w, b: m.b }] : [],
      normal: m ? { w: m.w, b: m.b, length: 0.22 } : null,
      points: CV.data.pts.map((p) => ({ ...p, wrong: m ? stepFn(zOf(m.w, m.b, p.x)) !== p.y : false })),
    });
    mistakesChart(chart, CV.history);
    const head = [L({ tr: "Boşluk", en: "Gap" }), L({ tr: "Düzeltme", en: "Corrections" }), "Epoch", L({ tr: "Teorem sınırı", en: "Theorem bound" })];
    table.replaceChildren(
      h("thead", {}, h("tr", {}, ...head.map((c) => h("th", { text: c })))),
      h("tbody", {}, ...(CV.runs.length ? CV.runs.map((r) => h("tr", { title: r.converged ? null : L({ tr: `${r.epochs} epoch'ta bitmedi: eğitim burada kesildi`, en: `Not finished after ${r.epochs} epochs: training was cut off here` }) },
        h("td", { text: nf(r.gap) }), h("td", { text: `${fmtNum(r.corrections, 0)}${r.converged ? "" : "+"}` }), h("td", { text: `${fmtNum(r.epochs, 0)}${r.converged ? "" : "+"}` }),
        h("td", { text: `≤ ${fmtNum(r.bound, 0)}` }))).reverse()
        : [h("tr", { class: "empty" }, h("td", { colspan: 4, text: L({ tr: "Eğit'e basın", en: "Press Train" }) }))])));
  }
  update();
  return demo;
}

// =====================================================================
// Theory 5 · XOR: one perceptron fails, two layers succeed
// =====================================================================
const XS = { mode: "one", model: null, i: 0, mistakes: 0, history: [], timer: 0 };
const XOR_START = { w: [0.3, -0.2], b: 0.1 };
const resetXor = () => Object.assign(XS, { model: { w: [...XOR_START.w], b: XOR_START.b }, i: 0, mistakes: 0, history: [] });
resetXor();
const HIDDEN = [
  { name: "OR", w: [1, 1], b: -0.5, color: "lineB" },
  { name: "NAND", w: [-1, -1], b: 1.5, color: "line" },
];
const twoLayer = (x) => {
  const hdn = HIDDEN.map((n) => stepFn(zOf(n.w, n.b, x)));
  return { hdn, out: stepFn(zOf([1, 1], -1.5, hdn)) };
};

function buildXorDemo() {
  const demo = h("div", { class: "t-demo" });
  const play = h("button", { type: "button", class: "btn small primary" });
  const reset = h("button", { type: "button", class: "btn small", text: L({ tr: "Baştan", en: "Start over" }) });
  const oneControls = h("div", { class: "t-controls" }, play, reset);
  const seg = segmented([["one", L({ tr: "Tek perceptron", en: "One perceptron" })], ["two", L({ tr: "İki katman", en: "Two layers" })]], XS.mode,
    (m) => { XS.mode = m; stop(); update(); }, L({ tr: "Model", en: "Model" }));
  demo.append(h("div", { class: "t-controls" }, seg), oneControls);
  const status = h("p", { class: "pc-msg", role: "status" });
  const grid = h("div", { class: "pc-two" });
  const stage = h("div", { class: "pc-stage" });
  const side = h("div", { class: "pc-side-stack" });
  grid.append(stage, side);
  demo.append(status, grid);
  const plane = createPlane(stage, { label: L({ tr: "XOR'un dört noktası", en: "The four points of XOR" }) });
  planeTooltip(plane, () => (XS.mode === "one" ? XS.model : null));
  const chart = chartHost(150);
  const chartBox = h("div", {}, h("div", { class: "mini-title", text: L({ tr: "Epoch başına hata (tek perceptron)", en: "Mistakes per epoch (one perceptron)" }) }), chart);
  const net = h("div", { class: "pc-net" });
  const table = h("table", { class: "data-table compact pc-truth" });
  const twoBox = h("div", { class: "pc-side-stack" }, net, h("div", { class: "table-wrap" }, table));
  side.append(chartBox, twoBox);

  function stop() {
    clearInterval(XS.timer);
    XS.timer = 0;
    play.textContent = L({ tr: "▶ Eğit", en: "▶ Train" });
  }
  const stepOnce = () => {
    const y = GATES.xor.y[XS.i];
    if (ruleStep(XS.model, INPUTS[XS.i], y, 0.25).e) XS.mistakes++;
    XS.i++;
    if (XS.i === 4) {
      XS.history.push(XS.mistakes);
      XS.i = 0;
      XS.mistakes = 0;
    }
  };
  play.addEventListener("click", () => {
    if (XS.timer) { stop(); return; }
    play.textContent = L({ tr: "❚❚ Durdur", en: "❚❚ Pause" });
    XS.timer = setInterval(() => {
      stepOnce();
      update();
      if (XS.history.length >= 60 || !demo.isConnected) stop();
    }, 120);
  });
  reset.addEventListener("click", () => { stop(); resetXor(); update(); });
  stop();
  XS.stop = () => { if (XS.timer) { stop(); update(); } };

  // The two-layer network, drawn once per language
  const drawNet = () => {
    const root = svg("svg", { viewBox: "0 0 360 170", role: "img", "aria-label": L({ tr: "İki katmanlı ağ: x₁ ve x₂, OR ve NAND nöronlarına, onlar da AND nöronuna bağlı", en: "Two-layer network: x₁ and x₂ feed an OR and a NAND neuron, which feed an AND neuron" }) });
    const pos = { x1: [40, 45], x2: [40, 125], h1: [170, 45], h2: [170, 125], o: [300, 85] };
    for (const [a, b] of [["x1", "h1"], ["x2", "h1"], ["x1", "h2"], ["x2", "h2"], ["h1", "o"], ["h2", "o"]]) {
      root.append(svg("line", { x1: pos[a][0] + 20, y1: pos[a][1], x2: pos[b][0] - 24, y2: pos[b][1], class: "pc-edge thin" }));
    }
    const node = (key, label, cls, sub) => {
      root.append(svg("circle", { cx: pos[key][0], cy: pos[key][1], r: key[0] === "x" ? 18 : 24, class: `pc-net-node ${cls}` }));
      root.append(svgText(pos[key][0], pos[key][1] + 4, label, "pc-net-label", "middle"));
      if (sub) root.append(svgText(pos[key][0], pos[key][1] + 40, sub, "pc-caption", "middle"));
    };
    node("x1", "x₁", "in");
    node("x2", "x₂", "in");
    node("h1", "OR", "h1");
    node("h2", "NAND", "h2", L({ tr: "gizli katman", en: "hidden layer" }));
    node("o", "AND", "o", L({ tr: "çıktı", en: "output" }));
    net.replaceChildren(root);
  };
  drawNet();

  function update() {
    seg.set(XS.mode);
    const one = XS.mode === "one";
    oneControls.hidden = !one;
    chartBox.hidden = !one;
    twoBox.hidden = one;
    if (one) {
      const m = XS.model;
      plane.draw({
        classify: (x, y) => stepFn(zOf(m.w, m.b, [x, y])),
        lines: [{ w: m.w, b: m.b }],
        normal: { w: m.w, b: m.b },
        points: INPUTS.map((x, i) => ({ x, y: GATES.xor.y[i], wrong: stepFn(zOf(m.w, m.b, x)) !== GATES.xor.y[i], current: XS.timer && XS.i === i })),
        coords: true,
      });
      mistakesChart(chart, XS.history, { max: 4 });
      const zeros = XS.history.filter((m_) => m_ === 0).length;
      status.className = `pc-msg ${XS.history.length >= 10 ? "bad" : ""}`;
      status.textContent = XS.history.length
        ? L({ tr: `${XS.history.length} epoch: epoch başına hata ${Math.min(...XS.history)} ile ${Math.max(...XS.history)} arasında kaldı, hatasız epoch sayısı ${zeros}. Sınır her epoch'ta yer değiştiriyor ama dört noktayı asla ayıramıyor.`,
          en: `${XS.history.length} epochs: the mistakes per epoch stayed between ${Math.min(...XS.history)} and ${Math.max(...XS.history)}, epochs without a mistake: ${zeros}. The boundary moves in every epoch but never separates the four points.` })
        : L({ tr: "Eğit'e basın ve perceptronun XOR'u öğrenmeye çalışmasını izleyin.", en: "Press Train and watch the perceptron try to learn XOR." });
    } else {
      plane.draw({
        classify: (x, y) => twoLayer([x, y]).out,
        lines: HIDDEN.map((n) => ({ w: n.w, b: n.b, color: COLORS[n.color], label: n.name })),
        points: INPUTS.map((x, i) => ({ x, y: GATES.xor.y[i] })),
        coords: true,
      });
      status.className = "pc-msg good";
      status.textContent = L({ tr: "İki doğrunun arasındaki şerit (OR = 1 ve NAND = 1) tam olarak XOR'un 1 olduğu yer: 4/4 doğru ✓", en: "The band between the two lines (OR = 1 and NAND = 1) is exactly where XOR is 1: 4/4 right ✓" });
      table.replaceChildren(
        h("thead", {}, h("tr", {}, ...["x₁", "x₂", "OR", "NAND", L({ tr: "AND → ŷ", en: "AND → ŷ" }), "XOR"].map((c) => h("th", { text: c })))),
        h("tbody", {}, ...INPUTS.map((x, i) => {
          const r = twoLayer(x);
          return h("tr", {}, h("td", { text: String(x[0]) }), h("td", { text: String(x[1]) }), h("td", { text: String(r.hdn[0]) }),
            h("td", { text: String(r.hdn[1]) }), h("td", { text: String(r.out) }), h("td", { class: "pc-ok", text: `${GATES.xor.y[i]} ✓` }));
        })));
    }
  }
  update();
  return demo;
}

// =====================================================================
// Theory tab: assembling the sections
// =====================================================================
const MATH_LINKS = { "p-neuron": "pm-notation", "p-boundary": "pm-geometry", "p-rule": "pm-rule", "p-converge": "pm-converge", "p-xor": "pm-xor" };

const sectionCard = (id, num, text, paragraphs, demo) => KIT.sectionCard(id, num, text.h, paragraphs, demo, MATH_LINKS[id]);

function buildTheory() {
  const panel = $("#panel-theory");
  const intro = TEXT.intro;
  const bio = h("div", { class: "pc-bio" }, h("div", { class: "mini-title", text: L(intro.bio.title) }),
    h("dl", {}, ...intro.bio.rows.flatMap(([a, b]) => [html("dt", {}, a), html("dd", {}, b)])));
  const fact = h("div", { role: "note", class: "tut-callout info" }, h("span", { class: "tut-callout-icon", text: "ℹ️", "aria-hidden": "true" }),
    h("div", {}, h("strong", { text: L(intro.fact.title) }), html("p", {}, intro.fact.text)));
  const introCard = h("div", { class: "card t-intro pc-intro", id: "p-intro" },
    h("div", { class: "t-intro-text" }, h("h2", { text: L(intro.h) }), ...intro.p.map((p) => html("p", {}, p))),
    h("div", { class: "pc-intro-side" }, bio, fact));

  const N = TEXT.neuron;
  const B = TEXT.boundary;
  const Rt = TEXT.rule;
  const C = TEXT.converge;
  const X = TEXT.xor;
  const cases = h("ul", { class: "pc-cases" }, ...Rt.cases.map((c) => html("li", {}, c)));
  panel.replaceChildren(
    introCard,
    sectionCard("p-neuron", 1, N, [html("p", {}, N.p1), formula(R`z = w_1 x_1 + w_2 x_2 + b`), html("p", {}, N.p2),
      formula(R`\hat y = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}`), html("p", {}, N.p3), html("p", {}, N.p4),
      html("div", { class: "in-model" }, N.model)], buildNeuronDemo()),
    sectionCard("p-boundary", 2, B, [html("p", {}, B.p1), formula(R`w_1 x_1 + w_2 x_2 + b = 0`), html("p", {}, B.p2), html("p", {}, B.p3),
      html("p", {}, B.p4), html("div", { class: "in-model" }, B.model)], buildBoundaryDemo()),
    sectionCard("p-rule", 3, Rt, [html("p", {}, Rt.p1), formula(R`\mathbf w \leftarrow \mathbf w + \eta\,(y - \hat y)\,\mathbf x \qquad b \leftarrow b + \eta\,(y - \hat y)`),
      cases, html("p", {}, Rt.p2), html("p", {}, Rt.p3), html("div", { class: "in-model" }, Rt.model)], buildRuleDemo()),
    sectionCard("p-converge", 4, C, [html("p", {}, C.p1), html("p", {}, C.p2), formula(R`\text{${L({ tr: "düzeltme sayısı", en: "corrections" })}} \;\le\; \left(\frac{R}{\gamma}\right)^{2}`),
      html("p", {}, C.p3), html("div", { class: "in-model" }, C.model)], buildConvergeDemo()),
    sectionCard("p-xor", 5, X, [html("p", {}, X.p1), html("p", {}, X.try), html("p", {}, X.p2), html("p", {}, X.p3), html("div", { class: "in-model" }, X.model),
      html("p", {}, X.recap), KIT.nextChapterCard()], buildXorDemo()));
}

// =====================================================================
// Playground
// =====================================================================
const PG = {
  data: [], cls: 1, preset: "blobs", eta: 0.2, speed: 8, shuffle: true,
  model: { w: [0.4, -0.3], b: 0.1 }, epoch: 0, i: 0, order: [], mistakes: 0, history: [], weights: [],
  running: false, raf: 0, acc: 0, status: "idle", current: -1, seed: 3,
};
const PRESETS = ["and", "or", "xor", "blobs", "overlap", "ring", "empty"];
const presetName = (k) => ({
  and: "AND", or: "OR", xor: "XOR",
  blobs: L({ tr: "İki küme", en: "Two clusters" }), overlap: L({ tr: "İç içe kümeler", en: "Overlapping clusters" }),
  ring: L({ tr: "Halka", en: "Ring" }), empty: L({ tr: "Boş", en: "Empty" }),
})[k];

function makePreset(key) {
  const rand = seeded(PG.seed);
  const pts = [];
  if (GATES[key]) return gatePoints(key);
  if (key === "blobs" || key === "overlap") {
    const sd = key === "blobs" ? 0.11 : 0.22;
    const [a, b] = key === "blobs" ? [[0.15, 0.25], [0.85, 0.8]] : [[0.3, 0.35], [0.7, 0.65]];
    for (let i = 0; i < 25; i++) pts.push({ x: [a[0] + sd * gauss(rand), a[1] + sd * gauss(rand)], y: 0 });
    for (let i = 0; i < 25; i++) pts.push({ x: [b[0] + sd * gauss(rand), b[1] + sd * gauss(rand)], y: 1 });
  } else if (key === "ring") {
    for (let i = 0; i < 22; i++) {
      const r = 0.28 * Math.sqrt(rand()); const t = rand() * Math.PI * 2;
      pts.push({ x: [0.5 + r * Math.cos(t), 0.5 + r * Math.sin(t)], y: 1 });
    }
    for (let i = 0; i < 30; i++) {
      const r = 0.55 + 0.25 * rand(); const t = rand() * Math.PI * 2;
      pts.push({ x: [0.5 + r * Math.cos(t), 0.5 + r * Math.sin(t)], y: 0 });
    }
  }
  return pts.map((p) => ({ ...p, x: p.x.map((v) => clamp(v, -0.45, 1.45)) }));
}

function pgResetRun() {
  Object.assign(PG, { epoch: 0, i: 0, order: [], mistakes: 0, history: [], weights: [{ w: [...PG.model.w], b: PG.model.b }], status: "idle", current: -1 });
}
function pgNewOrder() {
  PG.order = PG.data.map((_, k) => k);
  if (PG.shuffle) {
    for (let k = PG.order.length - 1; k > 0; k--) {
      const j = Math.floor(Math.random() * (k + 1));
      [PG.order[k], PG.order[j]] = [PG.order[j], PG.order[k]];
    }
  }
}
/** One example; returns false when training has ended. */
function pgStep() {
  if (!PG.data.length) { PG.status = "nodata"; return false; }
  if (PG.status === "converged" || PG.status === "limit") return false;
  if (PG.i === 0) {
    if (PG.epoch === 0 || PG.order.length !== PG.data.length) pgNewOrder();
    PG.epoch++;
    PG.mistakes = 0;
  }
  const k = PG.order[PG.i];
  const p = PG.data[k];
  PG.current = k;
  if (ruleStep(PG.model, p.x, p.y, PG.eta).e) {
    PG.mistakes++;
    PG.weights.push({ w: [...PG.model.w], b: PG.model.b });
  }
  PG.i++;
  if (PG.i >= PG.data.length) {
    PG.history.push(PG.mistakes);
    PG.i = 0;
    if (PG.mistakes === 0) { PG.status = "converged"; PG.current = -1; return false; }
    if (PG.epoch >= 300) { PG.status = "limit"; PG.current = -1; return false; }
    pgNewOrder();
  }
  PG.status = "running";
  return true;
}
const pgEpoch = () => { do { if (!pgStep()) return; } while (PG.i !== 0); };

function buildPlay() {
  const panel = $("#panel-play");
  pgStop();
  const intro = h("div", { class: "card pg-intro" },
    h("h2", { text: L({ tr: "Deneme alanı: perceptronu kendiniz eğitin", en: "Playground: train a perceptron yourself" }) }),
    html("p", { class: "sub" }, {
      tr: R`Hazır bir veri seçin ya da noktaları kendiniz yerleştirin: düzleme tıklamak seçili sınıftan bir nokta ekler, bir noktaya tıklamak onu siler. Sonra <b>Eğit</b>'e basıp [[perceptron-rule|perceptron kuralının]] sınırı nasıl taşıdığını izleyin. Hangi verilerde eğitimin bittiğine, hangilerinde hiç bitmediğine dikkat edin. <b>Deneyin:</b> Her veriden önce tahmin edin, sonra eğitin: İki küme, İç içe kümeler, Halka. Sonra İki küme'de öbür kümenin ortasına tek bir aykırı nokta ekleyin: eğitim yine biter mi? Son olarak η'yı değiştirin: İç içe kümelerde bir şeyi kurtarır mı?`,
      en: R`Pick a ready-made dataset or place points yourself: clicking the plane adds a point of the selected class, clicking a point removes it. Then press <b>Train</b> and watch the [[perceptron-rule|perceptron rule]] move the boundary. Notice on which data training ends and on which it never does. <b>Try it:</b> predict before each dataset, then train: Two clusters, Overlapping clusters, Ring. Then, in Two clusters, add a single outlier in the middle of the other cluster: does training still end? Finally change η: does it rescue anything on Overlapping clusters?` }));

  const presets = segmented(PRESETS.map((k) => [k, presetName(k)]), PG.preset, (k) => {
    PG.preset = k;
    PG.data = makePreset(k);
    pgStop();
    pgResetRun();
    update();
  }, L({ tr: "Hazır veri", en: "Dataset" }));
  const cls = segmented([["1", L({ tr: "● Sınıf 1", en: "● Class 1" })], ["0", L({ tr: "✕ Sınıf 0", en: "✕ Class 0" })]], String(PG.cls),
    (v) => { PG.cls = Number(v); }, L({ tr: "Eklenecek nokta", en: "Point to add" }));
  cls.classList.add("pg-cls");
  const reshuffle = h("button", { type: "button", class: "btn small", text: L({ tr: "Yeniden üret", en: "Regenerate" }) });
  reshuffle.addEventListener("click", () => { PG.seed++; if (PG.preset !== "custom") PG.data = makePreset(PG.preset); pgStop(); pgResetRun(); update(); });

  const stage = h("div", { class: "pc-stage pg-plane" });
  const stageCard = h("div", { class: "card pg-stage" },
    h("div", { class: "t-controls" }, presets, reshuffle),
    h("div", { class: "t-controls" }, h("span", { class: "mini-title pc-inline", text: L({ tr: "Tıklayınca ekle:", en: "Click adds:" }) }), cls),
    stage,
    h("p", { class: "sub", text: L({ tr: "Fareyle gezinince her noktanın z değeri görünür. Kesikli halka: şu an yanlış sınıflanan nokta; mavi halka: kuralın baktığı örnek.", en: "Hover to see the z value of any point. Dashed ring: a point misclassified right now; blue ring: the example the rule is looking at." }) }));
  const plane = createPlane(stage, { label: L({ tr: "Deneme alanı düzlemi", en: "Playground plane" }) });
  planeTooltip(plane, () => PG.model);
  plane.canvas.classList.add("pg-canvas");
  plane.canvas.addEventListener("click", (event) => {
    const box = plane.canvas.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    const hit = PG.data.findIndex((p) => {
      const [qx, qy] = plane.toPx(...p.x);
      return Math.hypot(qx - px, qy - py) < 11;
    });
    if (hit >= 0) PG.data.splice(hit, 1);
    else PG.data.push({ x: plane.toData(px, py), y: PG.cls });
    PG.preset = "custom";
    presets.set("custom");
    const was = PG.running;
    pgStop();
    pgResetRun();
    if (was) pgStart();
    update();
  });

  // Controls
  const eta = slider("η", { min: 0.02, max: 1, step: 0.02, value: PG.eta, onInput: (v) => { PG.eta = v; } });
  const speeds = [1, 4, 8, 30, 120, 1000];
  const speed = slider(L({ tr: "Hız", en: "Speed" }), { min: 0, max: speeds.length - 1, step: 1, value: speeds.indexOf(PG.speed),
    fmt: (i) => L({ tr: `${speeds[i]} örnek/sn`, en: `${speeds[i]} examples/s` }), onInput: (i) => { PG.speed = speeds[i]; } });
  const shuffle = h("input", { type: "checkbox" });
  shuffle.checked = PG.shuffle;
  shuffle.addEventListener("change", () => { PG.shuffle = shuffle.checked; });
  const play = h("button", { type: "button", class: "btn primary pg-play" });
  const one = h("button", { type: "button", class: "btn small", text: L({ tr: "Tek örnek", en: "One example" }) });
  const epoch = h("button", { type: "button", class: "btn small", text: L({ tr: "Tek epoch", en: "One epoch" }) });
  const randomize = h("button", { type: "button", class: "btn small", text: L({ tr: "Ağırlıkları rastgele yap", en: "Random weights" }) });
  play.addEventListener("click", () => (PG.running ? pgStop() : pgStart()));
  one.addEventListener("click", () => { pgStop(); if (pgFinished()) pgResetRun(); pgStep(); update(); });
  epoch.addEventListener("click", () => { pgStop(); if (pgFinished()) pgResetRun(); pgEpoch(); update(); });
  randomize.addEventListener("click", () => {
    pgStop();
    PG.model = { w: [Math.random() * 2 - 1, Math.random() * 2 - 1], b: Math.random() - 0.5 };
    pgResetRun();
    update();
  });
  const tiles = h("div", { class: "pg-tiles" });
  const status = h("p", { class: "pc-msg", role: "status" });
  const side = h("div", { class: "card pg-side" },
    h("h2", { text: L({ tr: "Eğitim", en: "Training" }) }),
    h("div", { class: "pg-controls" }, eta, speed,
      h("label", { class: "pg-check" }, shuffle, L({ tr: "Her epoch'ta örneklerin sırasını karıştır", en: "Shuffle the examples every epoch" }))),
    h("div", { class: "pg-buttons" }, play, one, epoch, randomize),
    status, tiles);
  const layout = h("div", { class: "pg-layout" }, stageCard, side);

  const chartM = chartHost(190);
  const chartW = chartHost(190);
  const charts = h("div", { class: "row two" },
    h("div", { class: "card chart-card" }, h("div", { class: "card-head" }, h("div", {},
      h("h2", { text: L({ tr: "Epoch başına hata", en: "Mistakes per epoch" }) }),
      h("p", { class: "sub", text: L({ tr: "Sıfıra inerse veri doğrusal ayrılabilir ve eğitim biter.", en: "If it reaches zero, the data is linearly separable and training ends." }) }))), chartM),
    h("div", { class: "card chart-card" }, h("div", { class: "card-head" }, h("div", {},
      h("h2", { text: L({ tr: "Ağırlıkların yolculuğu", en: "How the weights travel" }) }),
      h("p", { class: "sub", text: L({ tr: "Her düzeltmeden sonra w₁, w₂ ve b.", en: "w₁, w₂ and b after every correction." }) })),
      h("div", { class: "legend" },
        h("span", { class: "legend-item" }, h("span", { class: "key-line s1" }), "w₁"),
        h("span", { class: "legend-item" }, h("span", { class: "key-line s2" }), "w₂"),
        h("span", { class: "legend-item" }, h("span", { class: "key-line s3" }), "b"))), chartW));
  panel.replaceChildren(intro, layout, charts);

  PG.update = update;
  function update() {
    const m = PG.model;
    plane.draw({
      classify: (x, y) => stepFn(zOf(m.w, m.b, [x, y])),
      lines: [{ w: m.w, b: m.b }],
      normal: { w: m.w, b: m.b, length: 0.3 },
      points: PG.data.map((p, k) => ({ ...p, current: k === PG.current, wrong: stepFn(zOf(m.w, m.b, p.x)) !== p.y })),
      coords: GATES[PG.preset] != null,
    });
    play.textContent = PG.running ? L({ tr: "❚❚ Durdur", en: "❚❚ Pause" }) : L({ tr: "▶ Eğit", en: "▶ Train" });
    const right = PG.data.filter((p) => stepFn(zOf(m.w, m.b, p.x)) === p.y).length;
    const tile = (label, value) => h("div", { class: "pg-tile" }, h("span", { class: "tile-label", text: label }), h("span", { class: "pg-tile-value", text: value }));
    tiles.replaceChildren(
      tile("Epoch", fmtNum(PG.epoch, 0)),
      tile(L({ tr: "Bu epoch'ta hata", en: "Mistakes this epoch" }), PG.epoch ? fmtNum(PG.i === 0 ? (PG.history.at(-1) ?? 0) : PG.mistakes, 0) : "—"),
      tile(L({ tr: "Doğruluk", en: "Accuracy" }), PG.data.length ? fmtPct((right / PG.data.length) * 100, 0) : "—"),
      tile("w₁", nf(m.w[0])), tile("w₂", nf(m.w[1])), tile("b", nf(m.b)));
    const corrections = PG.weights.length - 1;
    const msgs = {
      idle: [L({ tr: `${PG.data.length} nokta. Eğit'e basın.`, en: `${PG.data.length} points. Press Train.` }), ""],
      nodata: [L({ tr: "Önce düzleme tıklayarak nokta ekleyin ya da hazır bir veri seçin.", en: "First add points by clicking the plane, or pick a dataset." }), "bad"],
      running: [L({ tr: `Eğitiliyor… ${corrections} düzeltme yapıldı.`, en: `Training… ${corrections} corrections so far.` }), ""],
      converged: [L({ tr: `Yakınsadı ✓ Epoch ${PG.epoch} hatasız geçti; toplam ${corrections} düzeltme. Veri doğrusal ayrılabilir.`, en: `Converged ✓ Epoch ${PG.epoch} had no mistakes; ${corrections} corrections in all. The data is linearly separable.` }), "good"],
      limit: [L({ tr: `300 epoch'ta hata sıfıra inmedi (en az ${Math.min(...(PG.history.length ? PG.history : [0]))}). Bu veri büyük olasılıkla doğrusal ayrılabilir değil; perceptron sonsuza kadar sınırı oynatır.`, en: `After 300 epochs the mistakes did not reach zero (lowest ${Math.min(...(PG.history.length ? PG.history : [0]))}). This data is most likely not linearly separable; the perceptron moves the boundary forever.` }), "bad"],
    };
    const [text, cls_] = msgs[PG.status] || msgs.idle;
    status.textContent = text;
    status.className = `pc-msg ${cls_}`;
    mistakesChart(chartM, PG.history);
    const ws = PG.weights;
    // At most about 400 points on the chart: a long run keeps every correction, the chart need not
    const stride = Math.ceil(ws.length / 400) || 1;
    const shown = [];
    for (let k = 0; k < ws.length; k += stride) shown.push(k);
    if (ws.length && shown.at(-1) !== ws.length - 1) shown.push(ws.length - 1);
    let lo = -1;
    let hi = 1;
    for (const v of ws) {
      lo = Math.min(lo, v.w[0], v.w[1], v.b);
      hi = Math.max(hi, v.w[0], v.w[1], v.b);
    }
    const yStep = niceStep(hi - lo, 4);
    const xMax = Math.max(ws.length, 5);
    lineChart(chartW, {
      label: L({ tr: "Ağırlıklar ve bias", en: "Weights and bias" }),
      empty: "",
      x: { min: 1, max: xMax, ticks: linearTicks(1, xMax, Math.max(1, niceStep(xMax, 6)), (v) => fmtNum(v - 1, 0)) },
      y: { min: Math.floor(lo / yStep) * yStep, max: Math.ceil(hi / yStep) * yStep, ticks: linearTicks(Math.floor(lo / yStep) * yStep, Math.ceil(hi / yStep) * yStep, yStep, (v) => fmtNum(v, 1)) },
      series: [
        { points: shown.map((k) => [k + 1, ws[k].w[0]]), color: "var(--series-1)" },
        { points: shown.map((k) => [k + 1, ws[k].w[1]]), color: "var(--series-2)" },
        { points: shown.map((k) => [k + 1, ws[k].b]), color: "var(--series-3)" },
      ],
      tooltip: (i) => {
        const k = shown[i];
        return { title: k ? L({ tr: `${k}. düzeltmeden sonra`, en: `After correction ${k}` }) : L({ tr: "Başlangıç", en: "Start" }), rows: [
          { value: nf(ws[k].w[0]), label: "w₁", color: "var(--series-1)" },
          { value: nf(ws[k].w[1]), label: "w₂", color: "var(--series-2)" },
          { value: nf(ws[k].b), label: "b", color: "var(--series-3)" },
        ] };
      },
    });
  }
  update();
}
const pgFinished = () => PG.status === "converged" || PG.status === "limit";

function pgStart() {
  if (pgFinished()) pgResetRun();
  if (!PG.data.length) { PG.status = "nodata"; PG.update?.(); return; }
  PG.running = true;
  PG.acc = 0;
  let last = performance.now();
  let lastDraw = 0;
  let drawGap = 40;   // ms between redraws; grows when drawing is slow, so the page stays responsive
  const frame = (now) => {
    if (!PG.running) return;
    PG.acc += ((now - last) / 1000) * PG.speed;
    last = now;
    let steps = Math.floor(PG.acc);
    PG.acc -= steps;
    let alive = true;
    const started = performance.now();
    while (steps-- > 0 && alive && performance.now() - started < 10) alive = pgStep();
    if (!alive) PG.running = false;
    if (!alive || now - lastDraw > drawGap) {
      const before = performance.now();
      PG.update?.();
      drawGap = Math.max(40, 2 * (performance.now() - before));
      lastDraw = now;
    }
    if (PG.running) PG.raf = requestAnimationFrame(frame);
  };
  PG.raf = requestAnimationFrame(frame);
  PG.update?.();
}
function pgStop() {
  PG.running = false;
  cancelAnimationFrame(PG.raf);
  PG.update?.();
}

// =====================================================================
// Wiring
// =====================================================================
PG.data = makePreset(PG.preset);
pgResetRun();

let built = false;
function buildAll() {
  for (const o of observers.splice(0)) o.disconnect();
  redrawers.clear();
  neuronViews.clear();
  clearInterval(RS.timer);
  clearInterval(XS.timer);
  RS.timer = 0;
  XS.timer = 0;
  buildTheory();
  buildPlay();
  built = true;
}

document.addEventListener("DOMContentLoaded", buildAll);
document.addEventListener("lang-change", () => { if (built) buildAll(); });
document.addEventListener("theme-change", () => {
  readColors();
  for (const redraw of redrawers) redraw();
  if (built) for (const view of neuronViews) view();
});
document.addEventListener("tab-shown", (event) => {
  if (event.detail !== "play") pgStop();
  if (event.detail !== "theory") {
    RS.stop?.();
    XS.stop?.();
  }
});
})();
