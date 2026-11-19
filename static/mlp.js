"use strict";
/*
 * Chapter 2: the multilayer network (MLP). Everything runs in the browser, on net.js.
 *
 * Theory tab: six sections, each with a live demo (activation functions, the hidden layer on
 * XOR, the loss and gradient descent, backpropagation step by step, how many neurons, an MLP on
 * images). Playground tab: train a small network on 2-D data and look inside it.
 *
 * Texts are { tr, en } pairs of HTML; \( TeX \) is typeset by math.js and [[key|text]] links a
 * glossary term. The panels are rebuilt on every language change; the demos keep their state.
 * Uses net.js (NET), kit.js (KIT), i18n.js (L), core.js (h, svg, $, fmtNum, lineChart, showTip, …),
 * math.js (texNum) and glossary.js (termify).
 */

/** This page's tabs (core.js navigates between them). */
const TABS = {
  theory: { tab: "#tab-theory", panel: "#panel-theory", hash: "#teori" },
  play: { tab: "#tab-play", panel: "#panel-play", hash: "#deneme" },
  math: { tab: "#tab-math", panel: "#panel-math", hash: "#matematik" },
  code: { tab: "#tab-code", panel: "#panel-code", hash: "#kodlama" },
};

(() => {
const { R, seeded, gauss, nf, tx, txp, html, formula, COLORS, readColors, redrawers, observers,
  createPlane, createGraph, slider, segmented, chartHost, sectionCard, nextChapterCard } = KIT;
const sigma = NET.sigmoid;
const XOR = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]].map(([a, b, y]) => ({ x: [a, b], y }));
const MATH_LINKS = { "ml-act": "mm-act", "ml-hidden": "mm-forward", "ml-loss": "mm-loss", "ml-backprop": "mm-backprop", "ml-capacity": "mm-universal" };
const card = (id, num, text, paragraphs, demo) => sectionCard(id, num, text.h, paragraphs, demo, MATH_LINKS[id]);
const button = (text, cls = "btn small") => h("button", { type: "button", class: cls, text });
const msg = (text, kind = "") => { const p = h("p", { class: `pc-msg ${kind}`, role: "status" }); p.textContent = text; return p; };
const setMsg = (node, text, kind = "") => { node.textContent = text; node.className = `pc-msg ${kind}`; };
const timers = new Set();          // animation timers of the demos (stopped on rebuild and when the tab is left)
const every = (ms, fn) => { const t = setInterval(fn, ms); timers.add(t); return t; };
const stop = (t) => { clearInterval(t); timers.delete(t); return 0; };

// =====================================================================
// Texts
// =====================================================================
const TEXT = {
  intro: {
    h: PLACE_TITLES["ml-intro"],
    p: [
      { tr: R`Perceptron XOR'u öğrenemedi. Çözüm perceptronları katman katman bağlamaktı; ama aradaki [[hidden-layer|gizli katmanın]] ağırlıklarını öğretecek bir kural yoktu. 1986'da David Rumelhart, Geoffrey Hinton ve Ronald Williams, <i>Nature</i>'da yayımladıkları makalede çok katmanlı ağların [[backprop|geri yayılımla]] eğitilebildiğini ve gizli katmanların kendi başlarına işe yarar özellikler öğrendiğini gösterdi.`,
        en: R`The perceptron could not learn XOR. The way out was to connect perceptrons in layers, but there was no rule to teach the weights of the [[hidden-layer|hidden layer]] in between. In 1986 David Rumelhart, Geoffrey Hinton and Ronald Williams showed, in a paper in <i>Nature</i>, that multilayer networks can be trained with [[backprop|backpropagation]] and that their hidden layers learn useful features by themselves.` },
      { tr: R`Bu bölümde perceptrondan üç farkı göreceğiz: basamak yerine yumuşak bir [[activation|aktivasyon]]; yalnızca yanlışlara bakan perceptron ölçütü yerine her örneği ölçen bir [[loss|kayıp fonksiyonu]]; ve hatayı katman katman geriye taşıyarak her ağırlığın [[gradient-descent|gradyan inişi]] için gereken türevini bulan geri yayılım. Sonunda bir [[mlp|MLP]]'nin neler öğrenebildiğine ve görüntülerde neden yetersiz kaldığına bakacağız; bu da bizi CNN'e götürecek.`,
        en: R`In this chapter we meet three differences from the perceptron: a smooth [[activation|activation]] instead of the step; a [[loss|loss function]] that measures every example instead of the perceptron criterion, which looks only at the mistakes; and backpropagation, which carries the error back layer by layer and finds the derivative [[gradient-descent|gradient descent]] needs for every weight. At the end we look at what an [[mlp|MLP]] can learn and why it falls short on images, which leads to the CNN.` },
      { tr: R`<b>Bu bölümün sonunda</b> şunları yapabileceksiniz: bir gizli katmanın XOR'u nasıl çözdüğünü gizli uzayda açıklamak; çapraz entropi kaybını ve çıktıdaki gradyanını yazmak; küçük bir ağda geri yayılımı adım adım hesaplamak; nöron sayısının ağın öğrenebileceklerini nasıl değiştirdiğini ve MLP'nin görüntülerde neden zorlandığını anlatmak.`,
        en: R`<b>By the end of this chapter</b> you will be able to: explain in the hidden space how a hidden layer solves XOR; write down the cross-entropy loss and its gradient at the output; compute backpropagation step by step on a small network; and describe how the number of neurons changes what a network can learn, and why an MLP struggles with images.` },
    ],
    compare: {
      title: { tr: "Perceptrondan MLP'ye", en: "From the perceptron to the MLP" },
      rows: [
        [{ tr: "Aktivasyon", en: "Activation" }, { tr: "basamak → sigmoid, tanh, ReLU", en: "step → sigmoid, tanh, ReLU" }],
        [{ tr: "Katmanlar", en: "Layers" }, { tr: "tek nöron → gizli katmanlar", en: "one neuron → hidden layers" }],
        [{ tr: "Hata ölçüsü", en: "Error measure" }, { tr: "perceptron ölçütü (yalnızca yanlışlar) → çapraz entropi (her örnek)", en: "perceptron criterion (only the mistakes) → cross-entropy (every example)" }],
        [{ tr: "Öğrenme", en: "Learning" }, { tr: "perceptron kuralı (tek katman) → geri yayılım + gradyan inişi (bütün katmanlar)", en: "perceptron rule (one layer) → backpropagation + gradient descent (all layers)" }],
        [{ tr: "Karar sınırı", en: "Decision boundary" }, { tr: "doğru → eğri", en: "a line → a curve" }],
      ],
    },
    fact: {
      title: { tr: "Tarihten", en: "History" },
      text: { tr: R`Geri yayılımın matematiği daha eskidir: Seppo Linnainmaa 1970'te otomatik türevin "ters mod"unu tanımladı, Paul Werbos 1974'teki doktora tezinde yöntemin sinir ağlarını eğitmek için kullanılabileceğini önerdi. 1986 makalesi ise gizli katmanların ne öğrendiğini gösterdi ve yöntemi yaygınlaştırdı.`,
        en: R`The mathematics of backpropagation is older: Seppo Linnainmaa described the "reverse mode" of automatic differentiation in 1970, and Paul Werbos proposed using the method to train neural networks in his 1974 PhD thesis. The 1986 paper showed what hidden layers learn and made the method widely known.` },
    },
  },
  act: {
    h: PLACE_TITLES["ml-act"],
    p1: { tr: R`1. bölümün sonunda gördüğümüz gibi, gizli katmanı öğretmek için hatayı gizli nöronlara kadar geriye taşımamız gerekir. Gradyan inişi her ağırlığı kaybı azaltan yönde biraz oynatır; bunun için "bu ağırlığı biraz değiştirirsem kayıp ne kadar değişir?" sorusunun cevabı, yani türev gerekir. Gizli bir ağırlığın etkisi kayba, gizli nöronun aktivasyonundan geçerek ulaşır. Basamak fonksiyonunun eğimi her yerde 0'dır (\(z = 0\)'da tanımsız): ağırlığı biraz oynatmak nöronun çıktısını ya hiç değiştirmez ya da birden 0'dan 1'e atlatır. Geriye taşınacak bir işaret yoktur. (Perceptron kuralı bu sorunu basamağın türevini hiç almadan, doğrudan \(z\)'ye bakarak atlıyordu; bu hile yalnızca tek katmanda işe yarar.)`,
      en: R`As we saw at the end of chapter 1, to teach the hidden layer we have to carry the error back to the hidden neurons. Gradient descent nudges every weight in the direction that lowers the loss; for that it needs the answer to "if I change this weight a little, how much does the loss change?", that is, a derivative. A hidden weight reaches the loss through the hidden neuron's activation. The slope of the step function is 0 everywhere (undefined at \(z = 0\)): nudging the weight either does not change the neuron's output at all or makes it jump from 0 to 1. There is no signal to carry back. (The perceptron rule got around this by looking at \(z\) directly, never differentiating the step; that trick works only for a single layer.)` },
    p2: { tr: R`[[sigmoid|Sigmoid]] basamağın yumuşatılmış hâlidir: büyük negatif \(z\)'de 0'a, büyük pozitif \(z\)'de 1'e yaklaşır, arada yumuşakça geçer ve her yerde türevlidir. Çıktısı 0 ile 1 arasında olduğu için "sınıf 1 olma olasılığı" gibi okunabilir. Sağda dikliği \(k\) büyütün: \(\sigma(kz)\) basamağa yaklaşır.`,
      en: R`The [[sigmoid|sigmoid]] is a smoothed step: it approaches 0 for large negative \(z\) and 1 for large positive \(z\), passes smoothly in between, and has a derivative everywhere. Its output lies between 0 and 1, so it reads like "the probability of class 1". On the right, raise the steepness \(k\): \(\sigma(kz)\) approaches the step.` },
    p3: { tr: R`[[tanh|tanh]] aynı şeklin −1 ile 1 arasına gerilmiş hâlidir; [[relu|ReLU]] ise \(\max(0, z)\): pozitif tarafta eğimi hep 1'dir. Kesikli turuncu eğri her fonksiyonun türevidir. Sigmoid'in türevi en fazla 0,25'tir ve iki uçta neredeyse 0'a iner: nöron "doyduğunda" öğrenme yavaşlar. Derin ağlarda ReLU'nun tercih edilmesinin bir nedeni budur ([[vanishing-gradient|kaybolan gradyan]]). <b>Deneyin:</b> önce tahmin edin, sigmoid'in \(z = 5\)'teki türevi kabaca kaçtır? Sonra \(z\) kaydırıcısıyla kontrol edin ve aynısını tanh ile ReLU için yapın.`,
      en: R`[[tanh|tanh]] is the same shape stretched to lie between −1 and 1; [[relu|ReLU]] is \(\max(0, z)\): its slope is always 1 on the positive side. The dashed orange curve is each function's derivative. The sigmoid's derivative is at most 0.25 and falls to almost 0 at both ends: when a neuron "saturates", learning slows down. That is one reason deep networks prefer ReLU ([[vanishing-gradient|vanishing gradient]]). <b>Try it:</b> first guess roughly what the sigmoid's derivative is at \(z = 5\). Then check with the \(z\) slider, and do the same for tanh and ReLU.` },
    model: { tr: R`<b>Perceptrondan farkı:</b> basamağı sigmoidle değiştirmek tek bir nöronun karar sınırını değiştirmez: \(\sigma(z) > 0{,}5 \iff z > 0\), sınır yine bir doğrudur. Yeni olan, artık her ağırlığın kaybı nasıl etkilediğinin hesaplanabilmesi.`,
      en: R`<b>The difference from the perceptron:</b> swapping the step for the sigmoid does not change a single neuron's decision boundary: \(\sigma(z) > 0.5 \iff z > 0\), the boundary is still a line. What is new is that we can now compute how every weight affects the loss.` },
  },
  hidden: {
    h: PLACE_TITLES["ml-hidden"],
    p1: { tr: R`XOR'a dönelim. Ağa iki nöronlu bir [[hidden-layer|gizli katman]] ekliyoruz: 2 girdi → 2 gizli nöron → 1 çıktı, hepsi sigmoid. Gizli nöronların her biri, perceptron gibi, kendi doğrusunu çizer (soldaki kesikli doğrular) ve her nokta için bir sayı üretir; ikisi birlikte her noktaya iki yeni koordinat verir: \(h_1\) ve \(h_2\). Çıktı nöronunun ağırlıklarını ayrı tutmak için \(v_1, v_2\) ve biasını \(c\) diye yazıyoruz.`,
      en: R`Back to XOR. We add a [[hidden-layer|hidden layer]] of two neurons to the network: 2 inputs → 2 hidden neurons → 1 output, all sigmoid. Each hidden neuron, like a perceptron, draws its own line (the dashed lines on the left) and gives every point one number; together they give every point two new coordinates: \(h_1\) and \(h_2\). To keep them apart, we write the output neuron's weights as \(v_1, v_2\) and its bias as \(c\).` },
    p2: { tr: R`Sağdaki düzlem noktaları bu yeni koordinatlarda gösterir: <b>gizli uzay</b>. <b>Eğit</b>'e basıp izleyin: ağ gizli katmanın ağırlıklarını öyle ayarlar ki dört nokta gizli uzayda yer değiştirir ve sonunda tek bir doğruyla ayrılabilir hâle gelir. Çıktı nöronu orada bir perceptron gibi çalışır; soldaki eğri sınır, o doğrunun girdi uzayındaki karşılığıdır.`,
      en: R`The plane on the right shows the points in these new coordinates: the <b>hidden space</b>. Press <b>Train</b> and watch: the network sets the hidden weights so that the four points move in the hidden space and finally become separable by a single line. The output neuron works there like a perceptron; the curved boundary on the left is that line seen in the input space.` },
    p3: { tr: R`İşin özü bu: gizli katmanlar veriyi, sonraki katmanın işini kolaylaştıran bir gösterime çevirir. Bu gösterimi biz tasarlamayız; ağ onu [[backprop|geri yayılımla]] kendisi öğrenir. 1986 makalesinin başlığı da bunu söyler: <i>Learning representations by back-propagating errors</i>, "geri yayılan hatalarla gösterim öğrenmek".`,
      en: R`That is the heart of it: hidden layers turn the data into a representation that makes the next layer's job easy. We do not design that representation; the network learns it by itself with [[backprop|backpropagation]]. The title of the 1986 paper says just that: <i>Learning representations by back-propagating errors</i>.` },
    model: { tr: R`<b>Bazen takılır:</b> <b>Yeni başlangıç</b>'a iki kez basıp (başlangıç 3) eğitin. Bazı başlangıçlarda kayıp 0,35 civarında durur ve iki nokta \(\hat y \approx 0{,}5\)'te kalır. Gizli nöronlar doymuştur, gradyan neredeyse sıfırdır: ağırlıklar kayıp yüzeyinin çok düz bir bölgesindedir ve küçük adımlar kaybı fark edilir biçimde azaltmaz. Uygulamada bir [[local-minimum|yerel minimum]] gibi davranır (bu küçük XOR ağında aslında gerçek bir yerel minimum olmadığı gösterilmiştir). Başka rastgele ağırlıklarla genelde çözülür; daha fazla gizli nöron da bu sorunu büyük ölçüde azaltır (Deneme alanında deneyin).`,
      en: R`<b>Sometimes it gets stuck:</b> press <b>New start</b> twice (start 3) and train. For some starts the loss stops near 0.35 and two points stay at \(\hat y \approx 0.5\). The hidden neurons have saturated and the gradient is almost zero: the weights are in a very flat region of the loss surface, and small steps do not lower the loss noticeably. In practice it behaves like a [[local-minimum|local minimum]] (for this small XOR network it has in fact been shown that there is no true local minimum). With other random weights it usually works; more hidden neurons also largely remove the problem (try it in the Playground).` },
  },
  loss: {
    h: PLACE_TITLES["ml-loss"],
    p1: { tr: R`1. bölümde perceptron kuralının da bir kaybı, perceptron ölçütünü azaltan bir gradyan inişi olduğunu gördük. Ama o kayıp yalnızca yanlış sınıflanan noktalara bakar: doğru taraftaki bir nokta, sınıra ne kadar yakın olursa olsun, kayba hiçbir şey katmaz. Çıktı artık 0 ile 1 arasında bir olasılık olduğuna göre daha iyisini yapabiliriz: [[loss|kayıp fonksiyonu]], modelin her örnekte doğru cevaba ne kadar olasılık verdiğini tek ve sürekli değişen bir sayıyla ölçsün. Olasılık veren bir çıktı için standart seçim [[cross-entropy|ikili çapraz entropidir]]:`,
      en: R`In chapter 1 we saw that the perceptron rule, too, is gradient descent on a loss, the perceptron criterion. But that loss looks only at the misclassified points: a point on the right side adds nothing to it, however close to the boundary it is. Now that the output is a probability between 0 and 1, we can do better: let the [[loss|loss function]] measure, with a single number that changes smoothly, how much probability the model gives the right answer on every example. For an output that gives a probability, the standard choice is the [[cross-entropy|binary cross-entropy]]:` },
    p2: { tr: R`Doğru sınıfa verilen olasılık 1'e yaklaştıkça kayıp 0'a iner; emin olunan yanlış bir cevap ise çok pahalıdır (\(y = 1\) iken \(\hat y \to 0\) ise \(L \to \infty\)). Eğitimde bütün örneklerin ortalaması alınır.`,
      en: R`As the probability given to the right class approaches 1, the loss falls to 0; a confident wrong answer is very expensive (if \(y = 1\) and \(\hat y \to 0\), then \(L \to \infty\)). Training uses the average over all examples.` },
    p3: { tr: R`Sağdaki örnekte modelin tek bir ağırlığı var: \(\hat y = \sigma(w x)\). Üstteki grafik ortalama kaybı \(w\)'nin fonksiyonu olarak çizer. [[gradient-descent|Gradyan inişi]] her adımda, 1. bölümdeki gibi, eğimin tersine gider: \(w \leftarrow w - \eta\, L'(w)\). Önce <b>Bir adım</b>'a basın: \(w\) hangi yöne gidecek? Sonra [[learning-rate|öğrenme oranı]] \(\eta\)'yı büyütüp <b>Baştan</b> ile yeniden deneyin: adımlar uzar; çok büyükse minimumun üstünden atlar.`,
      en: R`In the example on the right the model has a single weight: \(\hat y = \sigma(w x)\). The top graph plots the average loss as a function of \(w\). [[gradient-descent|Gradient descent]], as in chapter 1, steps against the slope every time: \(w \leftarrow w - \eta\, L'(w)\). First press <b>One step</b>: which way will \(w\) go? Then raise the [[learning-rate|learning rate]] \(\eta\) and try again with <b>Restart</b>: the steps get longer; too large and they jump over the minimum.` },
    model: { tr: R`<b>MSE'ye geçin:</b> ortalama kare hata da çalışır, ama \(w\) çok yanlışken (solda) eğri neredeyse düzdür: sigmoid doyduğunda MSE'nin gradyanı da küçülür ve öğrenme yavaşlar. Çapraz entropide bu olmaz, çünkü çıktıdaki gradyan sadece \(\hat y - y\)'dir (Matematik, 3. kısım).`,
      en: R`<b>Switch to MSE:</b> the mean squared error works too, but when \(w\) is very wrong (on the left) its curve is almost flat: when the sigmoid saturates, the MSE gradient shrinks too and learning slows down. Cross-entropy does not have this problem, because its gradient at the output is just \(\hat y - y\) (Math, section 3).` },
  },
  backprop: {
    h: PLACE_TITLES["ml-backprop"],
    p1: { tr: R`Gradyan inişi her ağırlık için \(\partial L / \partial w\)'yu ister. Çıktı katmanında bu kolay. Peki gizli bir ağırlık kaybı ne kadar etkiler? Etkisi bir zincir boyunca geçer: \(w_{11} \to z_1 \to h_1 \to z_o \to \hat y \to L\) (\(z_o\), çıktı nöronunun toplamı). [[chain-rule|Zincir kuralı]] der ki \(a\), \(b\)'yi ve \(b\) de \(c\)'yi etkiliyorsa \(\frac{\partial c}{\partial a} = \frac{\partial c}{\partial b}\cdot\frac{\partial b}{\partial a}\): zincirdeki yerel türevler çarpılır.`,
      en: R`Gradient descent asks for \(\partial L / \partial w\) for every weight. At the output layer that is easy. But how much does a hidden weight affect the loss? Its effect passes along a chain: \(w_{11} \to z_1 \to h_1 \to z_o \to \hat y \to L\) (\(z_o\) is the output neuron's sum). The [[chain-rule|chain rule]] says that if \(a\) affects \(b\) and \(b\) affects \(c\), then \(\frac{\partial c}{\partial a} = \frac{\partial c}{\partial b}\cdot\frac{\partial b}{\partial a}\): the local derivatives along the chain multiply.` },
    p2: { tr: R`[[backprop|Geri yayılım]] bunu düzenli bir sırayla yapar. Önce [[forward-pass|ileri hesapta]] her nöronun değeri bulunur. Sonra kayıptan başlayarak geriye doğru her nöronun hata sinyali \(\delta = \partial L/\partial z\) hesaplanır: bir nöronun \(\delta\)'sı, sonraki katmanın \(\delta\)'larının ağırlıklarla toplanıp nöronun kendi eğimiyle çarpılmasıdır. Her ağırlığın gradyanı da girdisiyle hedef nöronun \(\delta\)'sının çarpımıdır.`,
      en: R`[[backprop|Backpropagation]] does this in an orderly way. First the [[forward-pass|forward pass]] finds the value of every neuron. Then, starting from the loss and going backwards, it computes every neuron's error signal \(\delta = \partial L/\partial z\): a neuron's \(\delta\) is the next layer's \(\delta\)s summed through the weights, times the neuron's own slope. And every weight's gradient is its input times the \(\delta\) of the neuron it feeds.` },
    p3: { tr: R`Sağda bir örnek için hesabın tamamını adım adım izleyin: mavi sayılar ileri hesabın değerleri, turuncu sayılar gradyanlar. Son adımda bütün ağırlıklar aynı anda güncellenir ve aynı örneğin kaybı azalır. Gitmeden önce tahmin edin: \(\mathbf x = (1, 0)\) örneğinde hangi gizli ağırlıkların gradyanı 0 çıkacak? Sonra \(y = 0\) olan bir örnek seçin: \(\delta_o\)'nun işareti ne oluyor?`,
      en: R`On the right, follow the whole computation for one example, step by step: blue numbers are values of the forward pass, orange numbers are gradients. In the last step every weight is updated at once and the loss on the same example goes down. Before you start, predict: for the example \(\mathbf x = (1, 0)\), which hidden weights will get a zero gradient? Then pick an example with \(y = 0\): what is the sign of \(\delta_o\)?` },
    model: { tr: R`<b>Neden geriye?</b> Gradyanları sondan başa hesaplamak her ara sonucu bir kez kullanır: geri hesabın maliyeti, ağırlık sayısı ne olursa olsun, yaklaşık iki ileri hesap kadardır (Matematik, 4. kısım). Her ağırlığı tek tek oynatıp kaybı yeniden ölçmek (sayısal türev) ise en az ağırlık sayısı kadar ileri hesap ister: SimpleCNN için 582 bin.`,
      en: R`<b>Why backwards?</b> Computing the gradients from the end to the start uses every intermediate result once: the backward pass costs about two forward passes, whatever the number of weights (Math, section 4). Nudging each weight in turn and measuring the loss again (a numerical derivative) would take at least as many forward passes as there are weights: 582 thousand for SimpleCNN.` },
  },
  capacity: {
    h: PLACE_TITLES["ml-capacity"],
    p1: { tr: R`Tek gizli katmanlı bir ağ, girdiden bir sayı üreten bir fonksiyondur. Sağdaki ağ (1 girdi → \(H\) gizli tanh nöron → 1 doğrusal çıktı) seçilen eğriye uydurulmaya çalışılıyor. İnce renkli eğriler gizli nöronların tek tek katkılarıdır: her biri bir "yumuşak basamak", çıktı ise onların ağırlıklı toplamı.`,
      en: R`A network with one hidden layer is a function that turns an input into a number. The network on the right (1 input → \(H\) hidden tanh neurons → 1 linear output) is being fitted to the chosen curve. The thin colored curves are the hidden neurons' contributions one by one: each is a "soft step", and the output is their weighted sum.` },
    p2: { tr: R`Nöron sayısını artırın: daha çok basamak, daha ince ayrıntı. [[universal-approximation|Evrensel yaklaşım teoremi]] (Cybenko 1989, Hornik 1991) bunu kesinleştirir: tek gizli katmanlı bir ağ, yeterince nöronla, kapalı bir aralıktaki her sürekli fonksiyona istenen yakınlıkta yaklaşabilir.`,
      en: R`Add neurons: more steps, finer detail. The [[universal-approximation|universal approximation theorem]] (Cybenko 1989, Hornik 1991) makes this precise: a network with one hidden layer can, with enough neurons, approximate any continuous function on a closed interval as closely as we like.` },
    p3: { tr: R`Ama teorem kaç nöron gerektiğini ya da gradyan inişinin o ağırlıkları bulacağını söylemez. Pratikte derin (çok katmanlı) ağlar aynı işi çoğu zaman çok daha az nöronla yapar. Ayrıca eğitim noktalarına mükemmel uymak, yeni noktalarda iyi [[generalization|genelleme]] demek değildir.`,
      en: R`But the theorem does not say how many neurons it takes, or that gradient descent will find those weights. In practice deep (many-layer) networks often do the same job with far fewer neurons. And fitting the training points perfectly does not mean good [[generalization|generalization]] to new points.` },
    model: { tr: R`<b>Deneyin</b> (her seferinde önce tahmin edin, sonra eğitin): <b>Tepe</b>'yi seçip \(H = 1\) ve \(H = 2\) ile eğitin. Bir nöron neden yetmiyor da iki nöron yetiyor? (Biri yukarı, biri aşağı iki yumuşak basamak bir tepe yapar; Matematik, 6. kısım.) <b>Basamak</b>'ta \(H = 1\) yapın: tek nöronun kendisi basamak olur ve eğitim sürdükçe dikleşir. <b>Dalga</b>'da \(H = 1\)'den başlayıp artırın: kaç nörondan sonra iyi uyuyor?`,
      en: R`<b>Try it</b> (each time, predict first, then train): pick the <b>Bump</b> and train with \(H = 1\) and \(H = 2\). Why is one neuron not enough while two are? (Two soft steps, one up and one down, make a bump; Math, section 6.) For the <b>Step</b>, set \(H = 1\): the single neuron itself becomes the step and gets steeper the longer it trains. For the <b>Wave</b>, start at \(H = 1\) and go up: from how many neurons on is the fit good?` },
  },
  images: {
    h: PLACE_TITLES["ml-images"],
    p1: { tr: R`Bir MLP görüntüyü nasıl görür? 28×28 piksellik bir MNIST rakamı [[flatten|düzleştirilip]] 784 sayılık bir vektör olur ve her gizli nöron bu 784 sayının her birine ayrı bir ağırlık verir. 512 gizli nöronlu tek bir katman \(784 \cdot 512 + 512 = 401\,920\) [[parameter|parametre]] demektir. İyi eğitilmiş bir MLP, MNIST'te yaklaşık %98 doğruluğa ulaşabilir.`,
      en: R`How does an MLP see an image? A 28×28-pixel MNIST digit is [[flatten|flattened]] into a vector of 784 numbers, and every hidden neuron gives each of those 784 numbers its own weight. A single layer of 512 hidden neurons is \(784 \cdot 512 + 512 = 401{,}920\) [[parameter|parameters]]. A well-trained MLP can reach about 98% accuracy on MNIST.` },
    p2: { tr: R`Sorun şu: MLP için piksellerin komşuluğu yoktur. Vektörün sırasını (her görüntüde aynı şekilde) karıştırsanız, karıştırılmış görüntülerle eğitilen bir MLP aynı başarıya ulaşır; hangi pikselin hangisinin yanında olduğunu hiç kullanmaz. Sağdaki rakamın 32 mürekkepli pikseli var. Önce tahmin edin: rakamı bir piksel sağa kaydırınca kaçı yerinde kalır? Sonra deneyin, iki ve üç piksel de kaydırın: göz için hep aynı rakam, ama mürekkebin giderek daha büyük kısmı MLP'nin başka girdilerine, yani başka ağırlıklara düşer. Ağ aynı çizgiyi her konumda ayrı ayrı öğrenmek zorundadır.`,
      en: R`The problem: to an MLP, pixels have no neighbours. Shuffle the order of the vector (the same way for every image), and an MLP trained on the shuffled images does just as well; it never uses which pixel is next to which. The digit on the right has 32 inked pixels. Guess first: if you move it one pixel to the right, how many stay in place? Then try it, and move it by two and three pixels too: to the eye it is always the same digit, but a growing share of the ink lands on other inputs of the MLP, that is, on other weights. The network has to learn the same stroke separately at every position.` },
    p3: { tr: R`[[convolution|Evrişim]] iki sorunu birden çözer: küçük bir filtre (SimpleCNN'de 5×5 = 25 ağırlık) yalnızca komşu piksellere bakar ve aynı ağırlıklar görüntünün her yerinde kullanılır. SimpleCNN'in iki evrişim katmanında toplam yalnızca 52.096 parametre var. Sonraki bölümün konusu bu.`,
      en: R`[[convolution|Convolution]] solves both problems at once: a small filter (5×5 = 25 weights in SimpleCNN) looks only at neighbouring pixels, and the same weights are used everywhere in the image. The two convolution layers of SimpleCNN have only 52,096 parameters in all. That is the subject of the next chapter.` },
    model: { tr: R`<b>SimpleCNN'in sonu zaten bir MLP:</b> [[fully-connected|FC1]] (1024 → 512, ReLU) ve FC2 (512 → 10). Evrişim katmanları görüntüden özellikleri çıkarır, MLP de bu özelliklerden sınıfa karar verir. Parametrelerin çoğu da bu kısımdadır: FC1 tek başına 524.800 parametre, bütün modelin %90'ı.`,
      en: R`<b>SimpleCNN ends with an MLP:</b> [[fully-connected|FC1]] (1024 → 512, ReLU) and FC2 (512 → 10). The convolution layers extract features from the image and the MLP decides the class from those features. Most of the parameters are there too: FC1 alone has 524,800, 90% of the whole model.` },
    recap: { tr: R`<b>Özet:</b> Gizli katmanlar veriyi, bir doğrunun ayırabileceği yeni bir gösterime çevirir. Bunu öğrenmek için türevlenebilir bir aktivasyon, çapraz entropi gibi her örneği ölçen bir kayıp ve bütün gradyanları bir geri hesapta bulan geri yayılım gerekir. Yeterince nöronla bir MLP her sürekli fonksiyona yaklaşabilir; ama görüntülerde komşuluğu bilmez ve aynı deseni her konumda yeniden öğrenir. CNN bu eğitim düzeneğini aynen kullanır, yalnızca katmanları değişir.`,
      en: R`<b>Recap:</b> hidden layers turn the data into a new representation that a line can separate. Learning it takes a differentiable activation, a loss that measures every example, such as cross-entropy, and backpropagation, which finds all the gradients in one backward pass. With enough neurons an MLP can approximate any continuous function; but on images it knows nothing about neighbours and learns the same pattern again at every position. The CNN keeps exactly this training machinery; only its layers change.` },
  },
};

// =====================================================================
// Theory 1 · activation functions and their derivatives
// =====================================================================
const AS = { fn: "sigmoid", z: 1, k: 1 };
const ACTS = {
  step: { name: { tr: "Basamak", en: "Step" }, f: (z) => (z > 0 ? 1 : 0), d: (z) => (Math.abs(z) < 1e-9 ? null : 0) },
  sigmoid: { name: { tr: "Sigmoid", en: "Sigmoid" }, f: (z, k) => sigma(k * z), d: (z, k) => k * sigma(k * z) * (1 - sigma(k * z)) },
  tanh: { name: { tr: "tanh", en: "tanh" }, f: (z) => Math.tanh(z), d: (z) => 1 - Math.tanh(z) ** 2 },
  relu: { name: { tr: "ReLU", en: "ReLU" }, f: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : z < 0 ? 0 : null) },
};
const ACT_TEX = {
  step: R`H(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}`,
  sigmoid: R`\sigma(kz) = \frac{1}{1 + e^{-kz}}`,
  tanh: R`\tanh z = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}`,
  relu: R`\mathrm{ReLU}(z) = \max(0,\, z)`,
};

function buildActDemo() {
  const host = h("div", { class: "pc-demo" });
  const stage = h("div", { class: "pc-stage" });
  const graph = createGraph(stage, { x: [-6, 6], y: [-1.25, 1.6], xTicks: [-6, -4, -2, 0, 2, 4, 6], yTicks: [-1, -0.5, 0, 0.5, 1, 1.5],
    label: L({ tr: "Aktivasyon fonksiyonu ve türevi", en: "The activation function and its derivative" }), xName: "z" });
  const legend = h("div", { class: "legend" },
    h("span", { class: "legend-item" }, h("span", { class: "key-line s1" }), L({ tr: "fonksiyon f(z)", en: "function f(z)" })),
    h("span", { class: "legend-item" }, h("span", { class: "key-line s2 dashed" }), L({ tr: "türev f′(z)", en: "derivative f′(z)" })));
  const kSlider = slider(L({ tr: "Diklik k", en: "Steepness k" }), { min: 0.5, max: 10, step: 0.5, value: AS.k, onInput: (v) => { AS.k = v; update(); } });
  const zSlider = slider("z", { min: -6, max: 6, step: 0.1, value: AS.z, onInput: (v) => { AS.z = v; update(); } });
  const seg = segmented(Object.entries(ACTS).map(([k, a]) => [k, L(a.name)]), AS.fn, (v) => { AS.fn = v; update(); }, L({ tr: "Aktivasyon", en: "Activation" }));
  const out = h("div", { class: "pc-live" });
  const note = msg("");
  host.append(h("div", { class: "t-controls" }, seg), stage, legend, h("div", { class: "t-controls" }, zSlider, kSlider), out, note);

  function update() {
    const a = ACTS[AS.fn];
    const k = AS.fn === "sigmoid" ? AS.k : 1;
    kSlider.hidden = AS.fn !== "sigmoid";
    const f = (z) => a.f(z, k);
    const d = (z) => a.d(z, k);
    const z = AS.z;
    const fz = f(z);
    const dz = d(z);
    const tangent = dz == null ? [] : [{ from: [z - 1.3, fz - 1.3 * dz], to: [z + 1.3, fz + 1.3 * dz], color: COLORS.text, width: 1.4, dash: true }];
    graph.draw({
      curves: [{ f, color: COLORS.lineB, width: 2.8 }, { f: d, color: COLORS.line, width: 2, dash: true }],
      segments: tangent,
      vlines: [{ x: z }],
      points: [{ x: z, y: fz, color: COLORS.lineB, ring: true }],
    });
    const dTex = dz == null ? R`\text{${L({ tr: "tanımsız", en: "undefined" })}}` : tx(dz, 3);
    out.textContent = `\\(${ACT_TEX[AS.fn]}\\qquad f(${tx(z, 1)}) = ${tx(fz, 3)},\\quad f'(${tx(z, 1)}) = ${dTex}\\)`;
    const notes = {
      step: [L({ tr: "Eğim her yerde 0 (z = 0'da tanımsız): ağırlığı biraz değiştirmek çıktıyı değiştirmez, gradyan inişi ilerleyemez.", en: "The slope is 0 everywhere (undefined at z = 0): a small change of a weight does not change the output, gradient descent cannot move." }), "bad"],
      sigmoid: [AS.k > 4
        ? L({ tr: `k = ${nf(AS.k, 1)}: neredeyse basamak. Türev ortada çok büyük, başka her yerde neredeyse 0.`, en: `k = ${nf(AS.k, 1)}: almost a step. The derivative is very large in the middle and almost 0 everywhere else.` })
        : L({ tr: "Türev en fazla k/4'tür (z = 0'da). |z| büyüdükçe 0'a iner: nöron doyar.", en: "The derivative is at most k/4 (at z = 0). It falls to 0 as |z| grows: the neuron saturates." }), ""],
      tanh: [L({ tr: "Çıktı −1 ile 1 arasında, sıfır merkezli. Türevi en fazla 1, uçlarda yine 0'a iner.", en: "The output lies between −1 and 1, centred on zero. Its derivative is at most 1, and again falls to 0 at the ends." }), ""],
      relu: [L({ tr: "z > 0 iken eğim hep 1: aktivasyon, geriye giden gradyanı küçültmez. z < 0 iken 0: o nöron bu örnekte öğrenmez.", en: "For z > 0 the slope is always 1: the activation does not shrink the gradient going back. For z < 0 it is 0: that neuron does not learn from this example." }), "good"],
    };
    setMsg(note, ...notes[AS.fn]);
  }
  redrawers.add(update);
  update();
  return host;
}

// =====================================================================
// Theory 2 · the hidden layer on XOR, in the input space and the hidden space
// =====================================================================
const HS = { seed: 1, net: null, history: [], timer: 0, status: "idle" };
const HS_LR = 2;
const HS_MAX = 4000;
function resetHidden() {
  HS.net = NET.create([2, 2, 1], { act: "sigmoid", out: "sigmoid", rand: seeded(HS.seed) });
  HS.history = [NET.lossOf(HS.net, XOR)];
  HS.status = "idle";
}
resetHidden();

function buildHiddenDemo() {
  const host = h("div", { class: "pc-demo" });
  const leftHost = h("div", { class: "pc-stage" });
  const rightHost = h("div", { class: "pc-stage" });
  const two = h("div", { class: "pc-two ml-spaces" },
    h("div", {}, h("div", { class: "mini-title", text: L({ tr: "Girdi uzayı (x₁, x₂)", en: "Input space (x₁, x₂)" }) }), leftHost),
    h("div", {}, h("div", { class: "mini-title", text: L({ tr: "Gizli uzay (h₁, h₂)", en: "Hidden space (h₁, h₂)" }) }), rightHost));
  const left = createPlane(leftHost, { label: L({ tr: "Girdi uzayı", en: "Input space" }) });
  const right = createPlane(rightHost, { domain: [-0.15, 1.15, -0.15, 1.15], ticks: [0, 1], axes: ["h₁", "h₂"], label: L({ tr: "Gizli uzay", en: "Hidden space" }) });
  const play = button("", "btn primary");
  const again = button(L({ tr: "Yeni başlangıç", en: "New start" }));
  const status = msg("");
  const lossHost = chartHost(150);
  const net = h("div", { class: "ml-mini-net" });
  host.append(h("div", { class: "t-controls" }, play, again), two, status,
    h("div", { class: "mini-title", text: L({ tr: "Kayıp (ortalama çapraz entropi)", en: "Loss (mean cross-entropy)" }) }), lossHost, net);

  const running = () => Boolean(HS.timer);
  play.addEventListener("click", () => {
    if (running()) { HS.timer = stop(HS.timer); update(); return; }
    if (HS.status === "done" || HS.status === "stuck") resetHidden();
    HS.timer = every(30, () => {
      for (let k = 0; k < 8; k++) {
        NET.trainEpoch(HS.net, XOR, { lr: HS_LR });
        HS.history.push(NET.lossOf(HS.net, XOR));
      }
      const solved = NET.accuracy(HS.net, XOR) === 1 && HS.history.at(-1) < 0.05;
      // stuck: after 1000 epochs the loss has hardly moved in the last 400
      const h = HS.history;
      const flat = HS.net.epoch >= 1000 && h.length > 400 && h.at(-400) - h.at(-1) < 0.002 && h.at(-1) > 0.05;
      if (solved || flat || HS.net.epoch >= HS_MAX) {
        HS.status = solved ? "done" : "stuck";
        HS.timer = stop(HS.timer);
      } else HS.status = "running";
      update();
    });
    update();
  });
  again.addEventListener("click", () => {
    HS.timer = stop(HS.timer);
    HS.seed += 1;
    resetHidden();
    update();
  });

  function update() {
    const n = HS.net;
    const [L1, L2] = n.layers;
    const hiddenOf = (x) => NET.forward(n, x).as[1];
    const colors = [COLORS.lineB, COLORS.line];
    left.draw({
      prob: (x1, x2) => NET.predict(n, [x1, x2]),
      contour: true,
      lines: L1.W.map((w, j) => ({ w, b: L1.b[j], ghost: true, color: colors[j], label: `h${j ? "₂" : "₁"}` })),
      points: XOR.map((d) => ({ ...d, wrong: (NET.predict(n, d.x) > 0.5 ? 1 : 0) !== d.y })),
      coords: true,
    });
    const hiddenPts = XOR.map((d) => ({ x: hiddenOf(d.x), y: d.y, label: `(${d.x[0]}, ${d.x[1]})` }));
    hiddenPts.forEach((p, i) => {   // points that land on the same spot share one label
      const twin = hiddenPts.findIndex((q, k) => k < i && q.label && Math.hypot(q.x[0] - p.x[0], q.x[1] - p.x[1]) < 0.08);
      if (twin >= 0) { hiddenPts[twin].label += ` ${p.label}`; p.label = ""; }
    });
    right.draw({
      prob: (h1, h2) => sigma(L2.W[0][0] * h1 + L2.W[0][1] * h2 + L2.b[0]),
      lines: [{ w: L2.W[0], b: L2.b[0] }],
      points: hiddenPts,
      coords: true,
    });
    play.textContent = running() ? L({ tr: "❚❚ Durdur", en: "❚❚ Pause" }) : L({ tr: "▶ Eğit", en: "▶ Train" });
    const lossNow = HS.history.at(-1);
    const right_ = Math.round(NET.accuracy(n, XOR) * 4);
    const texts = {
      idle: [L({ tr: `Rastgele ağırlıklar (başlangıç ${HS.seed}). ${right_}/4 doğru. Eğit'e basın.`, en: `Random weights (start ${HS.seed}). ${right_}/4 right. Press Train.` }), ""],
      running: [L({ tr: `Epoch ${n.epoch} · kayıp ${nf(lossNow, 3)} · ${right_}/4 doğru`, en: `Epoch ${n.epoch} · loss ${nf(lossNow, 3)} · ${right_}/4 right` }), ""],
      done: [L({ tr: `Çözüldü ✓ ${n.epoch} epoch'ta. Sağda: dört nokta gizli uzayda artık bir doğruyla ayrılıyor.`, en: `Solved ✓ in ${n.epoch} epochs. On the right: in the hidden space the four points are now separated by a line.` }), "good"],
      stuck: [L({ tr: `${n.epoch} epoch'ta kayıp ${nf(lossNow, 3)}'te takıldı, ${right_}/4 doğru: gizli nöronlar doydu, gradyan neredeyse sıfır (düz bir bölge). Yeni başlangıç'a basın.`, en: `After ${n.epoch} epochs the loss is stuck at ${nf(lossNow, 3)}, ${right_}/4 right: the hidden neurons have saturated and the gradient is almost zero (a flat region). Press New start.` }), "bad"],
    };
    setMsg(status, ...texts[HS.status]);
    const hist = HS.history;
    const stepX = Math.max(1, Math.ceil(hist.length / 300));
    const pts = hist.filter((_, i) => i % stepX === 0).map((v, i) => [i + 1, v]);
    const xMax = Math.max(pts.length, 5);
    const epochAt = (i) => (i - 1) * stepX;
    lineChart(lossHost, {
      label: L({ tr: "Kayıp", en: "Loss" }), empty: "",
      x: { min: 1, max: xMax, ticks: linearTicks(1, xMax, Math.max(1, niceStep(xMax, 5)), (v) => fmtNum(epochAt(v), 0)) },
      y: { min: 0, max: 0.8, ticks: linearTicks(0, 0.8, 0.2, (v) => fmtNum(v, 1)) },
      series: [{ points: pts, color: "var(--series-2)" }],
      tooltip: (i) => ({ title: `Epoch ${epochAt(i + 1)}`, rows: [{ value: nf(pts[i][1], 3), label: L({ tr: "kayıp", en: "loss" }), color: "var(--series-2)" }] }),
    });
    net.replaceChildren(miniNetDiagram(n));
  }
  HS.update = update;
  redrawers.add(update);
  update();
  return host;
}

/** A small SVG of a 2-2-1 network with its weights (for the hidden-layer demo). */
function miniNetDiagram(n) {
  const [L1, L2] = n.layers;
  const pos = { x1: [40, 40], x2: [40, 120], h1: [190, 40], h2: [190, 120], y: [330, 80] };
  const root = svg("svg", { viewBox: "0 0 380 160", class: "ml-net-svg", role: "img", "aria-label": L({ tr: "Ağın ağırlıkları", en: "The network's weights" }) });
  const edge = (a, b, w, labelAt = 0.5) => {
    const [x1, y1] = pos[a];
    const [x2, y2] = pos[b];
    root.append(svg("line", { x1, y1, x2, y2, class: `ml-edge ${w >= 0 ? "pos" : "neg"}`, style: `stroke-width:${(1 + Math.min(4, Math.abs(w))).toFixed(1)}px` }));
    const lx = x1 + (x2 - x1) * labelAt;
    const ly = y1 + (y2 - y1) * labelAt;
    root.append(svgText(lx, ly - 5, nf(w, 1), "ml-edge-label", "middle"));
  };
  edge("x1", "h1", L1.W[0][0], 0.3); edge("x2", "h1", L1.W[0][1], 0.3);
  edge("x1", "h2", L1.W[1][0], 0.7); edge("x2", "h2", L1.W[1][1], 0.7);
  edge("h1", "y", L2.W[0][0]); edge("h2", "y", L2.W[0][1]);
  const node = (k, label, bias) => {
    const [x, y] = pos[k];
    root.append(svg("circle", { cx: x, cy: y, r: 17, class: "ml-node" }), svgText(x, y + 5, label, "ml-node-label", "middle"));
    if (bias != null) root.append(svgText(x, y + 34, `b = ${nf(bias, 1)}`, "ml-edge-label", "middle"));
  };
  node("x1", "x₁"); node("x2", "x₂"); node("h1", "h₁", L1.b[0]); node("h2", "h₂", L1.b[1]); node("y", "ŷ", L2.b[0]);
  return root;
}

// =====================================================================
// Theory 3 · the loss as a function of one weight, and gradient descent on it
// =====================================================================
const LD = { x: [-2.0, -1.2, -0.4, 0.3, 0.8, 1.5, 2.2], y: [0, 0, 1, 0, 1, 1, 1] };
const LS = { w: -2.5, eta: 1, loss: "bce", trail: [], timer: 0 };
const LS_START = -2.5;
const lossAt = (w, kind = LS.loss) => LD.x.reduce((s, x, i) => s + NET.lossOne(sigma(w * x), LD.y[i], kind), 0) / LD.x.length;
const gradAt = (w, kind = LS.loss) => LD.x.reduce((s, x, i) => {
  const p = sigma(w * x);
  return s + (kind === "mse" ? (p - LD.y[i]) * p * (1 - p) * x : (p - LD.y[i]) * x);
}, 0) / LD.x.length;

function buildLossDemo() {
  const host = h("div", { class: "pc-demo" });
  const top = h("div", { class: "pc-stage" });
  const bottom = h("div", { class: "pc-stage" });
  const gLoss = createGraph(top, { x: [-3, 7], y: [0, 3], xTicks: [-2, 0, 2, 4, 6], yTicks: [0, 1, 2, 3], xName: "w", yName: "L(w)",
    label: L({ tr: "Kayıp, w'nin fonksiyonu olarak", en: "The loss as a function of w" }) });
  const gData = createGraph(bottom, { x: [-2.6, 2.6], y: [-0.15, 1.15], xTicks: [-2, -1, 0, 1, 2], yTicks: [0, 0.5, 1], xName: "x", yName: "ŷ", ratio: 2.4,
    label: L({ tr: "Veri ve modelin olasılığı", en: "The data and the model's probability" }) });
  const lossSeg = segmented([["bce", L({ tr: "Çapraz entropi", en: "Cross-entropy" })], ["mse", "MSE"]], LS.loss, (v) => { LS.loss = v; reset(); }, L({ tr: "Kayıp", en: "Loss" }));
  const eta = slider("η", { min: 0.1, max: 20, step: 0.1, value: LS.eta, onInput: (v) => { LS.eta = v; update(); } });
  const one = button(L({ tr: "Bir adım", en: "One step" }));
  const auto = button("", "btn primary");
  const again = button(L({ tr: "Baştan", en: "Restart" }));
  const out = h("div", { class: "pc-live" });
  const status = msg("");
  host.append(h("div", { class: "t-controls" }, lossSeg, eta), top,
    h("div", { class: "t-controls" }, auto, one, again), out, status,
    h("div", { class: "mini-title", text: L({ tr: "Veri: ŷ = σ(w·x)", en: "Data: ŷ = σ(w·x)" }) }), bottom);

  const step = () => {
    LS.trail.push(LS.w);
    LS.w = clamp(LS.w - LS.eta * gradAt(LS.w), -50, 50);
  };
  const done = () => Math.abs(gradAt(LS.w)) < 2e-4 || LS.trail.length >= 200;
  function reset() {
    LS.timer = stop(LS.timer);
    LS.w = LS_START;
    LS.trail = [];
    update();
  }
  one.addEventListener("click", () => { LS.timer = stop(LS.timer); step(); update(); });
  again.addEventListener("click", reset);
  auto.addEventListener("click", () => {
    if (LS.timer) { LS.timer = stop(LS.timer); update(); return; }
    if (done()) { LS.w = LS_START; LS.trail = []; }
    LS.timer = every(350, () => { step(); if (done()) LS.timer = stop(LS.timer); update(); });
    update();
  });

  function update() {
    const bce = LS.loss === "bce";
    gLoss.y = bce ? [0, 3] : [0, 0.4];
    gLoss.yTicks = bce ? [0, 1, 2, 3] : [0, 0.1, 0.2, 0.3, 0.4];
    const w = LS.w;
    // the axis grows when a large step lands outside it
    const lo = Math.min(-3, Math.floor(Math.min(w, ...LS.trail.slice(-25))) - 1);
    const hi = Math.max(7, Math.ceil(Math.max(w, ...LS.trail.slice(-25))) + 1);
    const tickStep = hi - lo > 14 ? 5 : 2;
    gLoss.x = [lo, hi];
    gLoss.xTicks = [];
    for (let t = Math.ceil(lo / tickStep) * tickStep; t <= hi; t += tickStep) gLoss.xTicks.push(t);
    const L0 = lossAt(w);
    const g = gradAt(w);
    const span = bce ? 0.9 : 1.6;
    gLoss.draw({
      curves: [{ f: (v) => lossAt(v), color: COLORS.lineB, width: 2.6 }],
      segments: [{ from: [w - span, L0 - span * g], to: [w + span, L0 + span * g], color: COLORS.line, width: 2 }],
      points: [...LS.trail.slice(-25).map((v) => ({ x: v, y: lossAt(v), r: 3, color: COLORS.muted })), { x: w, y: L0, color: COLORS.line, ring: true }],
    });
    gData.draw({
      curves: [{ f: (x) => sigma(w * x), color: COLORS.lineB, width: 2.4 }],
      segments: [{ from: [-2.6, 0.5], to: [2.6, 0.5], color: COLORS.muted, width: 1, dash: true }],
      points: LD.x.map((x, i) => ({ x, y: LD.y[i], color: LD.y[i] ? COLORS.one : COLORS.zero })),
    });
    const next = w - LS.eta * g;
    out.textContent = `\\(L'(w) = ${tx(g, 3)}\\qquad w \\leftarrow w - \\eta\\,L'(w) = ${tx(w)} - ${tx(LS.eta, 1)} \\cdot ${txp(g, 3)} = ${tx(next)}\\)`;
    auto.textContent = LS.timer ? L({ tr: "❚❚ Durdur", en: "❚❚ Pause" }) : L({ tr: "▶ Otomatik", en: "▶ Run" });
    const steps = LS.trail.length;
    const lastJump = steps ? Math.abs(LS.trail.at(-1) - w) : 0;
    let text = L({ tr: `Adım ${steps} · w = ${nf(w)} · L = ${nf(L0, 3)}`, en: `Step ${steps} · w = ${nf(w)} · L = ${nf(L0, 3)}` });
    let kind = "";
    if (done() && steps) { text += L({ tr: " · minimumda ✓ (eğim ≈ 0)", en: " · at the minimum ✓ (slope ≈ 0)" }); kind = "good"; }
    else if (steps && Math.sign(gradAt(LS.trail.at(-1))) !== Math.sign(g) && lastJump > 0.5) {
      text += L({ tr: " · η büyük: adım minimumun üstünden atladı (eğimin işareti değişti)", en: " · η is large: the step jumped over the minimum (the slope changed sign)" }); kind = "bad";
    } else if (!bce && w < -1 && steps > 5) {
      text += L({ tr: " · MSE burada çok düz: adımlar küçük", en: " · MSE is very flat here: tiny steps" }); kind = "bad";
    }
    setMsg(status, text, kind);
  }
  LS.update = update;
  redrawers.add(update);
  update();
  return host;
}

// =====================================================================
// Theory 4 · backpropagation, step by step, on a 2-2-1 network
// =====================================================================
const BP_START = { W: [[0.6, -0.4], [-0.3, 0.8]], b: [0.1, -0.2], v: [0.9, -0.7], c: 0.2 };
const BP = { k: 0, ex: 2, eta: 0.5, p: null };
const resetBp = () => { BP.p = JSON.parse(JSON.stringify(BP_START)); BP.k = 0; };
resetBp();

/** Every number of one forward and backward pass. */
function bpCompute(p, x, y) {
  const z = p.W.map((row, j) => row[0] * x[0] + row[1] * x[1] + p.b[j]);
  const hh = z.map(sigma);
  const zo = p.v[0] * hh[0] + p.v[1] * hh[1] + p.c;
  const yh = sigma(zo);
  const loss = NET.lossOne(yh, y);
  const dzo = yh - y;
  const gv = hh.map((hj) => dzo * hj);
  const gc = dzo;
  const dh = p.v.map((vj) => dzo * vj);
  const dz = dh.map((d, j) => d * hh[j] * (1 - hh[j]));
  const gW = dz.map((d) => [d * x[0], d * x[1]]);
  const gb = [...dz];
  return { x, y, z, h: hh, zo, yh, loss, dzo, gv, gc, dh, dz, gW, gb };
}
function bpUpdated(p, c, eta) {
  return {
    W: p.W.map((row, j) => row.map((w, i) => w - eta * c.gW[j][i])),
    b: p.b.map((b, j) => b - eta * c.gb[j]),
    v: p.v.map((v, j) => v - eta * c.gv[j]),
    c: p.c - eta * c.gc,
  };
}

/** The steps: [first step at which a quantity shows] and the text of each step. */
const BP_SHOW = { z: 1, h: 2, zo: 3, yh: 4, loss: 5, dzo: 6, gv: 7, dh: 8, dz: 9, gW: 10, upd: 11 };
function bpSteps(c, p, eta) {
  const n2 = bpUpdated(p, c, eta);
  const c2 = bpCompute(n2, c.x, c.y);
  const j2 = (f) => `${f(0)},\\qquad ${f(1)}`;
  return [
    { title: { tr: "Başlangıç: ağırlıklar ve örnek", en: "Start: the weights and the example" },
      tex: R`\mathbf x = (${c.x[0]},\ ${c.x[1]}),\quad y = ${c.y}`,
      why: { tr: "Bir eğitim örneği ve ağın şu anki ağırlıkları. İleri'ye basarak hesabı adım adım ilerletin.", en: "One training example and the network's current weights. Press Next to go through the computation step by step." } },
    { title: { tr: "İleri 1: gizli nöronların toplamı", en: "Forward 1: the hidden neurons' sums" },
      tex: j2((j) => R`z_${j + 1} = w_{${j + 1}1}x_1 + w_{${j + 1}2}x_2 + b_${j + 1} = ${tx(c.z[j], 3)}`),
      why: { tr: "Her gizli nöron, perceptron gibi, girdilerin ağırlıklı toplamını alır.", en: "Each hidden neuron takes the weighted sum of the inputs, like a perceptron." } },
    { title: { tr: "İleri 2: gizli nöronların çıktısı", en: "Forward 2: the hidden neurons' outputs" },
      tex: j2((j) => R`h_${j + 1} = \sigma(z_${j + 1}) = ${tx(c.h[j], 3)}`),
      why: { tr: "Sigmoid toplamı 0 ile 1 arasına sıkıştırır.", en: "The sigmoid squeezes the sum between 0 and 1." } },
    { title: { tr: "İleri 3: çıktı nöronunun toplamı", en: "Forward 3: the output neuron's sum" },
      tex: R`z_o = v_1 h_1 + v_2 h_2 + c = ${tx(p.v[0])}\cdot ${tx(c.h[0], 3)} + ${txp(p.v[1])}\cdot ${tx(c.h[1], 3)} + ${tx(p.c)} = ${tx(c.zo, 3)}`,
      why: { tr: "Çıktı nöronunun girdileri gizli nöronların çıktılarıdır.", en: "The output neuron's inputs are the hidden neurons' outputs." } },
    { title: { tr: "İleri 4: tahmin", en: "Forward 4: the prediction" },
      tex: R`\hat y = \sigma(z_o) = ${tx(c.yh, 3)}`,
      why: { tr: "Ağın cevabı: sınıf 1 olma olasılığı.", en: "The network's answer: the probability of class 1." } },
    { title: { tr: "Kayıp", en: "The loss" },
      tex: R`L = -\big[y\ln\hat y + (1-y)\ln(1-\hat y)\big] = -\ln ${tx(c.y ? c.yh : 1 - c.yh, 3)} = ${tx(c.loss, 3)}`,
      why: { tr: "Doğru cevaba verilen olasılık ne kadar küçükse kayıp o kadar büyük. Şimdi geriye dönüyoruz.", en: "The smaller the probability given to the right answer, the larger the loss. Now we go back." } },
    { title: { tr: "Geri 1: çıktı nöronunun hatası", en: "Backward 1: the output neuron's error" },
      tex: R`\delta_o = \frac{\partial L}{\partial z_o} = \hat y - y = ${tx(c.yh, 3)} - ${c.y} = ${tx(c.dzo, 3)}`,
      why: c.dzo < 0
        ? { tr: "Çapraz entropi ile sigmoid birlikte bu kadar basit bir türev verir (Matematik, 3. kısım). Negatif: z_o büyürse kayıp azalır.", en: "Cross-entropy and the sigmoid together give this simple derivative (Math, section 3). Negative: if z_o grows, the loss falls." }
        : { tr: "Çapraz entropi ile sigmoid birlikte bu kadar basit bir türev verir (Matematik, 3. kısım). Pozitif: z_o küçülürse kayıp azalır.", en: "Cross-entropy and the sigmoid together give this simple derivative (Math, section 3). Positive: if z_o shrinks, the loss falls." } },
    { title: { tr: "Geri 2: çıktı ağırlıklarının gradyanı", en: "Backward 2: the gradients of the output weights" },
      tex: R`\frac{\partial L}{\partial v_j} = \delta_o\, h_j:\quad ${tx(c.gv[0], 3)},\ \ ${tx(c.gv[1], 3)}\qquad \frac{\partial L}{\partial c} = \delta_o = ${tx(c.gc, 3)}`,
      why: { tr: "Bir ağırlığın gradyanı = girdisi × hedef nöronun δ'sı. Büyük bir girdiden gelen ağırlık kaybı daha çok etkiler.", en: "A weight's gradient = its input × the δ of the neuron it feeds. A weight with a large input affects the loss more." } },
    { title: { tr: "Geri 3: hata gizli nöronlara geçer", en: "Backward 3: the error reaches the hidden neurons" },
      tex: j2((j) => R`\frac{\partial L}{\partial h_${j + 1}} = \delta_o\, v_${j + 1} = ${tx(c.dh[j], 3)}`),
      why: { tr: "Bu, geri yayılımın kalbi: gizli nöronun 'doğru cevabı' yok, ama kayba katkısı çıktı ağırlığı üzerinden ölçülebilir.", en: "This is the heart of backpropagation: a hidden neuron has no 'right answer', but its effect on the loss can be measured through the output weight." } },
    { title: { tr: "Geri 4: gizli nöronların hatası", en: "Backward 4: the hidden neurons' errors" },
      tex: j2((j) => R`\delta_${j + 1} = \frac{\partial L}{\partial h_${j + 1}}\, h_${j + 1}(1-h_${j + 1}) = ${tx(c.dz[j], 4)}`),
      why: { tr: "Zincir kuralı: sigmoid'in eğimiyle çarp. σ'(z) = σ(z)(1 − σ(z)) en fazla 0,25: hata geriye giderken küçülür.", en: "The chain rule: multiply by the sigmoid's slope. σ'(z) = σ(z)(1 − σ(z)) is at most 0.25: the error shrinks as it goes back." } },
    { title: { tr: "Geri 5: gizli ağırlıkların gradyanı", en: "Backward 5: the gradients of the hidden weights" },
      tex: R`\frac{\partial L}{\partial w_{ji}} = \delta_j\, x_i:\quad ${tx(c.gW[0][0], 4)},\ ${tx(c.gW[0][1], 4)},\ ${tx(c.gW[1][0], 4)},\ ${tx(c.gW[1][1], 4)}`,
      why: { tr: "Aynı kural: girdi × δ. Girdisi 0 olan ağırlıkların gradyanı 0: bu örnek onları değiştirmez.", en: "The same rule: input × δ. Weights whose input is 0 get a zero gradient: this example does not change them." } },
    { title: { tr: "Güncelleme", en: "The update" },
      tex: R`\theta \leftarrow \theta - \eta\,\frac{\partial L}{\partial \theta}\ \ (\eta = ${tx(eta, 1)}):\qquad \hat y = ${tx(c.yh, 3)} \to ${tx(c2.yh, 3)},\quad L = ${tx(c.loss, 3)} \to ${tx(c2.loss, 3)}`,
      why: { tr: "Bütün ağırlıklar aynı anda, gradyanlarının tersine biraz kaydırılır. Aynı örnekle yeniden hesaplayınca kayıp azaldı. Uygula'ya basarak yeni ağırlıklarla devam edebilirsiniz.", en: "All the weights move at once, a little against their gradients. Computing the same example again, the loss is lower. Press Apply to continue with the new weights." } },
  ];
}

function buildBackpropDemo() {
  const host = h("div", { class: "pc-demo" });
  const exSeg = segmented(XOR.map((d, i) => [String(i), `x = (${d.x[0]}, ${d.x[1]}), y = ${d.y}`]), String(BP.ex),
    (v) => { BP.ex = Number(v); BP.k = 0; update(); }, L({ tr: "Örnek", en: "Example" }));
  exSeg.classList.add("seg-wrap");
  const diagram = h("div", { class: "ml-bp-diagram" });
  const stepTitle = h("div", { class: "ml-bp-title" });
  const counter = h("span", { class: "m-derive-count" });
  const texBox = h("div", { class: "pc-live ml-bp-tex" });
  const why = h("p", { class: "m-why" });
  const prev = button(L({ tr: "← Geri", en: "← Back" }));
  const next = button(L({ tr: "İleri →", en: "Next →" }), "btn primary");
  const apply = button(L({ tr: "Uygula ve yeni tur", en: "Apply and go again" }));
  const reset = button(L({ tr: "Ağırlıkları sıfırla", en: "Reset the weights" }));
  const table = h("div", { class: "ml-bp-table" });
  host.append(h("div", { class: "t-controls" }, exSeg), diagram,
    h("div", { class: "ml-bp-step" }, h("div", { class: "ml-bp-head" }, stepTitle, counter), texBox, why),
    h("div", { class: "t-controls" }, prev, next, apply, reset), table);
  prev.addEventListener("click", () => { BP.k = Math.max(0, BP.k - 1); update(); });
  next.addEventListener("click", () => { BP.k = Math.min(BP_SHOW.upd, BP.k + 1); update(); });
  apply.addEventListener("click", () => {
    const d = XOR[BP.ex];
    BP.p = bpUpdated(BP.p, bpCompute(BP.p, d.x, d.y), BP.eta);
    BP.k = 0;
    update();
  });
  reset.addEventListener("click", () => { resetBp(); update(); });

  function update() {
    const d = XOR[BP.ex];
    const c = bpCompute(BP.p, d.x, d.y);
    const steps = bpSteps(c, BP.p, BP.eta);
    const s = steps[BP.k];
    stepTitle.textContent = L(s.title);
    counter.textContent = L({ tr: `${BP.k} / ${steps.length - 1}`, en: `${BP.k} of ${steps.length - 1}` });
    texBox.textContent = `\\(\\displaystyle ${s.tex}\\)`;
    why.innerHTML = termify(L(s.why));
    prev.disabled = BP.k === 0;
    next.disabled = BP.k === BP_SHOW.upd;
    apply.hidden = BP.k !== BP_SHOW.upd;
    diagram.replaceChildren(bpDiagram(c, BP.p, BP.k));
    table.replaceChildren(bpTable(c, BP.p, BP.k));
  }
  update();
  return host;
}

function bpDiagram(c, p, k) {
  const show = (key) => k >= BP_SHOW[key];
  const active = {
    1: ["h1", "h2"], 2: ["h1", "h2"], 3: ["o"], 4: ["o"], 5: ["L"], 6: ["o"], 7: ["o"], 8: ["h1", "h2"], 9: ["h1", "h2"], 10: ["h1", "h2"], 11: [],
  }[k] || [];
  const pos = { x1: [52, 70], x2: [52, 200], h1: [222, 70], h2: [222, 200], o: [392, 135], L: [496, 135] };
  const root = svg("svg", { viewBox: "0 0 540 270", class: "ml-net-svg ml-bp-svg", role: "img", "aria-label": L({ tr: "Geri yayılım şeması", en: "Backpropagation diagram" }) });
  const backEdges = k >= BP_SHOW.dh;
  const edge = (a, b, w, gradient, at) => {
    const [x1, y1] = pos[a];
    const [x2, y2] = pos[b];
    root.append(svg("line", { x1, y1, x2, y2, class: `ml-edge ${w >= 0 ? "pos" : "neg"}`, style: `stroke-width:${(1.2 + Math.min(3, Math.abs(w) * 2)).toFixed(1)}px` }));
    const lx = x1 + (x2 - x1) * at;
    const ly = y1 + (y2 - y1) * at;
    root.append(svgText(lx, ly - 6, nf(w), "ml-edge-label", "middle"));
    if (gradient != null) root.append(svgText(lx, ly + 13, nf(gradient, 3), "ml-grad", "middle"));
  };
  const gW = show("gW") ? c.gW : null;
  const gv = show("gv") ? c.gv : null;
  edge("x1", "h1", p.W[0][0], gW?.[0][0], 0.6); edge("x2", "h1", p.W[0][1], gW?.[0][1], 0.27);
  edge("x1", "h2", p.W[1][0], gW?.[1][0], 0.27); edge("x2", "h2", p.W[1][1], gW?.[1][1], 0.6);
  edge("h1", "o", p.v[0], gv?.[0], 0.5); edge("h2", "o", p.v[1], gv?.[1], 0.5);
  root.append(svg("line", { x1: pos.o[0] + 26, y1: 135, x2: pos.L[0] - 22, y2: 135, class: "ml-edge" }));
  if (backEdges) root.append(svg("path", { d: "M 470 160 C 430 240, 300 250, 245 222", class: "ml-back-arrow" }));
  const node = (key, label, above, below, bias) => {
    const [x, y] = pos[key];
    const r = key === "L" ? 20 : 26;
    root.append(svg("circle", { cx: x, cy: y, r, class: `ml-node${active.includes(key) ? " active" : ""}` }),
      svgText(x, bias == null ? y + 5 : y + 1, label, "ml-node-label", "middle"));
    if (bias != null) root.append(svgText(x, y + 16, bias, "ml-bias", "middle"));
    above.forEach((t, i) => root.append(svgText(x, y - r - 8 - (above.length - 1 - i) * 15, t, "ml-val", "middle")));
    below.forEach((t, i) => root.append(svgText(x, y + r + 16 + i * 15, t, "ml-grad", "middle")));
  };
  node("x1", "x₁", [`${c.x[0]}`], []);
  node("x2", "x₂", [`${c.x[1]}`], []);
  for (const j of [0, 1]) {
    const above = [];
    if (show("z")) above.push(`z = ${nf(c.z[j], 2)}`);
    if (show("h")) above.push(`h = ${nf(c.h[j], 2)}`);
    const below = [];
    if (show("dh")) below.push(`∂L/∂h = ${nf(c.dh[j], 3)}`);
    if (show("dz")) below.push(`δ = ${nf(c.dz[j], 4)}`);
    node(`h${j + 1}`, `h${j ? "₂" : "₁"}`, above, below, `b = ${nf(p.b[j])}`);
  }
  const oAbove = [];
  if (show("zo")) oAbove.push(`z = ${nf(c.zo, 2)}`);
  if (show("yh")) oAbove.push(`ŷ = ${nf(c.yh, 3)}`);
  node("o", "ŷ", oAbove, show("dzo") ? [`δ = ${nf(c.dzo, 3)}`] : [], `c = ${nf(p.c)}`);
  node("L", "L", show("loss") ? [nf(c.loss, 3)] : [], []);
  return root;
}

function bpTable(c, p, k) {
  const upd = k >= BP_SHOW.upd ? bpUpdated(p, c, BP.eta) : null;
  const rows = [
    ["w₁₁", p.W[0][0], k >= BP_SHOW.gW ? c.gW[0][0] : null, upd?.W[0][0]],
    ["w₁₂", p.W[0][1], k >= BP_SHOW.gW ? c.gW[0][1] : null, upd?.W[0][1]],
    ["b₁", p.b[0], k >= BP_SHOW.gW ? c.gb[0] : null, upd?.b[0]],
    ["w₂₁", p.W[1][0], k >= BP_SHOW.gW ? c.gW[1][0] : null, upd?.W[1][0]],
    ["w₂₂", p.W[1][1], k >= BP_SHOW.gW ? c.gW[1][1] : null, upd?.W[1][1]],
    ["b₂", p.b[1], k >= BP_SHOW.gW ? c.gb[1] : null, upd?.b[1]],
    ["v₁", p.v[0], k >= BP_SHOW.gv ? c.gv[0] : null, upd?.v[0]],
    ["v₂", p.v[1], k >= BP_SHOW.gv ? c.gv[1] : null, upd?.v[1]],
    ["c", p.c, k >= BP_SHOW.gv ? c.gc : null, upd?.c],
  ];
  const head = h("tr", {}, ...[L({ tr: "Parametre", en: "Parameter" }), L({ tr: "Değer", en: "Value" }), "∂L/∂θ", L({ tr: "Yeni değer", en: "New value" })].map((t) => h("th", { text: t })));
  const body = rows.map(([name, v, g, n]) => h("tr", {}, h("td", { text: name }), h("td", { class: "num", text: nf(v, 3) }),
    h("td", { class: "num ml-grad-cell", text: g == null ? "" : nf(g, 4) }), h("td", { class: "num", text: n == null ? "" : nf(n, 3) })));
  return h("table", { class: "pc-table" }, h("thead", {}, head), h("tbody", {}, ...body));
}

// =====================================================================
// Theory 5 · how many neurons: fitting a curve with H tanh neurons
// =====================================================================
const TARGETS = {
  wave: { name: { tr: "Dalga", en: "Wave" }, f: (x) => Math.sin(3 * x) * 0.9 + 0.2 * x },
  step: { name: { tr: "Basamak", en: "Step" }, f: (x) => (x > 0.1 ? 0.8 : -0.8) },
  bump: { name: { tr: "Tepe", en: "Bump" }, f: (x) => 1.6 * Math.exp(-(x * x) / 0.06) - 0.6 },
  abs: { name: { tr: "Mutlak değer", en: "Absolute value" }, f: (x) => 2 * Math.abs(x) - 1 },
};
const CP = { target: "wave", hidden: 1, net: null, state: {}, timer: 0, parts: true };
const CP_X = Array.from({ length: 41 }, (_, i) => -1 + i * 0.05);
const CP_MAX = 3000;
const cpData = () => CP_X.map((x) => ({ x: [x], y: TARGETS[CP.target].f(x) }));
function resetCapacity() {
  CP.net = NET.create([1, CP.hidden, 1], { act: "tanh", out: "linear", rand: seeded(11 + CP.hidden) });
  // spread the first guesses over the interval: steeper slopes, centres between −1 and 1
  CP.net.layers[0].W.forEach((row, j) => { row[0] *= 3; CP.net.layers[0].b[j] = (j / Math.max(1, CP.hidden - 1) * 2 - 1) * Math.abs(row[0]) * 0.8; });
  CP.state = {};
}
resetCapacity();

function buildCapacityDemo() {
  const host = h("div", { class: "pc-demo" });
  const stage = h("div", { class: "pc-stage" });
  const graph = createGraph(stage, { x: [-1.05, 1.05], y: [-1.6, 1.6], xTicks: [-1, -0.5, 0, 0.5, 1], yTicks: [-1, 0, 1], xName: "x", yName: "y",
    label: L({ tr: "Ağın çıktısı ve hedef eğri", en: "The network's output and the target curve" }) });
  const targetSeg = segmented(Object.entries(TARGETS).map(([k, t]) => [k, L(t.name)]), CP.target, (v) => { CP.target = v; restart(); }, L({ tr: "Hedef", en: "Target" }));
  const hidden = slider(L({ tr: "Gizli nöron H", en: "Hidden neurons H" }), { min: 1, max: 12, step: 1, value: CP.hidden, fmt: (v) => fmtNum(v, 0), onInput: (v) => { CP.hidden = v; restart(); } });
  const parts = h("input", { type: "checkbox" });
  parts.checked = CP.parts;
  parts.addEventListener("change", () => { CP.parts = parts.checked; update(); });
  const play = button("", "btn primary");
  const status = msg("");
  const legend = h("div", { class: "legend" },
    h("span", { class: "legend-item" }, h("span", { class: "key-line muted dashed" }), L({ tr: "hedef", en: "target" })),
    h("span", { class: "legend-item" }, h("span", { class: "key-line s1" }), L({ tr: "ağın çıktısı", en: "network output" })),
    h("span", { class: "legend-item" }, h("span", { class: "key-line thin" }), L({ tr: "nöronların katkıları", en: "neurons' contributions" })));
  host.append(h("div", { class: "t-controls" }, targetSeg), h("div", { class: "t-controls" }, hidden),
    stage, legend, h("div", { class: "t-controls" }, play, h("label", { class: "pg-check" }, parts, L({ tr: "Nöronların katkılarını göster", en: "Show the neurons' contributions" }))), status);

  function restart() {
    const was = Boolean(CP.timer);
    CP.timer = stop(CP.timer);
    resetCapacity();
    if (was) start();
    update();
  }
  function start() {
    if (CP.net.epoch >= CP_MAX) resetCapacity();
    const data = cpData();
    CP.timer = every(30, () => {
      for (let k = 0; k < 25; k++) NET.trainEpoch(CP.net, data, { lr: 0.03, loss: "mse", optimizer: "adam", state: CP.state });
      if (CP.net.epoch >= CP_MAX) CP.timer = stop(CP.timer);
      update();
    });
  }
  play.addEventListener("click", () => {
    if (CP.timer) CP.timer = stop(CP.timer); else start();
    update();
  });

  function update() {
    const n = CP.net;
    const [L1, L2] = n.layers;
    const tgt = TARGETS[CP.target].f;
    const palette = ["--series-1", "--series-2", "--series-3", "--critical", "--accent-strong", "--muted"].map(cssVar);
    const curves = [];
    if (CP.parts) {
      L1.W.forEach((row, j) => curves.push({ f: (x) => L2.W[0][j] * Math.tanh(row[0] * x + L1.b[j]), color: palette[j % palette.length], width: 1.2 }));
    }
    curves.push({ f: tgt, color: COLORS.muted, width: 2, dash: true });
    curves.push({ f: (x) => NET.predict(n, [x]), color: COLORS.lineB, width: 3 });
    graph.draw({ curves, points: CP_X.filter((_, i) => i % 2 === 0).map((x) => ({ x, y: tgt(x), r: 2.5, color: COLORS.text2 })) });
    play.textContent = CP.timer ? L({ tr: "❚❚ Durdur", en: "❚❚ Pause" }) : L({ tr: "▶ Eğit", en: "▶ Train" });
    const mse = NET.lossOf(n, cpData(), "mse") * 2;
    const good = mse < 0.005;
    setMsg(status, L({ tr: `H = ${CP.hidden} nöron (${NET.paramCount([1, CP.hidden, 1])} parametre) · epoch ${n.epoch} · ortalama kare hata ${nf(mse, 4)}${n.epoch >= CP_MAX ? (good ? " · iyi uyuyor ✓" : " · bu kadar nöronla daha iyisi zor") : ""}`,
      en: `H = ${CP.hidden} neurons (${NET.paramCount([1, CP.hidden, 1])} parameters) · epoch ${n.epoch} · mean squared error ${nf(mse, 4)}${n.epoch >= CP_MAX ? (good ? " · a good fit ✓" : " · hard to do better with this many neurons") : ""}` }),
      n.epoch >= CP_MAX ? (good ? "good" : "bad") : "");
  }
  CP.update = update;
  redrawers.add(update);
  update();
  return host;
}

// =====================================================================
// Theory 6 · an MLP on images: shift a digit by one pixel
// =====================================================================
const DIGIT = [
  "..............",
  "..............",
  "...########...",
  "...########...",
  ".........##...",
  "........##....",
  ".......##.....",
  "......##......",
  "......##......",
  ".....##.......",
  ".....##.......",
  "....##........",
  "..............",
  "..............",
];
const GN = DIGIT.length;
const IM = { dx: 0, dy: 0 };
const pixel = (r, c, dx = 0, dy = 0) => {
  const rr = r - dy;
  const cc = c - dx;
  return rr >= 0 && rr < GN && cc >= 0 && cc < GN && DIGIT[rr][cc] === "#" ? 1 : 0;
};

function buildImageDemo() {
  const host = h("div", { class: "pc-demo" });
  const grid = h("div", { class: "ml-digit", style: `--n:${GN}`, role: "img", "aria-label": L({ tr: "Kaydırılan rakam", en: "The shifted digit" }) });
  const strips = h("div", { class: "ml-strips" });
  const status = msg("");
  const move = (dx, dy, label) => {
    const b = button(label);
    b.addEventListener("click", () => { IM.dx = clamp(IM.dx + dx, -3, 3); IM.dy = clamp(IM.dy + dy, -2, 2); update(); });
    return b;
  };
  const reset = button(L({ tr: "Yerine koy", en: "Put back" }));
  reset.addEventListener("click", () => { IM.dx = 0; IM.dy = 0; update(); });
  host.append(h("div", { class: "ml-image-row" }, grid,
    h("div", { class: "ml-arrows" }, move(0, -1, "↑"), h("div", {}, move(-1, 0, "←"), move(1, 0, "→")), move(0, 1, "↓"), reset)),
  h("div", { class: "mini-title", text: L({ tr: `MLP'nin gördüğü: ${GN * GN} sayılık vektör (satır satır)`, en: `What the MLP sees: a vector of ${GN * GN} numbers (row by row)` }) }),
  strips, status);

  function strip(values, marks) {
    const row = h("div", { class: "ml-strip" });
    values.forEach((v, i) => row.append(h("span", { class: `${v ? "on" : ""}${marks?.[i] ? " changed" : ""}` })));
    return row;
  }
  function update() {
    const cells = [];
    const now = [];
    const orig = [];
    for (let r = 0; r < GN; r++) {
      for (let c = 0; c < GN; c++) {
        const v = pixel(r, c, IM.dx, IM.dy);
        const o = pixel(r, c);
        now.push(v);
        orig.push(o);
        cells.push(h("span", { class: `${v ? "on" : ""}${o && !v ? " ghost" : ""}` }));
      }
    }
    grid.replaceChildren(...cells);
    const changed = now.map((v, i) => v !== orig[i]);
    strips.replaceChildren(
      h("div", { class: "ml-strip-label", text: L({ tr: "Yerinde", en: "In place" }) }), strip(orig),
      h("div", { class: "ml-strip-label", text: L({ tr: `Kaydırılmış (${IM.dx >= 0 ? "+" : ""}${IM.dx}, ${IM.dy >= 0 ? "+" : ""}${IM.dy})`, en: `Shifted (${IM.dx >= 0 ? "+" : ""}${IM.dx}, ${IM.dy >= 0 ? "+" : ""}${IM.dy})` }) }), strip(now, changed));
    const ink = orig.filter(Boolean).length;
    const kept = orig.filter((v, i) => v && now[i]).length;
    const diff = changed.filter(Boolean).length;
    setMsg(status, IM.dx || IM.dy
      ? L({ tr: `${diff} girdi değişti. Mürekkepli ${ink} pikselden yalnızca ${kept} tanesi aynı yerde kaldı: MLP için bu, büyük ölçüde başka bir girdi.`, en: `${diff} inputs changed. Of the ${ink} inked pixels only ${kept} stayed in place: to an MLP this is largely a different input.` })
      : L({ tr: "Okları kullanarak rakamı kaydırın.", en: "Use the arrows to move the digit." }),
    IM.dx || IM.dy ? (kept / ink < 0.5 ? "bad" : "") : "");
  }
  update();
  return host;
}

// =====================================================================
// Theory tab: assembling the sections
// =====================================================================
function buildTheory() {
  const panel = $("#panel-theory");
  const I = TEXT.intro;
  const compare = h("div", { class: "pc-bio" }, h("div", { class: "mini-title", text: L(I.compare.title) }),
    h("dl", {}, ...I.compare.rows.flatMap(([a, b]) => [html("dt", {}, a), html("dd", {}, b)])));
  const fact = h("aside", { class: "tut-callout info" }, h("span", { class: "tut-callout-icon", text: "ℹ️", "aria-hidden": "true" }),
    h("div", {}, h("strong", { text: L(I.fact.title) }), html("p", {}, I.fact.text)));
  const introCard = h("div", { class: "card t-intro pc-intro", id: "ml-intro" },
    h("div", { class: "t-intro-text" }, h("h2", { text: L(I.h) }), ...I.p.map((p) => html("p", {}, p))),
    h("div", { class: "pc-intro-side" }, compare, fact));
  const A = TEXT.act;
  const Hd = TEXT.hidden;
  const Ls = TEXT.loss;
  const B = TEXT.backprop;
  const C = TEXT.capacity;
  const Im = TEXT.images;
  const inModel = (t) => html("div", { class: "in-model" }, t);
  panel.replaceChildren(
    introCard,
    card("ml-act", 1, A, [html("p", {}, A.p1), formula(R`\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big)`), html("p", {}, A.p2), html("p", {}, A.p3), inModel(A.model)], buildActDemo()),
    card("ml-hidden", 2, Hd, [html("p", {}, Hd.p1), formula(R`h_j = \sigma(w_{j1}x_1 + w_{j2}x_2 + b_j) \qquad \hat y = \sigma(v_1 h_1 + v_2 h_2 + c)`), html("p", {}, Hd.p2), html("p", {}, Hd.p3), inModel(Hd.model)], buildHiddenDemo()),
    card("ml-loss", 3, Ls, [html("p", {}, Ls.p1), formula(R`L = -\big[\,y \ln \hat y + (1 - y)\ln(1 - \hat y)\,\big]`), html("p", {}, Ls.p2), html("p", {}, Ls.p3), inModel(Ls.model)], buildLossDemo()),
    card("ml-backprop", 4, B, [html("p", {}, B.p1), formula(R`\frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial z_o}\cdot\frac{\partial z_o}{\partial h_1}\cdot\frac{\partial h_1}{\partial z_1}\cdot\frac{\partial z_1}{\partial w_{11}}`), html("p", {}, B.p2), html("p", {}, B.p3), inModel(B.model)], buildBackpropDemo()),
    card("ml-capacity", 5, C, [html("p", {}, C.p1), formula(R`\hat y(x) = c + \sum_{j=1}^{H} v_j \tanh(w_j x + b_j)`), html("p", {}, C.p2), html("p", {}, C.p3), inModel(C.model)], buildCapacityDemo()),
    card("ml-images", 6, Im, [html("p", {}, Im.p1), html("p", {}, Im.p2), html("p", {}, Im.p3), inModel(Im.model), html("p", {}, Im.recap), nextChapterCard()], buildImageDemo()));
}

// =====================================================================
// Playground: train a network on 2-D data and look inside it
// =====================================================================
const PG = {
  preset: "circle", seed: 1, data: [], cls: 1, layers: 1, width: 4, act: "tanh", lrIndex: 4, speedIndex: 2,
  net: null, history: [], running: false, raf: 0, focus: null,
};
const LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3];
const SPEEDS = [1, 5, 20, 60, 200];
const PRESETS = ["xor", "circle", "moons", "spiral", "blobs", "empty"];
const presetName = (k) => L({
  xor: { tr: "XOR", en: "XOR" }, circle: { tr: "Daire", en: "Circle" }, moons: { tr: "İki hilal", en: "Two moons" },
  spiral: { tr: "Spiral", en: "Spiral" }, blobs: { tr: "İki küme", en: "Two blobs" }, empty: { tr: "Boş", en: "Empty" }, custom: { tr: "Özel", en: "Custom" },
}[k]);

function makePreset(kind, seed, n = 160) {
  const r = seeded(seed * 7919 + kind.length);
  const pts = [];
  if (kind === "xor") {
    while (pts.length < n) {
      const a = r() * 2.4 - 1.2;
      const b = r() * 2.4 - 1.2;
      if (Math.abs(a) > 0.1 && Math.abs(b) > 0.1) pts.push({ x: [a, b], y: a * b > 0 ? 0 : 1 });
    }
  } else if (kind === "circle") {
    for (let i = 0; i < n; i++) {
      const inner = i % 2 === 0;
      const rad = inner ? Math.sqrt(r()) * 0.55 : 0.85 + r() * 0.4;
      const t = r() * 2 * Math.PI;
      pts.push({ x: [rad * Math.cos(t), rad * Math.sin(t)], y: inner ? 1 : 0 });
    }
  } else if (kind === "moons") {
    for (let i = 0; i < n; i++) {
      const k = i % 2;
      const t = r() * Math.PI;
      const x = k ? 1 - Math.cos(t) : Math.cos(t);
      const y = k ? 0.5 - Math.sin(t) : Math.sin(t);
      pts.push({ x: [(x - 0.5) * 1.15 + 0.1 * gauss(r), (y - 0.25) * 1.15 + 0.1 * gauss(r)], y: k });
    }
  } else if (kind === "spiral") {
    for (let i = 0; i < n; i++) {
      const k = i % 2;
      const t = 0.25 + (Math.floor(i / 2) / (n / 2)) * 3 * Math.PI;
      const rad = (t / (3 * Math.PI + 0.25)) * 1.35;
      pts.push({ x: [rad * Math.cos(t + k * Math.PI) + 0.04 * gauss(r), rad * Math.sin(t + k * Math.PI) + 0.04 * gauss(r)], y: k });
    }
  } else if (kind === "blobs") {
    for (let i = 0; i < n; i++) {
      const k = i % 2;
      pts.push({ x: [(k ? 0.55 : -0.55) + 0.32 * gauss(r), (k ? 0.45 : -0.45) + 0.32 * gauss(r)], y: k });
    }
  }
  return pts;
}

const pgSizes = () => [2, ...Array(PG.layers).fill(PG.width), 1];
function pgReset() {
  PG.net = NET.create(pgSizes(), { act: PG.act, out: "sigmoid", rand: seeded(PG.seed * 31 + PG.layers * 7 + PG.width) });
  PG.history = [];
  PG.focus = null;
}
PG.data = makePreset(PG.preset, PG.seed);
pgReset();

function pgEpoch() {
  if (!PG.data.length) return;
  NET.trainEpoch(PG.net, PG.data, { lr: LRS[PG.lrIndex], batch: 10, rand: seeded(PG.net.epoch + 1) });
  const loss = NET.lossOf(PG.net, PG.data);
  PG.history.push(Number.isFinite(loss) ? loss : 10);
}

function buildPlay() {
  const panel = $("#panel-play");
  pgStop();
  const intro = h("div", { class: "card pg-intro" },
    h("h2", { text: L({ tr: "Deneme alanı: bir sinir ağını kendiniz eğitin", en: "Playground: train a neural network yourself" }) }),
    html("p", { class: "sub" }, {
      tr: R`Bir veri seçin, ağın katman ve nöron sayısını ayarlayın ve <b>Eğit</b>'e basın. Arka plan ağın her noktaya verdiği olasılıktır, kalın çizgi karar sınırı (\(\hat y = 0{,}5\)). Sağdaki şemada her kare bir nöronun düzlemin her yerindeki çıktısını gösterir; bir nörona tıklayınca büyük düzlemde onu görürsünüz. 0 gizli katman bir perceptrondur (sigmoid'li): yalnızca doğru çizebilir. <b>Deneyin:</b>`,
      en: R`Pick a dataset, set the number of layers and neurons and press <b>Train</b>. The background is the probability the network gives every point, the thick line is the decision boundary (\(\hat y = 0.5\)). In the diagram on the right each square shows a neuron's output over the whole plane; click a neuron to see it on the big plane. 0 hidden layers is a perceptron (with a sigmoid): it can only draw a line. <b>Try this:</b>` }),
    html("ol", { class: "sub" }, {
      tr: R`<li><b>Daire</b>, 0 gizli katman: doğruluk neden %50–60 civarında kalıyor?</li><li>1 gizli katman, 1, 2, sonra 3 nöron: önce tahmin edin, daireyi çözmek için en az kaç nöron gerekir? (Her gizli nöron bir doğru çizer; iç noktaları çevrelemek için kaç doğru gerekir?) Sonra nöronlara tek tek tıklayıp doğrularına bakın.</li><li><b>Spiral</b>: 1 katmanda 8 nöron ile 2 katmanda 8'er nöronu karşılaştırın.</li><li>Aynı ağı sigmoid, tanh ve ReLU ile eğitin: hangisi daha hızlı öğreniyor?</li>`,
      en: R`<li><b>Circle</b> with 0 hidden layers: why does the accuracy stay around 50–60%?</li><li>1 hidden layer with 1, 2, then 3 neurons: predict first, what is the fewest neurons that solves the circle? (Each hidden neuron draws one line; how many lines does it take to fence in the inner points?) Then click the neurons one by one and look at their lines.</li><li><b>Spiral</b>: compare 1 layer of 8 neurons with 2 layers of 8.</li><li>Train the same network with sigmoid, tanh and ReLU: which one learns fastest?</li>` }));

  const presets = segmented(PRESETS.map((k) => [k, presetName(k)]), PG.preset, (k) => {
    PG.preset = k;
    PG.data = makePreset(k, PG.seed);
    restart();
  }, L({ tr: "Hazır veri", en: "Dataset" }));
  const cls = segmented([["1", L({ tr: "● Sınıf 1", en: "● Class 1" })], ["0", L({ tr: "✕ Sınıf 0", en: "✕ Class 0" })]], String(PG.cls),
    (v) => { PG.cls = Number(v); }, L({ tr: "Eklenecek nokta", en: "Point to add" }));
  const regen = button(L({ tr: "Yeniden üret", en: "Regenerate" }));
  regen.addEventListener("click", () => { PG.seed += 1; if (PG.preset !== "custom") PG.data = makePreset(PG.preset, PG.seed); restart(); });

  const stage = h("div", { class: "pc-stage pg-plane" });
  const focusNote = h("p", { class: "sub ml-focus-note" });
  const stageCard = h("div", { class: "card pg-stage" },
    h("div", { class: "t-controls" }, presets, regen),
    h("div", { class: "t-controls" }, h("span", { class: "mini-title pc-inline", text: L({ tr: "Tıklayınca ekle:", en: "Click adds:" }) }), cls),
    stage, focusNote);
  const plane = createPlane(stage, { domain: [-1.5, 1.5, -1.5, 1.5], ticks: [-1, 0, 1], label: L({ tr: "Deneme alanı düzlemi", en: "Playground plane" }) });
  plane.canvas.classList.add("pg-canvas");
  plane.canvas.addEventListener("click", (event) => {
    const box = plane.canvas.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    const hit = PG.data.findIndex((p) => {
      const [qx, qy] = plane.toPx(...p.x);
      return Math.hypot(qx - px, qy - py) < 8;
    });
    if (hit >= 0) PG.data.splice(hit, 1);
    else PG.data.push({ x: plane.toData(px, py), y: PG.cls });
    PG.preset = "custom";
    presets.set("custom");
    update(true);
  });
  plane.canvas.addEventListener("pointermove", (event) => {
    if (event.pointerType === "touch") return;
    const box = plane.canvas.getBoundingClientRect();
    const [x, y] = plane.toData(event.clientX - box.left, event.clientY - box.top);
    const p = NET.predict(PG.net, [x, y]);
    showTip(event.clientX, event.clientY, `x₁ = ${nf(x)}, x₂ = ${nf(y)}`, [
      { value: nf(p, 3), label: L({ tr: "ŷ: sınıf 1 olasılığı", en: "ŷ: probability of class 1" }), color: p > 0.5 ? COLORS.one : COLORS.zero },
    ]);
  });
  plane.canvas.addEventListener("pointerleave", hideTip);

  // Architecture and training controls
  const layers = segmented([0, 1, 2, 3].map((v) => [String(v), String(v)]), String(PG.layers), (v) => { PG.layers = Number(v); restart(); },
    L({ tr: "Gizli katman sayısı", en: "Number of hidden layers" }));
  const width = slider(L({ tr: "Nöron / katman", en: "Neurons / layer" }), { min: 1, max: 8, step: 1, value: PG.width, fmt: (v) => fmtNum(v, 0), onInput: (v) => { PG.width = v; restart(); } });
  const act = segmented([["sigmoid", "Sigmoid"], ["tanh", "tanh"], ["relu", "ReLU"]], PG.act, (v) => { PG.act = v; restart(); }, L({ tr: "Aktivasyon", en: "Activation" }));
  const lr = slider("η", { min: 0, max: LRS.length - 1, step: 1, value: PG.lrIndex, fmt: (i) => fmtNum(LRS[i], LRS[i] < 0.01 ? 3 : LRS[i] < 0.1 ? 2 : 1), onInput: (i) => { PG.lrIndex = i; } });
  const speed = slider(L({ tr: "Hız", en: "Speed" }), { min: 0, max: SPEEDS.length - 1, step: 1, value: PG.speedIndex,
    fmt: (i) => L({ tr: `${SPEEDS[i]} epoch/sn`, en: `${SPEEDS[i]} epochs/s` }), onInput: (i) => { PG.speedIndex = i; } });
  const play = button("", "btn primary pg-play");
  const one = button(L({ tr: "Tek epoch", en: "One epoch" }));
  const again = button(L({ tr: "Ağırlıkları sıfırla", en: "Reset the weights" }));
  play.addEventListener("click", () => (PG.running ? pgStop() : pgStart()));
  one.addEventListener("click", () => { pgStop(); pgEpoch(); update(true); });
  again.addEventListener("click", () => { PG.seed += 1; restart(); });
  const tiles = h("div", { class: "pg-tiles ml-tiles" });
  const status = msg("");
  const diagramHost = h("div", { class: "ml-diagram" });
  const side = h("div", { class: "card pg-side" },
    h("h2", { text: L({ tr: "Ağ", en: "Network" }) }),
    h("div", { class: "pg-controls" },
      h("div", { class: "ml-field" }, h("span", { class: "pc-slider-name", text: L({ tr: "Gizli katman", en: "Hidden layers" }) }), layers),
      width,
      h("div", { class: "ml-field" }, h("span", { class: "pc-slider-name", text: L({ tr: "Aktivasyon", en: "Activation" }) }), act)),
    diagramHost,
    h("h2", { text: L({ tr: "Eğitim", en: "Training" }) }),
    h("div", { class: "pg-controls" }, lr, speed),
    h("div", { class: "pg-buttons" }, play, one, again),
    status, tiles);
  const layout = h("div", { class: "pg-layout" }, stageCard, side);
  const chartLoss = chartHost(190);
  const charts = h("div", { class: "card chart-card" }, h("div", { class: "card-head" }, h("div", {},
    h("h2", { text: L({ tr: "Kayıp", en: "Loss" }) }),
    h("p", { class: "sub", text: L({ tr: "Her epoch'tan sonra bütün noktalar üzerindeki ortalama çapraz entropi.", en: "The mean cross-entropy over all the points after every epoch." }) }))), chartLoss);
  panel.replaceChildren(intro, layout, charts);

  // The network diagram: a canvas with every neuron's output over the plane
  const diagram = createNetDiagram(diagramHost, (focus) => {
    PG.focus = focus && PG.focus && focus.l === PG.focus.l && focus.j === PG.focus.j ? null : focus;
    update(true);
  });

  function restart() {
    const was = PG.running;
    pgStop();
    pgReset();
    if (was) pgStart();
    update(true);
  }

  let lastFull = 0;
  let lastChart = 0;
  PG.update = update;
  function update(full = true) {
    const n = PG.net;
    const now = performance.now();
    if (full || now - lastFull > 90) {
      lastFull = now;
      const f = PG.focus;
      const predict = NET.predictor(n);
      let prob = (x1, x2) => predict(x1, x2);
      if (f) {
        const hidden = new Float64Array(n.layers.reduce((s, L) => s + L.b.length, 0));
        const offset = n.layers.slice(0, f.l).reduce((s, L) => s + L.b.length, 0) + f.j;
        const scale = neuronScale(n, f.l);
        prob = (x1, x2) => { predict(x1, x2, hidden); return scale(hidden[offset]); };
      }
      plane.draw({
        prob,
        contour: true,
        points: PG.data.map((p) => ({ ...p })),
        radius: Math.max(3, Math.min(5, plane.size / 110)),
      });
      diagram.draw(n, PG.focus);
      focusNote.textContent = f
        ? L({ tr: `Gösterilen: ${f.l + 1}. katmanın ${f.j + 1}. nöronu. Ağın çıktısına dönmek için ona yeniden tıklayın.`, en: `Showing neuron ${f.j + 1} of layer ${f.l + 1}. Click it again to go back to the network's output.` })
        : L({ tr: "Fareyle gezinince ağın o noktadaki olasılığı görünür.", en: "Hover to see the network's probability at any point." });
    }
    play.textContent = PG.running ? L({ tr: "❚❚ Durdur", en: "❚❚ Pause" }) : L({ tr: "▶ Eğit", en: "▶ Train" });
    const acc = NET.accuracy(n, PG.data);
    const lossNow = PG.history.at(-1);
    const tile = (label, value) => h("div", { class: "pg-tile" }, h("span", { class: "tile-label", text: label }), h("span", { class: "pg-tile-value", text: value }));
    tiles.replaceChildren(
      tile("Epoch", fmtNum(n.epoch, 0)),
      tile(L({ tr: "Kayıp", en: "Loss" }), lossNow == null ? "—" : nf(lossNow, 3)),
      tile(L({ tr: "Doğruluk", en: "Accuracy" }), PG.data.length ? fmtPct(acc * 100, 0) : "—"),
      tile(L({ tr: "Parametre", en: "Parameters" }), fmtNum(NET.paramCount(n.sizes), 0)));
    if (!PG.data.length) setMsg(status, L({ tr: "Önce düzleme tıklayarak nokta ekleyin ya da hazır bir veri seçin.", en: "First add points by clicking the plane, or pick a dataset." }), "bad");
    else if (lossNow != null && !Number.isFinite(lossNow) || lossNow >= 5) setMsg(status, L({ tr: "Kayıp patladı: η çok büyük. Küçültüp ağırlıkları sıfırlayın.", en: "The loss blew up: η is too large. Lower it and reset the weights." }), "bad");
    else if (acc === 1 && n.epoch) setMsg(status, L({ tr: `Bütün noktalar doğru ✓ (${n.epoch}. epoch)`, en: `Every point is right ✓ (epoch ${n.epoch})` }), "good");
    else if (PG.layers === 0 && n.epoch > 200 && acc < 0.95) setMsg(status, L({ tr: "Gizli katman yok: ağ yalnızca bir doğru çizebilir. Bir gizli katman ekleyin.", en: "No hidden layer: the network can only draw a line. Add a hidden layer." }), "bad");
    else setMsg(status, PG.running ? L({ tr: "Eğitiliyor…", en: "Training…" }) : L({ tr: `${PG.data.length} nokta. Eğit'e basın.`, en: `${PG.data.length} points. Press Train.` }));
    if (full || now - lastChart > 250) { lastChart = now; drawLossChart(chartLoss, PG.history); }
  }
  update(true);
}

/** Maps a neuron's activation to 0..1 for shading (tanh: −1..1, ReLU: 0..its largest value on the plane). */
function neuronScale(net, l) {
  const act = net.layers[l].act;
  if (act === "tanh") return (a) => (a + 1) / 2;
  if (act === "relu") return (a) => 0.5 + Math.min(0.5, a / 2);
  return (a) => a;
}

function drawLossChart(host, history) {
  const stepX = Math.max(1, Math.ceil(history.length / 300));
  const pts = history.filter((_, i) => i % stepX === 0).map((v, i) => [i + 1, Math.min(v, 2)]);
  const xMax = Math.max(pts.length, 5);
  const top = Math.max(0.2, ...pts.map((p) => p[1]));
  const yStep = niceStep(top, 4);
  const epochAt = (i) => (i - 1) * stepX + 1;
  lineChart(host, {
    label: L({ tr: "Kayıp", en: "Loss" }),
    empty: L({ tr: "Henüz epoch yok", en: "No epochs yet" }),
    x: { min: 1, max: xMax, ticks: linearTicks(1, xMax, Math.max(1, niceStep(xMax, 6)), (v) => fmtNum(epochAt(v), 0)) },
    y: { min: 0, max: Math.ceil(top / yStep) * yStep, ticks: linearTicks(0, top, yStep, (v) => fmtNum(v, yStep < 0.1 ? 2 : 1)) },
    series: [{ points: pts, color: "var(--series-2)" }],
    tooltip: (i) => ({ title: `Epoch ${epochAt(i + 1)}`, rows: [{ value: nf(pts[i][1], 3), label: L({ tr: "kayıp", en: "loss" }), color: "var(--series-2)" }] }),
  });
}

/**
 * The playground's network: columns of neurons, each drawn as a small picture of its output over
 * the plane, joined by lines (blue = positive weight, orange = negative, thicker = larger).
 * onPick({ l, j }) is called when a hidden or output neuron is clicked.
 */
function createNetDiagram(host, onPick) {
  const canvas = h("canvas", { class: "ml-diagram-canvas", role: "img", "aria-label": L({ tr: "Ağın şeması", en: "The network's diagram" }) });
  host.append(canvas);
  const ctx = canvas.getContext("2d");
  const tile = document.createElement("canvas");
  const N = 18;
  tile.width = N;
  tile.height = N;
  let current = null;
  let focus = null;
  let boxes = [];
  const view = KIT.fitCanvas(canvas, () => render());

  function layout(net) {
    const cols = net.sizes.length;
    const W = view.w;
    const H = view.h;
    const maxN = Math.max(...net.sizes);
    const size = Math.max(16, Math.min(34, (H - 16) / maxN - 6, (W - 20) / (cols * 2)));
    const xs = net.sizes.map((_, c) => 26 + size / 2 + (c * (W - 36 - size)) / Math.max(1, cols - 1));
    return net.sizes.map((n, c) => Array.from({ length: n }, (_, j) => {
      const gap = Math.min(size + 8, (H - 8) / n);
      const y0 = H / 2 - ((n - 1) * gap) / 2;
      return { x: xs[c], y: y0 + j * gap, size };
    }));
  }

  function render() {
    if (!current || !view.w) return;
    const net = current;
    const dpr = canvas.width / view.w;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, view.w, view.h);
    const pos = layout(net);
    // edges
    net.layers.forEach((L, l) => {
      L.W.forEach((row, j) => row.forEach((w, i) => {
        const a = pos[l][i];
        const b = pos[l + 1][j];
        ctx.strokeStyle = w >= 0 ? COLORS.lineB : COLORS.line;
        ctx.globalAlpha = Math.min(1, 0.25 + Math.abs(w) / 2);
        ctx.lineWidth = Math.min(4, 0.5 + Math.abs(w));
        ctx.beginPath();
        ctx.moveTo(a.x + a.size / 2, a.y);
        ctx.bezierCurveTo((a.x + b.x) / 2, a.y, (a.x + b.x) / 2, b.y, b.x - b.size / 2, b.y);
        ctx.stroke();
      }));
    });
    ctx.globalAlpha = 1;
    // neurons: the output of each one over the plane
    const predict = NET.predictor(net);
    const total = net.layers.reduce((s, L) => s + L.b.length, 0);
    const values = Array.from({ length: total }, () => new Float64Array(N * N));
    const hidden = new Float64Array(total);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        predict(-1.5 + ((c + 0.5) / N) * 3, 1.5 - ((r + 0.5) / N) * 3, hidden);
        for (let k = 0; k < total; k++) values[k][r * N + c] = hidden[k];
      }
    }
    const one = hexToRgb(COLORS.one);
    const zero = hexToRgb(COLORS.zero);
    const surface = hexToRgb(COLORS.surface);
    const paint = (box, value, label, picked) => {
      const sctx = tile.getContext("2d");
      const img = sctx.createImageData(N, N);
      for (let k = 0; k < N * N; k++) {
        const v = value(k);
        const rgb = v > 0.5 ? one : zero;
        const s = Math.min(1, Math.abs(v - 0.5) * 2) * 0.75;
        img.data[k * 4] = Math.round(surface[0] + (rgb[0] - surface[0]) * s);
        img.data[k * 4 + 1] = Math.round(surface[1] + (rgb[1] - surface[1]) * s);
        img.data[k * 4 + 2] = Math.round(surface[2] + (rgb[2] - surface[2]) * s);
        img.data[k * 4 + 3] = 255;
      }
      sctx.putImageData(img, 0, 0);
      const x = box.x - box.size / 2;
      const y = box.y - box.size / 2;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(tile, x, y, box.size, box.size);
      ctx.strokeStyle = picked ? COLORS.accent : COLORS.axis;
      ctx.lineWidth = picked ? 3 : 1;
      ctx.strokeRect(x + 0.5, y + 0.5, box.size - 1, box.size - 1);
      if (label) {
        ctx.font = "italic 12px Cambria, 'Times New Roman', serif";
        ctx.fillStyle = COLORS.text2;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(label, box.x, box.y + box.size / 2 + 2);
      }
    };
    // inputs: x₁ grows to the right, x₂ upwards
    paint(pos[0][0], (k) => ((k % N) + 0.5) / N, "");
    paint(pos[0][1], (k) => 1 - (Math.floor(k / N) + 0.5) / N, "");
    ctx.font = "italic 13px Cambria, 'Times New Roman', serif";
    ctx.fillStyle = COLORS.text2;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    pos[0].forEach((box, i) => ctx.fillText(i ? "x₂" : "x₁", box.x - box.size / 2 - 4, box.y));
    boxes = [];
    let offset = 0;
    net.layers.forEach((L, l) => {
      const scale = neuronScale(net, l);
      L.b.forEach((_, j) => {
        const vals = values[offset + j];
        const box = pos[l + 1][j];
        paint(box, (k) => scale(vals[k]), l === net.layers.length - 1 ? "ŷ" : "", focus && focus.l === l && focus.j === j);
        boxes.push({ ...box, l, j });
      });
      offset += L.b.length;
    });
  }

  canvas.addEventListener("click", (event) => {
    const r = canvas.getBoundingClientRect();
    const x = event.clientX - r.left;
    const y = event.clientY - r.top;
    const hit = boxes.find((b) => Math.abs(b.x - x) <= b.size / 2 + 3 && Math.abs(b.y - y) <= b.size / 2 + 3);
    if (hit) onPick({ l: hit.l, j: hit.j });
  });
  canvas.addEventListener("pointermove", (event) => {
    if (event.pointerType === "touch") return;
    const r = canvas.getBoundingClientRect();
    const hit = boxes.find((b) => Math.abs(b.x - (event.clientX - r.left)) <= b.size / 2 + 3 && Math.abs(b.y - (event.clientY - r.top)) <= b.size / 2 + 3);
    canvas.style.cursor = hit ? "pointer" : "";
    if (!hit) return hideTip();
    const last = hit.l === current.layers.length - 1;
    showTip(event.clientX, event.clientY, last ? L({ tr: "Çıktı nöronu", en: "Output neuron" }) : L({ tr: `${hit.l + 1}. gizli katman, nöron ${hit.j + 1}`, en: `Hidden layer ${hit.l + 1}, neuron ${hit.j + 1}` }), [
      { value: L({ tr: "tıklayın", en: "click" }), label: L({ tr: "büyük düzlemde göster", en: "show it on the big plane" }) },
    ]);
  });
  canvas.addEventListener("pointerleave", hideTip);
  return {
    draw(net, f) { current = net; focus = f; render(); },
  };
}

function pgStart() {
  if (!PG.data.length) { PG.update?.(); return; }
  PG.running = true;
  let last = performance.now();
  let acc = 0;
  const frame = (now) => {
    if (!PG.running) return;
    acc += ((now - last) / 1000) * SPEEDS[PG.speedIndex];
    last = now;
    let steps = Math.min(40, Math.floor(acc));
    acc -= Math.floor(acc);
    while (steps-- > 0) pgEpoch();
    PG.update?.(false);
    PG.raf = requestAnimationFrame(frame);
  };
  PG.raf = requestAnimationFrame(frame);
  PG.update?.(true);
}
function pgStop() {
  const was = PG.running;
  PG.running = false;
  cancelAnimationFrame(PG.raf);
  if (was) PG.update?.(true);
}

// =====================================================================
// Wiring
// =====================================================================
let built = false;
function stopAll() {
  for (const t of [...timers]) stop(t);
  HS.timer = 0;
  LS.timer = 0;
  CP.timer = 0;
}
function buildAll() {
  stopAll();
  for (const o of observers.splice(0)) o.disconnect();
  redrawers.clear();
  buildTheory();
  buildPlay();
  built = true;
}

document.addEventListener("DOMContentLoaded", buildAll);
document.addEventListener("lang-change", () => { if (built) buildAll(); });
document.addEventListener("theme-change", () => {
  readColors();
  for (const redraw of redrawers) redraw();
  if (built) PG.update?.(true);
});
document.addEventListener("tab-shown", (event) => {
  if (event.detail !== "play") pgStop();
  if (event.detail !== "theory") {
    stopAll();
    for (const demo of [HS, LS, CP]) demo.update?.();
  }
});
})();
