"use strict";
/*
 * Info buttons (i) on the Training tab: each opens a short popup that explains the
 * concept next to it, in Turkish or English, with a link to the Theory or Coding tab.
 * Uses the helpers from i18n.js and app.js (L, t, h, $, navigate, fmtInt, fmtNum, form).
 * In the texts, [[key|text]] links a glossary term (glossary.js).
 */

const batchesPerEpoch = () => Math.ceil(48000 / Math.max(1, Number(form.elements.batch_size.value) || 256));

const INFO = {
  settings: {
    title: { tr: "Ayarlar (hiperparametreler)", en: "Settings (hyperparameters)" },
    body: {
      tr: "<p><b>Hiperparametreler</b>, eğitimden önce elle seçilen ve modelin kendisinin öğrenmediği değerlerdir: kaç [[epoch|epoch]] eğitileceği, adımların ne kadar büyük olacağı, verinin kaçarlık gruplar hâlinde işleneceği…</p><p>Varsayılanlar <code>SimpleCNN.py</code> ile aynıdır ve aynı [[seed|seed]] ile hep aynı sonucu (%99,29 test doğruluğu) verir. Bir değeri değiştirip yeniden eğiterek etkisini grafiklerde karşılaştırabilirsiniz.</p>",
      en: "<p><b>Hyperparameters</b> are values chosen by hand before training, which the model does not learn itself: how many [[epoch|epochs]] to train, how big the steps are, how many images are processed together…</p><p>The defaults match <code>SimpleCNN.py</code> and, with the same [[seed|seed]], always give the same result (99.29% test accuracy). Change one value and train again to compare its effect on the charts.</p>",
    },
    link: { tab: "code", section: "c-config", label: { tr: "Kodlama → Hiperparametreler", en: "Coding → Hyperparameters" } },
  },
  epoch: {
    title: { tr: "Epoch", en: "Epoch" },
    body: {
      tr: "<p>Bir <b>epoch</b>, eğitim verisinin tamamının modele bir kez gösterilmesidir. Veri tek seferde değil küçük gruplar ([[batch|batch]]) hâlinde verilir; bütün batch'ler bir kez işlendiğinde bir epoch biter.</p><p>Her epoch'ta model aynı görüntüleri yeniden görür ve ağırlıklarını biraz daha iyileştirir. Çok az epoch'ta model yeterince öğrenemez ([[underfitting|<i>underfitting</i>]]); çok fazlasında eğitim verisini ezberlemeye başlayabilir ([[overfitting|<i>overfitting</i>]]). Hangisinin olduğunu [[val-set|doğrulama]] kaybını izleyerek anlarız.</p>",
      en: "<p>An <b>epoch</b> is one pass of the whole training set through the model. The data is not fed all at once but in small groups ([[batch|batches]]); an epoch ends when every batch has been processed once.</p><p>In every epoch the model sees the same images again and improves its weights a little more. With too few epochs it does not learn enough ([[underfitting|<i>underfitting</i>]]); with too many it can start memorizing the training data ([[overfitting|<i>overfitting</i>]]). Watching the [[val-set|validation]] loss tells which one is happening.</p>",
    },
    live: () => {
      const epochs = Number(form.elements.epochs.value) || 0;
      const n = batchesPerEpoch();
      return {
        tr: `Bu ayarlarla: ${epochs} epoch × ${fmtInt(n)} batch = ${fmtInt(epochs * n)} eğitim adımı.`,
        en: `With these settings: ${epochs} epochs × ${fmtInt(n)} batches = ${fmtInt(epochs * n)} training steps.`,
      };
    },
    link: { tab: "code", section: "c-train-loop", label: { tr: "Kodlama → Eğitim döngüsü", en: "Coding → The training loop" } },
  },
  lr: {
    title: { tr: "Öğrenme oranı (learning rate)", en: "Learning rate" },
    body: {
      tr: "<p>[[optimizer|Optimizer]]'ın her adımda ağırlıkları ne kadar değiştireceğini belirler:</p><p class=\"info-formula\">\\(\\displaystyle \\theta \\leftarrow \\theta - \\eta\\, \\nabla L\\)</p><p>Çok küçükse eğitim yavaş ilerler; çok büyükse [[loss|kayıp]] salınır ya da hiç düşmez (ıraksar). SimpleCNN, [[adam|<b>Adam</b>]] optimizer ile 0,001 kullanır. Adam her ağırlığın adımını ayrıca ölçeklediği için bu değer güvenli bir başlangıçtır.</p>",
      en: "<p>Sets how much the [[optimizer|optimizer]] changes the weights at every step:</p><p class=\"info-formula\">\\(\\displaystyle \\theta \\leftarrow \\theta - \\eta\\, \\nabla L\\)</p><p>Too small and training crawls; too large and the [[loss|loss]] oscillates or never goes down (diverges). SimpleCNN uses 0.001 with the [[adam|<b>Adam</b>]] optimizer. Adam scales each weight's step separately, which makes this value a safe starting point.</p>",
    },
    live: () => {
      const lr = (Number(form.elements.lr.value) || 0).toLocaleString(locale(), { maximumFractionDigits: 6 });
      return { tr: `Şu anki değer: η = ${lr}`, en: `Current value: η = ${lr}` };
    },
    link: { tab: "theory", section: "t-opt", label: { tr: "Teori → Optimizasyon (canlı deneyin)", en: "Theory → Optimization (try it live)" } },
  },
  batch: {
    title: { tr: "Batch boyutu", en: "Batch size" },
    body: {
      tr: "<p>Model görüntüleri tek tek değil, gruplar (<b>batch</b>) hâlinde işler. Bir batch'teki tüm görüntülerin [[loss|kaybının]] ortalaması alınır ve ağırlıklar bu ortalamaya göre <b>bir kez</b> güncellenir.</p><p><b>Büyük batch:</b> daha kararlı [[gradient|gradyan]] ve [[gpu|GPU]]'nun verimli kullanımı, ama daha çok bellek ve epoch başına daha az güncelleme. <b>Küçük batch:</b> daha gürültülü ama daha sık güncelleme.</p>",
      en: "<p>The model processes images not one by one but in groups (<b>batches</b>). The [[loss|losses]] of all images in a batch are averaged and the weights are updated <b>once</b> using that average.</p><p><b>Large batch:</b> steadier [[gradient|gradients]] and efficient [[gpu|GPU]] use, but more memory and fewer updates per epoch. <b>Small batch:</b> noisier but more frequent updates.</p>",
    },
    live: () => {
      const b = Number(form.elements.batch_size.value) || 256;
      return {
        tr: `Bu ayarla: 48.000 eğitim görüntüsü / ${b} = ${fmtInt(batchesPerEpoch())} batch / epoch (son batch daha küçük olabilir).`,
        en: `With this setting: 48,000 training images / ${b} = ${fmtInt(batchesPerEpoch())} batches per epoch (the last one may be smaller).`,
      };
    },
    link: { tab: "code", section: "c-data", label: { tr: "Kodlama → Veriyi hazırlamak", en: "Coding → Preparing the data" } },
  },
  seed: {
    title: { tr: "Seed (rastgelelik tohumu)", en: "Seed (random seed)" },
    body: {
      tr: "<p>Eğitimde birçok şey rastgeledir: ağırlıkların ilk değerleri, verinin [[train-set|eğitim]]/[[val-set|doğrulama]] diye bölünmesi, her epoch'taki karıştırma sırası ve [[dropout|dropout]] maskeleri.</p><p><b>Seed</b>, rastgele sayı üretecinin başlangıç noktasını sabitler: aynı seed her çalıştırmada birebir aynı sonucu verir. Farklı bir seed denemek, sonucun şansa ne kadar bağlı olduğunu gösterir.</p>",
      en: "<p>Many things in training are random: the initial weights, the [[train-set|train]]/[[val-set|validation]] split, the shuffling order in each epoch and the [[dropout|dropout]] masks.</p><p>The <b>seed</b> fixes the starting point of the random number generator, so the same seed gives exactly the same result every run. Trying another seed shows how much the result depends on luck.</p>",
    },
    link: { tab: "code", section: "c-seed", label: { tr: "Kodlama → Tekrarlanabilirlik", en: "Coding → Reproducibility" } },
  },
  stages: {
    title: { tr: "Eğitimin aşamaları", en: "The stages of a training run" },
    body: {
      tr: "<p>Bir eğitim çalıştırması beş aşamadan geçer: <b>veri yükleme → eğitim → doğrulama → test → kayıt</b>.</p><p>Eğitim ve doğrulama her epoch'ta tekrarlanır: model önce eğitim verisiyle öğrenir, sonra hiç görmediği [[val-set|doğrulama verisiyle]] ölçülür. Test yalnızca en sonda, bir kez yapılır. Her aşamanın yanındaki <b>i</b> düğmesi o aşamayı ayrıca açıklar.</p>",
      en: "<p>A training run goes through five stages: <b>data loading → training → validation → test → saving</b>.</p><p>Training and validation repeat every epoch: the model first learns from the training data, then is measured on [[val-set|validation data]] it has never seen. The test happens only once, at the very end. The <b>i</b> button next to each stage explains it in detail.</p>",
    },
    link: { tab: "code", section: "c-main", label: { tr: "Kodlama → Hepsini birleştirmek", en: "Coding → Putting it all together" } },
  },
  "stage.loading": {
    title: { tr: "1 · Veri yükleme", en: "1 · Data loading" },
    body: {
      tr: "<p>[[mnist|MNIST]] indirilir (yalnızca ilk seferde) ve üçe ayrılır: <b>48.000 eğitim</b>, <b>12.000 doğrulama</b> ve <b>10.000 test</b> görüntüsü.</p><p>Her görüntü, değerleri 0 ile 1 arasında olan 1×28×28 boyutunda bir [[tensor|tensöre]] çevrilir. <code>DataLoader</code>'lar veriyi batch'ler hâlinde verecek şekilde hazırlanır; eğitim verisi her epoch'ta karıştırılır.</p>",
      en: "<p>[[mnist|MNIST]] is downloaded (only the first time) and split in three: <b>48,000 training</b>, <b>12,000 validation</b> and <b>10,000 test</b> images.</p><p>Each image becomes a 1×28×28 [[tensor|tensor]] with values between 0 and 1. <code>DataLoader</code>s are set up to hand out the data in batches; the training data is shuffled every epoch.</p>",
    },
    link: { tab: "code", section: "c-data", label: { tr: "Kodlama → Veriyi hazırlamak", en: "Coding → Preparing the data" } },
  },
  "stage.training": {
    title: { tr: "2 · Eğitim", en: "2 · Training" },
    body: {
      tr: "<p>Model eğitim verisini batch batch görür. Her batch için bir tahmin yapar, kaybı hesaplar, [[backprop|geri yayılımla]] [[gradient|gradyanları]] bulur ve [[optimizer|optimizer]] [[parameter|ağırlıkları]] günceller (aşağıdaki 5 adım).</p><p>Bu aşamada model [[train-eval|<b>eğitim modundadır</b>]] (<code>model.train()</code>): [[dropout|Dropout]] etkindir ve [[batchnorm|BatchNorm]] o batch'in istatistiklerini kullanır.</p>",
      en: "<p>The model goes through the training data batch by batch. For every batch it makes a prediction, computes the loss, finds the [[gradient|gradients]] with [[backprop|backpropagation]], and the [[optimizer|optimizer]] updates the [[parameter|weights]] (the 5 steps below).</p><p>In this stage the model is in [[train-eval|<b>training mode</b>]] (<code>model.train()</code>): [[dropout|Dropout]] is active and [[batchnorm|BatchNorm]] uses the batch's own statistics.</p>",
    },
    link: { tab: "code", section: "c-train-epoch", label: { tr: "Kodlama → Bir epoch eğitim", en: "Coding → Training for one epoch" } },
  },
  "stage.validating": {
    title: { tr: "3 · Doğrulama", en: "3 · Validation" },
    body: {
      tr: "<p>Her epoch sonunda model, eğitimde hiç görmediği <b>12.000 doğrulama görüntüsünde</b> ölçülür. Bu sırada ağırlıklar değişmez ve model [[train-eval|<b>değerlendirme modundadır</b>]] (<code>model.eval()</code>): Dropout kapalıdır ve BatchNorm, eğitimde biriktirdiği hareketli ortalamaları kullanır.</p><p>Doğrulama sonucu, modelin gerçekten öğrenip öğrenmediğini ya da ezberlediğini ([[overfitting|overfitting]]) gösterir. [[hyperparameter|Hiperparametreleri]] seçerken de bu sonuca bakılır.</p>",
      en: "<p>At the end of every epoch the model is measured on <b>12,000 validation images</b> it never saw during training. The weights do not change and the model is in [[train-eval|<b>evaluation mode</b>]] (<code>model.eval()</code>): Dropout is off and BatchNorm uses the running averages it collected during training.</p><p>The validation result shows whether the model is really learning or memorizing ([[overfitting|overfitting]]). It is also what we look at when choosing [[hyperparameter|hyperparameters]].</p>",
    },
    link: { tab: "code", section: "c-evaluate", label: { tr: "Kodlama → Değerlendirme", en: "Coding → Evaluation" } },
  },
  "stage.testing": {
    title: { tr: "4 · Test", en: "4 · Test" },
    body: {
      tr: "<p>Eğitim bittikten sonra model <b>10.000 test görüntüsünde</b> bir kez ölçülür.</p><p>[[test-set|Test setine]] eğitim boyunca hiç bakılmaz ve hiperparametreler de ona göre seçilmez. Bu yüzden test doğruluğu, modelin [[generalization|yeni görüntülerdeki başarısının]] en dürüst tahminidir.</p>",
      en: "<p>After training finishes, the model is measured once on <b>10,000 test images</b>.</p><p>The [[test-set|test set]] is never looked at during training, and the hyperparameters are not chosen with it. That makes the test accuracy the most honest estimate of [[generalization|how well the model does on new images]].</p>",
    },
    link: { tab: "code", section: "c-main", label: { tr: "Kodlama → Hepsini birleştirmek", en: "Coding → Putting it all together" } },
  },
  "stage.saving": {
    title: { tr: "5 · Kayıt", en: "5 · Saving" },
    body: {
      tr: "<p>Eğitilmiş ağırlıklar ([[checkpoint|<code>state_dict</code>]]) <code>checkpoints/simple_cnn.pt</code> dosyasına kaydedilir. Eğitimin başladığı ağırlıklar da karşılaştırma için aynı dosyaya eklenir.</p><p>Panel bir sonraki açılışta bu dosyayı yükler; böylece Çıkarım ve Teori sekmeleri yeniden eğitim yapmadan çalışır.</p>",
      en: "<p>The trained weights ([[checkpoint|<code>state_dict</code>]]) are saved to <code>checkpoints/simple_cnn.pt</code>. The weights training started from are stored in the same file for comparison.</p><p>The dashboard loads this file the next time it starts, so the Inference and Theory tabs work without training again.</p>",
    },
    link: { tab: "code", section: "c-predict", label: { tr: "Kodlama → Kaydetmek ve tahmin yapmak", en: "Coding → Saving and predicting" } },
  },
  progress: {
    title: { tr: "İlerleme çubukları", en: "Progress bars" },
    body: {
      tr: "<p><b>Epoch</b> çubuğu, içinde bulunulan epoch'ta kaç batch'in işlendiğini; <b>Toplam</b> çubuğu bütün eğitimin ne kadarının bittiğini gösterir.</p><p>Kalan süre, şimdiye kadarki hıza bakılarak tahmin edilir; ilk birkaç saniyede daha az isabetlidir.</p>",
      en: "<p>The <b>Epoch</b> bar shows how many batches of the current epoch have been processed; the <b>Total</b> bar shows how much of the whole run is done.</p><p>The remaining time is estimated from the speed so far; it is less accurate in the first few seconds.</p>",
    },
  },
  trainstep: {
    title: { tr: "Her batch'te çalışan eğitim adımı", en: "The training step run on every batch" },
    body: {
      tr: "<p>Her batch için aynı beş satır çalışır. Bir epoch'ta bu döngü batch sayısı kadar, yani varsayılan ayarlarla 188 kez döner.</p><p>Kutular bu döngüyü canlandırır; her kutunun kendi <b>i</b> düğmesi o satırın ne yaptığını açıklar. Sağdaki sayaç şu ana kadar atılan toplam adım sayısıdır.</p>",
      en: "<p>The same five lines run for every batch. In one epoch the loop runs once per batch, i.e. 188 times with the default settings.</p><p>The boxes animate this loop; each box has its own <b>i</b> button explaining what that line does. The counter on the right is the total number of steps taken so far.</p>",
    },
    link: { tab: "code", section: "c-train-epoch", label: { tr: "Kodlama → Bir epoch eğitim", en: "Coding → Training for one epoch" } },
  },
  "chip.zero": {
    title: { tr: "1 · optimizer.zero_grad()", en: "1 · optimizer.zero_grad()" },
    body: {
      tr: "<p>Önceki adımda hesaplanan <b>[[gradient|gradyanları]] sıfırlar</b>.</p><p>PyTorch, gradyanları her <code>backward()</code> çağrısında üst üste ekler (biriktirir). Sıfırlanmazsa her adım, öncekilerin toplamıyla güncellenir ve eğitim bozulur. Program hata vermediği için fark edilmesi zor, sık yapılan bir hatadır.</p>",
      en: "<p><b>Resets the [[gradient|gradients]]</b> computed in the previous step.</p><p>PyTorch adds gradients up on every <code>backward()</code> call (they accumulate). Without the reset, every step would use the sum of all previous gradients and training would go wrong. It is a common bug that is hard to spot, because the program does not crash.</p>",
    },
    link: { tab: "code", section: "c-train-epoch", label: { tr: "Kodlama → Bir epoch eğitim", en: "Coding → Training for one epoch" } },
  },
  "chip.forward": {
    title: { tr: "2 · İleri yayılım: model(x)", en: "2 · Forward pass: model(x)" },
    body: {
      tr: "<p>Batch, katmanlardan sırayla geçer: [[convolution|evrişim]] → [[batchnorm|BatchNorm]] → [[relu|ReLU]] → [[pooling|havuzlama]] → … → [[fully-connected|tam bağlı katmanlar]].</p><p>Çıktı, her görüntü için 10 skor ([[logit|<b>logit</b>]]) içeren bir [[tensor|tensördür]]: <code>(256, 10)</code>. En yüksek skor, modelin tahmin ettiği rakamdır.</p>",
      en: "<p>The batch flows through the layers in order: [[convolution|convolution]] → [[batchnorm|BatchNorm]] → [[relu|ReLU]] → [[pooling|pooling]] → … → [[fully-connected|fully connected layers]].</p><p>The output is a [[tensor|tensor]] with 10 scores ([[logit|<b>logits</b>]]) per image: <code>(256, 10)</code>. The highest score is the digit the model predicts.</p>",
    },
    link: { tab: "theory", section: "t-schematic", label: { tr: "Teori → Mimari şeması", en: "Theory → Architecture diagram" } },
  },
  "chip.loss": {
    title: { tr: "3 · Kaybı hesapla: criterion(out, y)", en: "3 · Compute the loss: criterion(out, y)" },
    body: {
      tr: "<p><b>Çapraz entropi kaybı</b> (cross-entropy), doğru sınıfa verilen [[softmax|olasılığın]] negatif logaritmasıdır:</p><p class=\"info-formula\">\\(\\displaystyle L = -\\log p_{\\text{doğru sınıf}}\\)</p><p>Model doğru rakamdan eminse kayıp 0'a yakındır; yanlış bir rakamdan eminse büyüktür. Batch'teki 256 görüntünün ortalaması alınır.</p>",
      en: "<p>The <b>cross-entropy loss</b> is the negative logarithm of the [[softmax|probability]] given to the correct class:</p><p class=\"info-formula\">\\(\\displaystyle L = -\\log p_{\\text{correct class}}\\)</p><p>If the model is sure of the right digit the loss is close to 0; if it is sure of a wrong one the loss is large. It is averaged over the 256 images of the batch.</p>",
    },
    link: { tab: "theory", section: "t-softmax", label: { tr: "Teori → Softmax ve kayıp", en: "Theory → Softmax and loss" } },
  },
  "chip.backward": {
    title: { tr: "4 · Geri yayılım: loss.backward()", en: "4 · Backpropagation: loss.backward()" },
    body: {
      tr: "<p>[[backprop|Zincir kuralıyla]], kaybın modeldeki <b>582.218 [[parameter|parametrenin]]</b> (ağırlıklar, bias'lar ve BatchNorm'un γ, β değerleri) her birine göre türevini ([[gradient|<b>gradyanını</b>]]) hesaplar ve parametrelerin <code>.grad</code> alanına yazar. Bölüm 2'de bir örnek için adım adım izlenen hesabın aynısıdır, yalnızca çok daha büyüktür.</p><p>Gradyan, bir parametre biraz artarsa kaybın hangi yönde ve ne kadar değişeceğini söyler.</p>",
      en: "<p>Using the [[backprop|chain rule]], it computes the derivative ([[gradient|<b>gradient</b>]]) of the loss with respect to each of the model's <b>582,218 [[parameter|parameters]]</b> (weights, biases and BatchNorm's γ and β) and stores it in their <code>.grad</code> field. It is the same computation chapter 2 followed step by step for one example, only much bigger.</p><p>A gradient tells in which direction, and how much, the loss changes when that parameter grows a little.</p>",
    },
    link: { tab: "theory", section: "t-opt", label: { tr: "Teori → Geri yayılım ve optimizasyon", en: "Theory → Backpropagation and optimization" } },
  },
  "chip.step": {
    title: { tr: "5 · Güncelleme: optimizer.step()", en: "5 · Update: optimizer.step()" },
    body: {
      tr: "<p>[[adam|<b>Adam</b>]], gradyanları kullanarak her ağırlığı kaybı azaltacak yönde küçük bir adım kaydırır.</p><p>Adımın büyüklüğünü [[learning-rate|öğrenme oranı]] (0,001) ve Adam'ın her ağırlık için tuttuğu geçmiş gradyan bilgisi birlikte belirler. Ağırlıkların gerçekten değiştiği tek satır budur.</p>",
      en: "<p>[[adam|<b>Adam</b>]] uses the gradients to move every weight a small step in the direction that lowers the loss.</p><p>The step size is set by the [[learning-rate|learning rate]] (0.001) together with the gradient history Adam keeps for each weight. This is the only line where the weights actually change.</p>",
    },
    link: { tab: "theory", section: "t-opt", label: { tr: "Teori → Optimizasyon", en: "Theory → Optimization" } },
  },
  "tile.loss": {
    title: { tr: "Eğitim kaybı", en: "Training loss" },
    body: {
      tr: "<p>Son batch'lerin [[loss|kaybının]] <b>[[moving-average|hareketli ortalaması]]</b>. Tek tek batch kayıpları çok dalgalandığı için yumuşatılmış hâli gösterilir.</p><p>Düşmesi, modelin eğitim verisini giderek daha iyi tahmin ettiğini gösterir. Tek başına yeterli değildir: modelin yeni veride de iyi olup olmadığını [[val-set|doğrulama]] sonucu söyler. Başlangıçtaki yaklaşık 2,3 değeri, her rakama eşit olasılık (%10) veren bir modelin kaybıdır: \\(-\\log 0{,}1 = \\ln 10 \\approx 2{,}3\\).</p>",
      en: "<p>A <b>[[moving-average|moving average]]</b> of the [[loss|loss]] of the most recent batches. Single batch losses fluctuate a lot, so a smoothed value is shown.</p><p>It going down means the model predicts the training data better and better. On its own that is not enough: the [[val-set|validation]] result tells whether it also does well on new data. The starting value of about 2.3 is the loss of a model that gives every digit the same probability (10%): \\(-\\log 0.1 = \\ln 10 \\approx 2.3\\).</p>",
    },
  },
  "tile.val": {
    title: { tr: "Doğrulama doğruluğu", en: "Validation accuracy" },
    body: {
      tr: "<p>Modelin hiç görmediği <b>12.000 doğrulama görüntüsünden</b> kaçını doğru sınıflandırdığı. Her epoch sonunda hesaplanır; altta en iyi epoch'un değeri yazar.</p><p>Bu sayı yükselmeyi bırakıp düşmeye başlarsa model ezberliyor ([[overfitting|overfitting]]) olabilir; eğitimi o noktada durdurmak gerekir.</p>",
      en: "<p>How many of the <b>12,000 validation images</b>, which the model never trained on, it classifies correctly. It is computed after every epoch; the best epoch's value is shown underneath.</p><p>If this number stops rising and starts falling, the model may be memorizing ([[overfitting|overfitting]]) and training should stop there.</p>",
    },
  },
  "tile.test": {
    title: { tr: "Test doğruluğu", en: "Test accuracy" },
    body: {
      tr: "<p>Eğitim bitince <b>10.000 test görüntüsünde</b> ölçülen doğruluk: modelin [[generalization|genelleme]] başarısının son ve en dürüst ölçüsü.</p><p>SimpleCNN, varsayılan ayarlarla %99,29'a ulaşır; yani 10.000 görüntüden yalnızca 71'ini yanlış sınıflandırır.</p>",
      en: "<p>The accuracy measured on <b>10,000 test images</b> after training: the final and most honest measure of how well the model [[generalization|generalizes]].</p><p>With the default settings SimpleCNN reaches 99.29%, i.e. it misclassifies only 71 of the 10,000 images.</p>",
    },
  },
  "tile.time": {
    title: { tr: "Süre", en: "Time" },
    body: {
      tr: "<p>Eğitimin başından beri geçen süre ve şimdiye kadarki hıza göre kalan süre tahmini.</p><p>[[gpu|Ekran kartıyla]] bir epoch, karta göre birkaç saniye ile yarım dakika arası sürer; CPU'da aynı eğitim birkaç kat daha uzundur. [[convolution|Evrişim]] gibi işlemler binlerce paralel çarpmadan oluştuğu için GPU'da çok hızlanır.</p>",
      en: "<p>The time since training started, and an estimate of the remaining time from the speed so far.</p><p>With a [[gpu|graphics card]] an epoch takes from a few seconds to half a minute, depending on the card; on a CPU the same training is several times longer. Operations such as [[convolution|convolution]] consist of thousands of parallel multiplications, which GPUs speed up enormously.</p>",
    },
  },
  batchloss: {
    title: { tr: "Batch kaybı grafiği", en: "Batch loss chart" },
    body: {
      tr: "<p>Her batch'ten sonraki eğitim [[loss|kaybı]]. <b>Gri çizgi</b> ham değerlerdir: her batch farklı görüntüler içerdiği için dalgalanır. <b>Mavi çizgi</b> [[moving-average|hareketli ortalamadır]] ve genel eğilimi gösterir.</p><p>Y ekseni [[log-scale|<b>logaritmiktir</b>]]: kayıp yaklaşık 2,3'ten 0,03'e indiği için sonlardaki küçük iyileşmeler ancak böyle görünür. Dikey çizgiler epoch sınırlarıdır.</p>",
      en: "<p>The training [[loss|loss]] after every batch. The <b>gray line</b> is the raw values: it fluctuates because every batch contains different images. The <b>blue line</b> is a [[moving-average|moving average]] showing the overall trend.</p><p>The y axis is [[log-scale|<b>logarithmic</b>]]: the loss falls from about 2.3 to 0.03, and only on this scale are the small late improvements visible. The vertical lines are epoch boundaries.</p>",
    },
  },
  losscurve: {
    title: { tr: "Epoch kaybı grafiği", en: "Epoch loss chart" },
    body: {
      tr: "<p>Her epoch sonundaki ortalama <b>eğitim</b> (mavi) ve <b>doğrulama</b> (turuncu) kaybı.</p><p>İkisi birlikte düşüyorsa model öğreniyordur. Eğitim kaybı düşmeye devam ederken doğrulama kaybı yükselirse model ezberlemeye başlamıştır ([[overfitting|overfitting]]). Burada eğitim kaybının doğrulamadan yüksek olması normaldir, iki nedeni var: eğitimde [[dropout|Dropout]] açıktır, doğrulamada kapalıdır; ve eğitim kaybı epoch boyunca, model henüz gelişirken ortalanır, doğrulama ise epoch sonundaki modelle ölçülür.</p>",
      en: "<p>The average <b>training</b> (blue) and <b>validation</b> (orange) loss at the end of each epoch.</p><p>If both go down, the model is learning. If the training loss keeps falling while the validation loss rises, the model has started memorizing ([[overfitting|overfitting]]). Here it is normal for the training loss to be higher than the validation loss, for two reasons: [[dropout|Dropout]] is on during training and off during validation; and the training loss is averaged over the epoch while the model is still improving, whereas validation uses the model as it is at the end of the epoch.</p>",
    },
    link: { tab: "theory", section: "t-drop", label: { tr: "Teori → Dropout", en: "Theory → Dropout" } },
  },
  accuracy: {
    title: { tr: "Doğruluk grafiği", en: "Accuracy chart" },
    body: {
      tr: "<p>Doğru sınıflandırılan görüntülerin yüzdesi. [[loss|Kayıptan]] farklı olarak yalnızca en yüksek [[logit|skorun]] doğru olup olmadığına bakar, modelin ne kadar emin olduğunu ölçmez.</p><p>[[mnist|MNIST]]'te doğrulama doğruluğu daha ilk epoch'un sonunda %97'yi geçer; sonraki epoch'lar kalan zor örnekleri düzeltir. Eğitim doğruluğu ilk epoch'ta daha düşük görünür, çünkü model henüz rastgeleyken başlayan batch'lerin ortalamasıdır.</p>",
      en: "<p>The percentage of correctly classified images. Unlike the [[loss|loss]], it only checks whether the highest [[logit|score]] is right; it does not measure how confident the model is.</p><p>On [[mnist|MNIST]] the validation accuracy passes 97% at the end of the very first epoch; later epochs fix the remaining hard examples. The training accuracy looks lower in the first epoch because it averages batches that started while the model was still random.</p>",
    },
  },
  filters: {
    title: { tr: "Conv1 filtreleri", en: "Conv1 filters" },
    body: {
      tr: "<p>İlk [[convolution|evrişim]] katmanının <b>32 adet 5×5 [[kernel|ağırlık matrisi]]</b>. Mavi negatif, kırmızı pozitif ağırlık demektir.</p><p>Her filtre görüntünün üzerinde kaydırılır ve kendi desenine benzeyen yerlerde güçlü tepki verir; bu tepkiler bir [[feature-map|öznitelik haritası]] oluşturur. Filtreler her adımda güncellenir; buradaki görünüm eğitim sırasında her 10 batch'te bir yenilenir, böylece nasıl değiştikleri izlenebilir. Bir filtreye tıklayarak büyütebilirsiniz.</p>",
      en: "<p>The first [[convolution|convolution]] layer's <b>32 [[kernel|weight matrices]] of 5×5</b>. Blue means a negative weight, red a positive one.</p><p>Each filter slides over the image and responds strongly where the image looks like its pattern; these responses form a [[feature-map|feature map]]. The filters are updated at every step; during training this view refreshes every 10 batches, so you can watch them change. Click a filter to enlarge it.</p>",
    },
    link: { tab: "theory", section: "t-conv", label: { tr: "Teori → Evrişim", en: "Theory → Convolution" } },
  },
  samples: {
    title: { tr: "Doğrulama örnekleri", en: "Validation samples" },
    body: {
      tr: "<p>[[val-set|Doğrulama setinden]] seçilmiş <b>16 sabit görüntü</b>. Model bunları eğitim başlamadan önce (rastgele [[parameter|ağırlıklarla]]) ve her epoch sonunda tahmin eder: ✓ doğru, ✗ yanlış (parantezde gerçek etiket).</p><p>Modelin eğitimle nasıl düzeldiğini gözle görmenin en kolay yolu: eğitim öncesinde tahminler rastgeledir, birkaç epoch sonra neredeyse hepsi doğrudur.</p>",
      en: "<p><b>16 fixed images</b> from the [[val-set|validation set]]. The model predicts them before training starts (with random [[parameter|weights]]) and after every epoch: ✓ correct, ✗ wrong (true label in brackets).</p><p>The easiest way to see the model improve: before training the predictions are random, a few epochs later almost all of them are right.</p>",
    },
  },
  table: {
    title: { tr: "Epoch tablosu", en: "Epoch table" },
    body: {
      tr: "<p>Her epoch'un özet sayıları: eğitim ve doğrulama kaybı, eğitim ve doğrulama doğruluğu, o ana kadar geçen süre.</p><p>En yüksek doğrulama doğruluğuna sahip epoch <b>kalın</b> yazılır. Grafiklerdeki değerlerin tam hâlidir.</p>",
      en: "<p>Each epoch's summary numbers: training and validation loss, training and validation accuracy, and the time elapsed so far.</p><p>The epoch with the highest validation accuracy is shown in <b>bold</b>. These are the exact values behind the charts.</p>",
    },
  },
  log: {
    title: { tr: "Kayıt", en: "Log" },
    body: {
      tr: "<p>Eğitimin metin çıktısı: <code>SimpleCNN.py</code>'yi terminalde çalıştırdığınızda göreceğiniz satırların aynısı (cihaz, veri boyutları, her epoch'un sonuçları, test sonucu).</p><p>Bir hata olursa ayrıntısı da burada görünür.</p>",
      en: "<p>The text output of training: the same lines you see when you run <code>SimpleCNN.py</code> in a terminal (device, data sizes, every epoch's results, the test result).</p><p>If something goes wrong, the details appear here too.</p>",
    },
    link: { tab: "code", section: "c-main", label: { tr: "Kodlama → Çalıştırmak ve çıktı", en: "Coding → Running it and the output" } },
  },
};

// =====================================================================
// The popup
// =====================================================================
const infoPop = { el: null, button: null };

function buildInfoPop() {
  const close = h("button", { type: "button", class: "btn small icon info-pop-close", text: "×" });
  close.addEventListener("click", closeInfo);
  const more = h("a", { href: "#", class: "info-pop-more" });
  more.addEventListener("click", (event) => {
    event.preventDefault();
    const link = INFO[infoPop.button?.dataset.info]?.link;
    closeInfo();
    if (link) navigate(link.tab, link.section);
  });
  infoPop.el = h("div", { class: "info-pop", role: "dialog", hidden: true },
    h("div", { class: "info-pop-head" }, h("strong", { class: "info-pop-title" }), close),
    h("div", { class: "info-pop-body" }),
    h("p", { class: "info-pop-live" }),
    more);
  document.body.append(infoPop.el);
}

function fillInfo(key) {
  const info = INFO[key];
  const pop = infoPop.el;
  pop.querySelector(".info-pop-title").textContent = L(info.title);
  pop.querySelector(".info-pop-body").innerHTML = termify(L(info.body));   // static, author-written markup; [[key|text]] → term links
  const live = pop.querySelector(".info-pop-live");
  live.textContent = info.live ? L(info.live()) : "";
  live.hidden = !info.live;
  const more = pop.querySelector(".info-pop-more");
  more.textContent = info.link ? `${L(info.link.label)} →` : "";
  more.hidden = !info.link;
  pop.setAttribute("aria-label", L(info.title));
  pop.querySelector(".info-pop-close").setAttribute("aria-label", L({ tr: "Kapat", en: "Close" }));
}

/** Places the popup under its button (or above it when there is no room), inside the window. */
function positionInfo() {
  placeNear(infoPop.el, infoPop.button);   // glossary.js
}

function openInfo(button) {
  if (infoPop.button === button) {   // a second click on the same button closes it
    closeInfo();
    return;
  }
  infoPop.button?.setAttribute("aria-expanded", "false");
  infoPop.button = button;
  button.setAttribute("aria-expanded", "true");
  fillInfo(button.dataset.info);
  hideTip();
  infoPop.el.hidden = false;
  positionInfo();
}

function closeInfo() {
  if (!infoPop.button) return;
  if (termCard.anchor && infoPop.el.contains(termCard.anchor)) closeTerm();   // its term card goes with it
  infoPop.button.setAttribute("aria-expanded", "false");
  infoPop.button = null;
  infoPop.el.hidden = true;
}

/** Every info button gets an accessible name from its topic, in the current language. */
function labelInfoButtons() {
  for (const button of document.querySelectorAll(".info-btn")) {
    const info = INFO[button.dataset.info];
    if (!info) continue;
    const name = `${L({ tr: "Açıklama", en: "About" })}: ${L(info.title)}`;
    button.setAttribute("aria-label", name);
    button.title = name;
  }
}

buildInfoPop();
labelInfoButtons();
document.addEventListener("click", (event) => {
  const button = event.target.closest?.(".info-btn");
  if (button) {
    event.preventDefault();
    openInfo(button);
  } else if (infoPop.button && !event.composedPath().some((node) => node === infoPop.el || node === termCard.el)) {
    closeInfo();   // a click anywhere else closes it
  }
});
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeInfo(); });
window.addEventListener("resize", () => { if (infoPop.button) positionInfo(); });
document.addEventListener("tab-shown", closeInfo);
document.addEventListener("lang-change", () => {
  labelInfoButtons();
  if (infoPop.button) {
    fillInfo(infoPop.button.dataset.info);
    positionInfo();
  }
});
