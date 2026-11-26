/*
 * Coding tab: a step-by-step tutorial that writes SimpleCNN.py from scratch.
 * Code excerpts are cut from the real SimpleCNN.py served by dashboard.py (located by
 * their first and last line, not by line numbers), so line numbers always match the file.
 * Uses codeview.js (codeEditor, terminal, rich), core.js (h, $, navigate)
 * and L() from i18n.js.
 * Every user-visible text is a plain string or a { tr, en } pair picked with L() at render time,
 * and the whole tab re-renders on "lang-change".
 */

// =====================================================================
// The tutorial
// Blocks (any block may carry engine: "browser" | "server", see shared/engine.ts):
//   { p }, { list, ordered }, { callout: "tip" | "warn" | "info", title, text },
//         { file: { from, to, fold, notes } }  (cut from SimpleCNN.py),
//         { code, name, notes, caption }      (a separate example file),
//         { term: title, lines }, { table: { head, rows } }, { links: [{ chapter, tab, section }] }
// Any text (including `code`, `name`, a note's `at` and a terminal line) may be a { tr, en } pair.
// =====================================================================
import SIMPLECNN_PY from "../../../../SimpleCNN.py";   // bundled as text (scripts/build.mjs)
import { hooks } from "../../shared/page.ts";
import { $, h, navigate } from "../../shared/core.js";
import { codeEditor, rich, terminal } from "../../shared/codeview.js";
import { L } from "../../shared/i18n.js";
import { placeLabel } from "../../shared/glossary/terms.js";

export const TRAIN_OUTPUT = [
  "Using device: cuda",
  "",
  "Epoch 1: Train Loss: 0.2790, Train Accuracy: 90.99% | Validation Loss: 0.0786, Validation Accuracy: 97.70%",
  "Epoch 2: Train Loss: 0.0904, Train Accuracy: 97.18% | Validation Loss: 0.0561, Validation Accuracy: 98.38%",
  "Epoch 3: Train Loss: 0.0685, Train Accuracy: 97.83% | Validation Loss: 0.0397, Validation Accuracy: 98.88%",
  "Epoch 4: Train Loss: 0.0553, Train Accuracy: 98.27% | Validation Loss: 0.0348, Validation Accuracy: 98.93%",
  "Epoch 5: Train Loss: 0.0453, Train Accuracy: 98.55% | Validation Loss: 0.0340, Validation Accuracy: 99.05%",
  "Epoch 6: Train Loss: 0.0415, Train Accuracy: 98.61% | Validation Loss: 0.0295, Validation Accuracy: 99.13%",
  "Epoch 7: Train Loss: 0.0378, Train Accuracy: 98.79% | Validation Loss: 0.0297, Validation Accuracy: 99.11%",
  "Epoch 8: Train Loss: 0.0329, Train Accuracy: 98.92% | Validation Loss: 0.0288, Validation Accuracy: 99.17%",
  "Epoch 9: Train Loss: 0.0335, Train Accuracy: 98.89% | Validation Loss: 0.0272, Validation Accuracy: 99.17%",
  "Epoch 10: Train Loss: 0.0301, Train Accuracy: 99.03% | Validation Loss: 0.0244, Validation Accuracy: 99.36%",
  "",
  "Test Loss: 0.0197, Test Accuracy: 99.29%",
];

export const STEPS = [
  {
    id: "c-intro", title: { tr: "Ne yapacağız?", en: "What are we building?" },
    blocks: [
      { p: {
        tr: "Bu rehberde `SimpleCNN.py` dosyasını baştan sona birlikte yazacağız: MNIST el yazısı rakamlarını tanıyan ve test setinde **%99'un üzerinde** doğruluğa ulaşan küçük bir evrişimli sinir ağı. Her adımda önce *neyi* ve *neden* yaptığımızı anlatıp, ardından kodu satır satır inceleyeceğiz.",
        en: "In this guide we will write `SimpleCNN.py` together from start to finish: a small convolutional neural network that recognizes handwritten MNIST digits and reaches **over 99%** accuracy on the test set. In every step we first explain *what* we are doing and *why*, then go through the code line by line.",
      } },
      { p: {
        tr: "Bu ilk ağınız değil: MLP bölümünde bir ağı önce NumPy ile elle, sonra PyTorch ile yazdınız (`nn.Linear`, bir kayıp ve `zero_grad` → `backward` → `step`). O dört satırlık eğitim döngüsü burada aynen geri gelecek, bu yüzden onu kısa geçeceğiz. Bu bölümde **yeni** olanlar: evrişim ve havuzlama katmanları (`Conv2d`, `MaxPool2d`), `BatchNorm2d` ve `Dropout` ile eğitim/değerlendirme modları, gerçek bir veri setini `DataLoader` ile batch'lere bölmek, GPU'da çalışmak ve modeli ayrı doğrulama ve test setlerinde ölçmek.",
        en: "This is not your first network: in the MLP chapter you wrote one by hand in NumPy and then in PyTorch (`nn.Linear`, a loss and `zero_grad` → `backward` → `step`). That four-line training loop comes back unchanged here, so we will go over it quickly. What is **new** in this chapter: convolution and pooling layers (`Conv2d`, `MaxPool2d`), the training/evaluation modes that come with `BatchNorm2d` and `Dropout`, splitting a real dataset into batches with `DataLoader`, running on the GPU, and measuring the model on separate validation and test sets.",
      } },
      { p: {
        tr: "Bu rehberin sonunda şunları yapabileceksiniz:",
        en: "By the end of this guide you will be able to:",
      } },
      { list: [
        { tr: "bir CNN'i `nn.Module` alt sınıfı olarak tanımlamak ve her katmandan sonra tensörün boyutunu hesaplamak;", en: "define a CNN as a subclass of `nn.Module` and work out the tensor's shape after every layer;" },
        { tr: "parametre sayısını elle bulup `named_parameters()` ile doğrulamak;", en: "count the parameters by hand and check the count with `named_parameters()`;" },
        { tr: "MNIST'i eğitim, doğrulama ve test diye bölüp `DataLoader` ile modele vermek;", en: "split MNIST into training, validation and test sets and feed it to the model with `DataLoader`;" },
        { tr: "eğitim ve değerlendirme döngülerini `model.train()`, `model.eval()` ve `torch.no_grad()` ile doğru yazmak;", en: "write the training and evaluation loops correctly with `model.train()`, `model.eval()` and `torch.no_grad()`;" },
        { tr: "eğitim çıktısını okuyup aşırı uyumu (overfitting) fark etmek.", en: "read the training output and recognize overfitting." },
      ] },
      { links: [
        { chapter: "mlp", tab: "code", section: "mc-torch" },
        { chapter: "mlp", tab: "code", section: "mc-fit" },
      ] },
      { p: {
        tr: "Bir derin öğrenme programı hemen her zaman aynı iskeleti izler. Adımlar da bu sırayla ilerliyor:",
        en: "A deep learning program almost always follows the same skeleton, and the steps follow it in this order:",
      } },
      { list: [
        { tr: "**Ortam ve kütüphaneler**: PyTorch'u kurup içe aktarmak", en: "**Environment and libraries**: installing and importing PyTorch" },
        { tr: "**Ayarlar**: hiperparametreler ve mimari sayıları", en: "**Settings**: hyperparameters and the architecture's numbers" },
        { tr: "**Veri**: MNIST'i indirip batch'lere bölmek", en: "**Data**: downloading MNIST and splitting it into batches" },
        { tr: "**Model**: katmanları tanımlamak ve verinin yolunu (`forward`) yazmak", en: "**Model**: defining the layers and writing the path the data takes (`forward`)" },
        { tr: "**Kayıp ve optimizer**: hatayı ölçmek ve ağırlıkları güncellemek", en: "**Loss and optimizer**: measuring the error and updating the weights" },
        { tr: "**Eğitim ve değerlendirme döngüleri**", en: "**Training and evaluation loops**" },
        { tr: "**Hepsini birleştirmek**, çalıştırmak ve sonucu yorumlamak", en: "**Putting it all together**, running it and interpreting the result" },
      ], ordered: true },
      { callout: "info", title: { tr: "Kodlar gerçek dosyadan geliyor", en: "The code comes from the real file" }, text: {
        tr: "Buradaki kod parçaları repodaki `SimpleCNN.py` dosyasından alınır (site derlenirken sayfaya eklenir); satır numaraları dosyadakiyle birebir aynıdır. Uzun açıklama metinleri (docstring) katlanmıştır, tıklayınca açılır.",
        en: "The code excerpts here are taken from the repository's `SimpleCNN.py` (added to the page when the site is built), so the line numbers match the file exactly. Long docstrings are folded; click one to expand it.",
      } },
      { callout: "tip", title: { tr: "Nasıl okunur?", en: "How to read this" }, text: {
        tr: "Kod satırlarının sonundaki numaralı mavi işaretler, bloğun altındaki aynı numaralı açıklamalara karşılık gelir. Bir açıklamanın üzerine gelince ilgili satır vurgulanır. Kavramların kendisi için [[theory|Teori]] sekmesine, sonuçların canlı hali için [[train|Eğitim]] sekmesine bağlantılar var.",
        en: "The numbered blue markers at the end of code lines match the numbered notes below each block. Hovering over a note highlights its line. There are links to the [[theory|Theory]] tab for the concepts themselves and to the [[train|Training]] tab for live results.",
      } },
      { callout: "info", title: { tr: "Temel fikir: tensör", en: "Key idea: the tensor" }, text: {
        tr: "PyTorch'ta her şey **tensör** (çok boyutlu dizi) üzerinde çalışır. Bir grup görüntü 4 boyutlu bir tensördür: `(N, C, H, W)`, yani örnek sayısı, kanal, yükseklik ve genişlik. Bir MNIST batch'i `(256, 1, 28, 28)` boyutundadır. Rehber boyunca boyutların nasıl değiştiğini takip edeceğiz.",
        en: "In PyTorch everything works on **tensors** (multi-dimensional arrays). A group of images is a 4-dimensional tensor: `(N, C, H, W)`, that is, number of samples, channels, height and width. An MNIST batch has shape `(256, 1, 28, 28)`. Throughout the guide we will track how these shapes change.",
      } },
    ],
  },
  {
    id: "c-setup", title: { tr: "Ortamı hazırlamak", en: "Setting up the environment" },
    blocks: [
      { p: {
        tr: "Projeyi diğer Python kurulumlarından ayırmak için bir **sanal ortam** (`.venv`) kullanıyoruz. İçine PyTorch'u, ekran kartıyla çalışabilen (CUDA) sürümüyle kuruyoruz:",
        en: "To keep the project separate from other Python installations we use a **virtual environment** (`.venv`). Into it we install PyTorch, in the build that can run on the graphics card (CUDA):",
      } },
      { term: { tr: "PowerShell · proje klasöründe", en: "PowerShell · in the project folder" }, lines: [
        "$ py -3.13 -m venv .venv",
        "$ .venv\\Scripts\\Activate.ps1",
        "$ python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126",
        "$ python -m pip install matplotlib",
      ] },
      { callout: "tip", title: { tr: "Kısa yol", en: "Shortcut" }, text: {
        tr: "Bu repodaki `start.bat`, sanal ortam yoksa bu adımları sizin yerinize yapar ve ardından repodaki Python panelini (`dashboard.py`, bu sayfanın PyTorch'la çalışan sürümü) başlatır.",
        en: "If there is no virtual environment yet, this repository's `start.bat` runs these steps for you and then starts the repository's Python dashboard (`dashboard.py`, a version of this page that runs on PyTorch).",
      } },
      { p: {
        tr: "Kurulumun çalıştığını ve ekran kartının görüldüğünü küçük bir denemeyle kontrol edelim:",
        en: "Let's check with a small test that the installation works and the graphics card is detected:",
      } },
      { code: {
        tr: String.raw`import torch

print(torch.__version__)              # PyTorch sürümü
print(torch.cuda.is_available())      # True ise ekran kartı kullanılabilir
print(torch.cuda.get_device_name(0))  # ekran kartının adı`,
        en: String.raw`import torch

print(torch.__version__)              # PyTorch version
print(torch.cuda.is_available())      # True if the GPU can be used
print(torch.cuda.get_device_name(0))  # name of the GPU`,
      }, name: { tr: "kontrol.py", en: "check.py" } },
      { term: { tr: "Çıktı (bu bilgisayarda)", en: "Output (on this computer)" }, lines: ["2.14.1+cu126", "True", "NVIDIA GeForce RTX 2060"] },
      { callout: "warn", title: { tr: "Ekran kartı yoksa", en: "If you have no GPU" }, text: {
        tr: "`torch.cuda.is_available()` `False` dönerse kod yine çalışır, yalnızca eğitim CPU'da birkaç kat daha uzun sürer. Kodun hiçbir yerini değiştirmeniz gerekmez (bkz. Adım 9).",
        en: "If `torch.cuda.is_available()` returns `False`, the code still runs; training just takes several times longer on the CPU. You don't need to change anything in the code (see Step 9).",
      } },
    ],
  },
  {
    id: "c-imports", title: { tr: "Kütüphaneleri içe aktarmak", en: "Importing the libraries" },
    blocks: [
      { p: {
        tr: "Dosyanın başında ihtiyaç duyacağımız araçları içe aktarıyoruz. PyTorch'un görevleri birkaç alt pakete ayrılmıştır: modeli kuranlar, eğitenler ve veriyi hazırlayanlar.",
        en: "At the top of the file we import the tools we will need. PyTorch splits its jobs across a few subpackages: those that build the model, those that train it and those that prepare the data.",
      } },
      { file: { from: "import torch", to: "import matplotlib.pyplot as plt", notes: [
        { at: "import torch", text: {
          tr: "PyTorch'un çekirdeği: tensörler, GPU desteği ve matematik işlemleri.",
          en: "The core of PyTorch: tensors, GPU support and math operations.",
        } },
        { at: "import torch.nn as nn", text: {
          tr: "Sinir ağı yapı taşları: `Conv2d`, `Linear`, `BatchNorm2d` gibi **öğrenilen ağırlıkları olan** katmanlar ve kayıp fonksiyonları.",
          en: "Neural network building blocks: layers **with learned weights** such as `Conv2d`, `Linear` and `BatchNorm2d`, plus the loss functions.",
        } },
        { at: "import torch.nn.functional as F", text: {
          tr: "Ağırlığı olmayan işlemlerin fonksiyon hâli, örneğin `F.relu`. Durum tutmadıkları için modelde önceden tanımlanmaları gerekmez.",
          en: "Function versions of operations without weights, for example `F.relu`. Since they hold no state, they don't need to be defined in the model beforehand.",
        } },
        { at: "import torch.optim as optim", text: {
          tr: "Optimizasyon algoritmaları (Adam, SGD…): gradyanları kullanarak ağırlıkları günceller.",
          en: "Optimization algorithms (Adam, SGD…): they use the gradients to update the weights.",
        } },
        { at: "from torch.utils.data import DataLoader, random_split", text: {
          tr: "`DataLoader` veriyi batch'ler hâlinde, istenirse karıştırarak verir; `random_split` bir veri setini rastgele parçalara böler.",
          en: "`DataLoader` serves the data in batches, shuffled if requested; `random_split` splits a dataset into random parts.",
        } },
        { at: "from torchvision import datasets", text: {
          tr: "Hazır veri setleri; `datasets.MNIST` MNIST'i indirir ve okur.",
          en: "Ready-made datasets; `datasets.MNIST` downloads and reads MNIST.",
        } },
        { at: "from torchvision.transforms import ToTensor", text: {
          tr: "Görüntüyü (0–255 tam sayı pikseller) 0–1 aralığında ondalıklı bir tensöre çevirir ve boyutunu `(1, 28, 28)` yapar.",
          en: "Converts an image (integer pixels from 0–255) into a floating-point tensor in the 0–1 range with shape `(1, 28, 28)`.",
        } },
        { at: "import matplotlib.pyplot as plt", text: {
          tr: "Eğitim sonunda kayıp grafiğini çizmek için.",
          en: "For drawing the loss plot at the end of training.",
        } },
      ] } },
      { callout: "info", title: { tr: "nn mi, F mi?", en: "nn or F?" }, text: {
        tr: "Kural basit: öğrenilen ağırlığı olan her şey (`Conv2d`, `Linear`, `BatchNorm2d`) `nn` katmanı olarak modelde saklanır. Ağırlığı olmayan işlemler (`relu`, `flatten`) doğrudan fonksiyon olarak çağrılabilir. Dropout bir istisnadır: ağırlığı yoktur ama eğitimde ve değerlendirmede farklı davranır. `nn.Dropout` katmanı `model.eval()` ile kendiliğinden kapanır; `F.dropout` kullansaydık `training=self.training`'i her seferinde elle vermemiz gerekirdi.",
        en: "The rule is simple: anything with learned weights (`Conv2d`, `Linear`, `BatchNorm2d`) is stored in the model as an `nn` layer. Operations without weights (`relu`, `flatten`) can be called directly as functions. Dropout is an exception: it has no weights, but it behaves differently in training and in evaluation. An `nn.Dropout` layer switches itself off with `model.eval()`; with `F.dropout` we would have to pass `training=self.training` by hand every time.",
      } },
    ],
  },
  {
    id: "c-config", title: { tr: "Hiperparametreler ve mimari", en: "Hyperparameters and architecture" },
    blocks: [
      { p: {
        tr: "**Hiperparametreler**, eğitimden önce elle seçtiğimiz ve modelin kendisinin öğrenmediği değerlerdir. Hepsini dosyanın başında tek bir yerde toplamak, denemeler yaparken (örneğin öğrenme oranını değiştirirken) kodun içinde aramaktan kurtarır.",
        en: "**Hyperparameters** are values we choose by hand before training and that the model does not learn itself. Collecting them all in one place at the top of the file saves you from hunting through the code when experimenting (for example, when changing the learning rate).",
      } },
      { file: { from: "#--------------- Hyperparameters", to: "dropout2_rate = 0.35", notes: [
        { at: "learning_rate = 1e-3", text: {
          tr: "**Öğrenme oranı**: optimizer'ın her adımda ağırlıkları ne kadar değiştireceği. Adam için 0,001 yaygın ve güvenli bir başlangıçtır.",
          en: "**Learning rate**: how much the optimizer changes the weights at each step. For Adam, 0.001 is a common and safe starting point.",
        } },
        { at: "num_epochs = 10", text: {
          tr: "**Epoch**: eğitim verisinin tamamının bir kez gösterilmesi. Model her görüntüyü 10 kez görür.",
          en: "**Epoch**: one pass over the entire training set. The model sees every image 10 times.",
        } },
        { at: "batch_size = 256", text: {
          tr: "Her adımda birlikte işlenen görüntü sayısı. Büyük batch daha kararlı gradyan ve GPU'nun verimli kullanımı demektir, ama daha çok bellek ister.",
          en: "The number of images processed together in each step. A larger batch means more stable gradients and better GPU utilization, but it needs more memory.",
        } },
        { at: "validation_split = 0.2", text: {
          tr: "Eğitim verisinin %20'si (12.000 görüntü) **doğrulama** için ayrılır; model bu görüntülerle eğitilmez.",
          en: "20% of the training data (12,000 images) is set aside for **validation**; the model is not trained on these images.",
        } },
        { at: "seed = 42", text: {
          tr: "Rastgelelik tohumu: aynı tohum her çalıştırmada aynı sonucu verir (Adım 5).",
          en: "The random seed: the same seed gives the same result on every run (Step 5).",
        } },
        { at: "conv1_out_channels = 32", text: {
          tr: "Filtre sayıları: ilk evrişim 32, ikincisi 64 öznitelik haritası üretir. Derinleştikçe kanal sayısını artırmak yaygın bir tasarımdır.",
          en: "Filter counts: the first convolution produces 32 feature maps, the second 64. Increasing the number of channels as the network gets deeper is a common design.",
        } },
        { at: "conv_kernel_size = 5", text: {
          tr: "Filtreler 5×5 piksel büyüklüğünde.",
          en: "The filters are 5×5 pixels.",
        } },
        { at: "pool_kernel_size = 2", text: {
          tr: "2×2 max pooling, stride 2: her havuzlama genişliği ve yüksekliği yarıya indirir.",
          en: "2×2 max pooling with stride 2: each pooling halves the width and the height.",
        } },
        { at: "fc1_in = conv2_out_channels * 4 * 4", text: {
          tr: "Düzleştirme sonrası vektör uzunluğu: 64 kanal × 4 × 4 = **1024**. Sabit 1024 yazmak yerine kanal sayısından hesaplıyoruz; `conv2_out_channels` değişirse kod bozulmaz.",
          en: "The vector length after flattening: 64 channels × 4 × 4 = **1024**. Instead of hard-coding 1024 we compute it from the channel count, so the code doesn't break if `conv2_out_channels` changes.",
        } },
        { at: "fc2_in = 512", text: {
          tr: "Gizli tam bağlı katmandaki nöron sayısı.",
          en: "The number of neurons in the hidden fully connected layer.",
        } },
        { at: "dropout1_rate = 0.45", text: {
          tr: "Eğitimde değerlerin %45'ini (ilk blok) ve %35'ini (ikinci blok) rastgele sıfırlayan dropout oranları.",
          en: "Dropout rates: during training they randomly zero 45% (first block) and 35% (second block) of the values.",
        } },
      ] } },
      { p: {
        tr: "`fc1_in`'deki `4 * 4` nereden geliyor? Görüntünün her katmanda nasıl küçüldüğünü izleyelim. Dolgusuz (padding 0) bir evrişim boyutu `k − 1` azaltır, 2×2 havuzlama ise yarıya indirir:",
        en: "Where does the `4 * 4` in `fc1_in` come from? Let's follow how the image shrinks at each layer. A convolution without padding (padding 0) reduces the size by `k − 1`, and 2×2 pooling halves it:",
      } },
      { table: {
        head: [
          { tr: "Katman", en: "Layer" },
          { tr: "Çıktı boyutu (C × H × W)", en: "Output shape (C × H × W)" },
          { tr: "Hesap", en: "Calculation" },
        ],
        rows: [
          [{ tr: "Giriş", en: "Input" }, "1 × 28 × 28", { tr: "MNIST görüntüsü", en: "an MNIST image" }],
          [{ tr: "Conv1 (5×5, 32 filtre)", en: "Conv1 (5×5, 32 filters)" }, "32 × 24 × 24", "28 − 5 + 1 = 24"],
          ["MaxPool 2×2", "32 × 12 × 12", "24 / 2 = 12"],
          [{ tr: "Conv2 (5×5, 64 filtre)", en: "Conv2 (5×5, 64 filters)" }, "64 × 8 × 8", "12 − 5 + 1 = 8"],
          ["MaxPool 2×2", "64 × 4 × 4", "8 / 2 = 4"],
          [{ tr: "Düzleştir (flatten)", en: "Flatten" }, "1024", "64 × 4 × 4"],
          ["FC1", "512", "fc2_in"],
          [{ tr: "FC2 (çıkış)", en: "FC2 (output)" }, "10", { tr: "her rakam için bir skor", en: "one score per digit" }],
        ],
      } },
      { callout: "tip", title: { tr: "Genel formül", en: "General formula" }, text: {
        tr: "Çıktı boyutu: \\(\\left\\lfloor \\frac{\\text{giriş} - \\text{çekirdek} + 2\\cdot\\text{padding}}{\\text{stride}} \\right\\rfloor + 1\\). [[theory:t-conv|Teori → Evrişim]] bölümünde padding ve stride'ı değiştirip formülü canlı görebilirsiniz.",
        en: "Output size: \\(\\left\\lfloor \\frac{\\text{input} - \\text{kernel} + 2\\cdot\\text{padding}}{\\text{stride}} \\right\\rfloor + 1\\). In the [[theory:t-conv|Theory → Convolution]] section you can change padding and stride and watch the formula live.",
      } },
    ],
  },
  {
    id: "c-seed", title: { tr: "Tekrarlanabilirlik: seed", en: "Reproducibility: the seed" },
    blocks: [
      { p: {
        tr: "Eğitimde birçok şey rastgeledir: ağırlıkların ilk değerleri, verinin eğitim/doğrulama diye bölünmesi, her epoch'taki karıştırma sırası ve dropout maskeleri. Sonuçları karşılaştırabilmek (ve öğrencilerin aynı sayıları görmesi) için bu rastgeleliği sabitliyoruz.",
        en: "Many things in training are random: the initial weights, the train/validation split, the shuffling order in every epoch and the dropout masks. To be able to compare results (and so that all students see the same numbers) we fix this randomness.",
      } },
      { file: { from: "def set_seed", to: "torch.backends.cudnn.benchmark = False", fold: true, notes: [
        { at: "torch.manual_seed(seed_value)", text: {
          tr: "PyTorch'un rastgele sayı üretecini sabitler; CPU ve tüm GPU'lar için geçerlidir.",
          en: "Fixes PyTorch's random number generator; it applies to the CPU and all GPUs.",
        } },
        { at: "torch.backends.cudnn.deterministic = True", text: {
          tr: "GPU'daki bazı hızlı evrişim algoritmaları çalıştırmadan çalıştırmaya küçük farklar üretir; bu ayar deterministik olanları seçer.",
          en: "Some fast convolution algorithms on the GPU give slightly different results from run to run; this setting picks deterministic ones.",
        } },
        { at: "torch.backends.cudnn.benchmark = False", text: {
          tr: "cuDNN'in her çalıştırmada \"en hızlı algoritmayı\" yeniden seçmesini kapatır; seçim değişirse sonuç da değişebilirdi.",
          en: "Stops cuDNN from re-selecting \"the fastest algorithm\" on every run; if that choice changed, the result could change too.",
        } },
      ] } },
      { callout: "info", engine: "browser", title: { tr: "Etkisi", en: "The effect" }, text: {
        tr: "Seed 42 ile her çalıştırma birebir aynı sonucu verir: test doğruluğu **%99,29**. Bu sayfanın [[train|Eğitim]] sekmesi aynı ayarları ve aynı adımları kullanır, ama tarayıcıda TensorFlow.js ile ve başka rastgele sayılarla çalıştığı için sonuçları bunlara yakındır, birebir aynı değildir.",
        en: "With seed 42 every run gives exactly the same result: a test accuracy of **99.29%**. This page's [[train|Training]] tab uses the same settings and the same steps, but it runs in the browser with TensorFlow.js and other random numbers, so its results come close to these, not digit for digit.",
      } },
      { callout: "info", engine: "server", title: { tr: "Etkisi", en: "The effect" }, text: {
        tr: "Seed 42 ile her çalıştırma birebir aynı sonucu verir: test doğruluğu **%99,29**. Bu sayfanın [[train|Eğitim]] sekmesi de PyTorch'ta aynı sırayı izlediği için aynı sayıları üretir.",
        en: "With seed 42 every run gives exactly the same result: a test accuracy of **99.29%**. This page's [[train|Training]] tab follows the same order in PyTorch, so it produces the same numbers.",
      } },
    ],
  },
  {
    id: "c-data", title: { tr: "Veriyi hazırlamak", en: "Preparing the data" },
    blocks: [
      { p: {
        tr: "MNIST, 28×28 piksellik gri tonlu el yazısı rakamlardan oluşur: 60.000 eğitim ve 10.000 test görüntüsü. Veriyi üç parçaya ayırıyoruz:",
        en: "MNIST consists of 28×28-pixel grayscale handwritten digits: 60,000 training and 10,000 test images. We split the data into three parts:",
      } },
      { list: [
        { tr: "**Eğitim** (48.000): model bunlardan öğrenir.", en: "**Training** (48,000): the model learns from these." },
        { tr: "**Doğrulama** (12.000): eğitim sırasında, görmediği verideki başarıyı izlemek için.", en: "**Validation** (12,000): to track, during training, how well the model does on data it hasn't seen." },
        { tr: "**Test** (10.000): en sonda yalnızca bir kez, son performansı ölçmek için.", en: "**Test** (10,000): used only once at the very end, to measure the final performance." },
      ] },
      { file: { from: "def get_mnist_dataloaders", to: "return training_loader, validation_loader, testing_loader", fold: true, notes: [
        { at: "full_train_dataset = datasets.MNIST(root='./data', train=True,", text: {
          tr: "60.000 eğitim görüntüsü. `root` verinin kaydedileceği klasör, `download=True` yoksa indirir, `transform` her görüntüye uygulanacak dönüşüm.",
          en: "The 60,000 training images. `root` is the folder where the data is stored, `download=True` downloads it if missing, and `transform` is the transformation applied to every image.",
        } },
        { at: "test_dataset = datasets.MNIST(root='./data', train=False,", text: {
          tr: "`train=False`: modelin eğitimde hiç görmeyeceği 10.000 test görüntüsü.",
          en: "`train=False`: the 10,000 test images the model never sees during training.",
        } },
        { at: "train_dataset, val_dataset = random_split(full_train_dataset,", text: {
          tr: "Eğitim setini rastgele %80 / %20 böler. Oranlar doğrudan kesir olarak verilebilir.",
          en: "Randomly splits the training set 80% / 20%. The proportions can be given directly as fractions.",
        } },
        { at: "training_loader = DataLoader(train_dataset,", text: {
          tr: "`shuffle=True`: her epoch'ta sıra karıştırılır, model örneklerin sırasını ezberleyemez.",
          en: "`shuffle=True`: the order is shuffled every epoch, so the model can't memorize the order of the samples.",
        } },
        { at: "validation_loader = DataLoader(val_dataset,", text: {
          tr: "Doğrulama ve testte karıştırmaya gerek yok; sonuç sıradan bağımsızdır.",
          en: "No need to shuffle for validation and testing; the result doesn't depend on the order.",
        } },
      ] } },
      { p: {
        tr: "`DataLoader` bir Python yineleyicisidir: `for` döngüsünde her turda bir batch verir. İlk batch'e bakarak boyutları doğrulayalım:",
        en: "A `DataLoader` is a Python iterable: in a `for` loop it yields one batch per iteration. Let's check the shapes by looking at the first batch:",
      } },
      { code: {
        tr: String.raw`from SimpleCNN import *

set_seed(seed)
train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size, validation_split)

images, labels = next(iter(train_loader))   # ilk batch
print(images.shape, images.dtype)           # (N, C, H, W)
print(images.min().item(), images.max().item())
print(labels[:10])                          # ilk 10 etiket
print(len(train_loader), "batch / epoch")`,
        en: String.raw`from SimpleCNN import *

set_seed(seed)
train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size, validation_split)

images, labels = next(iter(train_loader))   # the first batch
print(images.shape, images.dtype)           # (N, C, H, W)
print(images.min().item(), images.max().item())
print(labels[:10])                          # the first 10 labels
print(len(train_loader), "batches per epoch")`,
      }, name: { tr: "veriye_bakalim.py", en: "inspect_data.py" }, notes: [
        { at: "images, labels = next(iter(train_loader))", text: {
          tr: "`iter` yineleyiciyi başlatır, `next` ilk batch'i alır. Eğitim döngüsü de tam olarak bunu tekrar tekrar yapar.",
          en: "`iter` creates the iterator and `next` takes the first batch. The training loop does exactly this, over and over.",
        } },
        { at: { tr: "print(len(train_loader), \"batch / epoch\")", en: "print(len(train_loader), \"batches per epoch\")" }, text: {
          tr: "48.000 / 256 = 187,5, yukarı yuvarlanınca 188 batch. Son batch daha küçüktür (128 görüntü).",
          en: "48,000 / 256 = 187.5, which rounds up to 188 batches. The last batch is smaller (128 images).",
        } },
      ] },
      { term: { tr: "Çıktı", en: "Output" }, lines: [
        "torch.Size([256, 1, 28, 28]) torch.float32",
        "0.0 1.0",
        "tensor([5, 8, 4, 4, 4, 0, 3, 9, 1, 3])",
        { tr: "188 batch / epoch", en: "188 batches per epoch" },
      ] },
      { callout: "warn", title: { tr: "Test setini neden en sona saklıyoruz?", en: "Why do we save the test set for the very end?" }, text: {
        tr: "Hiperparametreleri (öğrenme oranı, epoch sayısı…) doğrulama sonuçlarına bakarak seçeriz. Test setine bakarak seçseydik, test sonucu artık \"hiç görülmemiş veri\" başarısını ölçmezdi.",
        en: "We choose the hyperparameters (learning rate, number of epochs…) by looking at the validation results. If we chose them by looking at the test set, the test result would no longer measure performance on \"never-seen data\".",
      } },
    ],
  },
  {
    id: "c-model", title: { tr: "Modeli tanımlamak: katmanlar", en: "Defining the model: layers" },
    blocks: [
      { p: {
        tr: "PyTorch'ta bir model, `nn.Module`'den türetilen bir sınıftır. İki yöntemi vardır: `__init__` katmanları **oluşturur**, `forward` verinin bu katmanlardan **hangi sırayla** geçeceğini söyler (Adım 8). MLP bölümündeki PyTorch örneğinde `nn.Sequential` yetiyordu, çünkü katmanlar yalnızca art arda uygulanıyordu. Burada `forward`'u kendimiz yazıyoruz: araya `torch.flatten` ve `F.relu` gibi işlemler giriyor, sıra da açıkça okunuyor.",
        en: "In PyTorch a model is a class derived from `nn.Module`. It has two methods: `__init__` **creates** the layers, and `forward` says **in which order** the data passes through them (Step 8). In the MLP chapter's PyTorch example `nn.Sequential` was enough, because the layers were simply applied one after the other. Here we write `forward` ourselves: operations such as `torch.flatten` and `F.relu` come in between, and the order can be read explicitly.",
      } },
      { file: { from: "class SimpleCNN(nn.Module):", to: "self.batch2 = nn.BatchNorm2d(num_features=conv2_out_channels)", fold: true, notes: [
        { at: "class SimpleCNN(nn.Module):", text: {
          tr: "`nn.Module`'den türetmek, katmanların ağırlıklarının otomatik bulunmasını (`model.parameters()`), modelin GPU'ya taşınabilmesini ve eğitim/değerlendirme modlarını sağlar.",
          en: "Deriving from `nn.Module` gives us automatic discovery of the layers' weights (`model.parameters()`), the ability to move the model to the GPU, and the training/evaluation modes.",
        } },
        { at: "super(SimpleCNN, self).__init__()", text: {
          tr: "`nn.Module`'ün kendi kurulumunu çalıştırır. Bu satır olmadan katmanlar modele kaydedilemez; her zaman ilk satırdır.",
          en: "Runs `nn.Module`'s own setup. Without this line the layers can't be registered with the model; it always comes first.",
        } },
        { at: "self.conv1 = nn.Conv2d(in_channels=input_channels,", text: {
          tr: "1 kanallı (gri) girişten 32 öznitelik haritası çıkaran 5×5 evrişim: 32 × (1×5×5) ağırlık + 32 bias = **832** parametre.",
          en: "A 5×5 convolution that turns the 1-channel (grayscale) input into 32 feature maps: 32 × (1×5×5) weights + 32 biases = **832** parameters.",
        } },
        { at: "self.conv2 = nn.Conv2d(in_channels=conv1_out_channels,", text: {
          tr: "32 haritayı girdi alıp 64 harita üretir. Her filtre 32 kanalın hepsine bakar: 64 × (32×5×5) + 64 = **51.264** parametre.",
          en: "Takes the 32 maps as input and produces 64 maps. Each filter looks at all 32 channels: 64 × (32×5×5) + 64 = **51,264** parameters.",
        } },
        { at: "self.max1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)", text: {
          tr: "Havuzlamanın öğrenilen parametresi yoktur; okunabilirlik için yine de katman olarak tanımlanır.",
          en: "Pooling has no learned parameters; it is still defined as a layer for readability.",
        } },
        { at: "self.fc1 = nn.Linear(in_features=fc1_in, out_features=fc2_in)", text: {
          tr: "1024 → 512 tam bağlı katman: 1024 × 512 + 512 = **524.800** parametre, modelin yaklaşık %90'ı. FC1 ve FC2 birlikte, MLP bölümünde yazdığınız türden bir MLP'dir; evrişim blokları ona ham pikseller yerine öznitelikler verir.",
          en: "A 1024 → 512 fully connected layer: 1024 × 512 + 512 = **524,800** parameters, about 90% of the model. Together FC1 and FC2 are an MLP of the kind you wrote in the MLP chapter; the convolution blocks feed it features instead of raw pixels.",
        } },
        { at: "self.fc2 = nn.Linear(in_features=fc2_in, out_features=num_classes)", text: {
          tr: "512 → 10: her rakam için bir skor (logit).",
          en: "512 → 10: one score (logit) per digit.",
        } },
        { at: "self.dropout1 = nn.Dropout(p=dropout1_rate)", text: {
          tr: "Dropout da parametresizdir ve yalnızca eğitim modunda çalışır.",
          en: "Dropout has no parameters either, and it is only active in training mode.",
        } },
        { at: "self.batch1 = nn.BatchNorm2d(num_features=conv1_out_channels)", text: {
          tr: "Kanal başına öğrenilen γ ve β (32 + 32 parametre) ile eğitim boyunca biriken ortalama ve varyans. Bu ikisi gradyanla öğrenilmez, batch'lerden hesaplanır: parametre değil \"buffer\"dır, bu yüzden aşağıdaki `named_parameters()` listesinde görünmezler ama `state_dict` ile kaydedilirler.",
          en: "A learned γ and β per channel (32 + 32 parameters), plus the running mean and variance accumulated during training. These two are not learned by gradients but computed from the batches: they are \"buffers\", not parameters, so they do not appear in the `named_parameters()` list below, but `state_dict` saves them.",
        } },
      ] } },
      { p: {
        tr: "Modelde kaç parametre olduğunu ve nerede yoğunlaştığını görmek için katmanları dolaşabiliriz:",
        en: "To see how many parameters the model has and where they are concentrated, we can loop over the layers:",
      } },
      { code: {
        tr: String.raw`model = SimpleCNN()
for name, p in model.named_parameters():
    print(f"{name:16s} {str(tuple(p.shape)):18s} {p.numel():>8,}")
print("Toplam:", f"{sum(p.numel() for p in model.parameters()):,}")`,
        en: String.raw`model = SimpleCNN()
for name, p in model.named_parameters():
    print(f"{name:16s} {str(tuple(p.shape)):18s} {p.numel():>8,}")
print("Total:", f"{sum(p.numel() for p in model.parameters()):,}")`,
      }, name: { tr: "parametreler.py", en: "parameters.py" } },
      { term: { tr: "Çıktı", en: "Output" }, lines: [
        "conv1.weight     (32, 1, 5, 5)           800",
        "conv1.bias       (32,)                    32",
        "conv2.weight     (64, 32, 5, 5)       51,200",
        "conv2.bias       (64,)                    64",
        "fc1.weight       (512, 1024)         524,288",
        "fc1.bias         (512,)                  512",
        "fc2.weight       (10, 512)             5,120",
        "fc2.bias         (10,)                    10",
        "batch1.weight    (32,)                    32",
        "batch1.bias      (32,)                    32",
        "batch2.weight    (64,)                    64",
        "batch2.bias      (64,)                    64",
        { tr: "Toplam: 582,218", en: "Total: 582,218" },
      ] },
      { callout: "info", title: { tr: "Evrişimin gücü", en: "The power of convolution" }, text: {
        tr: "İki evrişim katmanı toplam 52 bin parametreyle tüm görüntüyü tarıyor; tek bir tam bağlı katman (FC1) ise 524 bin parametre kullanıyor. Bunun nedeni **ağırlık paylaşımı**: aynı küçük filtre görüntünün her yerinde tekrar kullanılır.",
        en: "The two convolutional layers scan the whole image with 52 thousand parameters in total, while a single fully connected layer (FC1) uses 524 thousand. The reason is **weight sharing**: the same small filter is reused at every position in the image.",
      } },
    ],
  },
  {
    id: "c-forward", title: { tr: "İleri yayılım: forward", en: "The forward pass: forward" },
    blocks: [
      { p: {
        tr: "`forward`, bir batch'in modelden geçerken izlediği yolu tanımlar. Kod sıradan Python'dur: her satır bir önceki satırın çıktısını alır. Yanlarına boyutları yazdık; `N` batch büyüklüğüdür.",
        en: "`forward` defines the path a batch takes through the model. The code is ordinary Python: each line takes the output of the line before it. The notes give the shapes; `N` is the batch size.",
      } },
      { file: { from: "def forward(self, x):", to: "return x", fold: true, notes: [
        { at: "x = self.conv1(x)           # Convolution Layer 1", text: {
          tr: "`(N, 1, 28, 28)` → `(N, 32, 24, 24)`. Her çıktı kanalı bir filtrenin tepki haritasıdır.",
          en: "`(N, 1, 28, 28)` → `(N, 32, 24, 24)`. Each output channel is one filter's response map.",
        } },
        { at: "x = F.relu(self.batch1(x))  # Batch Normalization + ReLU", text: {
          tr: "Önce BatchNorm, sonra ReLU. ReLU'nun ağırlığı olmadığı için `F.relu` fonksiyon olarak çağrılır.",
          en: "BatchNorm first, then ReLU. ReLU has no weights, so it is called as the function `F.relu`.",
        } },
        { at: "x = self.max1(x)            # Max Pooling", text: {
          tr: "`(N, 32, 24, 24)` → `(N, 32, 12, 12)`.",
          en: "`(N, 32, 24, 24)` → `(N, 32, 12, 12)`.",
        } },
        { at: "x = self.dropout1(x)        # Dropout", text: {
          tr: "`model.train()` modunda değerlerin %45'ini sıfırlar; `model.eval()` modunda hiçbir şey yapmaz.",
          en: "In `model.train()` mode it zeroes 45% of the values; in `model.eval()` mode it does nothing.",
        } },
        { at: "x = self.conv2(x)           # Convolution Layer 2", text: {
          tr: "`(N, 32, 12, 12)` → `(N, 64, 8, 8)`.",
          en: "`(N, 32, 12, 12)` → `(N, 64, 8, 8)`.",
        } },
        { at: "x = self.max2(x)            # Max Pooling", text: {
          tr: "`(N, 64, 8, 8)` → `(N, 64, 4, 4)`.",
          en: "`(N, 64, 8, 8)` → `(N, 64, 4, 4)`.",
        } },
        { at: "x = torch.flatten(x, start_dim=1)  # Flatten for FC layer", text: {
          tr: "`(N, 64, 4, 4)` → `(N, 1024)`. `start_dim=1` batch boyutunu korur, gerisini tek bir vektöre açar.",
          en: "`(N, 64, 4, 4)` → `(N, 1024)`. `start_dim=1` keeps the batch dimension and unrolls the rest into a single vector.",
        } },
        { at: "x = F.relu(self.fc1(x))     # Fully Connected Layer 1 + ReLU", text: {
          tr: "`(N, 1024)` → `(N, 512)`, ardından ReLU.",
          en: "`(N, 1024)` → `(N, 512)`, followed by ReLU.",
        } },
        { at: "x = self.fc2(x)             # Fully Connected Layer 2 (logits)", text: {
          tr: "`(N, 512)` → `(N, 10)`: ham skorlar (logit).",
          en: "`(N, 512)` → `(N, 10)`: the raw scores (logits).",
        } },
      ] } },
      { callout: "warn", title: { tr: "Neden sonda softmax yok?", en: "Why is there no softmax at the end?" }, text: {
        tr: "`nn.CrossEntropyLoss` girdiyi **logit** olarak bekler ve softmax'ı (log-softmax olarak) kendi içinde uygular. Modelin sonuna softmax eklemek iki kez softmax uygulamak olurdu: hem yanlış hem de sayısal olarak kararsız. Olasılık gerektiğinde, tahmin anında `torch.softmax(logits, dim=1)` kullanılır (Adım 15).",
        en: "`nn.CrossEntropyLoss` expects its input as **logits** and applies softmax (as log-softmax) internally. Adding a softmax at the end of the model would apply softmax twice: both wrong and numerically unstable. When probabilities are needed, use `torch.softmax(logits, dim=1)` at prediction time (Step 15).",
      } },
      { p: {
        tr: "Boyutları tahmin etmek yerine sıfırlardan oluşan sahte bir görüntüyü katmanlardan geçirerek doğrulayabiliriz. Yeni bir mimari yazarken `fc1_in`'i bulmanın en güvenli yolu budur:",
        en: "Instead of guessing the shapes, we can check them by passing a dummy image of zeros through the layers. When writing a new architecture, this is the safest way to find `fc1_in`:",
      } },
      { code: {
        tr: String.raw`x = torch.zeros(1, 1, 28, 28)            # sahte bir görüntü
for layer in [model.conv1, model.max1, model.conv2, model.max2]:
    x = layer(x)
    print(f"{layer.__class__.__name__:10s} -> {tuple(x.shape)}")
x = torch.flatten(x, start_dim=1)
print(f"{'flatten':10s} -> {tuple(x.shape)}")`,
        en: String.raw`x = torch.zeros(1, 1, 28, 28)            # a dummy image
for layer in [model.conv1, model.max1, model.conv2, model.max2]:
    x = layer(x)
    print(f"{layer.__class__.__name__:10s} -> {tuple(x.shape)}")
x = torch.flatten(x, start_dim=1)
print(f"{'flatten':10s} -> {tuple(x.shape)}")`,
      }, name: { tr: "boyutlar.py", en: "shapes.py" } },
      { term: { tr: "Çıktı", en: "Output" }, lines: ["Conv2d     -> (1, 32, 24, 24)", "MaxPool2d  -> (1, 32, 12, 12)", "Conv2d     -> (1, 64, 8, 8)", "MaxPool2d  -> (1, 64, 4, 4)", "flatten    -> (1, 1024)"] },
      { callout: "tip", title: { tr: "Canlı görmek için", en: "See it live" }, text: {
        tr: "[[infer|Çıkarım]] sekmesindeki \"Katman katman ağ\" bölümü tam olarak bu adımların çıktısını, gerçek bir görüntü için gösterir.",
        en: "The \"Layer by layer\" section of the [[infer|Inference]] tab shows exactly the output of these steps for a real image.",
      } },
    ],
  },
  {
    id: "c-setup-train", title: { tr: "Kayıp fonksiyonu ve optimizer", en: "Loss function and optimizer" },
    blocks: [
      { p: {
        tr: "Eğitime başlamadan önce üç şey gerekir: modelin çalışacağı **cihaz**, tahminin ne kadar yanlış olduğunu ölçen **kayıp fonksiyonu** ve ağırlıkları güncelleyecek **optimizer**. Bunları tek bir yardımcı fonksiyonda kuruyoruz.",
        en: "Before training can start we need three things: the **device** the model runs on, a **loss function** that measures how wrong a prediction is, and an **optimizer** that updates the weights. We set them up in a single helper function.",
      } },
      { file: { from: "def _setup_model_for_training(", to: "return model, loss_function, optimizer, device", fold: true, notes: [
        { at: "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')", text: {
          tr: "Ekran kartı varsa GPU, yoksa CPU. Kodun geri kalanı ikisinde de aynı çalışır.",
          en: "The GPU if there is one, otherwise the CPU. The rest of the code works the same on both.",
        } },
        { at: "model = SimpleCNN(num_classes=num_classes).to(device)", text: {
          tr: "Modeli oluşturur ve tüm ağırlıklarını seçilen cihaza taşır. Model ve veri **aynı cihazda** olmalıdır; aksi halde hata alırsınız.",
          en: "Creates the model and moves all of its weights to the chosen device. The model and the data must be **on the same device**; otherwise you get an error.",
        } },
        { at: "loss_function = nn.CrossEntropyLoss().to(device)", text: {
          tr: "Sınıflandırma kaybı: logitlere softmax uygular ve doğru sınıfın olasılığının negatif logaritmasını alır. Batch ortalamasını döndürür. MLP bölümündeki `BCEWithLogitsLoss`'un 10 sınıflı karşılığıdır: o sigmoid'i, bu softmax'ı kayıpla birleştirir; logitlere göre gradyanı da \\(\\hat y - y\\) yerine \\(\\mathbf p - \\mathbf y\\) olur. Etiketler one-hot değil, doğrudan sınıf numarası (0–9) olarak verilir.",
          en: "The classification loss: it applies softmax to the logits and takes the negative logarithm of the correct class's probability. It returns the batch average. It is the 10-class counterpart of the MLP chapter's `BCEWithLogitsLoss`: that one merges the sigmoid into the loss, this one the softmax, and the gradient with respect to the logits becomes \\(\\mathbf p - \\mathbf y\\) instead of \\(\\hat y - y\\). The labels are given as class numbers (0–9), not one-hot vectors.",
        } },
        { at: "optimizer = optim.Adam(model.parameters(), lr=lr)", text: {
          tr: "`model.parameters()` modeldeki tüm öğrenilen ağırlıkları (582.218 sayı) optimizer'a verir. Adam her ağırlık için adım boyunu ayrı ayarlar.",
          en: "`model.parameters()` hands all of the model's learned weights (582,218 numbers) to the optimizer. Adam adapts the step size separately for each weight.",
        } },
      ] } },
      { callout: "info", title: { tr: "Kavramlar", en: "Concepts" }, text: {
        tr: "Çapraz entropinin neden \\(-\\log p\\) olduğu ve SGD, Momentum ile Adam'ın farkı için: [[theory:t-softmax|Teori → Softmax ve kayıp]] ile [[theory:t-opt|Teori → Optimizasyon]]. Gradyanın türetilmesi ve Adam'ın güncelleme kuralı: [[math:m-softmax|Matematik → Softmax ve çapraz entropi]] ile [[math:m-opt|Matematik → Optimizasyon]].",
        en: "For why cross-entropy is \\(-\\log p\\) and how SGD, Momentum and Adam differ, see [[theory:t-softmax|Theory → Softmax and loss]] and [[theory:t-opt|Theory → Optimization]]. For the derivation of the gradient and Adam's update rule: [[math:m-softmax|Math → Softmax and cross-entropy]] and [[math:m-opt|Math → Optimization]].",
      } },
    ],
  },
  {
    id: "c-train-epoch", title: { tr: "Bir epoch eğitim", en: "Training for one epoch" },
    blocks: [
      { p: {
        tr: "Eğitimin kalbi bu fonksiyondur. Eğitim verisindeki her batch için aynı **beş adım** tekrarlanır: gradyanları sıfırla, ileri yayılım, kaybı hesapla, geri yayılım, ağırlıkları güncelle. Kalan satırlar yalnızca doğruluğu ve ortalama kaybı hesaplamak içindir.",
        en: "This function is the heart of training. For every batch of training data the same **five steps** repeat: zero the gradients, forward pass, compute the loss, backward pass, update the weights. The remaining lines only compute the accuracy and the average loss.",
      } },
      { p: {
        tr: "Bu beş adımı MLP bölümündeki PyTorch döngüsünden tanıyorsunuz; orada `zero_grad` kayıptan sonra geliyordu, burada en başta. `backward`'dan önce çağrıldığı sürece yeri fark etmez. Yeni olanlar: veriyi bir `DataLoader`'dan batch batch almak, her batch'i `.to(device)` ile GPU'ya taşımak ve döngünün başındaki `model.train()`.",
        en: "You know these five steps from the MLP chapter's PyTorch loop; there `zero_grad` came after the loss, here it comes first. As long as it is called before `backward`, its position does not matter. What is new: taking the data batch by batch from a `DataLoader`, moving every batch to the GPU with `.to(device)`, and the `model.train()` at the start of the loop.",
      } },
      { file: { from: "def train_epoch(", to: "return epoch_accuracy, epoch_loss", fold: true, notes: [
        { at: "model.train()  # Training mode", text: {
          tr: "**Eğitim modu**: Dropout etkin, BatchNorm o batch'in istatistiklerini kullanır ve hareketli ortalamaları günceller.",
          en: "**Training mode**: Dropout is active, and BatchNorm uses the current batch's statistics and updates its running averages.",
        } },
        { at: "for inputs, labels in data_loader:", text: {
          tr: "Her turda bir batch: `inputs` `(256, 1, 28, 28)`, `labels` `(256,)`. 188 tur = 1 epoch.",
          en: "One batch per iteration: `inputs` is `(256, 1, 28, 28)`, `labels` is `(256,)`. 188 iterations = 1 epoch.",
        } },
        { at: "inputs, labels = inputs.to(device), labels.to(device)", text: {
          tr: "Veriyi modelin bulunduğu cihaza taşı.",
          en: "Move the data to the device the model is on.",
        } },
        { at: "optimizer.zero_grad()", text: {
          tr: "**1. Gradyanları sıfırla.** PyTorch gradyanları biriktirir; sıfırlanmazsa her adımın gradyanı öncekilerin üstüne eklenir.",
          en: "**1. Zero the gradients.** PyTorch accumulates gradients; if they aren't reset, each step's gradient is added on top of the previous ones.",
        } },
        { at: "outputs = model(inputs)", text: {
          tr: "**2. İleri yayılım.** `model(inputs)` aslında `forward`'u çağırır. Çıktı `(256, 10)` logit.",
          en: "**2. Forward pass.** `model(inputs)` actually calls `forward`. The output is `(256, 10)` logits.",
        } },
        { at: "loss = criterion(outputs, labels)", text: {
          tr: "**3. Kayıp.** Tahminin ne kadar yanlış olduğunu ölçen tek bir sayı.",
          en: "**3. Loss.** A single number that measures how wrong the prediction is.",
        } },
        { at: "loss.backward()", text: {
          tr: "**4. Geri yayılım.** Kaybın her ağırlığa göre gradyanını zincir kuralıyla hesaplar ve ağırlıkların `.grad` alanına **ekler** (üzerine yazmaz; 1. adımın nedeni budur). MLP bölümünde elle yazdığınız `backward`'ı, her katman türü için [[math:m-backprop|Matematik → Geri yayılım]] tablosundaki kuralla otomatik yapar.",
          en: "**4. Backward pass.** Uses the chain rule to compute the gradient of the loss with respect to every weight and **adds** it to each weight's `.grad` field (it does not overwrite it; that is the reason for step 1). It does automatically what you wrote by hand as `backward` in the MLP chapter, using the rule for each kind of layer from the [[math:m-backprop|Math → Backpropagation]] table.",
        } },
        { at: "optimizer.step()", text: {
          tr: "**5. Güncelleme.** Adam, gradyanları kullanarak tüm ağırlıkları bir adım değiştirir.",
          en: "**5. Update.** Adam uses the gradients to move all the weights one step.",
        } },
        { at: "total_train_loss += loss.item()", text: {
          tr: "`.item()` tek elemanlı tensörü sıradan bir Python sayısına çevirir.",
          en: "`.item()` turns a one-element tensor into an ordinary Python number.",
        } },
        { at: "_, predicted = outputs.max(1)", text: {
          tr: "Her satırdaki en büyük logitin indeksi, tahmin edilen rakamdır. `max(1)` (değer, indeks) döndürür; değeri `_` ile atarız.",
          en: "The index of the largest logit in each row is the predicted digit. `max(1)` returns (value, index); we discard the value with `_`.",
        } },
        { at: "correct_train += predicted.eq(labels).sum().item()", text: {
          tr: "Doğru tahmin sayısı: `eq` eşit olanlarda `True` verir, `sum` bunları sayar.",
          en: "The number of correct predictions: `eq` gives `True` where the values are equal, and `sum` counts them.",
        } },
      ] } },
      { callout: "tip", engine: "browser", title: { tr: "Bu beş adımı canlı izleyin", en: "Watch these five steps live" }, text: {
        tr: "[[train|Eğitim]] sekmesindeki \"Her batch'te çalışan eğitim adımı\" kartı tam olarak bu beş satırı gösterir. Orada eğitim tarayıcıda çalışır ve TensorFlow.js aynı beş adımı tek bir çağrıda (`model.trainOnBatch`) yapar.",
        en: "The \"The training step run on every batch\" card in the [[train|Training]] tab shows exactly these five lines. There training runs in the browser, and TensorFlow.js performs the same five steps in a single call (`model.trainOnBatch`).",
      } },
      { callout: "tip", engine: "server", title: { tr: "Bu beş adımı canlı izleyin", en: "Watch these five steps live" }, text: {
        tr: "[[train|Eğitim]] sekmesindeki \"Her batch'te çalışan eğitim adımı\" kartı tam olarak bu beş satırı gösterir.",
        en: "The \"The training step run on every batch\" card in the [[train|Training]] tab shows exactly these five lines.",
      } },
      { callout: "warn", title: { tr: "Sık yapılan hata", en: "A common mistake" }, text: {
        tr: "`optimizer.zero_grad()` satırını unutmak hata vermez ama model kötü öğrenir: gradyanlar her adımda birikerek büyür. Programın çökmediği için fark edilmesi zor bir hatadır.",
        en: "Forgetting `optimizer.zero_grad()` raises no error, but the model learns badly: the gradients keep accumulating and growing at every step. Because the program doesn't crash, this bug is hard to notice.",
      } },
    ],
  },
  {
    id: "c-evaluate", title: { tr: "Değerlendirme", en: "Evaluation" },
    blocks: [
      { p: {
        tr: "Değerlendirme, eğitimin \"yalnızca ölçen\" hâlidir: ileri yayılım ve kayıp aynıdır, ama geri yayılım ve güncelleme yoktur. İki önemli farkı var:",
        en: "Evaluation is the \"measure only\" version of training: the forward pass and the loss are the same, but there is no backward pass and no update. There are two important differences:",
      } },
      { file: { from: "def evaluate_model(", to: "return eval_accuracy, eval_loss", fold: true, notes: [
        { at: "model.eval()  # Evaluation mode", text: {
          tr: "**Değerlendirme modu**: Dropout kapanır, BatchNorm eğitimde biriktirdiği hareketli ortalamaları kullanır.",
          en: "**Evaluation mode**: Dropout is switched off, and BatchNorm uses the running averages it accumulated during training.",
        } },
        { at: "with torch.no_grad():", text: {
          tr: "Gradyan takibini kapatır: hesap grafiği tutulmaz, bellek ve zaman kazanılır. `backward` yapmayacağımız için gerekmez.",
          en: "Turns off gradient tracking: no computation graph is kept, which saves memory and time. We don't need it because we won't call `backward`.",
        } },
        { at: "logits = model(inputs)", text: {
          tr: "Yalnızca ileri yayılım. `zero_grad`, `backward` ve `step` yok, yani ağırlıklar değişmez.",
          en: "Only the forward pass. There is no `zero_grad`, `backward` or `step`, so the weights don't change.",
        } },
      ] } },
      { callout: "warn", title: { tr: "model.eval()'i unutursanız", en: "If you forget model.eval()" }, text: {
        tr: "Dropout değerlendirmede de rastgele değer sıfırlamaya devam eder ve BatchNorm batch istatistiklerini kullanır. Sonuçlar hem kötüleşir hem de her çalıştırmada değişir. `model.train()` / `model.eval()` çiftini her zaman doğru yerde çağırın.",
        en: "Dropout keeps zeroing random values during evaluation, and BatchNorm uses the batch statistics. The results get worse and also change from run to run. Always call the `model.train()` / `model.eval()` pair in the right places.",
      } },
    ],
  },
  {
    id: "c-train-loop", title: { tr: "Eğitim döngüsü", en: "The training loop" },
    blocks: [
      { p: {
        tr: "Şimdi iki fonksiyonu birleştiriyoruz: her epoch'ta önce bir tur eğitim, ardından doğrulama. Kayıpları listelerde saklıyoruz ki sonunda grafiğini çizebilelim.",
        en: "Now we combine the two functions: in every epoch, first one round of training, then validation. We keep the losses in lists so we can plot them at the end.",
      } },
      { file: { from: "def train_model(", to: "return train_loss, validation_loss", fold: true, notes: [
        { at: "for epoch in range(num_epochs):", text: {
          tr: "10 epoch, her biri tüm eğitim verisinden bir geçiş.",
          en: "10 epochs, each one a full pass over the training data.",
        } },
        { at: "train_accuracy, epoch_train_loss = train_epoch(model,", text: {
          tr: "Adım 10'daki fonksiyon: bir epoch eğitir, doğruluğu ve ortalama kaybı döndürür.",
          en: "The function from Step 10: it trains for one epoch and returns the accuracy and the average loss.",
        } },
        { at: "validation_accuracy, epoch_validation_loss = evaluate_model(model,", text: {
          tr: "Adım 11'deki fonksiyon, bu sefer doğrulama verisiyle: modelin görmediği veride ne kadar başarılı?",
          en: "The function from Step 11, this time with the validation data: how well does the model do on data it hasn't seen?",
        } },
        { at: "print(f\"Epoch {epoch + 1}: Train Loss: {train_loss[epoch]:.4f}, \"", text: {
          tr: "f-string: `{değer:.4f}` sayıyı 4 ondalık basamakla yazar. Alt alta yazılan f-string'leri Python otomatik birleştirir.",
          en: "f-string: `{value:.4f}` prints the number with 4 decimal places. Python automatically joins f-strings written one under the other.",
        } },
      ] } },
      { callout: "info", title: { tr: "Neyi izliyoruz?", en: "What are we watching?" }, text: {
        tr: "Eğitim kaybı düşerken doğrulama kaybı da düşüyorsa model öğreniyor demektir. Eğitim kaybı düşmeye devam ederken doğrulama kaybı **yükselmeye** başlarsa aşırı uyum (overfitting) başlamıştır: model eğitim verisini ezberliyordur; o noktada eğitimi durdurmak gerekir.",
        en: "If the validation loss falls along with the training loss, the model is learning. If the training loss keeps falling while the validation loss starts to **rise**, the model has started to memorize (overfitting); that is the point to stop training.",
      } },
    ],
  },
  {
    id: "c-plot", title: { tr: "Kayıp grafiği", en: "The loss plot" },
    blocks: [
      { p: {
        tr: "Sayıları tek tek okumak yerine eğrileri yan yana görmek, eğitimin gidişatını anlamanın en hızlı yoludur.",
        en: "Seeing the curves side by side, instead of reading the numbers one by one, is the quickest way to understand how training is going.",
      } },
      { file: { from: "def plot_training_losses(", to: "plt.show()", fold: true, notes: [
        { at: "eps = range(1, len(train_loss_epochs) + 1)", text: {
          tr: "x ekseni: 1'den başlayan epoch numaraları.",
          en: "The x-axis: epoch numbers starting from 1.",
        } },
        { at: "plt.plot(eps, train_loss_epochs, linestyle='-', color='#1f77b4',", text: {
          tr: "Eğitim kaybı eğrisi (mavi).",
          en: "The training loss curve (blue).",
        } },
        { at: "plt.plot(eps, validation_loss_epochs, linestyle='-', color='#d62728',", text: {
          tr: "Doğrulama kaybı eğrisi (kırmızı).",
          en: "The validation loss curve (red).",
        } },
        { at: "plt.show()", text: {
          tr: "Grafiği bir pencerede açar; program pencere kapanana kadar bekler.",
          en: "Opens the plot in a window; the program waits until the window is closed.",
        } },
      ] } },
    ],
  },
  {
    id: "c-main", title: { tr: "Hepsini birleştirmek", en: "Putting it all together" },
    blocks: [
      { p: {
        tr: "Son olarak bütün parçaları doğru sırayla çağıran ana bölümü yazıyoruz. Dosya çalıştırıldığında olan her şey burada.",
        en: "Finally we write the main section, which calls all the pieces in the right order. Everything that happens when the file is run is here.",
      } },
      { file: { from: "if __name__ == \"__main__\":", to: "plot_training_losses(train_losses, validation_losses)", notes: [
        { at: "if __name__ == \"__main__\":", text: {
          tr: "Bu blok yalnızca dosya doğrudan çalıştırıldığında (`python SimpleCNN.py`) çalışır; başka bir dosya `import SimpleCNN` yaptığında çalışmaz. Repodaki Python paneli `dashboard.py` bu sayede buradaki fonksiyonları yeniden kullanabiliyor.",
          en: "This block only runs when the file is executed directly (`python SimpleCNN.py`), not when another file does `import SimpleCNN`. That is what lets the repository's Python dashboard, `dashboard.py`, reuse the functions defined here.",
        } },
        { at: "set_seed(seed)", text: {
          tr: "Önce seed. Sıra önemli: modelin ilk ağırlıkları ve veri bölünmesi rastgele sayı tükettiği için seed onlardan önce sabitlenmeli.",
          en: "The seed comes first. The order matters: the model's initial weights and the data split both consume random numbers, so the seed must be fixed before them.",
        } },
        { at: "cnn_model, loss_fn, adam_optimizer, h_device = _setup_model_for_training(", text: {
          tr: "Model, kayıp, optimizer ve cihaz (Adım 9).",
          en: "Model, loss, optimizer and device (Step 9).",
        } },
        { at: "train_loader, val_loader, test_loader = get_mnist_dataloaders(", text: {
          tr: "Veri (Adım 6).",
          en: "Data (Step 6).",
        } },
        { at: "train_losses, validation_losses = train_model(cnn_model, loss_fn, adam_optimizer,", text: {
          tr: "Eğitim ve doğrulama (Adım 12).",
          en: "Training and validation (Step 12).",
        } },
        { at: "test_accuracy, test_loss = evaluate_model(cnn_model, loss_fn,", text: {
          tr: "Test: model hazır olduktan sonra, **yalnızca bir kez**.",
          en: "Test: once the model is ready, and **only once**.",
        } },
        { at: "plot_training_losses(train_losses, validation_losses)", text: {
          tr: "Kayıp grafiği (Adım 13).",
          en: "The loss plot (Step 13).",
        } },
      ] } },
      { p: {
        tr: "Çalıştıralım. İlk seferde MNIST indirilir (birkaç saniye), ardından eğitim başlar. RTX 2060'ta yaklaşık 1,5–2 dakika sürer:",
        en: "Let's run it. The first time, MNIST is downloaded (a few seconds), then training starts. On an RTX 2060 it takes about 1.5–2 minutes:",
      } },
      { term: "PowerShell", lines: ["$ .venv\\Scripts\\python.exe SimpleCNN.py", ...TRAIN_OUTPUT] },
      { callout: "info", title: { tr: "Sonucu yorumlamak", en: "Interpreting the result" }, text: {
        tr: "İlk epoch'ta bile doğrulama doğruluğu %97,7. Sonraki epoch'lar küçük ama düzenli iyileştirmeler getiriyor. Eğitim kaybının doğrulamadan yüksek olması bir hata değil, iki nedeni var: eğitimde Dropout açık, doğrulamada kapalı; ayrıca eğitim kaybı, ağırlıklar epoch boyunca hâlâ iyileşirken alınan bir ortalamadır, doğrulama ise epoch sonundaki modelle ölçülür. Doğrulama kaybı 10 epoch boyunca (7. epoch'taki küçük bir dalgalanma dışında) düşmeye devam ediyor, yani aşırı uyum belirtisi yok. Test doğruluğu (%99,29) da doğrulamaya (%99,36) çok yakın: doğrulama sonucu, hiç görülmemiş verideki başarının iyi bir tahminiymiş.",
        en: "Even after the first epoch the validation accuracy is 97.7%. The following epochs bring small but steady improvements. The training loss being higher than the validation loss is not a bug, and there are two reasons for it: Dropout is on during training and off during validation, and the training loss is an average taken while the weights are still improving during the epoch, whereas validation is measured with the model at the end of the epoch. The validation loss keeps falling over the 10 epochs (apart from a tiny wobble at epoch 7), so there is no sign of overfitting. The test accuracy (99.29%) is also very close to the validation accuracy (99.36%): the validation result was a good estimate of the performance on never-seen data.",
      } },
      { callout: "tip", title: { tr: "Özet: MLP bölümünden ne değişti?", en: "Recap: what changed since the MLP chapter?" }, text: {
        tr: "Eğitim döngüsü aynı: ileri yayılım, kayıp, `backward`, `step`. Değişenler: **model** (evrişim, havuzlama, BatchNorm, Dropout ve sonda bir MLP), **kayıp** (10 sınıflı `CrossEntropyLoss`), **optimizer** (Adam), **veri** (`DataLoader` ile 188 batch'lik epoch'lar), **cihaz** (`.to(device)`) ve **ölçüm** (`model.eval()`, `torch.no_grad()`, ayrı doğrulama ve test setleri).",
        en: "The training loop is the same: forward pass, loss, `backward`, `step`. What changed: the **model** (convolution, pooling, BatchNorm, Dropout and an MLP at the end), the **loss** (10-class `CrossEntropyLoss`), the **optimizer** (Adam), the **data** (epochs of 188 batches from a `DataLoader`), the **device** (`.to(device)`) and the **measurement** (`model.eval()`, `torch.no_grad()`, separate validation and test sets).",
      } },
    ],
  },
  {
    id: "c-predict", title: { tr: "Ek: kaydetmek ve tahmin yapmak", en: "Extra: saving and predicting" },
    blocks: [
      { engine: "browser", p: {
        tr: "`SimpleCNN.py` eğitimden sonra modeli saklamaz. Eğitilmiş bir modeli tekrar kullanmak için ağırlıklarını kaydedip başka bir programda yükleyebilirsiniz. Repodaki Python paneli `dashboard.py` tam olarak bunu yapıyor; bu sayfa ise tarayıcıda eğittiğiniz modeli tarayıcınızda saklar.",
        en: "`SimpleCNN.py` does not keep the model after training. To reuse a trained model, you can save its weights and load them in another program. The repository's Python dashboard, `dashboard.py`, does exactly that; this page instead keeps the model you train here in your browser.",
      } },
      { engine: "server", p: {
        tr: "`SimpleCNN.py` eğitimden sonra modeli saklamaz. Eğitilmiş bir modeli tekrar kullanmak için ağırlıklarını kaydedip başka bir programda yükleyebilirsiniz. Bu sayfayı sunan Python paneli `dashboard.py` tam olarak bunu yapıyor: eğittiğiniz modeli `checkpoints/simple_cnn.pt` dosyasına kaydeder.",
        en: "`SimpleCNN.py` does not keep the model after training. To reuse a trained model, you can save its weights and load them in another program. The Python dashboard serving this page, `dashboard.py`, does exactly that: it saves the model you train to `checkpoints/simple_cnn.pt`.",
      } },
      { code: {
        tr: String.raw`# Eğitimden sonra: yalnızca ağırlıkları (state_dict) kaydet
torch.save(cnn_model.state_dict(), "simple_cnn.pt")

# Başka bir programda: aynı mimariyi kur, ağırlıkları yükle
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn.pt", map_location="cpu"))
model.eval()

image, label = test_loader.dataset[0]        # (1, 28, 28) tensör ve etiket
with torch.no_grad():
    logits = model(image.unsqueeze(0))       # batch boyutu ekle: (1, 1, 28, 28)
    probs = torch.softmax(logits, dim=1)[0]  # logit -> olasılık
print("Tahmin:", probs.argmax().item(), "| Gerçek:", label, f"| Güven: %{probs.max().item() * 100:.1f}")`,
        en: String.raw`# After training: save only the weights (state_dict)
torch.save(cnn_model.state_dict(), "simple_cnn.pt")

# In another program: build the same architecture, load the weights
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn.pt", map_location="cpu"))
model.eval()

image, label = test_loader.dataset[0]        # (1, 28, 28) tensor and its label
with torch.no_grad():
    logits = model(image.unsqueeze(0))       # add a batch dimension: (1, 1, 28, 28)
    probs = torch.softmax(logits, dim=1)[0]  # logits -> probabilities
print("Prediction:", probs.argmax().item(), "| True:", label, f"| Confidence: {probs.max().item() * 100:.1f}%")`,
      }, name: { tr: "tahmin.py", en: "predict.py" }, notes: [
        { at: "torch.save(cnn_model.state_dict(), \"simple_cnn.pt\")", text: {
          tr: "`state_dict` katman adlarını ağırlık tensörlerine eşleyen bir sözlüktür. Modelin kodu değil, yalnızca sayıları kaydedilir.",
          en: "`state_dict` is a dictionary that maps layer names to weight tensors. Only the numbers are saved, not the model's code.",
        } },
        { at: "model.load_state_dict(torch.load(\"simple_cnn.pt\", map_location=\"cpu\"))", text: {
          tr: "Önce aynı sınıftan boş bir model kurulur, sonra ağırlıklar yüklenir. `map_location=\"cpu\"` GPU'da kaydedilmiş modeli GPU'suz bir bilgisayarda açmayı sağlar.",
          en: "First a fresh model of the same class is built, then the weights are loaded into it. `map_location=\"cpu\"` lets you open a model saved on a GPU on a computer without one.",
        } },
        { at: "model.eval()", text: {
          tr: "Tahminden önce mutlaka değerlendirme moduna alın (Adım 11).",
          en: "Always switch to evaluation mode before predicting (Step 11).",
        } },
        { at: {
          tr: "logits = model(image.unsqueeze(0))       # batch boyutu ekle: (1, 1, 28, 28)",
          en: "logits = model(image.unsqueeze(0))       # add a batch dimension: (1, 1, 28, 28)",
        }, text: {
          tr: "Model her zaman bir batch bekler; tek görüntüye `unsqueeze(0)` ile boyu 1 olan bir batch boyutu ekleriz.",
          en: "The model always expects a batch; `unsqueeze(0)` adds a batch dimension of size 1 to the single image.",
        } },
        { at: {
          tr: "probs = torch.softmax(logits, dim=1)[0]  # logit -> olasılık",
          en: "probs = torch.softmax(logits, dim=1)[0]  # logits -> probabilities",
        }, text: {
          tr: "Olasılık istediğimiz yer burası: logitlere softmax uygulayıp toplamı 1 olan 10 olasılık elde ederiz.",
          en: "This is where we want probabilities: applying softmax to the logits gives 10 probabilities that sum to 1.",
        } },
      ] },
      { term: { tr: "Çıktı", en: "Output" }, lines: [{ tr: "Tahmin: 7 | Gerçek: 7 | Güven: %100.0", en: "Prediction: 7 | True: 7 | Confidence: 100.0%" }] },
      { callout: "tip", title: { tr: "Çizdiğiniz rakamla deneyin", en: "Try it with a digit you draw" }, text: {
        tr: "[[infer|Çıkarım]] sekmesi bunu sizin çiziminizle yapar. Çizim önce MNIST'teki gibi 20×20'lik bir kutuya sığdırılıp ortalanır; aksi halde model, eğitimde gördüğünden farklı bir görüntüyle karşılaşırdı.",
        en: "The [[infer|Inference]] tab does this with your own drawing. The drawing is first scaled to fit a 20×20 box and centered, as in MNIST; otherwise the model would face an image unlike the ones it saw in training.",
      } },
    ],
  },
  {
    id: "c-full", title: { tr: "Tam dosya", en: "The full file" },
    blocks: [
      { p: {
        tr: "Yazdığımız her şey bir arada. Dosya repodan canlı okunuyor; `SimpleCNN.py`'yi değiştirirseniz burası (ve önceki adımlar) da güncellenir.",
        en: "Everything we wrote, all in one place. The file is read live from the repository; if you change `SimpleCNN.py`, this view (and the previous steps) update too.",
      } },
      { file: { whole: true, tall: true } },
    ],
  },
  {
    id: "c-exercises", title: { tr: "Alıştırmalar", en: "Exercises" },
    blocks: [
      { p: {
        tr: "Kodu gerçekten anlamanın en iyi yolu değiştirip ne olduğunu görmektir. Her alıştırmadan sonra kayıp grafiğine ve test doğruluğuna bakın. Alıştırmalar kolaydan zora sıralı: ★ bir sayıyı değiştirip sonucu yorumlamak, ★★ modeli değiştirip sonucu önce elle hesaplamak, ★★★ yeni kod yazmak.",
        en: "The best way to really understand the code is to change it and see what happens. After each exercise, look at the loss plot and the test accuracy. The exercises go from easy to hard: ★ change a number and interpret the result, ★★ change the model and work out the result by hand first, ★★★ write new code.",
      } },
      { list: [
        {
          tr: "★ **Öğrenme oranı**: `learning_rate`'i `1e-2` ve `1e-4` yapın. Hangisi daha hızlı öğreniyor, hangisi kararsız? [[theory:t-opt|Teori → Optimizasyon]] ile karşılaştırın. *İpucu:* ilk 2–3 epoch'un doğrulama kaybına bakmak yeterli. Adam'da her ağırlığın adımı yaklaşık η'dır ([[math:m-opt|Matematik → Optimizasyon]]), yani η'yı 10 kat büyütmek adımı da yaklaşık 10 kat büyütür.",
          en: "★ **Learning rate**: set `learning_rate` to `1e-2` and to `1e-4`. Which one learns faster, and which one is unstable? Compare with [[theory:t-opt|Theory → Optimization]]. *Hint:* looking at the validation loss of the first 2–3 epochs is enough. With Adam every weight's step is about η ([[math:m-opt|Math → Optimization]]), so making η 10 times larger makes the step about 10 times larger too.",
        },
        {
          tr: "★ **Dropout'suz**: iki dropout oranını da `0` yapın. Eğitim ve doğrulama kaybı arasındaki fark nasıl değişiyor? *İpucu:* şu anki çıktıda eğitim kaybı doğrulamadan yüksek (Adım 14) ve Dropout bunun iki nedeninden biriydi. Kapatınca eğitim kaybı hangi yöne kayıyor?",
          en: "★ **No dropout**: set both dropout rates to `0`. How does the gap between the training and validation loss change? *Hint:* in the current output the training loss is higher than the validation loss (Step 14), and Dropout was one of the two reasons. Which way does the training loss move when it is switched off?",
        },
        {
          tr: "★ **Optimizer**: Adam yerine `optim.SGD(model.parameters(), lr=0.05, momentum=0.9)` deneyin. *İpucu:* SGD'nin adımı gradyanın büyüklüğüyle orantılıdır, Adam'ınki değildir; bu yüzden SGD için `lr` Adam'dakinden çok daha büyük seçilir. `momentum=0.9`, yönü tutarlı bileşenlerde adımı yaklaşık 10 kat büyütür ([[math:m-opt|Matematik → Optimizasyon]]).",
          en: "★ **Optimizer**: try `optim.SGD(model.parameters(), lr=0.05, momentum=0.9)` instead of Adam. *Hint:* SGD's step is proportional to the size of the gradient, Adam's is not; that is why SGD gets a much larger `lr` than Adam. `momentum=0.9` makes the step about 10 times larger in consistent directions ([[math:m-opt|Math → Optimization]]).",
        },
        {
          tr: "★★ **Daha geniş model**: `conv1_out_channels`'ı 64 yapın. Parametre sayısı ve süre nasıl değişiyor? (`fc1_in` neden değişmiyor?) *İpucu:* önce elle hesaplayın: \\(P = C_{\\text{out}}(C_{\\text{in}}k^2 + 1)\\) ile Conv1, BatchNorm1 ve Conv2 değişir. Sonucu Adım 7'deki `parametreler.py` ile kontrol edin (doğru cevap: 634.314).",
          en: "★★ **A wider model**: set `conv1_out_channels` to 64. How do the parameter count and the training time change? (Why doesn't `fc1_in` change?) *Hint:* work it out by hand first: with \\(P = C_{\\text{out}}(C_{\\text{in}}k^2 + 1)\\), Conv1, BatchNorm1 and Conv2 change. Check your result with `parameters.py` from Step 7 (the answer: 634,314).",
        },
        {
          tr: "★★ **BatchNorm'suz**: `forward`'da `self.batch1(x)` ve `self.batch2(x)` çağrılarını kaldırın (`F.relu(x)` kalsın). Eğitim hızı ve Conv1 filtreleri nasıl değişiyor? *İpucu:* [[math:m-notation|Matematik → Gösterim]] bölümündeki \"Başlangıç ağırlıkları\" paragrafına bakın: PyTorch'un varsayılan başlangıcıyla, BatchNorm olmadan sinyal her katmanda küçülür. En büyük farkı ilk epoch'ta arayın. Fark beklediğinizden küçük olabilir; neden? (Modelde yalnızca iki evrişim katmanı var ve Adam'ın adımı gradyanın büyüklüğüne bağlı değildir.)",
          en: "★★ **No BatchNorm**: remove the `self.batch1(x)` and `self.batch2(x)` calls in `forward` (keep `F.relu(x)`). How do the training speed and the Conv1 filters change? *Hint:* see the \"Initial weights\" paragraph of [[math:m-notation|Math → Notation]]: with PyTorch's default initialization and no BatchNorm, the signal shrinks at every layer. Look for the biggest difference in the first epoch. The difference may be smaller than you expect; why? (The model has only two convolution layers, and Adam's step does not depend on the size of the gradient.)",
        },
        {
          tr: "★★ **MLP ile karşılaştırma**: evrişim bloklarını kaldırıp MLP bölümündeki gibi bir ağ kurun: `torch.flatten` → `nn.Linear(784, 512)` → ReLU → `nn.Linear(512, 10)`. Parametre sayısı ve test doğruluğu nasıl değişiyor? *İpucu:* `fc1_in = 28 * 28` yapın ve evrişim, BatchNorm, havuzlama ve dropout katmanlarını yalnızca `forward`'dan değil `__init__`'ten de silin; yoksa kullanılmayan katmanlar da parametre sayısına girer (doğru cevap: 407.050).",
          en: "★★ **Compare with an MLP**: remove the convolution blocks and build a network like the one in the MLP chapter: `torch.flatten` → `nn.Linear(784, 512)` → ReLU → `nn.Linear(512, 10)`. How do the parameter count and the test accuracy change? *Hint:* set `fc1_in = 28 * 28` and delete the convolution, BatchNorm, pooling and dropout layers not only from `forward` but also from `__init__`; otherwise the unused layers are counted too (the answer: 407,050).",
        },
        {
          tr: "★★★ **Üçüncü blok**: `padding=2` ile 5×5 bir `conv3` (64 → 128) ve 2×2 havuzlama ekleyin. `fc1_in` ne olmalı? Sahte görüntü yöntemiyle (Adım 8) doğrulayın. *İpucu:* Adım 4'teki formülü yeni katmanlara uygulayın: `padding=2` olan 5×5 bir evrişim boyutu değiştirmez, havuzlama yarıya indirir. Yeni katmanları hem `__init__`'e hem `forward`'a ekleyin.",
          en: "★★★ **A third block**: add a 5×5 `conv3` (64 → 128) with `padding=2` and a 2×2 pooling layer. What should `fc1_in` be? Check it with the dummy-image method (Step 8). *Hint:* apply the formula from Step 4 to the new layers: a 5×5 convolution with `padding=2` keeps the size, pooling halves it. Add the new layers to both `__init__` and `forward`.",
        },
        {
          tr: "★★★ **Hatalı tahminler**: test setinde `predicted != labels` olan görüntüleri bulun ve matplotlib ile gösterin. Model hangi rakamları karıştırıyor? *İpucu:* `evaluate_model`'deki döngüyü kopyalayın; `model.eval()` ve `torch.no_grad()` içinde `inputs[predicted != labels]` ile yanlışları toplayın ve her birini `plt.imshow(img[0].cpu(), cmap=\"gray\")` ile, başlığında tahmin ve gerçek etiketle çizin.",
          en: "★★★ **Wrong predictions**: find the test images where `predicted != labels` and display them with matplotlib. Which digits does the model confuse? *Hint:* copy the loop from `evaluate_model`; inside `model.eval()` and `torch.no_grad()`, collect the mistakes with `inputs[predicted != labels]` and draw each one with `plt.imshow(img[0].cpu(), cmap=\"gray\")`, with the prediction and the true label in its title.",
        },
      ], ordered: true },
      { callout: "tip", title: { tr: "Seed'i unutmayın", en: "Don't forget the seed" }, text: {
        tr: "Karşılaştırmaların adil olması için her denemede seed'i aynı bırakın; böylece fark yalnızca değiştirdiğiniz şeyden gelir.",
        en: "To keep comparisons fair, leave the seed the same in every experiment; that way any difference comes only from what you changed.",
      } },
    ],
  },
];

// =====================================================================
// Rendering and the tab's lifecycle
// =====================================================================
export const tut = { started: false, rendered: false, source: null, observer: null };

/** Cuts lines from SimpleCNN.py, located by the text of their first and last line. */
export function sourceRange(spec) {
  const all = tut.source.map((text, k) => ({ n: k + 1, text }));
  if (spec.whole) return all;
  const start = all.findIndex((l) => l.text.includes(spec.from));
  const end = start === -1 ? -1 : all.findIndex((l, k) => k >= start && l.text.includes(spec.to));
  if (start === -1 || end === -1) return null;
  return all.slice(start, end + 1);
}

/** Renders one block in the current language (every text field may be a { tr, en } pair). */
export function renderBlock(block) {
  if (block.p) return h("p", {}, rich(block.p));
  if (block.list) {
    const list = h(block.ordered ? "ol" : "ul", { class: "tut-list" });
    for (const item of block.list) list.append(h("li", {}, rich(item)));
    return list;
  }
  if (block.callout) {
    const icons = { tip: "💡", warn: "⚠️", info: "ℹ️" };
    return h("div", { role: "note", class: `tut-callout ${block.callout}` },
      h("span", { class: "tut-callout-icon", text: icons[block.callout], "aria-hidden": "true" }),
      h("div", {}, h("strong", { text: L(block.title) }), h("p", {}, rich(block.text))));
  }
  if (block.file) {
    if (!tut.source) {
      return h("p", { class: "placeholder", text: L({
        tr: "SimpleCNN.py okunamadı. Sayfayı yeniden yükleyin.",
        en: "Could not read SimpleCNN.py. Reload the page.",
      }) });
    }
    const lines = sourceRange(block.file);
    if (!lines) return h("p", { class: "placeholder", text: `${L({ tr: "Kod parçası bulunamadı", en: "Code excerpt not found" })}: ${block.file.from}` });
    return codeEditor(lines, { file: "SimpleCNN.py", fromFile: true, fold: block.file.fold, notes: block.file.notes, tall: block.file.tall });
  }
  if (block.code) {
    const lines = L(block.code).split("\n").map((text, k) => ({ n: k + 1, text }));
    return codeEditor(lines, { file: L(block.name), notes: block.notes, caption: block.caption });
  }
  if (block.term) return terminal(L(block.term), block.lines.map((line) => L(line)));
  if (block.links) {
    // Links to places in other chapters ({ chapter, tab, section }); core.js follows data-go/data-chapter
    return h("div", { class: "m-links" }, h("span", { text: L({ tr: "Ayrıca bakın:", en: "See also:" }) }),
      ...block.links.map((place) => h("a", { href: "#", "data-go": `${place.tab}:${place.section || ""}`, "data-chapter": place.chapter,
        text: `${placeLabel(place)} →` })));
  }
  if (block.table) {
    return h("div", { class: "table-wrap" }, h("table", { class: "data-table tut-table" },
      h("thead", {}, h("tr", {}, ...block.table.head.map((c) => h("th", { text: L(c) })))),
      h("tbody", {}, ...block.table.rows.map((row) => h("tr", {}, ...row.map((c) => h("td", {}, rich(c))))))));
  }
  return null;
}

/** Jumps to a step as a history step, so Back returns to where you were reading. */
export const goToStep = (id) => navigate("code", id);

export function renderTutorial() {
  const main = $("#tut-main");
  const toc = $("#tut-toc");
  main.replaceChildren();
  toc.replaceChildren();
  const title = (step) => L(step.title);
  STEPS.forEach((step, k) => {
    const kicker = L({ tr: `Adım ${k + 1} / ${STEPS.length}`, en: `Step ${k + 1} of ${STEPS.length}` });
    const article = h("article", { class: "card tut-step", id: step.id },
      h("header", { class: "tut-step-head" },
        h("span", { class: "tut-num", text: String(k + 1) }),
        h("div", {}, h("span", { class: "tut-kicker", text: kicker }), h("h2", { text: title(step) }))));
    for (const block of step.blocks) {
      const node = renderBlock(block);
      if (node && block.engine) node.dataset.engine = block.engine;   // true only in the browser or on dashboard.py
      if (node) article.append(node);
    }
    const nav = h("footer", { class: "tut-nav" });
    if (k > 0) {
      const prev = h("button", { type: "button", class: "btn small", text: `← ${title(STEPS[k - 1])}` });
      prev.addEventListener("click", () => goToStep(STEPS[k - 1].id));
      nav.append(prev);
    }
    if (k < STEPS.length - 1) {
      const next = h("button", { type: "button", class: "btn small primary", text: `${title(STEPS[k + 1])} →` });
      next.addEventListener("click", () => goToStep(STEPS[k + 1].id));
      nav.append(next);
    }
    article.append(nav);
    main.append(article);

    const link = h("button", { type: "button", "data-step": step.id },
      h("span", { class: "toc-num", text: String(k + 1) }), h("span", { text: title(step) }));
    link.addEventListener("click", () => goToStep(step.id));
    toc.append(h("li", {}, link));
  });

  // Highlight the step being read in the table of contents
  tut.observer?.disconnect();
  tut.observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const b of toc.querySelectorAll("button")) b.classList.toggle("active", b.dataset.step === entry.target.id);
    }
  }, { rootMargin: "-25% 0px -65% 0px" });
  for (const article of main.querySelectorAll(".tut-step")) tut.observer.observe(article);
  tut.rendered = true;
}

/** The step article nearest the top of the viewport and its current offset (null if the tab is hidden). */
export function readingPosition() {
  const steps = [...document.querySelectorAll("#tut-main .tut-step")];
  if (!steps.length || !steps[0].getClientRects().length) return null;
  const topbar = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--topbar-h")) || 59;
  let anchor = steps[0];
  for (const step of steps) {
    if (step.getBoundingClientRect().top <= topbar + 40) anchor = step;
    else break;
  }
  return { id: anchor.id, top: anchor.getBoundingClientRect().top };
}

export async function startTutorial() {
  if (tut.started) return;
  tut.started = true;
  tut.source = SIMPLECNN_PY.replace(/\r\n/g, "\n").replace(/\n$/, "").split("\n");   // bundled with the page
  renderTutorial();
}

document.addEventListener("tab-shown", (event) => {
  if (event.detail === "code") startTutorial();
});

// Language switch: re-render in the new language and keep the reader at the same step
document.addEventListener("lang-change", () => {
  if (!tut.rendered) return;   // not opened yet (or still loading): the first render uses the new language
  const position = readingPosition();
  renderTutorial();
  if (!position) return;
  const anchor = document.getElementById(position.id);
  if (!anchor) return;
  if (position.top >= 0) {
    // The step's header was on screen: put it back exactly where it was
    window.scrollBy({ top: anchor.getBoundingClientRect().top - position.top, behavior: "instant" });
  } else {
    // Reading inside a long step: the text length changed, so bring the step's start into view
    anchor.scrollIntoView({ behavior: "instant", block: "start" });
  }
});

// The glossary names the Coding tab's steps in its links (shared/glossary/terms.js)
hooks.codeSteps = STEPS;
