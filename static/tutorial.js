"use strict";
/*
 * Coding tab: a step-by-step tutorial that writes SimpleCNN.py from scratch.
 * Code excerpts are cut from the real SimpleCNN.py served by dashboard.py (located by
 * their first and last line, not by line numbers), so line numbers always match the file.
 * Uses the helpers defined in app.js (h, $, api, selectTab, fmtNum).
 */

// =====================================================================
// Python syntax highlighting (small tokenizer, VS Code "Dark+" colors)
// =====================================================================
const PY_CONTROL = new Set(["if", "elif", "else", "for", "while", "return", "with", "try", "except", "finally",
  "raise", "import", "from", "as", "pass", "break", "continue", "yield", "in", "not", "and", "or", "is",
  "assert", "del", "global", "nonlocal", "async", "await"]);
const PY_STORAGE = new Set(["def", "class", "lambda"]);
const PY_CONST = new Set(["True", "False", "None"]);
const PY_BUILTIN = new Set(["print", "range", "len", "int", "float", "str", "list", "dict", "tuple", "set",
  "super", "sum", "min", "max", "abs", "round", "type", "next", "iter", "isinstance", "enumerate", "zip", "open"]);

/** Splits one line into [text, class] tokens. `state` carries open triple-quoted strings and bracket depth. */
function tokenizePython(text, state) {
  const out = [];
  const push = (s, cls) => { if (s) out.push([s, cls]); };
  let i = 0;
  if (state.triple) {
    const end = text.indexOf(state.triple);
    if (end === -1) return [[text, "tok-doc"]];
    push(text.slice(0, end + 3), "tok-doc");
    i = end + 3;
    state.triple = null;
  }
  let previous = "";
  while (i < text.length) {
    const rest = text.slice(i);
    let m;
    if (rest[0] === "#") { push(rest, "tok-com"); break; }
    if ((m = rest.match(/^[rRbBuUfF]{0,2}("""|''')/))) {
      const end = rest.indexOf(m[1], m[0].length);
      if (end === -1) { push(rest, "tok-doc"); state.triple = m[1]; break; }
      push(rest.slice(0, end + 3), "tok-doc");
      i += end + 3;
      continue;
    }
    if ((m = rest.match(/^([rRbBuUfF]{0,2})(["'])/))) {
      let j = m[0].length;
      while (j < rest.length && rest[j] !== m[2]) j += rest[j] === "\\" ? 2 : 1;
      const str = rest.slice(0, j + 1);
      if (/f/i.test(m[1])) {
        // f-string: the {expressions} inside keep normal colors
        for (const part of str.split(/(\{[^{}]*\})/)) {
          if (/^\{[^{}]*\}$/.test(part)) { push("{", "tok-brace"); push(part.slice(1, -1), "tok-var"); push("}", "tok-brace"); }
          else push(part, "tok-str");
        }
      } else {
        push(str, "tok-str");
      }
      i += str.length;
      continue;
    }
    if ((m = rest.match(/^\d[\d_]*(\.\d+)?([eE][+-]?\d+)?/)) && !/\w/.test(text[i - 1] || "")) {
      push(m[0], "tok-num");
      i += m[0].length;
      continue;
    }
    if ((m = rest.match(/^@[\w.]+/))) { push(m[0], "tok-dec"); i += m[0].length; continue; }
    if ((m = rest.match(/^[A-Za-z_]\w*/))) {
      const word = m[0];
      const after = rest.slice(word.length);
      let cls = "tok-var";
      if (previous === "def") cls = "tok-fn";
      else if (previous === "class") cls = "tok-cls";
      else if (PY_STORAGE.has(word) || PY_CONST.has(word) || word === "self") cls = "tok-def";
      else if (PY_CONTROL.has(word)) cls = "tok-kw";
      else if (/^[A-Z]/.test(word) && /[a-z]/.test(word)) cls = "tok-cls";        // CamelCase: a class
      else if (state.depth > 0 && /^\s*=(?!=)/.test(after)) cls = "tok-param";   // keyword argument
      else if (/^\s*\(/.test(after)) cls = PY_BUILTIN.has(word) ? "tok-bi" : "tok-fn";   // a call
      push(word, cls);
      previous = PY_STORAGE.has(word) ? word : "";
      i += word.length;
      continue;
    }
    const ch = rest[0];
    if ("([{".includes(ch)) state.depth++;
    if (")]}".includes(ch)) state.depth = Math.max(0, state.depth - 1);
    push(ch, /[=+\-*/<>%!&|^~]/.test(ch) ? "tok-op" : null);
    i++;
  }
  return out;
}

function highlightInto(node, text, state, lang) {
  if (lang !== "python") {
    node.textContent = text;
    return;
  }
  for (const [part, cls] of tokenizePython(text, state)) {
    node.append(cls ? h("span", { class: cls, text: part }) : document.createTextNode(part));
  }
}

// =====================================================================
// Rich text: `code`, **bold**, *italic*, [[tab:section|link text]]
// =====================================================================
function rich(text) {
  const fragment = document.createDocumentFragment();
  for (const part of text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\])/)) {
    if (!part) continue;
    if (part.startsWith("`")) fragment.append(h("code", { class: "ic", text: part.slice(1, -1) }));
    else if (part.startsWith("**")) fragment.append(h("strong", { text: part.slice(2, -2) }));
    else if (part.startsWith("[[")) {
      const [target, label] = part.slice(2, -2).split("|");
      const [tab, section] = target.split(":");
      const link = h("a", { href: "#", class: "tut-link", text: label });
      link.addEventListener("click", (e) => {
        e.preventDefault();
        selectTab(tab);
        if (section) setTimeout(() => document.getElementById(section)?.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
      });
      fragment.append(link);
    } else if (part.startsWith("*") && part.endsWith("*") && part.length > 2) fragment.append(h("em", { text: part.slice(1, -1) }));
    else fragment.append(part);
  }
  return fragment;
}


// =====================================================================
// Code editor component
// =====================================================================
/**
 * lines: [{ n, text }]; options: { file, lang, notes: [{ at, text }], fold: true (hide the first docstring),
 * caption, tall }. A note's `at` is the line's text (exact match first, then substring).
 */
function codeEditor(lines, options) {
  const lang = options.lang || "python";
  const state = { triple: null, depth: 0 };

  // Notes: find their lines and number them in reading order
  const notes = (options.notes || []).map((note) => {
    let index = lines.findIndex((l) => l.text.trim() === note.at.trim());
    if (index === -1) index = lines.findIndex((l) => l.text.includes(note.at));
    if (index === -1) console.warn("Tutorial note not found:", note.at);
    return { ...note, index };
  }).filter((note) => note.index >= 0).sort((a, b) => a.index - b.index);
  notes.forEach((note, k) => { note.mark = String(k + 1); });

  // A docstring right after the def/class line can be folded away
  let foldStart = -1;
  let foldEnd = -1;
  if (options.fold) {
    foldStart = lines.findIndex((l, k) => k > 0 && /^\s*(r?""")/.test(l.text));
    if (foldStart > 0) {
      const opener = lines[foldStart].text.trim();
      foldEnd = opener.length > 3 && opener.endsWith('"""') ? foldStart
        : lines.findIndex((l, k) => k > foldStart && l.text.includes('"""'));
    }
    if (foldEnd < foldStart) foldStart = -1;
  }

  const body = h("div", { class: "code-body" });
  const rows = [];
  let foldGroup = null;
  lines.forEach((line, k) => {
    const row = h("div", { class: "code-line" });
    const gutter = h("span", { class: "code-gutter", text: line.n != null ? String(line.n) : "" });
    const src = h("span", { class: "code-src" });
    highlightInto(src, line.text || " ", state, lang);
    row.append(gutter, src);
    const note = notes.find((nt) => nt.index === k);
    if (note) {
      row.classList.add("has-note");
      row.append(h("span", { class: "code-mark", text: note.mark, "aria-hidden": "true" }));
    }
    rows[k] = row;
    if (k === foldStart) {
      // Fold bar standing in for the hidden docstring lines
      const hidden = foldEnd - foldStart + 1;
      const bar = h("button", { type: "button", class: "code-fold", "aria-expanded": "false" },
        h("span", { class: "code-gutter", text: "⋯" }),
        h("span", { class: "code-fold-text", text: `"""…"""  açıklama metni (docstring) · ${hidden} satır · göstermek için tıklayın` }));
      foldGroup = h("div", { class: "code-fold-lines", hidden: true });
      bar.addEventListener("click", () => {
        const open = foldGroup.hidden;
        foldGroup.hidden = !open;
        bar.setAttribute("aria-expanded", String(open));
        bar.querySelector(".code-fold-text").textContent = open
          ? `"""…"""  açıklama metnini gizle`
          : `"""…"""  açıklama metni (docstring) · ${hidden} satır · göstermek için tıklayın`;
      });
      body.append(bar, foldGroup);
    }
    if (foldGroup && k >= foldStart && k <= foldEnd) foldGroup.append(row);
    else body.append(row);
  });

  const copy = h("button", { type: "button", class: "code-copy", text: "Kopyala" });
  copy.addEventListener("click", async () => {
    const text = lines.map((l) => l.text).join("\n");
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const area = h("textarea", {});
      area.value = text;
      document.body.append(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    copy.textContent = "Kopyalandı ✓";
    setTimeout(() => { copy.textContent = "Kopyala"; }, 1500);
  });

  const numbered = lines.filter((l) => l.n != null);
  const range = numbered.length && options.file === "SimpleCNN.py"
    ? `satır ${numbered[0].n}–${numbered[numbered.length - 1].n}` : (lang === "python" ? "Python" : "");
  const head = h("div", { class: "code-head" },
    h("span", { class: "code-dots", "aria-hidden": "true" }, h("i"), h("i"), h("i")),
    h("span", { class: "code-tab" }, h("span", { class: "code-tab-icon", text: lang === "python" ? "py" : ">_" }), options.file || "kod"),
    h("span", { class: "code-meta", text: range }),
    copy);

  const figure = h("figure", { class: `code${options.tall ? " tall" : ""}` }, head, body);
  if (notes.length) {
    const list = h("ol", { class: "code-notes" });
    for (const note of notes) {
      const item = h("li", {}, h("span", { class: "code-note-mark", text: note.mark }), h("span", {}, rich(note.text)));
      // Hovering a note lights up its line
      item.addEventListener("pointerenter", () => rows[note.index].classList.add("lit"));
      item.addEventListener("pointerleave", () => rows[note.index].classList.remove("lit"));
      list.append(item);
    }
    figure.append(list);
  }
  if (options.caption) figure.append(h("figcaption", {}, rich(options.caption)));
  return figure;
}

/** A terminal window. lines: ["$ command", "output", ...]; lines starting with "$ " are commands. */
function terminal(title, lines) {
  const body = h("div", { class: "term-body" });
  for (const line of lines) {
    if (line.startsWith("$ ")) body.append(h("div", { class: "term-cmd" }, h("span", { class: "term-prompt", text: "PS>" }), " ", line.slice(2)));
    else body.append(h("div", { class: "term-out", text: line || " " }));
  }
  return h("figure", { class: "term" },
    h("div", { class: "term-head" }, h("span", { class: "code-dots", "aria-hidden": "true" }, h("i"), h("i"), h("i")), h("span", { text: title })),
    body);
}

// =====================================================================
// The tutorial
// Blocks: { p }, { list, ordered }, { callout: "tip" | "warn" | "info", title, text },
//         { file: { from, to, fold, notes } }  (cut from SimpleCNN.py),
//         { code, name, notes, caption }      (a separate example file),
//         { term: title, lines }, { table: { head, rows } }
// =====================================================================
const TRAIN_OUTPUT = [
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

const STEPS = [
  {
    id: "c-intro", title: "Ne yapacağız?",
    blocks: [
      { p: "Bu rehberde `SimpleCNN.py` dosyasını baştan sona birlikte yazacağız: MNIST el yazısı rakamlarını tanıyan ve test setinde **%99'un üzerinde** doğruluğa ulaşan küçük bir evrişimli sinir ağı. Her adımda önce *neyi* ve *neden* yaptığımızı anlatıp, ardından kodu satır satır inceleyeceğiz." },
      { p: "Bir derin öğrenme programı hemen her zaman aynı iskeleti izler. Adımlar da bu sırayla ilerliyor:" },
      { list: ["**Ortam ve kütüphaneler**: PyTorch'u kurup içe aktarmak", "**Ayarlar**: hiperparametreler ve mimari sayıları", "**Veri**: MNIST'i indirip batch'lere bölmek", "**Model**: katmanları tanımlamak ve verinin yolunu (`forward`) yazmak", "**Kayıp ve optimizer**: hatayı ölçmek ve ağırlıkları güncellemek", "**Eğitim ve değerlendirme döngüleri**", "**Hepsini birleştirmek**, çalıştırmak ve sonucu yorumlamak"], ordered: true },
      { callout: "info", title: "Kodlar gerçek dosyadan geliyor", text: "Buradaki kod parçaları repodaki `SimpleCNN.py` dosyasından canlı okunur; satır numaraları dosyadakiyle birebir aynıdır. Uzun açıklama metinleri (docstring) katlanmıştır, tıklayınca açılır." },
      { callout: "tip", title: "Nasıl okunur?", text: "Kod satırlarının sonundaki numaralı mavi işaretler, bloğun altındaki aynı numaralı açıklamalara karşılık gelir. Bir açıklamanın üzerine gelince ilgili satır vurgulanır. Kavramların kendisi için [[theory|Teori]] sekmesine, sonuçların canlı hali için [[train|Eğitim]] sekmesine bağlantılar var." },
      { callout: "info", title: "Temel fikir: tensör", text: "PyTorch'ta her şey **tensör** (çok boyutlu dizi) üzerinde çalışır. Bir grup görüntü 4 boyutlu bir tensördür: `(N, C, H, W)`, yani örnek sayısı, kanal, yükseklik ve genişlik. Bir MNIST batch'i `(256, 1, 28, 28)` boyutundadır. Rehber boyunca boyutların nasıl değiştiğini takip edeceğiz." },
    ],
  },
  {
    id: "c-setup", title: "Ortamı hazırlamak",
    blocks: [
      { p: "Projeyi diğer Python kurulumlarından ayırmak için bir **sanal ortam** (`.venv`) kullanıyoruz. İçine PyTorch'u, ekran kartıyla çalışabilen (CUDA) sürümüyle kuruyoruz:" },
      { term: "PowerShell · proje klasöründe", lines: [
        "$ py -3.13 -m venv .venv",
        "$ .venv\\Scripts\\Activate.ps1",
        "$ python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126",
        "$ python -m pip install matplotlib",
      ] },
      { callout: "tip", title: "Kısa yol", text: "Bu repodaki `start.bat`, sanal ortam yoksa bu adımları sizin yerinize yapar ve paneli başlatır." },
      { p: "Kurulumun çalıştığını ve ekran kartının görüldüğünü küçük bir denemeyle kontrol edelim:" },
      { code: String.raw`import torch

print(torch.__version__)              # PyTorch sürümü
print(torch.cuda.is_available())      # True ise ekran kartı kullanılabilir
print(torch.cuda.get_device_name(0))  # ekran kartının adı`, name: "kontrol.py" },
      { term: "Çıktı (bu bilgisayarda)", lines: ["2.14.1+cu126", "True", "NVIDIA GeForce RTX 2060"] },
      { callout: "warn", title: "Ekran kartı yoksa", text: "`torch.cuda.is_available()` `False` dönerse kod yine çalışır, yalnızca eğitim CPU'da birkaç kat daha uzun sürer. Kodun hiçbir yerini değiştirmeniz gerekmez (bkz. Adım 9)." },
    ],
  },
  {
    id: "c-imports", title: "Kütüphaneleri içe aktarmak",
    blocks: [
      { p: "Dosyanın başında ihtiyaç duyacağımız araçları içe aktarıyoruz. PyTorch'un görevleri birkaç alt pakete ayrılmıştır: modeli kuranlar, eğitenler ve veriyi hazırlayanlar." },
      { file: { from: "import torch", to: "import matplotlib.pyplot as plt", notes: [
        { at: "import torch", text: "PyTorch'un çekirdeği: tensörler, GPU desteği ve matematik işlemleri." },
        { at: "import torch.nn as nn", text: "Sinir ağı yapı taşları: `Conv2d`, `Linear`, `BatchNorm2d` gibi **öğrenilen ağırlıkları olan** katmanlar ve kayıp fonksiyonları." },
        { at: "import torch.nn.functional as F", text: "Ağırlığı olmayan işlemlerin fonksiyon hâli, örneğin `F.relu`. Durum tutmadıkları için modelde önceden tanımlanmaları gerekmez." },
        { at: "import torch.optim as optim", text: "Optimizasyon algoritmaları (Adam, SGD…): gradyanları kullanarak ağırlıkları günceller." },
        { at: "from torch.utils.data import DataLoader, random_split", text: "`DataLoader` veriyi batch'ler hâlinde, istenirse karıştırarak verir; `random_split` bir veri setini rastgele parçalara böler." },
        { at: "from torchvision import datasets", text: "Hazır veri setleri; `datasets.MNIST` MNIST'i indirir ve okur." },
        { at: "from torchvision.transforms import ToTensor", text: "Görüntüyü (0–255 tam sayı pikseller) 0–1 aralığında ondalıklı bir tensöre çevirir ve boyutunu `(1, 28, 28)` yapar." },
        { at: "import matplotlib.pyplot as plt", text: "Eğitim sonunda kayıp grafiğini çizmek için." },
      ] } },
      { callout: "info", title: "nn mi, F mi?", text: "Kural basit: öğrenilen ağırlığı olan her şey (`Conv2d`, `Linear`, `BatchNorm2d`) `nn` katmanı olarak modelde saklanır. Ağırlığı olmayan işlemler (`relu`, `flatten`) doğrudan fonksiyon olarak çağrılabilir." },
    ],
  },
  {
    id: "c-config", title: "Hiperparametreler ve mimari",
    blocks: [
      { p: "**Hiperparametreler**, eğitimden önce elle seçtiğimiz ve modelin kendisinin öğrenmediği değerlerdir. Hepsini dosyanın başında tek bir yerde toplamak, denemeler yaparken (örneğin öğrenme oranını değiştirirken) kodun içinde aramaktan kurtarır." },
      { file: { from: "#--------------- Hyperparameters", to: "dropout2_rate = 0.35", notes: [
        { at: "learning_rate = 1e-3", text: "**Öğrenme oranı**: optimizer'ın her adımda ağırlıkları ne kadar değiştireceği. Adam için 0,001 yaygın ve güvenli bir başlangıçtır." },
        { at: "num_epochs = 10", text: "**Epoch**: eğitim verisinin tamamının bir kez gösterilmesi. Model her görüntüyü 10 kez görür." },
        { at: "batch_size = 256", text: "Her adımda birlikte işlenen görüntü sayısı. Büyük batch daha kararlı gradyan ve GPU'nun verimli kullanımı demektir, ama daha çok bellek ister." },
        { at: "validation_split = 0.2", text: "Eğitim verisinin %20'si (12.000 görüntü) **doğrulama** için ayrılır; model bu görüntülerle eğitilmez." },
        { at: "seed = 42", text: "Rastgelelik tohumu: aynı tohum her çalıştırmada aynı sonucu verir (Adım 5)." },
        { at: "conv1_out_channels = 32", text: "Filtre sayıları: ilk evrişim 32, ikincisi 64 öznitelik haritası üretir. Derinleştikçe kanal sayısını artırmak yaygın bir tasarımdır." },
        { at: "conv_kernel_size = 5", text: "Filtreler 5×5 piksel büyüklüğünde." },
        { at: "pool_kernel_size = 2", text: "2×2 max pooling, stride 2: her havuzlama genişliği ve yüksekliği yarıya indirir." },
        { at: "fc1_in = conv2_out_channels * 4 * 4", text: "Düzleştirme sonrası vektör uzunluğu: 64 kanal × 4 × 4 = **1024**. Sabit 1024 yazmak yerine kanal sayısından hesaplıyoruz; `conv2_out_channels` değişirse kod bozulmaz." },
        { at: "fc2_in = 512", text: "Gizli tam bağlı katmandaki nöron sayısı." },
        { at: "dropout1_rate = 0.45", text: "Eğitimde değerlerin %45'ini (ilk blok) ve %35'ini (ikinci blok) rastgele sıfırlayan dropout oranları." },
      ] } },
      { p: "`fc1_in`'deki `4 * 4` nereden geliyor? Görüntünün her katmanda nasıl küçüldüğünü izleyelim. Dolgusuz (padding 0) bir evrişim boyutu `k − 1` azaltır, 2×2 havuzlama ise yarıya indirir:" },
      { table: { head: ["Katman", "Çıktı boyutu (C × H × W)", "Hesap"], rows: [
        ["Giriş", "1 × 28 × 28", "MNIST görüntüsü"],
        ["Conv1 (5×5, 32 filtre)", "32 × 24 × 24", "28 − 5 + 1 = 24"],
        ["MaxPool 2×2", "32 × 12 × 12", "24 / 2 = 12"],
        ["Conv2 (5×5, 64 filtre)", "64 × 8 × 8", "12 − 5 + 1 = 8"],
        ["MaxPool 2×2", "64 × 4 × 4", "8 / 2 = 4"],
        ["Düzleştir (flatten)", "1024", "64 × 4 × 4"],
        ["FC1", "512", "fc2_in"],
        ["FC2 (çıkış)", "10", "her rakam için bir skor"],
      ] } },
      { callout: "tip", title: "Genel formül", text: "Çıktı boyutu = ⌊(giriş − çekirdek + 2·padding) / stride⌋ + 1. [[theory:t-conv|Teori → Evrişim]] bölümünde padding ve stride'ı değiştirip formülü canlı görebilirsiniz." },
    ],
  },
  {
    id: "c-seed", title: "Tekrarlanabilirlik: seed",
    blocks: [
      { p: "Eğitimde birçok şey rastgeledir: ağırlıkların ilk değerleri, verinin eğitim/doğrulama diye bölünmesi, her epoch'taki karıştırma sırası ve dropout maskeleri. Sonuçları karşılaştırabilmek (ve öğrencilerin aynı sayıları görmesi) için bu rastgeleliği sabitliyoruz." },
      { file: { from: "def set_seed", to: "torch.backends.cudnn.benchmark = False", fold: true, notes: [
        { at: "torch.manual_seed(seed_value)", text: "PyTorch'un rastgele sayı üretecini sabitler; CPU ve tüm GPU'lar için geçerlidir." },
        { at: "torch.backends.cudnn.deterministic = True", text: "GPU'daki bazı hızlı evrişim algoritmaları çalıştırmadan çalıştırmaya küçük farklar üretir; bu ayar deterministik olanları seçer." },
        { at: "torch.backends.cudnn.benchmark = False", text: "cuDNN'in her çalıştırmada \"en hızlı algoritmayı\" yeniden seçmesini kapatır; seçim değişirse sonuç da değişebilirdi." },
      ] } },
      { callout: "info", title: "Etkisi", text: "Seed 42 ile her çalıştırma birebir aynı sonucu verir: test doğruluğu **%99,29**. Panelin [[train|Eğitim]] sekmesi de aynı sırayı izlediği için aynı sayıları üretir." },
    ],
  },
  {
    id: "c-data", title: "Veriyi hazırlamak",
    blocks: [
      { p: "MNIST, 28×28 piksellik gri tonlu el yazısı rakamlardan oluşur: 60.000 eğitim ve 10.000 test görüntüsü. Veriyi üç parçaya ayırıyoruz:" },
      { list: ["**Eğitim** (48.000): model bunlardan öğrenir.", "**Doğrulama** (12.000): eğitim sırasında, görmediği verideki başarıyı izlemek için.", "**Test** (10.000): en sonda yalnızca bir kez, son performansı ölçmek için."] },
      { file: { from: "def get_mnist_dataloaders", to: "return training_loader, validation_loader, testing_loader", fold: true, notes: [
        { at: "full_train_dataset = datasets.MNIST(root='./data', train=True,", text: "60.000 eğitim görüntüsü. `root` verinin kaydedileceği klasör, `download=True` yoksa indirir, `transform` her görüntüye uygulanacak dönüşüm." },
        { at: "test_dataset = datasets.MNIST(root='./data', train=False,", text: "`train=False`: modelin eğitimde hiç görmeyeceği 10.000 test görüntüsü." },
        { at: "train_dataset, val_dataset = random_split(full_train_dataset,", text: "Eğitim setini rastgele %80 / %20 böler. Oranlar doğrudan kesir olarak verilebilir." },
        { at: "training_loader = DataLoader(train_dataset,", text: "`shuffle=True`: her epoch'ta sıra karıştırılır, model örneklerin sırasını ezberleyemez." },
        { at: "validation_loader = DataLoader(val_dataset,", text: "Doğrulama ve testte karıştırmaya gerek yok; sonuç sıradan bağımsızdır." },
      ] } },
      { p: "`DataLoader` bir Python yineleyicisidir: `for` döngüsünde her turda bir batch verir. İlk batch'e bakarak boyutları doğrulayalım:" },
      { code: String.raw`from SimpleCNN import *

set_seed(seed)
train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size, validation_split)

images, labels = next(iter(train_loader))   # ilk batch
print(images.shape, images.dtype)           # (N, C, H, W)
print(images.min().item(), images.max().item())
print(labels[:10])                          # ilk 10 etiket
print(len(train_loader), "batch / epoch")`, name: "veriye_bakalim.py", notes: [
        { at: "images, labels = next(iter(train_loader))", text: "`iter` yineleyiciyi başlatır, `next` ilk batch'i alır. Eğitim döngüsü de tam olarak bunu tekrar tekrar yapar." },
        { at: "print(len(train_loader), \"batch / epoch\")", text: "48.000 / 256 = 187,5, yukarı yuvarlanınca 188 batch. Son batch daha küçüktür (128 görüntü)." },
      ] },
      { term: "Çıktı", lines: ["torch.Size([256, 1, 28, 28]) torch.float32", "0.0 1.0", "tensor([5, 8, 4, 4, 4, 0, 3, 9, 1, 3])", "188 batch / epoch"] },
      { callout: "warn", title: "Test setini neden en sona saklıyoruz?", text: "Hiperparametreleri (öğrenme oranı, epoch sayısı…) doğrulama sonuçlarına bakarak seçeriz. Test setine bakarak seçseydik, test sonucu artık \"hiç görülmemiş veri\" başarısını ölçmezdi." },
    ],
  },
  {
    id: "c-model", title: "Modeli tanımlamak: katmanlar",
    blocks: [
      { p: "PyTorch'ta bir model, `nn.Module`'den türetilen bir sınıftır. İki yöntemi vardır: `__init__` katmanları **oluşturur**, `forward` verinin bu katmanlardan **hangi sırayla** geçeceğini söyler (Adım 8)." },
      { file: { from: "class SimpleCNN(nn.Module):", to: "self.batch2 = nn.BatchNorm2d(num_features=conv2_out_channels)", fold: true, notes: [
        { at: "class SimpleCNN(nn.Module):", text: "`nn.Module`'den türetmek, katmanların ağırlıklarının otomatik bulunmasını (`model.parameters()`), modelin GPU'ya taşınabilmesini ve eğitim/değerlendirme modlarını sağlar." },
        { at: "super(SimpleCNN, self).__init__()", text: "`nn.Module`'ün kendi kurulumunu çalıştırır. Bu satır olmadan katmanlar modele kaydedilemez; her zaman ilk satırdır." },
        { at: "self.conv1 = nn.Conv2d(in_channels=input_channels,", text: "1 kanallı (gri) girişten 32 öznitelik haritası çıkaran 5×5 evrişim: 32 × (1×5×5) ağırlık + 32 bias = **832** parametre." },
        { at: "self.conv2 = nn.Conv2d(in_channels=conv1_out_channels,", text: "32 haritayı girdi alıp 64 harita üretir. Her filtre 32 kanalın hepsine bakar: 64 × (32×5×5) + 64 = **51.264** parametre." },
        { at: "self.max1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)", text: "Havuzlamanın öğrenilen parametresi yoktur; okunabilirlik için yine de katman olarak tanımlanır." },
        { at: "self.fc1 = nn.Linear(in_features=fc1_in, out_features=fc2_in)", text: "1024 → 512 tam bağlı katman: 1024 × 512 + 512 = **524.800** parametre, modelin yaklaşık %90'ı." },
        { at: "self.fc2 = nn.Linear(in_features=fc2_in, out_features=num_classes)", text: "512 → 10: her rakam için bir skor (logit)." },
        { at: "self.dropout1 = nn.Dropout(p=dropout1_rate)", text: "Dropout da parametresizdir ve yalnızca eğitim modunda çalışır." },
        { at: "self.batch1 = nn.BatchNorm2d(num_features=conv1_out_channels)", text: "Kanal başına öğrenilen γ ve β (32 + 32 parametre) ile eğitim boyunca biriken ortalama ve varyans." },
      ] } },
      { p: "Modelde kaç parametre olduğunu ve nerede yoğunlaştığını görmek için katmanları dolaşabiliriz:" },
      { code: String.raw`model = SimpleCNN()
for name, p in model.named_parameters():
    print(f"{name:16s} {str(tuple(p.shape)):18s} {p.numel():>8,}")
print("Toplam:", f"{sum(p.numel() for p in model.parameters()):,}")`, name: "parametreler.py" },
      { term: "Çıktı", lines: [
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
        "Toplam: 582,218",
      ] },
      { callout: "info", title: "Evrişimin gücü", text: "İki evrişim katmanı toplam 52 bin parametreyle tüm görüntüyü tarıyor; tek bir tam bağlı katman (FC1) ise 524 bin parametre kullanıyor. Bunun nedeni **ağırlık paylaşımı**: aynı küçük filtre görüntünün her yerinde tekrar kullanılır." },
    ],
  },
  {
    id: "c-forward", title: "İleri yayılım: forward",
    blocks: [
      { p: "`forward`, bir batch'in modelden geçerken izlediği yolu tanımlar. Kod sıradan Python'dur: her satır bir önceki satırın çıktısını alır. Yanlarına boyutları yazdık; `N` batch büyüklüğüdür." },
      { file: { from: "def forward(self, x):", to: "return x", fold: true, notes: [
        { at: "x = self.conv1(x)           # Convolution Layer 1", text: "`(N, 1, 28, 28)` → `(N, 32, 24, 24)`. Her çıktı kanalı bir filtrenin tepki haritasıdır." },
        { at: "x = F.relu(self.batch1(x))  # Batch Normalization + ReLU", text: "Önce BatchNorm, sonra ReLU. ReLU'nun ağırlığı olmadığı için `F.relu` fonksiyon olarak çağrılır." },
        { at: "x = self.max1(x)            # Max Pooling", text: "`(N, 32, 24, 24)` → `(N, 32, 12, 12)`." },
        { at: "x = self.dropout1(x)        # Dropout", text: "`model.train()` modunda değerlerin %45'ini sıfırlar; `model.eval()` modunda hiçbir şey yapmaz." },
        { at: "x = self.conv2(x)           # Convolution Layer 2", text: "`(N, 32, 12, 12)` → `(N, 64, 8, 8)`." },
        { at: "x = self.max2(x)            # Max Pooling", text: "`(N, 64, 8, 8)` → `(N, 64, 4, 4)`." },
        { at: "x = torch.flatten(x, start_dim=1)  # Flatten for FC layer", text: "`(N, 64, 4, 4)` → `(N, 1024)`. `start_dim=1` batch boyutunu korur, gerisini tek bir vektöre açar." },
        { at: "x = F.relu(self.fc1(x))     # Fully Connected Layer 1 + ReLU", text: "`(N, 1024)` → `(N, 512)`, ardından ReLU." },
        { at: "x = self.fc2(x)             # Fully Connected Layer 2 (logits)", text: "`(N, 512)` → `(N, 10)`: ham skorlar (logit)." },
      ] } },
      { callout: "warn", title: "Neden sonda softmax yok?", text: "`nn.CrossEntropyLoss` girdiyi **logit** olarak bekler ve softmax'ı (log-softmax olarak) kendi içinde uygular. Modelin sonuna softmax eklemek iki kez softmax uygulamak olurdu: hem yanlış hem de sayısal olarak kararsız. Olasılık gerektiğinde, tahmin anında `torch.softmax(logits, dim=1)` kullanılır (Adım 15)." },
      { p: "Boyutları tahmin etmek yerine sıfırlardan oluşan sahte bir görüntüyü katmanlardan geçirerek doğrulayabiliriz. Yeni bir mimari yazarken `fc1_in`'i bulmanın en güvenli yolu budur:" },
      { code: String.raw`x = torch.zeros(1, 1, 28, 28)            # sahte bir görüntü
for layer in [model.conv1, model.max1, model.conv2, model.max2]:
    x = layer(x)
    print(f"{layer.__class__.__name__:10s} -> {tuple(x.shape)}")
x = torch.flatten(x, start_dim=1)
print(f"{'flatten':10s} -> {tuple(x.shape)}")`, name: "boyutlar.py" },
      { term: "Çıktı", lines: ["Conv2d     -> (1, 32, 24, 24)", "MaxPool2d  -> (1, 32, 12, 12)", "Conv2d     -> (1, 64, 8, 8)", "MaxPool2d  -> (1, 64, 4, 4)", "flatten    -> (1, 1024)"] },
      { callout: "tip", title: "Canlı görmek için", text: "[[infer|Çıkarım]] sekmesindeki \"Katman katman ağ\" bölümü tam olarak bu adımların çıktısını, gerçek bir görüntü için gösterir." },
    ],
  },
  {
    id: "c-setup-train", title: "Kayıp fonksiyonu ve optimizer",
    blocks: [
      { p: "Eğitime başlamadan önce üç şey gerekir: modelin çalışacağı **cihaz**, tahminin ne kadar yanlış olduğunu ölçen **kayıp fonksiyonu** ve ağırlıkları güncelleyecek **optimizer**. Bunları tek bir yardımcı fonksiyonda kuruyoruz." },
      { file: { from: "def _setup_model_for_training(", to: "return model, loss_function, optimizer, device", fold: true, notes: [
        { at: "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')", text: "Ekran kartı varsa GPU, yoksa CPU. Kodun geri kalanı ikisinde de aynı çalışır." },
        { at: "model = SimpleCNN(num_classes=num_classes).to(device)", text: "Modeli oluşturur ve tüm ağırlıklarını seçilen cihaza taşır. Model ve veri **aynı cihazda** olmalıdır; aksi halde hata alırsınız." },
        { at: "loss_function = nn.CrossEntropyLoss().to(device)", text: "Sınıflandırma kaybı: logitlere softmax uygular ve doğru sınıfın olasılığının negatif logaritmasını alır. Batch ortalamasını döndürür." },
        { at: "optimizer = optim.Adam(model.parameters(), lr=lr)", text: "`model.parameters()` modeldeki tüm öğrenilen ağırlıkları (582.218 sayı) optimizer'a verir. Adam her ağırlık için adım boyunu ayrı ayarlar." },
      ] } },
      { callout: "info", title: "Kavramlar", text: "Cross-Entropy'nin neden −log p olduğu ve SGD, Momentum ile Adam'ın farkı için: [[theory:t-softmax|Teori → Softmax ve kayıp]] ile [[theory:t-opt|Teori → Optimizasyon]]." },
    ],
  },
  {
    id: "c-train-epoch", title: "Bir epoch eğitim",
    blocks: [
      { p: "Eğitimin kalbi bu fonksiyondur. Eğitim verisindeki her batch için aynı **beş adım** tekrarlanır: gradyanları sıfırla, ileri yayılım, kaybı hesapla, geri yayılım, ağırlıkları güncelle. Kalan satırlar yalnızca doğruluğu ve ortalama kaybı hesaplamak içindir." },
      { file: { from: "def train_epoch(", to: "return epoch_accuracy, epoch_loss", fold: true, notes: [
        { at: "model.train()  # Training mode", text: "**Eğitim modu**: Dropout etkin, BatchNorm o batch'in istatistiklerini kullanır ve hareketli ortalamaları günceller." },
        { at: "for inputs, labels in data_loader:", text: "Her turda bir batch: `inputs` `(256, 1, 28, 28)`, `labels` `(256,)`. 188 tur = 1 epoch." },
        { at: "inputs, labels = inputs.to(device), labels.to(device)", text: "Veriyi modelin bulunduğu cihaza taşı." },
        { at: "optimizer.zero_grad()", text: "**1. Gradyanları sıfırla.** PyTorch gradyanları biriktirir; sıfırlanmazsa her adımın gradyanı öncekilerin üstüne eklenir." },
        { at: "outputs = model(inputs)", text: "**2. İleri yayılım.** `model(inputs)` aslında `forward`'u çağırır. Çıktı `(256, 10)` logit." },
        { at: "loss = criterion(outputs, labels)", text: "**3. Kayıp.** Tahminin ne kadar yanlış olduğunu ölçen tek bir sayı." },
        { at: "loss.backward()", text: "**4. Geri yayılım.** Kaybın her ağırlığa göre gradyanını zincir kuralıyla hesaplar ve ağırlıkların `.grad` alanına yazar." },
        { at: "optimizer.step()", text: "**5. Güncelleme.** Adam, gradyanları kullanarak tüm ağırlıkları bir adım değiştirir." },
        { at: "total_train_loss += loss.item()", text: "`.item()` tek elemanlı tensörü sıradan bir Python sayısına çevirir." },
        { at: "_, predicted = outputs.max(1)", text: "Her satırdaki en büyük logitin indeksi, tahmin edilen rakamdır. `max(1)` (değer, indeks) döndürür; değeri `_` ile atarız." },
        { at: "correct_train += predicted.eq(labels).sum().item()", text: "Doğru tahmin sayısı: `eq` eşit olanlarda `True` verir, `sum` bunları sayar." },
      ] } },
      { callout: "tip", title: "Bu beş adımı canlı izleyin", text: "[[train|Eğitim]] sekmesindeki \"Her batch'te çalışan eğitim adımı\" kartı tam olarak bu beş satırı gösterir." },
      { callout: "warn", title: "Sık yapılan hata", text: "`optimizer.zero_grad()` satırını unutmak hata vermez ama model kötü öğrenir: gradyanlar her adımda birikerek büyür. Programın çökmediği için fark edilmesi zor bir hatadır." },
    ],
  },
  {
    id: "c-evaluate", title: "Değerlendirme",
    blocks: [
      { p: "Değerlendirme, eğitimin \"yalnızca ölçen\" hâlidir: ileri yayılım ve kayıp aynıdır, ama geri yayılım ve güncelleme yoktur. İki önemli farkı var:" },
      { file: { from: "def evaluate_model(", to: "return eval_accuracy, eval_loss", fold: true, notes: [
        { at: "model.eval()  # Evaluation mode", text: "**Değerlendirme modu**: Dropout kapanır, BatchNorm eğitimde biriktirdiği hareketli ortalamaları kullanır." },
        { at: "with torch.no_grad():", text: "Gradyan takibini kapatır: hesap grafiği tutulmaz, bellek ve zaman kazanılır. `backward` yapmayacağımız için gerekmez." },
        { at: "logits = model(inputs)", text: "Yalnızca ileri yayılım. `zero_grad`, `backward` ve `step` yok, yani ağırlıklar değişmez." },
      ] } },
      { callout: "warn", title: "model.eval()'i unutursanız", text: "Dropout değerlendirmede de rastgele değer sıfırlamaya devam eder ve BatchNorm batch istatistiklerini kullanır. Sonuçlar hem kötüleşir hem de her çalıştırmada değişir. `model.train()` / `model.eval()` çiftini her zaman doğru yerde çağırın." },
    ],
  },
  {
    id: "c-train-loop", title: "Eğitim döngüsü",
    blocks: [
      { p: "Şimdi iki fonksiyonu birleştiriyoruz: her epoch'ta önce bir tur eğitim, ardından doğrulama. Kayıpları listelerde saklıyoruz ki sonunda grafiğini çizebilelim." },
      { file: { from: "def train_model(", to: "return train_loss, validation_loss", fold: true, notes: [
        { at: "for epoch in range(num_epochs):", text: "10 epoch, her biri tüm eğitim verisinden bir geçiş." },
        { at: "train_accuracy, epoch_train_loss = train_epoch(model,", text: "Adım 10'daki fonksiyon: bir epoch eğitir, doğruluğu ve ortalama kaybı döndürür." },
        { at: "validation_accuracy, epoch_validation_loss = evaluate_model(model,", text: "Adım 11'deki fonksiyon, bu sefer doğrulama verisiyle: modelin görmediği veride ne kadar başarılı?" },
        { at: "print(f\"Epoch {epoch + 1}: Train Loss: {train_loss[epoch]:.4f}, \"", text: "f-string: `{değer:.4f}` sayıyı 4 ondalık basamakla yazar. Alt alta yazılan f-string'leri Python otomatik birleştirir." },
      ] } },
      { callout: "info", title: "Neyi izliyoruz?", text: "Eğitim kaybı düşerken doğrulama kaybı da düşüyorsa model öğreniyor demektir. Eğitim kaybı düşmeye devam ederken doğrulama kaybı **yükselmeye** başlarsa model ezberlemeye (overfitting) başlamıştır; o noktada eğitimi durdurmak gerekir." },
    ],
  },
  {
    id: "c-plot", title: "Kayıp grafiği",
    blocks: [
      { p: "Sayıları tek tek okumak yerine eğrileri yan yana görmek, eğitimin gidişatını anlamanın en hızlı yoludur." },
      { file: { from: "def plot_training_losses(", to: "plt.show()", fold: true, notes: [
        { at: "eps = range(1, len(train_loss_epochs) + 1)", text: "x ekseni: 1'den başlayan epoch numaraları." },
        { at: "plt.plot(eps, train_loss_epochs, linestyle='-', color='#1f77b4',", text: "Eğitim kaybı eğrisi (mavi)." },
        { at: "plt.plot(eps, validation_loss_epochs, linestyle='-', color='#d62728',", text: "Doğrulama kaybı eğrisi (kırmızı)." },
        { at: "plt.show()", text: "Grafiği bir pencerede açar; program pencere kapanana kadar bekler." },
      ] } },
    ],
  },
  {
    id: "c-main", title: "Hepsini birleştirmek",
    blocks: [
      { p: "Son olarak bütün parçaları doğru sırayla çağıran ana bölümü yazıyoruz. Dosya çalıştırıldığında olan her şey burada." },
      { file: { from: "if __name__ == \"__main__\":", to: "plot_training_losses(train_losses, validation_losses)", notes: [
        { at: "if __name__ == \"__main__\":", text: "Bu blok yalnızca dosya doğrudan çalıştırıldığında (`python SimpleCNN.py`) çalışır; başka bir dosya `import SimpleCNN` yaptığında çalışmaz. `dashboard.py` bu sayede buradaki fonksiyonları yeniden kullanabiliyor." },
        { at: "set_seed(seed)", text: "Önce seed. Sıra önemli: modelin ilk ağırlıkları ve veri bölünmesi rastgele sayı tükettiği için seed onlardan önce sabitlenmeli." },
        { at: "cnn_model, loss_fn, adam_optimizer, h_device = _setup_model_for_training(", text: "Model, kayıp, optimizer ve cihaz (Adım 9)." },
        { at: "train_loader, val_loader, test_loader = get_mnist_dataloaders(", text: "Veri (Adım 6)." },
        { at: "train_losses, validation_losses = train_model(cnn_model, loss_fn, adam_optimizer,", text: "Eğitim ve doğrulama (Adım 12)." },
        { at: "test_accuracy, test_loss = evaluate_model(cnn_model, loss_fn,", text: "Test: model hazır olduktan sonra, **yalnızca bir kez**." },
        { at: "plot_training_losses(train_losses, validation_losses)", text: "Kayıp grafiği (Adım 13)." },
      ] } },
      { p: "Çalıştıralım. İlk seferde MNIST indirilir (birkaç saniye), ardından eğitim başlar. RTX 2060'ta yaklaşık 1,5–2 dakika sürer:" },
      { term: "PowerShell", lines: ["$ .venv\\Scripts\\python.exe SimpleCNN.py", ...TRAIN_OUTPUT] },
      { callout: "info", title: "Sonucu yorumlamak", text: "İlk epoch'ta bile doğrulama doğruluğu %97,7. Sonraki epoch'lar küçük ama düzenli iyileştirmeler getiriyor. Eğitim kaybının doğrulamadan yüksek olması bir hata değil: eğitimde Dropout açık, doğrulamada kapalı. Test doğruluğu (%99,29) doğrulamaya (%99,36) çok yakın, yani model ezberlemeden genelleme yapıyor." },
    ],
  },
  {
    id: "c-predict", title: "Ek: kaydetmek ve tahmin yapmak",
    blocks: [
      { p: "`SimpleCNN.py` eğitimden sonra modeli saklamaz. Eğitilmiş bir modeli tekrar kullanmak için ağırlıklarını kaydedip başka bir programda yükleyebilirsiniz. Panelin `dashboard.py` dosyası da tam olarak bunu yapıyor." },
      { code: String.raw`# Eğitimden sonra: yalnızca ağırlıkları (state_dict) kaydet
torch.save(cnn_model.state_dict(), "simple_cnn.pt")

# Başka bir programda: aynı mimariyi kur, ağırlıkları yükle
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn.pt", map_location="cpu"))
model.eval()

image, label = test_loader.dataset[0]        # (1, 28, 28) tensör ve etiket
with torch.no_grad():
    logits = model(image.unsqueeze(0))       # batch boyutu ekle: (1, 1, 28, 28)
    probs = torch.softmax(logits, dim=1)[0]  # logit -> olasılık
print("Tahmin:", probs.argmax().item(), "| Gerçek:", label, f"| Güven: %{probs.max().item() * 100:.1f}")`, name: "tahmin.py", notes: [
        { at: "torch.save(cnn_model.state_dict(), \"simple_cnn.pt\")", text: "`state_dict` katman adlarını ağırlık tensörlerine eşleyen bir sözlüktür. Modelin kodu değil, yalnızca sayıları kaydedilir." },
        { at: "model.load_state_dict(torch.load(\"simple_cnn.pt\", map_location=\"cpu\"))", text: "Önce aynı sınıftan boş bir model kurulur, sonra ağırlıklar yüklenir. `map_location=\"cpu\"` GPU'da kaydedilmiş modeli GPU'suz bir bilgisayarda açmayı sağlar." },
        { at: "model.eval()", text: "Tahminden önce mutlaka değerlendirme moduna alın (Adım 11)." },
        { at: "logits = model(image.unsqueeze(0))       # batch boyutu ekle: (1, 1, 28, 28)", text: "Model her zaman bir batch bekler; tek görüntüye `unsqueeze(0)` ile boyu 1 olan bir batch boyutu ekleriz." },
        { at: "probs = torch.softmax(logits, dim=1)[0]  # logit -> olasılık", text: "Olasılık istediğimiz yer burası: logitlere softmax uygulayıp toplamı 1 olan 10 olasılık elde ederiz." },
      ] },
      { term: "Çıktı", lines: ["Tahmin: 7 | Gerçek: 7 | Güven: %100.0"] },
      { callout: "tip", title: "Çizdiğiniz rakamla deneyin", text: "[[infer|Çıkarım]] sekmesi bunu sizin çiziminizle yapar. Çizim önce MNIST'teki gibi 20×20'lik bir kutuya sığdırılıp ortalanır; aksi halde model, eğitimde gördüğünden farklı bir görüntüyle karşılaşırdı." },
    ],
  },
  {
    id: "c-full", title: "Tam dosya",
    blocks: [
      { p: "Yazdığımız her şey bir arada. Dosya repodan canlı okunuyor; `SimpleCNN.py`'yi değiştirirseniz burası (ve önceki adımlar) da güncellenir." },
      { file: { whole: true, tall: true } },
    ],
  },
  {
    id: "c-exercises", title: "Alıştırmalar",
    blocks: [
      { p: "Kodu gerçekten anlamanın en iyi yolu değiştirip ne olduğunu görmektir. Her alıştırmadan sonra kayıp grafiğine ve test doğruluğuna bakın:" },
      { list: [
        "**Öğrenme oranı**: `learning_rate`'i `1e-2` ve `1e-4` yapın. Hangisi daha hızlı öğreniyor, hangisi kararsız? [[theory:t-opt|Teori → Optimizasyon]] ile karşılaştırın.",
        "**Dropout'suz**: iki dropout oranını da `0` yapın. Eğitim ve doğrulama kaybı arasındaki fark nasıl değişiyor?",
        "**BatchNorm'suz**: `forward`'da `self.batch1(x)` ve `self.batch2(x)` çağrılarını kaldırın (`F.relu(x)` kalsın). Eğitim hızı ve Conv1 filtreleri nasıl değişiyor?",
        "**Optimizer**: Adam yerine `optim.SGD(model.parameters(), lr=0.05, momentum=0.9)` deneyin.",
        "**Daha geniş model**: `conv1_out_channels`'ı 64 yapın. Parametre sayısı ve süre nasıl değişiyor? (`fc1_in` neden değişmiyor?)",
        "**Üçüncü blok**: `padding=2` ile 5×5 bir `conv3` (64 → 128) ve 2×2 havuzlama ekleyin. `fc1_in` ne olmalı? Sahte görüntü yöntemiyle (Adım 8) doğrulayın.",
        "**Hatalı tahminler**: test setinde `predicted != labels` olan görüntüleri bulun ve matplotlib ile gösterin. Model hangi rakamları karıştırıyor?",
      ], ordered: true },
      { callout: "tip", title: "Seed'i unutmayın", text: "Karşılaştırmaların adil olması için her denemede seed'i aynı bırakın; böylece fark yalnızca değiştirdiğiniz şeyden gelir." },
    ],
  },
];

// =====================================================================
// Rendering and the tab's lifecycle
// =====================================================================
const tut = { started: false, source: null, observer: null };

/** Cuts lines from SimpleCNN.py, located by the text of their first and last line. */
function sourceRange(spec) {
  const all = tut.source.map((text, k) => ({ n: k + 1, text }));
  if (spec.whole) return all;
  const start = all.findIndex((l) => l.text.includes(spec.from));
  const end = start === -1 ? -1 : all.findIndex((l, k) => k >= start && l.text.includes(spec.to));
  if (start === -1 || end === -1) return null;
  return all.slice(start, end + 1);
}

function renderBlock(block) {
  if (block.p) return h("p", {}, rich(block.p));
  if (block.list) {
    const list = h(block.ordered ? "ol" : "ul", { class: "tut-list" });
    for (const item of block.list) list.append(h("li", {}, rich(item)));
    return list;
  }
  if (block.callout) {
    const icons = { tip: "💡", warn: "⚠️", info: "ℹ️" };
    return h("aside", { class: `tut-callout ${block.callout}` },
      h("span", { class: "tut-callout-icon", text: icons[block.callout], "aria-hidden": "true" }),
      h("div", {}, h("strong", { text: block.title }), h("p", {}, rich(block.text))));
  }
  if (block.file) {
    if (!tut.source) return h("p", { class: "placeholder", text: "SimpleCNN.py okunamadı. Panel sunucusu eski sürümse yeniden başlatın." });
    const lines = sourceRange(block.file);
    if (!lines) return h("p", { class: "placeholder", text: `Kod parçası bulunamadı: ${block.file.from}` });
    return codeEditor(lines, { file: "SimpleCNN.py", fold: block.file.fold, notes: block.file.notes, tall: block.file.tall });
  }
  if (block.code) {
    const lines = block.code.split("\n").map((text, k) => ({ n: k + 1, text }));
    return codeEditor(lines, { file: block.name, notes: block.notes, caption: block.caption });
  }
  if (block.term) return terminal(block.term, block.lines);
  if (block.table) {
    return h("div", { class: "table-wrap" }, h("table", { class: "data-table tut-table" },
      h("thead", {}, h("tr", {}, ...block.table.head.map((c) => h("th", { text: c })))),
      h("tbody", {}, ...block.table.rows.map((row) => h("tr", {}, ...row.map((c) => h("td", {}, rich(c))))))));
  }
  return null;
}

function goToStep(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderTutorial() {
  const main = $("#tut-main");
  const toc = $("#tut-toc");
  main.replaceChildren();
  toc.replaceChildren();
  STEPS.forEach((step, k) => {
    const article = h("article", { class: "card tut-step", id: step.id },
      h("header", { class: "tut-step-head" },
        h("span", { class: "tut-num", text: String(k + 1) }),
        h("div", {}, h("span", { class: "tut-kicker", text: `Adım ${k + 1} / ${STEPS.length}` }), h("h2", { text: step.title }))));
    for (const block of step.blocks) {
      const node = renderBlock(block);
      if (node) article.append(node);
    }
    const nav = h("footer", { class: "tut-nav" });
    if (k > 0) {
      const prev = h("button", { type: "button", class: "btn small", text: `← ${STEPS[k - 1].title}` });
      prev.addEventListener("click", () => goToStep(STEPS[k - 1].id));
      nav.append(prev);
    }
    if (k < STEPS.length - 1) {
      const next = h("button", { type: "button", class: "btn small primary", text: `${STEPS[k + 1].title} →` });
      next.addEventListener("click", () => goToStep(STEPS[k + 1].id));
      nav.append(next);
    }
    article.append(nav);
    main.append(article);

    const link = h("button", { type: "button", "data-step": step.id },
      h("span", { class: "toc-num", text: String(k + 1) }), h("span", { text: step.title }));
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
}

async function startTutorial() {
  if (tut.started) return;
  tut.started = true;
  try {
    const response = await fetch("/api/source/SimpleCNN.py", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    tut.source = (await response.text()).replace(/\r\n/g, "\n").replace(/\n$/, "").split("\n");
  } catch {
    tut.source = null;
    showOutdatedServer();
  }
  renderTutorial();
}

document.addEventListener("tab-shown", (event) => {
  if (event.detail === "code") startTutorial();
});
