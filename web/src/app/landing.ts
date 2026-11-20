/**
 * The start page (index.html): what the course is, the chapters of the learning path, and how to
 * use it without a server. Built from the chapter list, again on every language change.
 */
import { I18N, L, applyStaticTranslations } from "../shared/i18n.js";
import { wireSwitches } from "../shared/switches.ts";
import { CHAPTERS, type Text } from "../shared/chapters.ts";
import { onServer } from "../shared/engine.ts";

const REPO = "https://github.com/itumekanik/myz303";
/** The downloadable copy of this site (put next to index.html by the deployment). */
const ZIP = "ai-learning-path.zip";

type Child = Node | string | null | false;
function el(tag: string, attrs: Record<string, string> = {}, ...children: Child[]): HTMLElement {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  for (const child of children) if (child) node.append(child);
  return node;
}

const TABS: Text[] = [
  { tr: "<b>Teori</b>: kavramlar, her biri canlı bir demoyla", en: "<b>Theory</b>: the ideas, each with a live demo" },
  {
    tr: "<b>Deneme alanı</b>: kendi verinizle eğitin, ağın içine bakın",
    en: "<b>Playground</b>: train on your own data, look inside the network",
  },
  { tr: "<b>Matematik</b>: formüller ve adım adım türetmeler", en: "<b>Math</b>: the formulas and step-by-step derivations" },
  { tr: "<b>Kodlama</b>: aynı modeli Python'da sıfırdan yazmak", en: "<b>Coding</b>: writing the same model in Python from scratch" },
];

function html(tag: string, text: Text, attrs: Record<string, string> = {}): HTMLElement {
  const node = el(tag, attrs);
  node.innerHTML = L(text); // static, author-written markup
  return node;
}

function chapterCard(c: (typeof CHAPTERS)[number]): HTMLElement {
  const body = [
    el("span", { class: "lp-num" }, String(c.number)),
    el(
      "span",
      { class: "lp-text" },
      el("span", { class: "lp-title" }, L(c.title)),
      el("span", { class: "lp-sub" }, L(c.sub)),
      el("span", { class: "lp-meta" }, c.url ? c.year : `${c.year} · ${L({ tr: "yakında", en: "coming soon" })}`),
    ),
    c.url ? el("span", { class: "lp-go", "aria-hidden": "true" }, "→") : null,
  ];
  return el("li", {}, c.url ? el("a", { class: "lp-chapter", href: c.url }, ...body) : el("div", { class: "lp-chapter soon" }, ...body));
}

/** How the site can be used without anything installed (GitHub Pages, or the downloaded copy). */
function offlineCard(offline: boolean): HTMLElement {
  return el(
    "section",
    { class: "card lp-offline" },
    html("h2", { tr: "Kurulum gerekmez", en: "Nothing to install" }),
    html("p", {
      tr: "Her şey tarayıcınızda çalışır: sunucu, Python ya da internet bağlantısı gerekmez. Hesaplar bu bilgisayarda yapılır, hiçbir veri bir yere gönderilmez.",
      en: "Everything runs in your browser: no server, no Python and no internet connection needed. All computations happen on this computer; no data is sent anywhere.",
    }),
    offline
      ? html("p", {
          tr: "Şu an bu sitenin bilgisayarınızdaki kopyasını kullanıyorsunuz.",
          en: "You are using the copy of this site on your computer.",
        })
      : html("p", {
          tr: `İnternetsiz kullanmak için <a href="${ZIP}" download>sitenin ZIP dosyasını indirin</a>, bir klasöre açın ve <code>index.html</code>'e çift tıklayın.`,
          en: `To use it offline, <a href="${ZIP}" download>download the site as a ZIP file</a>, unpack it into a folder and double-click <code>index.html</code>.`,
        }),
    html("p", {
      tr: `Kaynak kod, Python sürümleri ve sunuculu panel: <a href="${REPO}" rel="noopener">GitHub</a>.`,
      en: `Source code, the Python versions and the dashboard with a server: <a href="${REPO}" rel="noopener">GitHub</a>.`,
    }),
  );
}

/** The same pages served by the Python dashboard (dashboard.py), which trains the CNN with PyTorch. */
function serverCard(): HTMLElement {
  return el(
    "section",
    { class: "card lp-offline" },
    html("h2", { tr: "Python paneli", en: "The Python dashboard" }),
    html("p", {
      tr: "Bu sayfaları bu bilgisayardaki Python paneli (<code>dashboard.py</code>) sunuyor. CNN bölümünde eğitim, tarayıcı yerine burada PyTorch ile çalışır ve model <code>checkpoints/simple_cnn.pt</code> dosyasına kaydedilir; diğer her şey sitedekiyle aynıdır.",
      en: "These pages are served by the Python dashboard (<code>dashboard.py</code>) on this computer. In the CNN chapter, training runs here with PyTorch instead of in the browser, and the model is saved to <code>checkpoints/simple_cnn.pt</code>; everything else is the same as on the site.",
    }),
    html("p", {
      tr: `Kaynak kod ve çevrimiçi sürüm: <a href="${REPO}" rel="noopener">GitHub</a>.`,
      en: `Source code and the online version: <a href="${REPO}" rel="noopener">GitHub</a>.`,
    }),
  );
}

function render(): void {
  const main = document.getElementById("landing");
  if (!main) return;
  const offline = location.protocol === "file:";
  const first = CHAPTERS.find((c) => c.url);
  main.replaceChildren(
    el(
      "section",
      { class: "card lp-hero" },
      html("h2", { tr: "Tek bir nörondan derin ağlara", en: "From a single neuron to deep networks" }),
      html("p", {
        tr: "Bu derste yapay sinir ağlarını tarihsel sırayla, her adımı tarayıcıda deneyerek öğreneceksiniz. Her bölüm bir öncekinin bıraktığı sorundan başlar ve dört sekmeden oluşur:",
        en: "In this course you learn neural networks in historical order, trying every step in the browser. Each chapter starts from the problem the previous one left open and has four tabs:",
      }),
      el("ul", { class: "lp-tabs" }, ...TABS.map((t) => html("li", t))),
      first
        ? el(
            "a",
            { class: "btn primary lp-start", href: first.url ?? "#" },
            L({ tr: `Başlayın: ${L(first.title)} →`, en: `Start: ${L(first.title)} →` }),
          )
        : null,
    ),
    el(
      "section",
      { class: "lp-path" },
      html("h2", { tr: "Öğrenme yolu", en: "Learning path" }),
      el("ol", { class: "lp-list" }, ...CHAPTERS.map(chapterCard)),
    ),
    onServer ? serverCard() : offlineCard(offline),
  );
  document.documentElement.lang = I18N.lang;
}

wireSwitches();
applyStaticTranslations();
render();
document.addEventListener("lang-change", render);
