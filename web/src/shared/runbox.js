/*
 * "How do I run this file?": a Coding tab's Python file to download, then three ways to run it,
 * each folded until opened: Google Colab with nothing to install, Windows, and macOS or Linux.
 * The file comes from the text bundled into the page, so the download works offline, from the
 * ZIP, too. Lesson blocks { run: { … } } in lesson.js and the CNN tutorial render it.
 *
 *   runBox({ file: "Perceptron.py", text, packages: "numpy",
 *            colabStep?: { tr, en }, // a step before pasting into Colab (e.g. choosing a GPU)
 *            colab?: { tr, en },     // what Colab already has
 *            install?: { tr, en },   // a note under the Windows install commands (e.g. the GPU build)
 *            extra?: { tr, en },     // a note at the end (e.g. another file of the chapter)
 *            args?: "train seg",     // command-line arguments the file needs
 *            writefile?: true })     // Colab: save the file with %%writefile and run it with !python
 *                                    // (a file that reads its command line cannot run as a pasted cell)
 */
import { L } from "./i18n.js";
import { h } from "./core.js";
import { copyButton, rich, terminal } from "./codeview.js";

const LINKS = {
  colab: "https://colab.research.google.com/",
  python: "https://www.python.org/downloads/",
  pytorch: "https://pytorch.org/get-started/locally/",
};

/** Rich text whose {name} places become links to the pages above, opened in a new tab. */
function withLinks(text, labels = {}) {
  const fragment = document.createDocumentFragment();
  L(text).split(/\{(\w+)\}/).forEach((part, k) => {
    if (k % 2 === 0) fragment.append(rich(part));
    else fragment.append(h("a", { href: LINKS[part], target: "_blank", rel: "noopener", text: L(labels[part]) || LINKS[part].replace(/^https:\/\/|\/$/g, "") }));
  });
  return fragment;
}

const steps = (items, labels) => h("ol", { class: "tut-list" }, ...items.map((item) => h("li", {}, withLinks(item, labels))));
const note = (text) => (text ? h("p", { class: "run-note" }, withLinks(text)) : null);

function save(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/x-python" }));
  const link = h("a", { href: url, download: name, hidden: true });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function runBox({ file, text, packages, colabStep, colab, install, extra, args = "", writefile = false }) {
  const command = args ? `${file} ${args}` : file;
  const download = h("button", { type: "button", class: "btn small primary run-download", text: `⬇ ${L({ tr: `${file} dosyasını indir`, en: `Download ${file}` })}` });
  download.addEventListener("click", () => save(file, text));
  const copy = copyButton(() => text.replace(/\r\n/g, "\n"), "btn small", { tr: "Tümünü kopyala", en: "Copy all" });

  // The body has its own box: a browser may wrap the content of <details> in a box of its own
  const way = (title, ...body) => h("details", { class: "run-way" }, h("summary", { text: L(title) }), h("div", { class: "run-way-body" }, ...body));
  const run = (prefix) => [
    `$ ${prefix.venv}`,
    `$ ${prefix.python} -m pip install ${packages}`,
    `$ ${prefix.python} ${command}`,
  ];
  const colabSteps = writefile
    ? [
        { tr: `İlk hücrenin en üstüne \`%%writefile ${file}\` yazın, alt satırına **Tümünü kopyala** ile kopyaladığınız dosyayı yapıştırın ve hücreyi **Shift + Enter** ile çalıştırın: dosya Colab'ın diskine kaydedilir.`, en: `At the top of the first cell write \`%%writefile ${file}\`, paste the file copied with **Copy all** below it and run the cell with **Shift + Enter**: the file is saved to Colab's disk.` },
        { tr: "Yeni bir hücrede şu iki satırı çalıştırın (baştaki `!`, satırı Python yerine kabuğa gönderir):", en: "In a new cell run these two lines (the leading `!` sends the line to the shell instead of Python):" },
      ]
    : [
        { tr: "**Tümünü kopyala** ile dosyayı kopyalayıp ilk hücreye yapıştırın.", en: "Copy the file with **Copy all** and paste it into the first cell." },
        { tr: "Hücreyi **Shift + Enter** ile çalıştırın. Çıktı ve grafikler hücrenin altında görünür.", en: "Run the cell with **Shift + Enter**. The output and the plots appear under the cell." },
      ];

  return h("section", { class: "run-box", "aria-label": L({ tr: "Bu dosyayı çalıştırmak", en: "Running this file" }) },
    h("h3", { text: L({ tr: "Bu dosyayı nasıl çalıştırırım?", en: "How do I run this file?" }) }),
    h("p", {}, rich({
      tr: `Proje klasörünüz varsa \`${file}\` kök klasörde hazır. Yoksa dosyayı buradan indirin ya da kopyalayın:`,
      en: `If you have the project folder, \`${file}\` is ready at its root. If not, download or copy the file here:`,
    })),
    h("div", { class: "run-actions" }, download, copy),
    way({ tr: "Hiçbir şey kurmadan: Google Colab", en: "With nothing to install: Google Colab" },
      steps([
        { tr: "{colab} adresinde Google hesabınızla yeni bir not defteri açın.", en: "Open a new notebook at {colab} with your Google account." },
        ...(colabStep ? [colabStep] : []),
        ...colabSteps,
      ]),
      writefile ? terminal(L({ tr: "Colab hücresi", en: "Colab cell" }), [`$ !pip install ${packages}`, `$ !python ${command}`], "") : null,
      note(colab),
      note({
        tr: "Colab, Google'ın ücretsiz bir hizmetidir: kod Google'ın bilgisayarlarında çalışır ve not defteri Google Drive'ınızda saklanır.",
        en: "Colab is a free Google service: the code runs on Google's computers, and the notebook is kept in your Google Drive.",
      })),
    way({ tr: "Kendi bilgisayarımda: Windows", en: "On my computer: Windows" },
      steps([
        { tr: "Python yoksa {python} adresinden kurun (3.10 ya da daha yenisi). Kurulumun ilk ekranında **Add python.exe to PATH** kutusunu işaretleyin.", en: "If you have no Python, install it from {python} (3.10 or newer). On the installer's first screen, tick **Add python.exe to PATH**." },
        { tr: `\`${file}\` dosyasını bir klasöre koyun. Dosya Gezgini'nde klasöre sağ tıklayıp **Terminalde aç**'ı seçin.`, en: `Put \`${file}\` in a folder. In File Explorer, right-click the folder and choose **Open in Terminal**.` },
        { tr: "Şu üç komutu yazın. İlk ikisi yalnızca bir kez gerekir: sanal ortamı (`.venv`) kurar ve kütüphaneleri içine yükler.", en: "Type these three commands. The first two are needed only once: they create the virtual environment (`.venv`) and install the libraries into it." },
      ]),
      terminal(L({ tr: "PowerShell · dosyanın klasöründe", en: "PowerShell · in the file's folder" }), run({ venv: "py -m venv .venv", python: ".venv\\Scripts\\python.exe" })),
      note(install)),
    way({ tr: "Kendi bilgisayarımda: macOS ya da Linux", en: "On my computer: macOS or Linux" },
      steps([
        { tr: "Terminalde `python3 --version` yazın. Python yoksa ya da 3.10'dan eskiyse macOS'ta {python} adresinden, Linux'ta paket yöneticisiyle kurun (Ubuntu'da `sudo apt install python3-venv`).", en: "Type `python3 --version` in a terminal. If there is no Python or it is older than 3.10, install it from {python} on macOS, or with the package manager on Linux (`sudo apt install python3-venv` on Ubuntu)." },
        { tr: `Terminalde \`${file}\` dosyasının bulunduğu klasöre geçin (\`cd\`) ve şu üç komutu yazın. İlk ikisi yalnızca bir kez gerekir.`, en: `In the terminal, change to the folder with \`${file}\` (\`cd\`) and type these three commands. The first two are needed only once.` },
      ]),
      terminal(L({ tr: "Terminal · dosyanın klasöründe", en: "Terminal · in the file's folder" }), run({ venv: "python3 -m venv .venv", python: ".venv/bin/python" }), "$")),
    note(extra));
}
