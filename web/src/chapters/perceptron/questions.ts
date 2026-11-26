/**
 * Chapter 1, Practice tab: the questions, in the order of the Theory sections. Every number here
 * is worked out again by `compute` or `run` with the chapter's own code: tests/unit/practice.test.ts
 * fails if a stated answer and the model disagree.
 */
import type { PracticeSection } from "../../shared/practice.ts";
import { ruleStep, stepFn, trainFromZero, zOf, type Example } from "./model.ts";

const R = String.raw;
const INPUTS = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
] as const;
const table = (labels: readonly number[]): Example[] => INPUTS.map((x, i) => ({ x, y: labels[i]! }));

const AND = [0, 0, 0, 1];
const AND_OPTIONS: [number, number, number][] = [
  [1, 1, -0.5],
  [1, 1, -1.5],
  [-1, -1, 1.5],
  [1, 1, 0.5],
];

export const PRACTICE: PracticeSection[] = [
  {
    id: "px-neuron",
    title: { tr: "Perceptron nasıl karar verir?", en: "How does a perceptron decide?" },
    intro: {
      tr: R`Çıktı iki adımda hesaplanır: \(z = w_1 x_1 + w_2 x_2 + b\), sonra \(z > 0\) ise \(\hat y = 1\), değilse \(\hat y = 0\).`,
      en: R`The output takes two steps: \(z = w_1 x_1 + w_2 x_2 + b\), then \(\hat y = 1\) if \(z > 0\), otherwise \(\hat y = 0\).`,
    },
    read: { tab: "theory", section: "p-neuron" },
    questions: [
      {
        id: "n-z",
        kind: "number",
        q: {
          tr: R`Ağırlıklar \(w_1 = 0{,}6\) ve \(w_2 = -0{,}4\), bias \(b = 0{,}1\). Girdi \(\mathbf x = (1, 1)\) için \(z\) kaçtır?`,
          en: R`The weights are \(w_1 = 0.6\) and \(w_2 = -0.4\), the bias is \(b = 0.1\). What is \(z\) for the input \(\mathbf x = (1, 1)\)?`,
        },
        label: { tr: R`\(z =\)`, en: R`\(z =\)` },
        answer: 0.3,
        compute: () => zOf([0.6, -0.4], 0.1, [1, 1]),
        hint: { tr: R`Üç terimi toplayın: \(w_1 x_1 + w_2 x_2 + b\).`, en: R`Add the three terms: \(w_1 x_1 + w_2 x_2 + b\).` },
        explain: {
          tr: R`\(z = 0{,}6\cdot 1 + (-0{,}4)\cdot 1 + 0{,}1 = 0{,}3\). \(z > 0\) olduğu için çıktı \(\hat y = 1\) olur.`,
          en: R`\(z = 0.6\cdot 1 + (-0.4)\cdot 1 + 0.1 = 0.3\). Since \(z > 0\), the output is \(\hat y = 1\).`,
        },
      },
      {
        id: "n-zero",
        kind: "choice",
        q: {
          tr: R`\(z\) tam olarak 0 çıkarsa perceptronun çıktısı nedir?`,
          en: R`If \(z\) is exactly 0, what does the perceptron output?`,
        },
        options: [
          { text: { tr: R`\(\hat y = 0\)`, en: R`\(\hat y = 0\)` } },
          {
            text: { tr: R`\(\hat y = 1\)`, en: R`\(\hat y = 1\)` },
            why: {
              tr: R`Çıktının 1 olması için \(z > 0\) gerekir; eşitlik bu koşula girmez.`,
              en: R`An output of 1 needs \(z > 0\); equality does not meet that condition.`,
            },
          },
          {
            text: { tr: "Tanımsızdır", en: "It is undefined" },
            why: {
              tr: R`Basamak fonksiyonu her \(z\) için bir değer verir, 0 için de.`,
              en: R`The step function gives a value for every \(z\), including 0.`,
            },
          },
        ],
        answer: 0,
        compute: () => stepFn(0),
        explain: {
          tr: R`Kural \(z > 0\) ise 1, \(z \le 0\) ise 0'dır. Yani tam sınırın üzerindeki noktalar sınıf 0'a düşer.`,
          en: R`The rule is 1 if \(z > 0\) and 0 if \(z \le 0\), so points exactly on the boundary fall into class 0.`,
        },
      },
      {
        id: "n-and",
        kind: "choice",
        q: { tr: R`Hangi \((w_1, w_2, b)\) üçlüsü AND kapısını hesaplar?`, en: R`Which \((w_1, w_2, b)\) computes the AND gate?` },
        options: [
          {
            text: { tr: R`\((1,\ 1,\ -0{,}5)\)`, en: R`\((1,\ 1,\ -0.5)\)` },
            why: {
              tr: R`Girdilerden biri 1 olunca bile \(z = 0{,}5 > 0\) olur: bu OR kapısıdır.`,
              en: R`Even with one input at 1, \(z = 0.5 > 0\): this is the OR gate.`,
            },
          },
          { text: { tr: R`\((1,\ 1,\ -1{,}5)\)`, en: R`\((1,\ 1,\ -1.5)\)` } },
          {
            text: { tr: R`\((-1,\ -1,\ 1{,}5)\)`, en: R`\((-1,\ -1,\ 1.5)\)` },
            why: {
              tr: R`Yalnızca \((1, 1)\) için \(z < 0\) olur, yani çıktı AND'in tersidir: bu NAND kapısıdır.`,
              en: R`Only \((1, 1)\) gives \(z < 0\), so the output is the opposite of AND: this is the NAND gate.`,
            },
          },
          {
            text: { tr: R`\((1,\ 1,\ 0{,}5)\)`, en: R`\((1,\ 1,\ 0.5)\)` },
            why: {
              tr: R`\((0, 0)\) için bile \(z = 0{,}5 > 0\) olur: bu perceptron her girdiye 1 der.`,
              en: R`Even \((0, 0)\) gives \(z = 0.5 > 0\): this perceptron says 1 for every input.`,
            },
          },
        ],
        answer: 1,
        compute: () => AND_OPTIONS.findIndex(([w1, w2, b]) => INPUTS.every((x, i) => stepFn(zOf([w1, w2], b, x)) === AND[i])),
        explain: {
          tr: R`\(z = x_1 + x_2 - 1{,}5\) yalnızca iki girdi de 1 iken pozitiftir: \(1 + 1 - 1{,}5 = 0{,}5\). Diğer üç girdide \(z < 0\) olur.`,
          en: R`\(z = x_1 + x_2 - 1.5\) is positive only when both inputs are 1: \(1 + 1 - 1.5 = 0.5\). The other three inputs give \(z < 0\).`,
        },
      },
    ],
  },
  {
    id: "px-boundary",
    title: { tr: "Karar sınırının geometrisi", en: "The geometry of the boundary" },
    intro: {
      tr: R`Karar sınırı \(z = 0\) olan noktalardır. İki girdide bu bir doğrudur: \(w_1 x_1 + w_2 x_2 + b = 0\).`,
      en: R`The decision boundary is the set of points where \(z = 0\). With two inputs it is a line: \(w_1 x_1 + w_2 x_2 + b = 0\).`,
    },
    read: { tab: "theory", section: "p-boundary" },
    questions: [
      {
        id: "b-cross",
        kind: "number",
        q: {
          tr: R`\(\mathbf w = (2, 1)\) ve \(b = -1\) olan perceptronun sınırı \(x_2\) eksenini hangi \(x_2\) değerinde keser?`,
          en: R`The perceptron has \(\mathbf w = (2, 1)\) and \(b = -1\). At which value of \(x_2\) does its boundary cross the \(x_2\) axis?`,
        },
        label: { tr: R`\(x_2 =\)`, en: R`\(x_2 =\)` },
        answer: 1,
        compute: () => -zOf([2, 1], -1, [0, 0]) / 1, // on the axis z = w2·x2 + b, so x2 = -b / w2
        hint: {
          tr: R`\(x_2\) ekseninde \(x_1 = 0\)'dır. Bunu sınır denklemine koyun.`,
          en: R`On the \(x_2\) axis, \(x_1 = 0\). Put that into the boundary equation.`,
        },
        explain: {
          tr: R`\(2\cdot 0 + 1\cdot x_2 - 1 = 0\) denkleminden \(x_2 = 1\) bulunur.`,
          en: R`From \(2\cdot 0 + 1\cdot x_2 - 1 = 0\) we get \(x_2 = 1\).`,
        },
      },
      {
        id: "b-normal",
        kind: "choice",
        q: {
          tr: R`\(\mathbf w\) vektörü karar sınırına göre nasıl durur?`,
          en: R`How does the vector \(\mathbf w\) sit relative to the decision boundary?`,
        },
        options: [
          {
            text: { tr: "Sınıra paraleldir", en: "It is parallel to the boundary" },
            why: {
              tr: R`Sınır boyunca yürürken \(z\) değişmez; bu ancak \(\mathbf w\) yürüyüş yönüne dik ise mümkündür.`,
              en: R`Walking along the boundary leaves \(z\) unchanged, which is only possible if \(\mathbf w\) is perpendicular to that direction.`,
            },
          },
          {
            text: { tr: "Sınıra diktir ve sınıf 0 tarafını gösterir", en: "It is perpendicular and points to the class 0 side" },
            why: {
              tr: R`\(\mathbf w\) yönünde ilerlemek \(z\)'yi artırır, yani sınıf 1 tarafına götürür.`,
              en: R`Moving along \(\mathbf w\) increases \(z\), so it leads to the class 1 side.`,
            },
          },
          { text: { tr: "Sınıra diktir ve sınıf 1 tarafını gösterir", en: "It is perpendicular and points to the class 1 side" } },
          {
            text: { tr: R`Yönü yalnızca \(b\)'ye bağlıdır`, en: R`Its direction depends only on \(b\)` },
            why: {
              tr: R`\(b\) sınırı kaydırır ama \(\mathbf w\)'nin yönünü değiştirmez.`,
              en: R`\(b\) shifts the boundary but does not change the direction of \(\mathbf w\).`,
            },
          },
        ],
        answer: 2,
        explain: {
          tr: R`\(\mathbf w\) sınırın normalidir. Onun yönünde gidildikçe \(z\) büyür, bu yüzden gösterdiği taraf \(z > 0\) yani sınıf 1 tarafıdır.`,
          en: R`\(\mathbf w\) is the normal of the boundary. Moving along it makes \(z\) grow, so the side it points to is \(z > 0\), the class 1 side.`,
        },
      },
      {
        id: "b-negate",
        kind: "choice",
        q: {
          tr: R`\(\mathbf w\) ve \(b\)'nin ikisini birden \(-1\) ile çarparsak ne olur?`,
          en: R`What happens if we multiply both \(\mathbf w\) and \(b\) by \(-1\)?`,
        },
        options: [
          {
            text: { tr: "Hiçbir şey değişmez", en: "Nothing changes" },
            why: {
              tr: R`Sınır aynı kalır ama her noktada \(z\)'nin işareti ters döner, dolayısıyla kararlar da değişir.`,
              en: R`The boundary stays, but the sign of \(z\) flips at every point, so the decisions change too.`,
            },
          },
          { text: { tr: "Sınır aynı kalır, iki taraf yer değiştirir", en: "The boundary stays, the two sides swap" } },
          {
            text: { tr: "Sınır 90° döner", en: "The boundary turns by 90°" },
            why: {
              tr: R`\(-z = 0\) ile \(z = 0\) aynı noktaları verir, yani doğru dönmez.`,
              en: R`\(-z = 0\) and \(z = 0\) describe the same points, so the line does not turn.`,
            },
          },
          {
            text: { tr: "Sınır orijinin öbür yanına geçer", en: "The boundary moves to the other side of the origin" },
            why: {
              tr: R`\(-z = 0\) ile \(z = 0\) aynı noktaları verir, yani doğru yerinden oynamaz.`,
              en: R`\(-z = 0\) and \(z = 0\) describe the same points, so the line does not move.`,
            },
          },
        ],
        answer: 1,
        explain: {
          tr: R`Yeni çıktı için \(z' = -z\) olur. \(z' = 0\) ile \(z = 0\) aynı doğrudur, ama pozitif taraf artık öbür yandır.`,
          en: R`The new value is \(z' = -z\). \(z' = 0\) and \(z = 0\) are the same line, but the positive side is now the other one.`,
        },
      },
    ],
  },
  {
    id: "px-rule",
    title: { tr: "Öğrenme kuralı", en: "The learning rule" },
    intro: {
      tr: R`Her örnekte: \(\mathbf w \leftarrow \mathbf w + \eta\,(y - \hat y)\,\mathbf x\) ve \(b \leftarrow b + \eta\,(y - \hat y)\).`,
      en: R`For each example: \(\mathbf w \leftarrow \mathbf w + \eta\,(y - \hat y)\,\mathbf x\) and \(b \leftarrow b + \eta\,(y - \hat y)\).`,
    },
    read: { tab: "theory", section: "p-rule" },
    questions: [
      {
        id: "r-correct",
        kind: "choice",
        q: {
          tr: "Perceptron bir örneği doğru sınıflandırırsa ağırlıklara ne olur?",
          en: "If the perceptron classifies an example correctly, what happens to the weights?",
        },
        options: [
          {
            text: { tr: R`\(\eta\) kadar artarlar`, en: R`They grow by \(\eta\)` },
            why: {
              tr: R`Güncelleme \(y - \hat y\) ile çarpılır; doğru tahminde bu çarpan 0'dır.`,
              en: R`The update is multiplied by \(y - \hat y\), which is 0 for a correct prediction.`,
            },
          },
          { text: { tr: "Hiç değişmezler", en: "They do not change" } },
          {
            text: { tr: "Sınırı örneğe yaklaştıracak kadar değişirler", en: "They move the boundary closer to the example" },
            why: {
              tr: R`Perceptron yalnızca hatalardan öğrenir; doğru örnekte hiçbir düzeltme yapmaz.`,
              en: R`The perceptron learns only from mistakes; a correct example causes no correction.`,
            },
          },
          {
            text: { tr: "Sıfırlanırlar", en: "They are reset to zero" },
            why: {
              tr: "Kural ağırlıkları hiçbir zaman sıfırlamaz, yalnızca hata varsa üzerine ekleme yapar.",
              en: "The rule never resets the weights; it only adds to them after a mistake.",
            },
          },
        ],
        answer: 1,
        compute: () => {
          const model = { w: [1, 1], b: -1.5 };
          ruleStep(model, [1, 1], 1, 0.5);
          return model.w[0] === 1 && model.w[1] === 1 && model.b === -1.5 ? 1 : -1;
        },
        explain: {
          tr: R`Doğru tahminde \(y - \hat y = 0\) olduğu için \(\mathbf w\) ve \(b\) aynen kalır. Perceptron yalnızca hata yapınca öğrenir.`,
          en: R`With a correct prediction \(y - \hat y = 0\), so \(\mathbf w\) and \(b\) stay as they are. The perceptron learns only when it makes a mistake.`,
        },
      },
      {
        id: "r-w1",
        kind: "number",
        q: {
          tr: R`\(\mathbf w = (0{,}5,\ -0{,}5)\), \(b = -0{,}1\), \(\eta = 0{,}2\). Örnek: \(\mathbf x = (1, 1)\), \(y = 1\). Bu adımdan sonra \(w_1\) kaç olur?`,
          en: R`\(\mathbf w = (0.5,\ -0.5)\), \(b = -0.1\), \(\eta = 0.2\). The example is \(\mathbf x = (1, 1)\), \(y = 1\). What is \(w_1\) after this step?`,
        },
        label: { tr: R`yeni \(w_1 =\)`, en: R`new \(w_1 =\)` },
        answer: 0.7,
        compute: () => ruleStep({ w: [0.5, -0.5], b: -0.1 }, [1, 1], 1, 0.2).after.w[0]!,
        hint: {
          tr: R`Önce \(z\)'yi ve \(\hat y\)'yi bulun. Tahmin yanlışsa \(w_1\)'e \(\eta\,(y - \hat y)\,x_1\) eklenir.`,
          en: R`First find \(z\) and \(\hat y\). If the prediction is wrong, \(\eta\,(y - \hat y)\,x_1\) is added to \(w_1\).`,
        },
        explain: {
          tr: R`\(z = 0{,}5 - 0{,}5 - 0{,}1 = -0{,}1\), yani \(\hat y = 0\) ve hata \(y - \hat y = 1\). Yeni \(w_1 = 0{,}5 + 0{,}2\cdot 1\cdot 1 = 0{,}7\).`,
          en: R`\(z = 0.5 - 0.5 - 0.1 = -0.1\), so \(\hat y = 0\) and the error is \(y - \hat y = 1\). The new \(w_1 = 0.5 + 0.2\cdot 1\cdot 1 = 0.7\).`,
        },
      },
      {
        id: "r-b",
        kind: "number",
        q: {
          tr: R`\(\mathbf w = (0{,}5,\ 0{,}5)\), \(b = 0{,}1\), \(\eta = 0{,}25\). Örnek: \(\mathbf x = (1, 0)\), \(y = 0\). Bu adımdan sonra \(b\) kaç olur?`,
          en: R`\(\mathbf w = (0.5,\ 0.5)\), \(b = 0.1\), \(\eta = 0.25\). The example is \(\mathbf x = (1, 0)\), \(y = 0\). What is \(b\) after this step?`,
        },
        label: { tr: R`yeni \(b =\)`, en: R`new \(b =\)` },
        answer: -0.15,
        compute: () => ruleStep({ w: [0.5, 0.5], b: 0.1 }, [1, 0], 0, 0.25).after.b,
        hint: {
          tr: R`\(b\)'nin güncellemesinde \(\mathbf x\) yoktur: \(b \leftarrow b + \eta\,(y - \hat y)\). Hatanın işaretine dikkat edin.`,
          en: R`The update of \(b\) has no \(\mathbf x\): \(b \leftarrow b + \eta\,(y - \hat y)\). Watch the sign of the error.`,
        },
        explain: {
          tr: R`\(z = 0{,}5 + 0{,}1 = 0{,}6\), yani \(\hat y = 1\) ve hata \(y - \hat y = -1\). Yeni \(b = 0{,}1 - 0{,}25 = -0{,}15\). \(x_2 = 0\) olduğu için \(w_2\) değişmez.`,
          en: R`\(z = 0.5 + 0.1 = 0.6\), so \(\hat y = 1\) and the error is \(y - \hat y = -1\). The new \(b = 0.1 - 0.25 = -0.15\). Since \(x_2 = 0\), \(w_2\) does not change.`,
        },
      },
    ],
  },
  {
    id: "px-converge",
    title: { tr: "Ne zaman öğrenebilir?", en: "When can it learn?" },
    intro: {
      tr: R`Veri doğrusal olarak ayrılabiliyorsa kural sonlu sayıda düzeltmeyle durur: düzeltme sayısı en çok \((R/\gamma)^2\) olur. Bu kısımdaki tahmin sorularında önce kararınızı verin, sonra gerçek model çalışsın.`,
      en: R`If the data is linearly separable, the rule stops after a finite number of corrections, at most \((R/\gamma)^2\). In the prediction questions here, decide first, then let the real model run.`,
    },
    read: { tab: "theory", section: "p-converge" },
    questions: [
      {
        id: "c-eta",
        kind: "predict",
        q: {
          tr: R`AND kapısını sıfır ağırlıklarla başlatıp iki kez eğitiyoruz: \(\eta = 0{,}25\) ve \(\eta = 1\). Hangisi daha erken hatasız bir epoch'a ulaşır?`,
          en: R`We start the AND gate from zero weights and train it twice: with \(\eta = 0.25\) and with \(\eta = 1\). Which one reaches an epoch without mistakes first?`,
        },
        options: [
          { tr: R`\(\eta = 1\)`, en: R`\(\eta = 1\)` },
          { tr: R`\(\eta = 0{,}25\)`, en: R`\(\eta = 0.25\)` },
          { tr: "İkisi aynı epoch'ta", en: "Both in the same epoch" },
        ],
        answer: 2,
        run: () => {
          const small = trainFromZero(table(AND), 0.25, 100);
          const large = trainFromZero(table(AND), 1, 100);
          const answer = small.epochs === large.epochs ? 2 : large.epochs < small.epochs ? 0 : 1;
          return {
            answer,
            result: {
              tr: R`\(\eta = 0{,}25\): ${small.epochs}. epoch'ta hatasız. \(\eta = 1\): ${large.epochs}. epoch'ta hatasız.`,
              en: R`\(\eta = 0.25\): no mistakes in epoch ${small.epochs}. \(\eta = 1\): no mistakes in epoch ${large.epochs}.`,
            },
          };
        },
        tryIt: { tab: "math", section: "pm-converge" },
        explain: {
          tr: R`Sıfırdan başlayınca \(\eta\) ağırlıkları yalnızca ölçekler: \(\mathbf w\) ve \(b\) hep aynı oranda büyür, \(z\)'nin işareti değişmez. Bu yüzden iki eğitim aynı hataları yapar.`,
          en: R`Starting from zero, \(\eta\) only scales the weights: \(\mathbf w\) and \(b\) grow by the same factor and the sign of \(z\) does not change. So both runs make the same mistakes.`,
        },
      },
      {
        id: "c-flip",
        kind: "predict",
        q: {
          tr: R`AND tablosunda yalnızca \((0, 0)\) örneğinin etiketini 1 yapıyoruz. Perceptron 200 epoch içinde hatasız bir epoch'a ulaşır mı?`,
          en: R`In the AND table we change only the label of \((0, 0)\) to 1. Does the perceptron reach an epoch without mistakes within 200 epochs?`,
        },
        options: [
          { tr: "Evet, birkaç epoch'ta", en: "Yes, in a few epochs" },
          { tr: "Evet, ama 100 epoch'tan sonra", en: "Yes, but after 100 epochs" },
          { tr: "Hayır, hiçbir zaman", en: "No, never" },
        ],
        answer: 2,
        run: () => {
          const r = trainFromZero(table([1, 0, 0, 1]), 0.25, 200);
          const answer = !r.converged ? 2 : r.epochs <= 100 ? 0 : 1;
          return {
            answer,
            result: r.converged
              ? { tr: `${r.epochs}. epoch'ta hatasız.`, en: `No mistakes in epoch ${r.epochs}.` }
              : {
                  tr: `200. epoch'ta hâlâ ${r.lastMistakes} hata var.`,
                  en: `Epoch 200 still has ${r.lastMistakes} mistakes.`,
                },
          };
        },
        tryIt: { tab: "play" },
        explain: {
          tr: R`Artık \((0, 0)\) ve \((1, 1)\) sınıf 1, \((0, 1)\) ve \((1, 0)\) sınıf 0. Aynı sınıflar çapraz durduğu için tek bir doğru onları ayıramaz. Tek bir etiket, veriyi ayrılamaz hâle getirdi.`,
          en: R`Now \((0, 0)\) and \((1, 1)\) are class 1, \((0, 1)\) and \((1, 0)\) class 0. The classes sit on opposite corners, so no single line separates them. One label made the data inseparable.`,
        },
      },
      {
        id: "c-bound",
        kind: "number",
        q: {
          tr: R`Bir veri kümesinde \(R = 2\) ve marj \(\gamma = 0{,}25\). Yakınsama teoremine göre düzeltme sayısının üst sınırı kaçtır?`,
          en: R`A dataset has \(R = 2\) and margin \(\gamma = 0.25\). By the convergence theorem, what is the upper bound on the number of corrections?`,
        },
        label: { tr: "üst sınır =", en: "upper bound =" },
        answer: 64,
        compute: () => (2 / 0.25) ** 2,
        hint: {
          tr: R`Sınır \((R/\gamma)^2\)'dir. Önce bölün, sonra karesini alın.`,
          en: R`The bound is \((R/\gamma)^2\). Divide first, then square.`,
        },
        explain: {
          tr: R`\((2 / 0{,}25)^2 = 8^2 = 64\). Marj yarıya inerse sınır dört katına çıkar: sınıflar birbirine ne kadar yakınsa öğrenme o kadar çok düzeltme isteyebilir.`,
          en: R`\((2 / 0.25)^2 = 8^2 = 64\). Halving the margin makes the bound four times larger: the closer the classes, the more corrections learning may need.`,
        },
      },
    ],
  },
  {
    id: "px-xor",
    title: { tr: "Sınır: XOR", en: "The limit: XOR" },
    intro: {
      tr: R`XOR, tek perceptronun neyi yapamadığını gösterir ve bir sonraki bölüme, çok katmanlı ağlara, kapıyı açar.`,
      en: R`XOR shows what a single perceptron cannot do, and opens the door to the next chapter, multilayer networks.`,
    },
    read: { tab: "theory", section: "p-xor" },
    questions: [
      {
        id: "x-why",
        kind: "choice",
        q: { tr: "Tek bir perceptron XOR'u neden öğrenemez?", en: "Why can a single perceptron not learn XOR?" },
        options: [
          {
            text: { tr: "Öğrenme oranı yeterince küçük seçilmediği için", en: "Because the learning rate is not small enough" },
            why: {
              tr: R`\(\eta\) adımın boyunu değiştirir, ama doğru bir sınır zaten yoksa hiçbir adım onu bulamaz.`,
              en: R`\(\eta\) changes the size of the step, but if no correct boundary exists, no step can find one.`,
            },
          },
          {
            text: { tr: "Dört örnek öğrenmek için çok az olduğu için", en: "Because four examples are too few to learn from" },
            why: {
              tr: "AND ve OR da dört örnektir ve perceptron onları kolayca öğrenir.",
              en: "AND and OR also have four examples, and the perceptron learns them easily.",
            },
          },
          {
            text: {
              tr: "Aynı sınıftaki köşeler çapraz durduğu için tek doğru onları ayıramaz",
              en: "Because each class sits on opposite corners, so no single line separates them",
            },
          },
          {
            text: { tr: "Ağırlıklar sıfırdan başladığı için", en: "Because the weights start from zero" },
            why: {
              tr: "Başlangıç değeri sonucu değiştirmez: hangi ağırlıklarla başlanırsa başlansın XOR'u ayıran bir doğru yoktur.",
              en: "The starting point does not matter: whatever the weights, no line separates XOR.",
            },
          },
        ],
        answer: 2,
        explain: {
          tr: R`Perceptronun kararı bir doğrudur. XOR'da \((0, 1)\) ve \((1, 0)\) bir sınıfta, \((0, 0)\) ve \((1, 1)\) öbür sınıftadır; bu dört noktayı tek doğruyla ayırmak imkânsızdır.`,
          en: R`A perceptron's decision is a line. In XOR, \((0, 1)\) and \((1, 0)\) are in one class, \((0, 0)\) and \((1, 1)\) in the other; no single line can separate these four points.`,
        },
      },
      {
        id: "x-eta",
        kind: "predict",
        q: {
          tr: R`XOR'u üç farklı öğrenme oranıyla 300'er epoch eğitiyoruz: \(\eta = 0{,}01\), \(0{,}25\) ve \(1\). Sonuç ne olur?`,
          en: R`We train XOR for 300 epochs with three learning rates: \(\eta = 0.01\), \(0.25\) and \(1\). What happens?`,
        },
        options: [
          { tr: R`En küçük \(\eta\) öğrenir`, en: R`The smallest \(\eta\) learns it` },
          { tr: R`En büyük \(\eta\) öğrenir`, en: R`The largest \(\eta\) learns it` },
          { tr: "Hiçbiri öğrenemez", en: "None of them learns it" },
        ],
        answer: 2,
        run: () => {
          const xor = table([0, 1, 1, 0]);
          const runs = [0.01, 0.25, 1].map((eta) => trainFromZero(xor, eta, 300));
          const [a, b, c] = runs.map((r) => r.lastMistakes);
          const answer = runs.every((r) => !r.converged) ? 2 : runs[0]!.converged ? 0 : 1;
          return {
            answer,
            result: {
              tr: R`Son epoch'taki hata sayısı: \(\eta = 0{,}01\) için ${a}, \(\eta = 0{,}25\) için ${b}, \(\eta = 1\) için ${c}.`,
              en: R`Mistakes in the last epoch: ${a} for \(\eta = 0.01\), ${b} for \(\eta = 0.25\), ${c} for \(\eta = 1\).`,
            },
          };
        },
        tryIt: { tab: "theory", section: "p-xor" },
        explain: {
          tr: R`Sorun adım boyunda değil, modelde: XOR'u ayıran bir doğru olmadığı için hiçbir \(\eta\) hatasız bir epoch'a ulaşamaz. Ağırlıklar sonsuza kadar döner durur.`,
          en: R`The problem is the model, not the step size: since no line separates XOR, no \(\eta\) can reach an epoch without mistakes. The weights keep cycling forever.`,
        },
      },
      {
        id: "x-feature",
        kind: "predict",
        q: {
          tr: R`Girdilere üçüncü bir özellik ekliyoruz: \(x_3 = x_1 x_2\). Tek perceptron artık XOR'u öğrenir mi?`,
          en: R`We add a third feature to the inputs: \(x_3 = x_1 x_2\). Can a single perceptron learn XOR now?`,
        },
        options: [
          { tr: "Evet, öğrenir", en: "Yes, it can" },
          { tr: "Hayır, yine öğrenemez", en: "No, it still cannot" },
          { tr: R`Yalnızca \(\eta\) çok küçükse öğrenir`, en: R`Only if \(\eta\) is very small` },
        ],
        answer: 0,
        run: () => {
          const points = INPUTS.map(([x1, x2], i) => ({ x: [x1, x2, x1 * x2], y: [0, 1, 1, 0][i]! }));
          const r = trainFromZero(points, 0.25, 200);
          return {
            answer: r.converged ? 0 : 1,
            result: r.converged
              ? {
                  tr: R`\(\eta = 0{,}25\) ile ${r.epochs}. epoch'ta hatasız.`,
                  en: R`With \(\eta = 0.25\), no mistakes in epoch ${r.epochs}.`,
                }
              : { tr: "200 epoch içinde hatasız bir epoch olmadı.", en: "No epoch without mistakes in 200 epochs." },
          };
        },
        tryIt: { tab: "code", section: "pc-exercises" },
        explain: {
          tr: R`Üç boyutta \((1, 1, 1)\) köşesi diğerlerinden ayrılır ve dört nokta bir düzlemle ayrılabilir hâle gelir. Burada doğru özelliği biz seçtik; MLP'nin gizli katmanı böyle özellikleri kendisi öğrenir.`,
          en: R`In three dimensions the corner \((1, 1, 1)\) lifts away from the others, and a plane can separate the four points. Here we chose the right feature by hand; the hidden layer of an MLP learns such features by itself.`,
        },
      },
    ],
  },
];
