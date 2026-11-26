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
const XOR_LABELS = [0, 1, 1, 0];
const OR_UNIT: [number, number, number] = [1, 1, -0.5];
const NAND_UNIT: [number, number, number] = [-1, -1, 1.5];
const unit = ([w1, w2, b]: readonly number[], x: readonly number[]) => stepFn(zOf([w1!, w2!], b!, x));
/** The four inputs as the two hidden perceptrons (OR, NAND) see them. */
const HIDDEN = INPUTS.map((x) => [unit(OR_UNIT, x), unit(NAND_UNIT, x)] as const);
const OUTPUT_OPTIONS: [number, number, number][] = [OR_UNIT, NAND_UNIT, [1, 1, -1.5]];
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
        mistakes: [
          {
            value: 0.2,
            why: {
              tr: R`Bias'ı unutmuş olabilirsiniz: \(z\)'ye \(b = 0{,}1\) de eklenir.`,
              en: R`You may have left out the bias: \(b = 0.1\) is added to \(z\) too.`,
            },
          },
          {
            value: 1.1,
            why: {
              tr: R`\(w_2\) negatif: \(w_2 x_2 = -0{,}4\) toplamdan çıkar.`,
              en: R`\(w_2\) is negative: \(w_2 x_2 = -0.4\) takes away from the sum.`,
            },
          },
          {
            value: 1,
            why: {
              tr: R`Bu, çıktı \(\hat y\). Soru, basamaktan önceki \(z\) değerini soruyor.`,
              en: R`That is the output \(\hat y\). The question asks for \(z\), the value before the step.`,
            },
          },
        ],
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
        mistakes: [
          {
            value: 0.5,
            why: {
              tr: R`Bu, sınırın \(x_1\) eksenini kestiği yer (\(x_2 = 0\) iken \(2 x_1 - 1 = 0\)). \(x_2\) ekseninde \(x_1 = 0\)'dır.`,
              en: R`That is where the boundary crosses the \(x_1\) axis (with \(x_2 = 0\), \(2 x_1 - 1 = 0\)). On the \(x_2\) axis, \(x_1 = 0\).`,
            },
          },
          {
            value: -1,
            why: {
              tr: R`İşarete dikkat: \(x_2 - 1 = 0\) denkleminden \(x_2 = +1\) çıkar.`,
              en: R`Watch the sign: \(x_2 - 1 = 0\) gives \(x_2 = +1\).`,
            },
          },
        ],
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
        mistakes: [
          {
            value: 0.5,
            why: {
              tr: R`Tahmin doğru değil: \(z = -0{,}1\) ve \(z \le 0\) olduğu için \(\hat y = 0\), oysa \(y = 1\). Ağırlıklar güncellenir.`,
              en: R`The prediction is not right: \(z = -0.1\), and since \(z \le 0\), \(\hat y = 0\) while \(y = 1\). The weights are updated.`,
            },
          },
          {
            value: 0.3,
            why: {
              tr: R`İşarete dikkat: hata \(y - \hat y = 1 - 0 = +1\), yani \(w_1\) büyür.`,
              en: R`Watch the sign: the error is \(y - \hat y = 1 - 0 = +1\), so \(w_1\) grows.`,
            },
          },
          {
            value: 0.2,
            why: {
              tr: R`Bu yalnızca eklenen miktar. Soru, \(w_1\)'in yeni değerini soruyor.`,
              en: R`That is only the amount added. The question asks for the new value of \(w_1\).`,
            },
          },
        ],
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
        mistakes: [
          {
            value: 0.35,
            why: {
              tr: R`İşarete dikkat: model 1 dedi ama \(y = 0\), yani hata \(y - \hat y = -1\) ve \(b\) küçülür.`,
              en: R`Watch the sign: the model said 1 but \(y = 0\), so the error is \(y - \hat y = -1\) and \(b\) shrinks.`,
            },
          },
          {
            value: 0.1,
            why: {
              tr: R`Tahmin doğru değil: \(z = 0{,}6 > 0\) olduğu için \(\hat y = 1\), oysa \(y = 0\). Bias da güncellenir.`,
              en: R`The prediction is not right: \(z = 0.6 > 0\), so \(\hat y = 1\) while \(y = 0\). The bias is updated too.`,
            },
          },
          {
            value: -0.25,
            why: {
              tr: R`Bu yalnızca eklenen miktar. Soru, \(b\)'nin yeni değerini soruyor.`,
              en: R`That is only the amount added. The question asks for the new value of \(b\).`,
            },
          },
        ],
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
          tr: R`Bir veri kümesinde \(R = 2\) ve marj \(\gamma = 0{,}25\). İki sınıfı birbirine yaklaştırıyoruz: noktalar aynı bölgede kalıyor (\(R\) değişmiyor), marj yarıya iniyor. Yakınsama teoremine göre düzeltme sayısının üst sınırı şimdi kaçtır?`,
          en: R`A dataset has \(R = 2\) and margin \(\gamma = 0.25\). We move the two classes closer: the points stay in the same region (\(R\) does not change) and the margin is halved. By the convergence theorem, what is the upper bound on the number of corrections now?`,
        },
        label: { tr: "yeni üst sınır =", en: "new upper bound =" },
        answer: 256,
        compute: () => (2 / (0.25 / 2)) ** 2,
        hint: {
          tr: R`Sınır \((R/\gamma)^2\)'dir. Yeni marjı bulun, önce bölün, sonra karesini alın.`,
          en: R`The bound is \((R/\gamma)^2\). Find the new margin, divide first, then square.`,
        },
        mistakes: [
          {
            value: 64,
            why: {
              tr: R`Bu, sınıfları yaklaştırmadan önceki sınır: \((2/0{,}25)^2\). Marj artık \(0{,}125\).`,
              en: R`That is the bound before the classes moved: \((2/0.25)^2\). The margin is now \(0.125\).`,
            },
          },
          {
            value: 128,
            why: {
              tr: R`Sınır marjla ters orantılı değil, marjın karesiyle ters orantılı: marj yarıya inince sınır iki değil, dört katına çıkar.`,
              en: R`The bound goes with one over the margin squared, not one over the margin: halving the margin makes it four times larger, not twice.`,
            },
          },
          {
            value: 16,
            why: { tr: R`\(R/\gamma = 16\) doğru; sınır bunun karesidir.`, en: R`\(R/\gamma = 16\) is right; the bound is its square.` },
          },
        ],
        explain: {
          tr: R`Yeni marj \(0{,}125\) ve sınır \((2 / 0{,}125)^2 = 16^2 = 256\): önceki 64'ün dört katı. Sınıflar birbirine ne kadar yakınsa öğrenme o kadar çok düzeltme isteyebilir. Bu bir üst sınırdır; gerçek eğitim çoğu zaman çok daha az düzeltmeyle biter.`,
          en: R`The new margin is \(0.125\) and the bound is \((2 / 0.125)^2 = 16^2 = 256\): four times the earlier 64. The closer the classes, the more corrections learning may need. It is an upper bound; real training often ends with far fewer corrections.`,
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
        q: {
          tr: R`XOR'u hesaplayan bir \((w_1, w_2, b)\) olmadığını hangi akıl yürütme kanıtlar?`,
          en: R`Which argument proves that no \((w_1, w_2, b)\) computes XOR?`,
        },
        options: [
          {
            text: {
              tr: R`Hangi \(\eta\) seçilirse seçilsin eğitimde hatalar bitmiyor, ağırlıklar dönüp duruyor`,
              en: R`Whatever \(\eta\) we pick, training never runs out of mistakes and the weights keep cycling`,
            },
            why: {
              tr: R`Bu bir belirti, kanıt değil: birkaç yüz epoch'ta bitmeyen bir eğitim, daha uzun sürünce bitebilirdi. Kanıt, doğru bir sınırın hiç olmadığını göstermeli.`,
              en: R`That is a symptom, not a proof: training that has not finished after a few hundred epochs might finish later. A proof has to show that no correct boundary exists at all.`,
            },
          },
          {
            text: {
              tr: R`Dört örnek, iki ağırlığı ve bir bias'ı belirlemeye yetmiyor; daha çok örnek gerekiyor`,
              en: R`Four examples are not enough to pin down two weights and a bias; it needs more examples`,
            },
            why: {
              tr: "AND ve OR da dört örnektir ve perceptron onları kolayca öğrenir. Örnek eklemek bir sınırı var etmez.",
              en: "AND and OR also have four examples, and the perceptron learns them easily. Adding examples does not make a boundary exist.",
            },
          },
          {
            text: {
              tr: R`\((0, 1)\) ve \((1, 0)\) için \(w_1 + w_2 + 2b > 0\) gerekiyor, \((0, 0)\) ve \((1, 1)\) için ise \(w_1 + w_2 + 2b \le 0\)`,
              en: R`\((0, 1)\) and \((1, 0)\) need \(w_1 + w_2 + 2b > 0\), while \((0, 0)\) and \((1, 1)\) need \(w_1 + w_2 + 2b \le 0\)`,
            },
          },
          {
            text: {
              tr: R`Ağırlıklar sıfırdan başlayınca \((0, 0)\) için \(z\) hep 0 kalıyor, bu nokta hiç öğrenilemiyor`,
              en: R`With weights that start at zero, \(z\) stays 0 for \((0, 0)\), so that point is never learned`,
            },
            why: {
              tr: R`\((0, 0)\) için \(z = b\)'dir ve \(b\) her hatada değişir. Üstelik başlangıç noktası sonucu değiştirmez: XOR'u ayıran bir doğru hiç yoktur.`,
              en: R`For \((0, 0)\), \(z = b\), and \(b\) changes with every mistake. Besides, the start does not matter: no line separates XOR at all.`,
            },
          },
        ],
        answer: 2,
        explain: {
          tr: R`Sınıf 1 için: \(w_2 + b > 0\) ve \(w_1 + b > 0\); topları \(w_1 + w_2 + 2b > 0\). Sınıf 0 için: \(b \le 0\) ve \(w_1 + w_2 + b \le 0\); topları \(w_1 + w_2 + 2b \le 0\). Aynı sayı hem pozitif hem pozitif olmayan olamaz. Geometrik olarak: aynı sınıftaki köşeler çapraz durur ve tek bir doğru onları ayıramaz.`,
          en: R`For class 1: \(w_2 + b > 0\) and \(w_1 + b > 0\); their sum is \(w_1 + w_2 + 2b > 0\). For class 0: \(b \le 0\) and \(w_1 + w_2 + b \le 0\); their sum is \(w_1 + w_2 + 2b \le 0\). The same number cannot be both positive and not positive. Geometrically: each class sits on opposite corners, and no single line separates them.`,
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
  {
    id: "px-transfer",
    title: { tr: "Hepsi bir arada", en: "Putting it together" },
    intro: {
      tr: R`Bu kısımdaki sorular birkaç kısmı birlikte kullanır: bir sınırı tersinden kurmak, kuralı iki adım izlemek, iki perceptronu bir araya getirmek. Son üç soru bir sonraki bölüme, çok katmanlı ağlara, köprüdür.`,
      en: R`The questions here use several sections at once: building a boundary backwards, following the rule for two steps, putting two perceptrons together. The last three lead into the next chapter, multilayer networks.`,
    },
    read: { tab: "theory" },
    questions: [
      {
        id: "t-line",
        kind: "number",
        q: {
          tr: R`Bir perceptronun sınırı \(x_1\) eksenini 2'de, \(x_2\) eksenini 4'te kesiyor ve \((0, 0)\) noktası sınıf 0'da. \(w_1 = 2\) ise \(b\) kaçtır?`,
          en: R`A perceptron's boundary crosses the \(x_1\) axis at 2 and the \(x_2\) axis at 4, and the point \((0, 0)\) is in class 0. If \(w_1 = 2\), what is \(b\)?`,
        },
        label: { tr: R`\(b =\)`, en: R`\(b =\)` },
        answer: -4,
        compute: () => {
          const b = -2 * 2; // the point (2, 0) is on the boundary: 2·2 + b = 0
          const w2 = -b / 4; // so is (0, 4): 4·w2 + b = 0
          return zOf([2, w2], b, [2, 0]) === 0 && zOf([2, w2], b, [0, 4]) === 0 && stepFn(zOf([2, w2], b, [0, 0])) === 0 ? b : NaN;
        },
        hint: {
          tr: R`\((2, 0)\) noktası sınırın üzerinde: orada \(z = 0\) olur. Sonra \((0, 0)\)'ın hangi tarafta kaldığına bakın.`,
          en: R`The point \((2, 0)\) is on the boundary, so \(z = 0\) there. Then check which side \((0, 0)\) is on.`,
        },
        mistakes: [
          {
            value: 4,
            why: {
              tr: R`İşarete dikkat: \((2, 0)\) sınırda olduğu için \(2\cdot 2 + b = 0\), yani \(b = -4\). Böylece \((0, 0)\)'da \(z = b < 0\) olur ve orijin sınıf 0'da kalır.`,
              en: R`Watch the sign: \((2, 0)\) is on the boundary, so \(2\cdot 2 + b = 0\) and \(b = -4\). Then \(z = b < 0\) at \((0, 0)\), and the origin stays in class 0.`,
            },
          },
          {
            value: -2,
            why: {
              tr: R`Kesişim noktası ağırlıkla çarpılır: sınırda \(w_1\cdot 2 + b = 0\), \(w_1 = 2\) olduğu için \(b = -4\).`,
              en: R`The crossing point is multiplied by the weight: on the boundary \(w_1\cdot 2 + b = 0\), and with \(w_1 = 2\), \(b = -4\).`,
            },
          },
        ],
        explain: {
          tr: R`\((2, 0)\) sınırda: \(2\cdot 2 + b = 0\), yani \(b = -4\). \((0, 4)\) da sınırda: \(4 w_2 - 4 = 0\), yani \(w_2 = 1\). Perceptron \((2,\ 1,\ -4)\) olur; \((0, 0)\)'da \(z = -4 < 0\), orijin sınıf 0'da. Aynı doğruyu \((-2,\ -1,\ 4)\) de verir, ama o zaman orijin sınıf 1'e geçer.`,
          en: R`\((2, 0)\) is on the boundary: \(2\cdot 2 + b = 0\), so \(b = -4\). So is \((0, 4)\): \(4 w_2 - 4 = 0\), so \(w_2 = 1\). The perceptron is \((2,\ 1,\ -4)\); at \((0, 0)\), \(z = -4 < 0\), so the origin is in class 0. \((-2,\ -1,\ 4)\) gives the same line, but then the origin moves to class 1.`,
        },
      },
      {
        id: "t-trace",
        kind: "number",
        q: {
          tr: R`Sıfır ağırlıklarla (\(\mathbf w = (0, 0)\), \(b = 0\)) ve \(\eta = 0{,}5\) ile başlıyoruz. Önce \(\mathbf x = (1, 0)\), \(y = 1\) örneği, sonra \(\mathbf x = (1, 1)\), \(y = 0\) örneği geliyor. Bu iki adımdan sonra ilk örnek \((1, 0)\) için \(z\) kaçtır?`,
          en: R`We start from zero weights (\(\mathbf w = (0, 0)\), \(b = 0\)) with \(\eta = 0.5\). First the example \(\mathbf x = (1, 0)\), \(y = 1\) comes, then \(\mathbf x = (1, 1)\), \(y = 0\). After these two steps, what is \(z\) for the first example \((1, 0)\)?`,
        },
        label: { tr: R`\(z =\)`, en: R`\(z =\)` },
        answer: 0,
        compute: () => {
          const model = { w: [0, 0], b: 0 };
          ruleStep(model, [1, 0], 1, 0.5);
          ruleStep(model, [1, 1], 0, 0.5);
          return zOf(model.w, model.b, [1, 0]);
        },
        hint: {
          tr: R`Her adımda önce \(z\)'yi ve \(\hat y\)'yi bulun, tahmin yanlışsa kuralı uygulayın. İkinci adımda ilk adımın ağırlıklarıyla başlayın.`,
          en: R`In each step find \(z\) and \(\hat y\) first, and apply the rule if the prediction is wrong. Start the second step from the weights of the first.`,
        },
        mistakes: [
          {
            value: 1,
            why: {
              tr: R`İkinci örnek de bir hata: ilk adımdan sonra \((1, 1)\) için \(z = 0{,}5 + 0 + 0{,}5 = 1 > 0\), yani \(\hat y = 1\), oysa \(y = 0\). Ağırlıklar yeniden güncellenir.`,
              en: R`The second example is a mistake too: after the first step, \(z = 0.5 + 0 + 0.5 = 1 > 0\) for \((1, 1)\), so \(\hat y = 1\) while \(y = 0\). The weights are updated again.`,
            },
          },
          {
            value: 2,
            why: {
              tr: R`İkinci adımda işarete dikkat: hata \(y - \hat y = 0 - 1 = -1\), yani ağırlıklar ve bias küçülür.`,
              en: R`Watch the sign in the second step: the error is \(y - \hat y = 0 - 1 = -1\), so the weights and the bias shrink.`,
            },
          },
        ],
        explain: {
          tr: R`1. adım: \(z = 0\), \(\hat y = 0\), hata \(+1\): \(\mathbf w = (0{,}5;\ 0)\), \(b = 0{,}5\). 2. adım: \(z = 1 > 0\), \(\hat y = 1\), hata \(-1\): \(\mathbf w = (0;\ -0{,}5)\), \(b = 0\). Şimdi \((1, 0)\) için \(z = 0\) ve \(\hat y = 0\): ikinci düzeltme birinci örneği yeniden yanlış yaptı. Kural bu yüzden veriyi epoch epoch dolaşır; veri ayrılabiliyorsa sonunda hepsini birden doğru sınıflayan ağırlıklara varır.`,
          en: R`Step 1: \(z = 0\), \(\hat y = 0\), error \(+1\): \(\mathbf w = (0.5,\ 0)\), \(b = 0.5\). Step 2: \(z = 1 > 0\), \(\hat y = 1\), error \(-1\): \(\mathbf w = (0,\ -0.5)\), \(b = 0\). Now \(z = 0\) and \(\hat y = 0\) for \((1, 0)\): the second correction made the first example wrong again. That is why the rule goes over the data epoch after epoch; if the data is separable, it ends at weights that get every example right at once.`,
        },
      },
      {
        id: "t-hidden",
        kind: "number",
        q: {
          tr: R`İki perceptronu yan yana koyuyoruz: \(h_1\) OR kapısıdır, \((1,\ 1,\ -0{,}5)\); \(h_2\) NAND kapısıdır, \((-1,\ -1,\ 1{,}5)\). XOR'un dört girdisini \((h_1, h_2)\) noktalarına çeviriyoruz. Kaç farklı nokta çıkar?`,
          en: R`We put two perceptrons side by side: \(h_1\) is the OR gate, \((1,\ 1,\ -0.5)\); \(h_2\) is the NAND gate, \((-1,\ -1,\ 1.5)\). We turn the four XOR inputs into points \((h_1, h_2)\). How many different points come out?`,
        },
        label: { tr: "farklı nokta sayısı =", en: "number of different points =" },
        answer: 3,
        compute: () => new Set(HIDDEN.map((h) => h.join(","))).size,
        hint: {
          tr: R`Dört girdinin her biri için \(h_1\) ve \(h_2\)'yi ayrı ayrı hesaplayın, sonra aynı çıkan noktaları bir sayın.`,
          en: R`Work out \(h_1\) and \(h_2\) for each of the four inputs, then count the points that come out the same only once.`,
        },
        mistakes: [
          {
            value: 4,
            why: {
              tr: R`\((0, 1)\) ve \((1, 0)\) girdilerini hesaplayın: ikisinde de OR 1, NAND 1 der. İkisi aynı noktaya, \((1, 1)\)'e gider.`,
              en: R`Work out the inputs \((0, 1)\) and \((1, 0)\): for both, OR says 1 and NAND says 1. They go to the same point, \((1, 1)\).`,
            },
          },
        ],
        explain: {
          tr: R`\((0, 0) \to (0, 1)\); \((0, 1)\) ve \((1, 0) \to (1, 1)\); \((1, 1) \to (1, 0)\). XOR'un 1 dediği iki girdi aynı noktaya düştü, 0 dediği iki girdi ise ayrı iki köşeye. Gizli katman veriyi böyle yeni bir gösterime çevirir. Bu gösterimde sınıflar artık çapraz durmuyor.`,
          en: R`\((0, 0) \to (0, 1)\); \((0, 1)\) and \((1, 0) \to (1, 1)\); \((1, 1) \to (1, 0)\). The two inputs where XOR says 1 fell on the same point, the two where it says 0 on two other corners. A hidden layer turns the data into a new representation like this. In it, the classes no longer sit on opposite corners.`,
        },
      },
      {
        id: "t-and",
        kind: "choice",
        q: {
          tr: R`Önceki sorudaki \((h_1, h_2)\) noktalarını üçüncü bir perceptrona veriyoruz. Hangi \((w_1, w_2, b)\) üçlüsü dört girdinin hepsinde XOR'u verir?`,
          en: R`We give the points \((h_1, h_2)\) of the last question to a third perceptron. Which \((w_1, w_2, b)\) gives XOR for all four inputs?`,
        },
        options: [
          {
            text: { tr: R`\((1,\ 1,\ -0{,}5)\), OR`, en: R`\((1,\ 1,\ -0.5)\), OR` },
            why: {
              tr: R`OR, \((0, 1)\) ve \((1, 0)\) noktalarında da 1 der. Oysa bu noktalar \((0, 0)\) ve \((1, 1)\) girdilerinden gelir, XOR onlara 0 der.`,
              en: R`OR says 1 at the points \((0, 1)\) and \((1, 0)\) too. But those points come from the inputs \((0, 0)\) and \((1, 1)\), where XOR says 0.`,
            },
          },
          {
            text: { tr: R`\((-1,\ -1,\ 1{,}5)\), NAND`, en: R`\((-1,\ -1,\ 1.5)\), NAND` },
            why: {
              tr: R`NAND yalnızca \((1, 1)\) noktasında 0 der: XOR'un tam tersini verir.`,
              en: R`NAND says 0 only at the point \((1, 1)\): it gives the exact opposite of XOR.`,
            },
          },
          { text: { tr: R`\((1,\ 1,\ -1{,}5)\), AND`, en: R`\((1,\ 1,\ -1.5)\), AND` } },
          {
            text: { tr: "Hiçbiri: noktalar hâlâ tek doğruyla ayrılamaz", en: "None: the points still cannot be separated by one line" },
            why: {
              tr: R`Gizli katman XOR'un 1 dediği iki girdiyi \((1, 1)\) noktasında topladı. Bu noktayı öbür ikisinden tek bir doğru ayırır.`,
              en: R`The hidden layer gathered the two inputs where XOR says 1 at the point \((1, 1)\). One line separates that point from the other two.`,
            },
          },
        ],
        answer: 2,
        compute: () => {
          const k = OUTPUT_OPTIONS.findIndex((u) => HIDDEN.every((h, i) => unit(u, h) === XOR_LABELS[i]));
          return k < 0 ? 3 : k;
        },
        explain: {
          tr: R`XOR = AND(OR, NAND): "en az biri 1" ve "ikisi birden 1 değil". Üç perceptrondan oluşan bu iki katmanlı ağ XOR'u hesaplar. Ağırlıkları burada biz seçtik; MLP bölümünde ağ onları kendisi öğrenir.`,
          en: R`XOR = AND(OR, NAND): "at least one is 1" and "not both are 1". This two-layer network of three perceptrons computes XOR. Here we chose the weights by hand; in the MLP chapter the network learns them by itself.`,
        },
      },
      {
        id: "t-target",
        kind: "choice",
        q: {
          tr: "Aynı iki katmanlı ağı ağırlıkları vermeden, perceptron kuralıyla eğitmek istiyoruz. Çıktı nöronu kuralı uygulayabilir. Gizli nöronlar neden uygulayamaz?",
          en: "We want to train the same two-layer network with the perceptron rule, without giving it the weights. The output neuron can apply the rule. Why can the hidden neurons not?",
        },
        options: [
          {
            text: {
              tr: "Gizli katmanda ağırlık çok fazla olduğu için kural çok yavaşlar",
              en: "The hidden layer has too many weights, so the rule gets too slow",
            },
            why: {
              tr: R`Sayı sorun değil: kural her ağırlığı ayrı ayrı ve kolayca günceller. Eksik olan, kuralın içindeki \(y\)'dir.`,
              en: R`The number is no problem: the rule updates each weight on its own, cheaply. What is missing is the \(y\) in the rule.`,
            },
          },
          {
            text: {
              tr: "Basamak fonksiyonunun türevi olmadığı için kural hesaplanamaz",
              en: "The step function has no derivative, so the rule cannot be computed",
            },
            why: {
              tr: R`Perceptron kuralı basamağın türevini kullanmaz, yalnızca \(y - \hat y\) farkına bakar. Gizli nöronda eksik olan bu farkın \(y\)'sidir. Türev, MLP bölümünde gizli katmanı gradyanla eğitmeye geçince önem kazanır.`,
              en: R`The perceptron rule does not use the step's derivative; it only looks at the difference \(y - \hat y\). In a hidden neuron, the \(y\) of that difference is what is missing. The derivative matters once the MLP chapter trains the hidden layer with gradients.`,
            },
          },
          {
            text: {
              tr: "Gizli nöronların her biri XOR'u tek başına çözemediği için",
              en: "Each hidden neuron cannot solve XOR on its own",
            },
            why: {
              tr: "Çözmeleri gerekmez: önceki soruda biri OR, biri NAND hesaplıyordu ve birlikte XOR'u çözdüler.",
              en: "They do not need to: in the last question one computed OR, the other NAND, and together they solved XOR.",
            },
          },
          {
            text: {
              tr: R`Kural her nöron için bir doğru cevap \(y\) ister; gizli nöronlar için veride böyle bir hedef yok`,
              en: R`The rule needs a right answer \(y\) for each neuron; the data has no such target for the hidden neurons`,
            },
          },
        ],
        answer: 3,
        explain: {
          tr: R`Kural \(y - \hat y\) ile çalışır. Çıktı nöronunun \(y\)'si veride yazar; bir gizli nöronun ne vermesi gerektiğini ise veri söylemez. Yukarıda OR ile NAND'i biz seçmiştik. MLP bölümü bu sorunu çözer: basamak yerine türevi olan sigmoid kullanır ve çıktıdaki hatayı gizli nöronlara geri yayar.`,
          en: R`The rule works with \(y - \hat y\). The output neuron's \(y\) is in the data; what a hidden neuron should give, the data does not say. Above, we chose OR and NAND ourselves. The MLP chapter solves this: it uses the sigmoid, which has a derivative, instead of the step, and passes the error at the output back to the hidden neurons.`,
        },
      },
    ],
  },
];
