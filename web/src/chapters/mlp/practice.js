/* Chapter 2, Practice tab: the questions of questions.ts, shown by shared/practice-view.js. */
import { createPractice } from "../../shared/practice-view.js";
import { PRACTICE } from "./questions.ts";

createPractice({ main: "#px-main", toc: "#px-toc", summary: "#px-summary", sections: PRACTICE });
