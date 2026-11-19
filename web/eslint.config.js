import js from "@eslint/js";
import { defineConfig } from "eslint/config";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig(
  { ignores: ["dist/", "node_modules/", "test-results/", "playwright-report/"] },
  js.configs.recommended,
  tseslint.configs.recommended,
  {
    languageOptions: { ecmaVersion: 2023, sourceType: "module", globals: { ...globals.browser } },
    rules: {
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrors: "none" }],
    },
  },
  {
    files: ["scripts/**/*.mjs", "tests/**/*.ts", "*.config.{js,ts}"],
    languageOptions: { globals: { ...globals.node } },
  },
);
