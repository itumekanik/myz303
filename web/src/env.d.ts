/** Python sources are bundled as text (esbuild's text loader, scripts/build.mjs). */
declare module "*.py" {
  const text: string;
  export default text;
}
