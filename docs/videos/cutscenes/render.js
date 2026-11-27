// Render docs/videos/cutscenes/frames/*.svg to 1920x1080 PNGs (Playwright).
// lower-third.png keeps a transparent background, for overlaying.
// Usage (from the repository root):
//   NODE_PATH=$(npm root -g) node docs/videos/cutscenes/render.js [prefix ...]
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const dir = path.join(__dirname, "frames");
  const only = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
  });
  const files = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".svg"))
    .filter((f) => !only.length || only.some((o) => f.startsWith(o)));
  for (const file of files) {
    await page.goto("file://" + path.join(dir, file));
    await page.waitForTimeout(300);
    const png = path.join(dir, file.replace(/\.svg$/, ".png"));
    await page.screenshot({ path: png, omitBackground: file.startsWith("lower") });
    console.log("wrote", path.relative(process.cwd(), png));
  }
  await browser.close();
})();
