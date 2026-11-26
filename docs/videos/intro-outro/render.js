// Render docs/videos/intro-outro/frames/*.svg to 1920x1080 PNGs (Playwright).
// Usage (from the repository root):
//   NODE_PATH=$(npm root -g) node docs/videos/intro-outro/render.js
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const dir = path.join(__dirname, "frames");
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
  });
  for (const file of fs.readdirSync(dir).filter((f) => f.endsWith(".svg"))) {
    await page.goto("file://" + path.join(dir, file));
    await page.waitForTimeout(300);
    const png = path.join(dir, file.replace(/\.svg$/, ".png"));
    await page.screenshot({ path: png });
    console.log("wrote", path.relative(process.cwd(), png));
  }
  await browser.close();
})();
