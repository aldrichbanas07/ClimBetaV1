import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(msg.text());
});
page.on("pageerror", (err) => errors.push(String(err)));

await page.goto("http://localhost:5173");
await page.waitForSelector("text=Beta sequence");
await page.waitForSelector(".move-row", { timeout: 10000 });
await page.screenshot({ path: "scripts/screenshot-1.png" });

// Select a climb with more moves and drag the angle slider to exercise it.
await page.selectOption("#climb-select", { label: (await page.locator("#climb-select option").allTextContents())[1] });
await page.waitForTimeout(600);
await page.fill("#angle-slider", "70");
await page.locator("#angle-slider").dispatchEvent("input");
await page.waitForTimeout(800);
await page.screenshot({ path: "scripts/screenshot-2.png" });

// Click a move row to open its detail panel.
await page.locator(".move-row").first().click();
await page.waitForTimeout(200);
await page.screenshot({ path: "scripts/screenshot-3.png" });

console.log("console errors:", errors);
await browser.close();
