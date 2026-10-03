"""Record the KL-spring page as a smooth GIF by driving its animation clock frame by frame."""
import io, sys, pathlib
from PIL import Image
from playwright.sync_api import sync_playwright

PAGE = pathlib.Path(sys.argv[1]).resolve().as_uri()
OUT = sys.argv[2]
FPS, SECONDS, WIDTH = 20, 10, 960

# Replace the browser's clock: the page only advances when we call __advance(ms).
FAKE_CLOCK = """
(() => {
  let t = 0, cbs = [];
  performance.now = () => t;
  window.requestAnimationFrame = cb => { cbs.push(cb); return cbs.length; };
  window.__advance = ms => {
    const steps = Math.max(1, Math.round(ms / (1000 / 60)));
    for (let i = 0; i < steps; i++) {
      t += ms / steps;
      const run = cbs; cbs = [];
      run.forEach(cb => cb(t));
    }
  };
})();
"""

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=1)
    page.add_init_script(FAKE_CLOCK)
    page.goto(PAGE)
    page.evaluate("document.documentElement.setAttribute('data-theme', 'light')")
    page.wait_for_function("document.fonts.status === 'loaded'", polling=100)   # rAF is faked, so poll by timer
    # light damping and a big starting offset: clear oscillation that rings down
    page.click("button[data-p='under']")
    page.fill("#offset", "2.5")
    page.dispatch_event("#offset", "input")
    page.click("#release")
    page.evaluate("window.__advance(16)")

    region = page.locator(".stage")
    frames = []
    for _ in range(FPS * SECONDS):
        page.evaluate(f"window.__advance({1000 / FPS})")
        img = Image.open(io.BytesIO(region.screenshot())).convert("RGB")
        img = img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS)
        frames.append(img)
    browser.close()

# shared palette keeps the file small and colors stable across frames
palette = frames[len(frames) // 3].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
q = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
q[0].save(OUT, save_all=True, append_images=q[1:], duration=1000 // FPS, loop=0, optimize=True)
print(OUT, len(q), "frames,", f"{pathlib.Path(OUT).stat().st_size / 1e6:.1f} MB", q[0].size)
