"""Record the end-to-end walkthrough (screenshots with click markers + a video) of a RUNNING LocateNet.

Needs: pip install playwright (uses an installed Microsoft Edge or Chrome, no browser download) and
ffmpeg on PATH (only for the mp4 / gif conversion).

    # 1. empty demo database + accounts, then start the app on it
    LOCATENET_DB=demo.db LOCATENET_SECRET_KEY=demo python scripts/seed_walkthrough.py
    LOCATENET_DB=demo.db LOCATENET_SECRET_KEY=demo python app.py
    # 2. in another terminal
    python scripts/make_walkthrough.py --url http://127.0.0.1:5000

Inputs are the synthetic faces in docs/walkthrough/inputs (see its README for source and licence).
Everything in the story is fictional: the people are computer-rendered, the names are invented.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(ROOT, "docs", "walkthrough", "inputs")
OUT = os.path.join(ROOT, "docs", "walkthrough")
SHOTS = os.path.join(OUT, "screenshots")
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
TOTAL_STEPS = 14

OVERLAY = r"""
(() => {
  const build = () => {
    if (document.getElementById('__cur')) return;
    const s = document.createElement('style');
    s.textContent = `
      #__cur{position:fixed;left:0;top:0;width:30px;height:30px;z-index:2147483647;pointer-events:none;transition:transform .6s cubic-bezier(.22,.8,.3,1)}
      .__rip{position:fixed;z-index:2147483646;width:16px;height:16px;border-radius:50%;border:4px solid #ff3b30;pointer-events:none;animation:__r .8s ease-out forwards}
      @keyframes __r{from{transform:translate(-50%,-50%) scale(1);opacity:1}to{transform:translate(-50%,-50%) scale(5);opacity:0}}
      #__cap{position:fixed;left:0;right:0;bottom:0;z-index:2147483645;background:rgba(15,23,42,.93);color:#fff;font:600 20px/1.35 system-ui,Segoe UI,sans-serif;padding:12px 24px;pointer-events:none}
      #__cap small{display:block;font-weight:400;font-size:15px;color:#a9bdd3;margin-top:2px}
      #__step{position:fixed;bottom:16px;right:20px;z-index:2147483645;background:#2563eb;color:#fff;font:700 14px system-ui;padding:5px 13px;border-radius:14px;pointer-events:none}`;
    document.head.appendChild(s);
    const c = document.createElement('div');
    c.id = '__cur';
    c.innerHTML = '<svg width="30" height="30" viewBox="0 0 24 24"><path d="M3 2l7 19 3-8 8-3z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    const p = JSON.parse(sessionStorage.getItem('__pos') || '[40,40]');
    c.style.transition = 'none';
    c.style.transform = `translate(${p[0]}px,${p[1]}px)`;
    document.body.appendChild(c);
    requestAnimationFrame(() => requestAnimationFrame(() => { c.style.transition = ''; }));
    const cap = document.createElement('div'); cap.id = '__cap'; cap.style.display = 'none'; document.body.appendChild(cap);
    const st = document.createElement('div'); st.id = '__step'; st.style.display = 'none'; document.body.appendChild(st);
    const last = JSON.parse(sessionStorage.getItem('__cap') || 'null');
    if (last) window.__cap(last[0], last[1], last[2]);
  };
  window.__moveTo = (x, y) => { sessionStorage.setItem('__pos', JSON.stringify([x - 4, y - 2]));
    const c = document.getElementById('__cur'); if (c) c.style.transform = `translate(${x - 4}px,${y - 2}px)`; };
  window.__ripple = (x, y) => { const r = document.createElement('div'); r.className = '__rip'; r.style.left = x + 'px'; r.style.top = y + 'px';
    document.body.appendChild(r); setTimeout(() => r.remove(), 900); };
  window.__cap = (t, s, n) => { sessionStorage.setItem('__cap', JSON.stringify([t, s, n]));
    const cap = document.getElementById('__cap'), st = document.getElementById('__step'); if (!cap) return;
    cap.style.display = 'block'; cap.innerHTML = t + (s ? '<small>' + s + '</small>' : '');
    st.style.display = n ? 'block' : 'none'; st.textContent = n || ''; };
  document.addEventListener('DOMContentLoaded', build);
  if (document.readyState !== 'loading') build();
})();
"""


class Tour:
    def __init__(self, page, delay):
        self.pg, self.delay, self.n, self.stepn = page, delay, 0, None

    def pause(self, ms=1.0):
        self.pg.wait_for_timeout(int(ms * 1000 * self.delay))

    def caption(self, title, sub="", step=None):
        step = step or self.stepn
        label = f"Step {step} / {TOTAL_STEPS}" if step else ""
        self.pg.evaluate("([t, s, n]) => window.__cap(t, s, n)", [title, sub, label])

    def shot(self, name):
        self.n += 1
        os.makedirs(SHOTS, exist_ok=True)
        path = os.path.join(SHOTS, f"{self.n:02d}_{name}.png")
        self.pg.screenshot(path=path)
        print("  screenshot", os.path.relpath(path, ROOT))

    def _center(self, selector):
        el = self.pg.locator(selector).first
        el.wait_for(state="visible", timeout=15000)
        el.scroll_into_view_if_needed()
        box = el.bounding_box()
        if box["y"] + box["height"] > 720 - 110:  # keep the element above the caption bar
            self.pg.evaluate("(d) => window.scrollBy(0, d)", box["y"] + box["height"] - (720 - 110) + 40)
            self.pg.wait_for_timeout(200)
            box = el.bounding_box()
        return el, box["x"] + box["width"] / 2, box["y"] + min(box["height"] / 2, 30)

    def move(self, selector):
        el, x, y = self._center(selector)
        self.pg.evaluate("([x, y]) => window.__moveTo(x, y)", [x, y])
        self.pause(0.8)
        return el, x, y

    def click(self, selector, shot=None, wait=0.9):
        el, x, y = self.move(selector)
        self.pg.evaluate("([x, y]) => window.__ripple(x, y)", [x, y])
        if shot:
            self.pause(0.25)
            self.shot(shot)
        self.pause(0.35)
        el.click()
        self.pause(wait)

    def type(self, selector, text, delay=45):
        el, x, y = self.move(selector)
        self.pg.evaluate("([x, y]) => window.__ripple(x, y)", [x, y])
        el.click()
        el.fill("")
        self.pg.keyboard.type(text, delay=delay)
        self.pause(0.3)

    def upload(self, selector, path, label):
        el, x, y = self.move(selector)
        self.pg.evaluate("([x, y]) => window.__ripple(x, y)", [x, y])
        self.pause(0.4)
        el.set_input_files(path)
        self.caption(label[0], label[1])
        self.pause(0.8)


def find_browser():
    for path in BROWSERS:
        if os.path.exists(path):
            return path
    sys.exit("No Edge or Chrome found; edit BROWSERS in this script.")


def convert(webm, out_dir):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found: left the .webm, skipped mp4/gif")
        return
    mp4 = os.path.join(out_dir, "walkthrough.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", mp4], check=True)
    gif = os.path.join(out_dir, "walkthrough_preview.gif")
    palette = os.path.join(out_dir, "_palette.png")
    flt = "setpts=PTS/5,fps=8,scale=720:-1:flags=lanczos"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", mp4, "-vf", f"{flt},palettegen=max_colors=96", palette], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", mp4, "-i", palette, "-lavfi",
                    f"{flt}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", gif], check=True)
    os.remove(palette)
    print("video:", os.path.relpath(mp4, ROOT), f"({os.path.getsize(mp4) / 1e6:.1f} MB)")
    print("gif:  ", os.path.relpath(gif, ROOT), f"({os.path.getsize(gif) / 1e6:.1f} MB)")


def run(url, delay, headless):
    inp = lambda name: os.path.join(INPUTS, name)  # noqa: E731
    for old in glob.glob(os.path.join(SHOTS, "*.png")):
        os.remove(old)
    tmp_video = os.path.join(OUT, "_video")
    shutil.rmtree(tmp_video, ignore_errors=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=find_browser(), headless=headless)
        ctx = browser.new_context(viewport={"width": 1280, "height": 720}, record_video_dir=tmp_video,
                                  record_video_size={"width": 1280, "height": 720})
        ctx.add_init_script(OVERLAY)
        pg = ctx.new_page()
        pg.on("dialog", lambda d: d.accept())
        t = Tour(pg, delay)

        def step(n, title, sub=""):
            t.stepn = n
            t.caption(title, sub, n)

        # 1 ------------------------------------------------------------ public home
        pg.goto(url + "/")
        step(1, "A family member or passer-by opens LocateNet", "Anyone can report a sighting or search the public registry; no account needed.")
        t.pause(1.8)
        t.shot("public_home")

        # 2 ------------------------------------------------------------ empty registry
        step(2, "The public registry lists only people who are still missing", "Name, city, date and photo: nothing else is public. It is empty for now.")
        t.click("a:has-text('Search Cases')", wait=1.2)
        t.shot("registry_empty")

        # 3 ------------------------------------------------------------ officer login
        pg.goto(url + "/login")
        step(3, "Inspector Meera Rao logs in", "Passwords are hashed, the session is reset on login and every form carries a CSRF token.")
        t.pause(0.8)
        t.type("#username", "officer_meera")
        t.type("#password", "demo-officer-pass")
        t.click("button[type=submit]", shot="officer_login", wait=1.5)

        # 4 ------------------------------------------------------------ dashboard
        step(4, "Officer dashboard: no alerts and no cases yet", "She registers a missing person: Anil Verma, last seen in Aligarh.")
        t.pause(1.5)
        t.shot("dashboard_empty")
        t.click("a:has-text('Register New Case')", wait=1.5)

        # 5 ------------------------------------------------------------ register case
        step(5, "Registering the case with a reference photo", "A 2017 photo, his date of birth and the city. Each photo is quality-checked (clear, large enough face).")
        t.type("input[name=person_name]", "Anil Verma")
        t.type("#city_search", "Aligarh")
        pg.keyboard.press("Tab")
        pg.fill("#dob", "1996-04-12")
        t.pause(0.5)
        t.upload("input[name=images] >> nth=0", inp("reference_2017.png"), ("Reference photo 1 chosen (taken in 2017)", "Up to 5 photos per case; the right-hand field is the approximate month the photo was taken."))
        pg.locator("input[name=photo_dates]").first.fill("2017-03")
        t.pause(0.8)
        t.click("#submitBtn", shot="register_case_filled", wait=2.5)

        # 6 ------------------------------------------------------------ case page + age warning
        step(6, "Case page: the system warns that the photo is old", "The miss rate shown comes from the age-gap evaluation (FG-NET), not from a hard-coded number.")
        t.pause(2.0)
        t.shot("case_age_warning")

        # 7 ------------------------------------------------------------ add recent photo
        step(7, "Adding a recent reference photo (2025)", "More photos from different years are the most effective way to find someone years later.")
        t.upload("form[action$='/photos'] input[name=image]", inp("reference_2025.png"), ("Recent photo chosen", "Approximate date: February 2025"))
        pg.locator("form[action$='/photos'] input[name=photo_date]").fill("2025-02")
        t.click("form[action$='/photos'] button[type=submit]", wait=2.0)
        step(7, "Two reference photos, the warning is gone", "Every change is written to the case's audit log.")
        t.pause(1.5)
        t.shot("case_two_photos")

        # 8 ------------------------------------------------------------ public registry with the case
        pg.goto(url + "/search")
        step(8, "The case now appears in the public registry", "Citizens can recognise him and report a sighting.")
        t.pause(1.8)
        t.shot("registry_with_case")
        t.click("a:has-text('Report a Sighting')", wait=1.5)

        # 9 ------------------------------------------------------------ three sightings
        sightings = [("Mohan Gupta", "9876501234", "sighting_1_platform.jpg", "Platform camera, Aligarh Junction (Oct 20)"),
                     ("Sana Khan", "9876505678", "sighting_2_bus_stand.jpg", "Bus-stand camera (Oct 21, morning)"),
                     ("Ravi Singh", "9876509012", "sighting_3_market.jpg", "Central Market camera (Oct 21, evening)")]
        for k, (name, mobile, image, where) in enumerate(sightings, 1):
            if k > 1:
                pg.goto(url + "/report_sighting")
            step(9, f"Sighting report {k} of 3 from a member of the public", f"{where}. A group photo with several people in it.")
            t.pause(0.6)
            t.type("input[name=name]", name, delay=25)
            t.type("input[name=mobile]", mobile, delay=25)
            t.type("input[name=location]", "Aligarh", delay=25)
            t.upload("input[name=image]", inp(image), ("Photo chosen: " + image, "All faces in the photo are detected and compared with every open case."))
            t.click("button:has-text('Submit Sighting')", shot=f"report_sighting_{k}" if k == 1 else None, wait=3.0)
            if k == 1:
                step(10, "Result for the reporter: faces found, officer notified", "The public never sees who was matched; the officer does.")
                t.pause(1.5)
                t.shot("report_result")
            if k == 3:
                step(11, "Hotspot: the same person matched 3 times in one city within 48 hours", "Hotspot rule: same case, same city, 3 matches, 48 hours.")
                t.pause(2.0)
                t.shot("hotspot")

        # 12 ----------------------------------------------------------- dashboard alerts
        pg.goto(url + "/dashboard")
        step(12, "The officer's dashboard shows three alerts", "Similarity, an estimated match likelihood (an estimate!) and 'Needs human confirmation'.")
        t.pause(2.0)
        t.shot("dashboard_alerts")
        t.click("a:has-text('View Faces') >> nth=0", wait=2.0)

        # 13 ----------------------------------------------------------- review
        step(13, "Reviewing the detected faces: a human decides", "One face is a possible match; the others are strangers and score as 'No Match'.")
        t.pause(2.0)
        t.shot("sighting_faces")
        pg.evaluate("() => { document.querySelector('.border-success').scrollIntoView({block: 'start'}); window.scrollBy(0, -24); }")
        t.pause(0.6)
        t.move("input[name=add_as_template] >> nth=0")
        t.caption("She confirms the match and adds the face as a new reference photo", "Opt-in, with guards: only the case's officer, a quality check, and it must resemble the enrolled photos.", 13)
        pg.check("input[name=add_as_template] >> nth=0")
        t.pause(0.6)
        t.click("form:has(input[name=add_as_template]) button[value=confirm_match] >> nth=0", shot="confirm_match", wait=2.0)
        t.shot("review_saved")

        # 14 ----------------------------------------------------------- case page, admin, resolve
        pg.goto(url + "/dashboard")
        t.pause(0.6)
        t.click("a:has-text('Anil Verma') >> nth=0", wait=2.0)
        step(14, "The case page now holds a photo that came from the sighting", "It is labelled 'confirmed sighting', can be deactivated, and appears in the audit log.")
        t.pause(2.0)
        t.shot("case_photos_from_sighting")
        pg.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        t.caption("Every change is recorded in the case's audit log", "Who added or deactivated which photo, and when.")
        t.pause(1.5)
        t.shot("case_audit_log")

        t.click("a:has-text('Back to Dashboard')", wait=1.2)
        t.click("a:has-text('Logout')", wait=1.0)
        pg.goto(url + "/login")
        step(14, "An Admin checks model health", "Field precision, calibration and alert volume built from officer reviews. Rows with too few reviews say 'insufficient data'.")
        t.type("#username", "admin_demo")
        t.type("#password", "demo-admin-pass")
        t.click("button[type=submit]", wait=1.5)
        t.click("a:has-text('Model health')", wait=2.0)
        t.pause(1.5)
        t.shot("model_health")
        t.click("a:has-text('Back to Dashboard')", wait=1.0)
        t.click("a:has-text('Logout')", wait=1.0)

        pg.goto(url + "/login")
        step(14, "Anil is found safe: the officer closes the case", "'Mark Found' removes the case from matching and from the public registry.")
        t.type("#username", "officer_meera")
        t.type("#password", "demo-officer-pass")
        t.click("button[type=submit]", wait=1.5)
        t.click("button:has-text('Mark Found')", shot="mark_found", wait=2.0)
        t.shot("case_resolved")
        pg.goto(url + "/search")
        t.caption("Case closed", "The public registry no longer lists him, and his photos can no longer raise alerts.", 14)
        t.pause(2.0)
        t.shot("registry_after_resolve")
        t.pause(1.0)

        video_path = pg.video.path()
        ctx.close()
        browser.close()
    convert(video_path, OUT)
    shutil.rmtree(tmp_video, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--speed", type=float, default=1.0, help="pause multiplier (0.5 = faster)")
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    run(a.url, a.speed, headless=not a.headed)
