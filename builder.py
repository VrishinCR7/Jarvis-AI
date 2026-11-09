"""
builder.py — Jarvis's HTML builder.

Pipeline:   PLAN  ->  GENERATE  ->  VERIFY  ->  REPAIR (up to 3x)  ->  REVIEW  ->  OPEN

VERIFY runs real checks, not vibes:
  * structure  balanced tags, not truncated, no external / missing resources
  * JS syntax  `node --check` on every inline script                  (needs Node.js)
  * runtime    headless Chromium: loads the page, fills inputs, clicks every control,
               presses keys, tests a phone-sized viewport, collects every error
                                                  (pip install playwright && playwright install chromium)
Every problem found is fed back to the model to fix. Jarvis only claims success when the checks
actually passed, and says so honestly when they didn't.

Standalone use:   python builder.py "a pomodoro timer with stats"
"""
import os
import re
import json
import time
import html as htmllib
import shutil
import tempfile
import subprocess
import webbrowser
from html.parser import HTMLParser
from pathlib import Path

BUILDS_DIR = Path(__file__).parent / "builds"
LAST_FILE = BUILDS_DIR / ".last.json"
CODE_MODEL = os.environ.get("JARVIS_CODE_MODEL", "openai/gpt-oss-120b")
MAX_REPAIRS = 3
MAX_TOKENS = 16000

# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
SPEC_PROMPT = """You are a senior product designer and software architect. Turn the user's request into a tight build spec for a SINGLE-FILE web app (HTML + inline CSS + inline JS, no libraries).

Output markdown, max ~300 words, with EXACTLY these sections:
## Goal
One sentence.
## Must-have features
5-9 bullets. Interpret vague requests ambitiously but sensibly: include the features a polished, real product of this type would have.
## UX & states
Empty / error / success states, keyboard and touch support, what feedback the user sees.
## Edge cases
What could break and how it is handled.
## Design direction
Mood, 4-5 hex colors (light + dark), type scale using system fonts, one signature visual detail.
## Acceptance checklist
6-10 concrete, testable statements (e.g. "Entering 2+2 and pressing = shows 4").

No code. No questions. No preamble."""

BUILDER_SYSTEM_PROMPT = """You are a world-class front-end engineer and product designer. You ship single-file web apps that feel like polished products and work on the first run.

OUTPUT: ONLY one complete HTML document starting with <!DOCTYPE html>. No markdown fences, no commentary.

HARD REQUIREMENTS
1. Fully self-contained: inline <style> and <script>. ZERO external resources: no CDNs, no web fonts, no remote images, no separate files. Use a system font stack, inline SVG, emoji, CSS and <canvas> for visuals.
2. Works the instant it is opened from disk (file://). Put the script at the end of <body>. Zero console errors, zero uncaught exceptions.
3. Complete: implement EVERY feature in the spec. No TODOs, no placeholders, no lorem ipsum. Real logic, not a mockup. Realistic sample data is fine where useful.
4. Robust: validate all input; handle empty, huge, invalid and rapid-fire input; never let NaN/undefined/Infinity reach the UI; guard every DOM lookup; wrap localStorage and navigator.clipboard (and any other optional browser API) in feature checks + try/catch.
5. Use 'use strict' and keep code inside an IIFE: no accidental globals. Use addEventListener (no inline onclick). Buttons inside forms must call preventDefault so the page never reloads.

DESIGN BAR
- Design tokens as CSS variables (colors, radius, shadow, spacing) with light AND dark themes via prefers-color-scheme (+ color-scheme). Example foundation, adapt the palette to the spec:
  :root{color-scheme:light dark;--bg:#f6f7fb;--surface:#fff;--text:#0f172a;--muted:#64748b;--accent:#6366f1;--radius:16px;--shadow:0 10px 30px rgba(15,23,42,.08)}
  @media (prefers-color-scheme:dark){:root{--bg:#0b1020;--surface:#141a2f;--text:#e7eaf6;--muted:#97a0bd;--accent:#8b8ff8;--shadow:0 10px 30px rgba(0,0,0,.4)}}
- System font stack, a clear type scale, generous spacing, one signature visual detail. Never default-browser-ugly.
- Mobile-first responsive: flawless at 360px, NO horizontal scroll at any width, sensible max-width on desktop, touch targets >= 44px.
- Polished states: hover, active, :focus-visible, disabled, empty, error, success. Subtle transitions; honour prefers-reduced-motion.
- Accessible: semantic landmarks (<header>, <main>), a label for every input, aria-live for dynamic results, full keyboard operability, WCAG AA contrast.

BY TYPE
- Games: requestAnimationFrame loop with delta-time; start/pause/restart; keyboard AND touch/pointer controls; score + persisted high score; crisp canvas (devicePixelRatio); clear game-over state.
- Tools / calculators / converters: instant live results, copy-to-clipboard where useful, keyboard shortcuts, sensible defaults, input formatting.
- Landing pages / sites: real persuasive copy, clear hierarchy, sticky nav with working anchor links, working forms with inline validation.
- Dashboards / trackers: realistic sample data, working filters and sorting, localStorage persistence, charts drawn with inline SVG or canvas."""

REVIEW_PROMPT = """You are a ruthless QA lead and product designer. You get the user's request, the build spec with its acceptance checklist, and the HTML that was built.

Decide whether the build genuinely satisfies the request. Check that every acceptance item is really implemented in the code (not just claimed); look for logic bugs, missing features, broken edge cases, UX gaps, visual polish problems and basic accessibility gaps.

Reply with ONLY JSON: {"pass": true or false, "issues": ["specific, actionable defect", ...]}
List at most 6 issues, most important first. Be strict about REAL defects but never invent nitpicks: if it is genuinely solid, pass it."""

# --------------------------------------------------------------------------------------
# LLM helper
# --------------------------------------------------------------------------------------
def _llm(client, messages, temperature=0.3, effort="medium", max_tokens=MAX_TOKENS):
    """Returns the reply text. Retries on rate limits; degrades gracefully if reasoning_effort is unsupported."""
    use_effort = True
    last_err = None
    for attempt in range(4):
        try:
            c = client.with_options(timeout=240.0) if hasattr(client, "with_options") else client
            kwargs = dict(model=CODE_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
            if use_effort:
                kwargs["reasoning_effort"] = effort
            resp = c.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            if use_effort and ("reasoning" in msg or isinstance(e, TypeError)):
                use_effort = False
                continue
            if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
                wait = 10 * (attempt + 1)
                print(f"[build] rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"model unavailable after retries: {last_err}")


def extract_html(text):
    """Pull the HTML document out of a model reply, tolerating fences, chatter and truncation."""
    text = (text or "").strip()
    start = re.search(r"<!DOCTYPE html|<html", text, re.IGNORECASE)
    if start:
        text = text[start.start():]
        end = text.lower().rfind("</html>")
        if end != -1:
            text = text[: end + len("</html>")]
        else:
            text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _slugify(text, max_len=40):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].strip("-")) or "build"


def _title_of(html, fallback):
    m = re.search(r"<title[^>]*>\s*(.*?)\s*</title>", html, re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip() if m and m.group(1).strip() else fallback)[:80]


# --------------------------------------------------------------------------------------
# Auto-fix of trivial things (saves a round trip to the model)
# --------------------------------------------------------------------------------------
def _autofix(html, title):
    if not re.match(r"\s*<!doctype html", html, re.IGNORECASE):
        html = "<!DOCTYPE html>\n" + html
    if re.search(r"<html\b", html, re.IGNORECASE) and not re.search(r"<html[^>]*\blang=", html, re.IGNORECASE):
        html = re.sub(r"<html\b", '<html lang="en"', html, count=1, flags=re.IGNORECASE)
    inject = ""
    if not re.search(r"<meta[^>]+charset", html, re.IGNORECASE):
        inject += '<meta charset="utf-8">'
    if not re.search(r"<meta[^>]+name=[\"']viewport", html, re.IGNORECASE):
        inject += '<meta name="viewport" content="width=device-width, initial-scale=1">'
    if not re.search(r"<title[^>]*>\s*\S", html, re.IGNORECASE):
        inject += f"<title>{htmllib.escape(title)}</title>"
    if inject and re.search(r"<head[^>]*>", html, re.IGNORECASE):
        html = re.sub(r"(<head[^>]*>)", lambda m: m.group(1) + inject, html, count=1, flags=re.IGNORECASE)
    return html


# --------------------------------------------------------------------------------------
# VERIFY 1: structure scan
# --------------------------------------------------------------------------------------
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
OPTIONAL_CLOSE = {"p", "li", "dt", "dd", "tr", "td", "th", "thead", "tbody", "tfoot", "option", "optgroup",
                  "colgroup", "caption", "html", "body", "head"}
SCRIPT_TYPES = {"", "text/javascript", "application/javascript", "module"}


class _Scan(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.issues = []
        self.scripts = []          # (is_module, code)
        self._script_type = None
        self._buf = []

    def _flag(self, msg):
        if msg not in self.issues:
            self.issues.append(msg)

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        ref = None
        if tag in ("script", "img", "source", "iframe", "video", "audio", "embed") and a.get("src"):
            ref = a["src"]
        elif tag == "link" and a.get("href"):
            ref = a["href"]
        if ref and not ref.startswith(("data:", "#", "blob:")):
            if re.match(r"^(https?:)?//", ref, re.IGNORECASE):
                self._flag(f"External resource <{tag}> '{ref[:70]}' — everything must be inline so it works offline.")
            else:
                self._flag(f"<{tag}> references local file '{ref[:70]}' which does not exist — inline it instead.")
        if tag == "script" and not a.get("src"):
            self._script_type = a.get("type", "").lower().strip()
            self._buf = []
        if tag not in VOID:
            self.stack.append(tag)

    def handle_data(self, data):
        if self._script_type is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if tag == "script" and self._script_type is not None:
            if self._script_type in SCRIPT_TYPES:
                self.scripts.append((self._script_type == "module", "".join(self._buf)))
            self._script_type = None
        if tag not in self.stack:
            self._flag(f"Stray closing tag </{tag}> with no matching opening tag.")
            return
        while self.stack:
            top = self.stack.pop()
            if top == tag:
                break
            if top not in OPTIONAL_CLOSE:
                self._flag(f"Unclosed <{top}> element (closed by </{tag}>).")

    def finish(self):
        for tag in self.stack:
            if tag not in OPTIONAL_CLOSE:
                self._flag(f"Unclosed <{tag}> element at end of document.")


def _scan(html):
    s = _Scan()
    try:
        s.feed(html)
        s.close()
    except Exception as e:  # noqa: BLE001
        s._flag(f"HTML could not be parsed: {e}")
    s.finish()
    if "</html>" not in html.lower():
        s._flag("Document is truncated: no closing </html>. Output was cut off — rewrite it MORE COMPACTLY.")
    if re.search(r"url\(\s*[\"']?(?:https?:)?//", html, re.IGNORECASE) or re.search(r"@import", html, re.IGNORECASE):
        s._flag("CSS pulls in a remote resource (url(http...) or @import) — remove it; keep everything inline.")
    if not re.search(r"<body", html, re.IGNORECASE):
        s._flag("Missing <body>.")
    s.issues = s.issues[:6]
    return s


# --------------------------------------------------------------------------------------
# VERIFY 2: JS syntax via node --check
# --------------------------------------------------------------------------------------
def _check_js(scripts):
    """Returns (issues, ran)."""
    node = shutil.which("node")
    if not node:
        return [], False
    issues = []
    for i, (is_module, code) in enumerate(scripts, 1):
        if not code.strip():
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".mjs" if is_module else ".js", delete=False, encoding="utf-8") as f:
            f.write(code)
            name = f.name
        try:
            r = subprocess.run([node, "--check", name], capture_output=True, text=True, timeout=20)
            if r.returncode != 0:
                lines = [l for l in r.stderr.strip().splitlines() if l.strip()]
                loc = re.search(r":(\d+)\s*$", lines[0]) if lines else None
                err = next((l for l in lines if "Error" in l), lines[-1] if lines else "syntax error")
                code_line = lines[1].strip()[:100] if len(lines) > 1 else ""
                where = f" at script line {loc.group(1)}" if loc else ""
                issues.append(f"JavaScript syntax error in inline script #{i}{where}: {err.strip()[:140]}"
                              + (f" | near: {code_line}" if code_line else ""))
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                os.unlink(name)
            except OSError:
                pass
    return issues, True


# --------------------------------------------------------------------------------------
# VERIFY 3: real browser run (headless Chromium via Playwright)
# --------------------------------------------------------------------------------------
SMOKE_KEYS = ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft", "Space", "Enter", "w", "a", "s", "d", "1", "Escape"]
CLICK_SELECTORS = ("button:visible:not([disabled])", "[role=button]:visible", "input[type=button]:visible:not([disabled])",
                   "input[type=submit]:visible:not([disabled])", "summary:visible", "input[type=checkbox]:visible",
                   "input[type=radio]:visible", "canvas:visible")
SKIP_INPUT_TYPES = {"button", "submit", "reset", "file", "color", "date", "time", "datetime-local", "month", "week",
                    "image", "hidden", "range", "checkbox", "radio"}
INPUT_VALUES = {"number": "7", "email": "test@example.com", "url": "https://example.com", "tel": "5551234567"}


def _try(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        pass


def _smoke_interact(page, url, issues, deadline):
    # 1) fill inputs and selects
    n = min(page.locator("input:visible, textarea:visible").count(), 12)
    for i in range(n):
        if time.monotonic() > deadline:
            return
        el = page.locator("input:visible, textarea:visible").nth(i)

        def fill(el=el):
            t = (el.get_attribute("type") or "text").lower()
            if t not in SKIP_INPUT_TYPES:
                el.fill(INPUT_VALUES.get(t, "test"), timeout=800)
        _try(fill)
    for i in range(min(page.locator("select:visible").count(), 6)):
        _try(lambda i=i: page.locator("select:visible").nth(i).select_option(index=1, timeout=800))

    # 2) click every control
    navigated = False
    for sel in CLICK_SELECTORS:
        count = min(page.locator(sel).count(), 25)
        for i in range(count):
            if time.monotonic() > deadline:
                return
            _try(lambda sel=sel, i=i: page.locator(sel).nth(i).click(timeout=800))
            if page.url.split("#")[0] != url.split("#")[0]:
                if not navigated:
                    issues.append("Clicking a control navigated/reloaded the page (a form submit without preventDefault?). "
                                  "State is lost — handle submit events with preventDefault.")
                    navigated = True
                _try(lambda: page.goto(url, timeout=3000))

    # 3) keyboard mash (games, shortcuts)
    for k in SMOKE_KEYS:
        if time.monotonic() > deadline:
            return
        _try(lambda k=k: page.keyboard.press(k))
        page.wait_for_timeout(30)


def _runtime_check(html_path, shot_path):
    """Returns (issues, ran, note)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return [], False, "playwright not installed"
    errors, issues = [], []
    url = Path(html_path).resolve().as_uri()
    deadline = time.monotonic() + 30
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception:  # noqa: BLE001
                return [], False, "Chromium not installed (run: playwright install chromium)"
            try:
                ctx = browser.new_context(viewport={"width": 1280, "height": 800})
                ctx.set_default_timeout(3000)
                page = ctx.new_page()
                page.on("pageerror", lambda e: errors.append(f"Uncaught exception: {str(e)[:200]}"))
                page.on("console", lambda m: errors.append(f"console.error: {m.text[:200]}") if m.type == "error" else None)
                page.on("requestfailed", lambda r: errors.append(f"Resource failed to load: {r.url[:100]}"))
                page.on("dialog", lambda d: _try(d.accept))
                page.on("popup", lambda pp: _try(pp.close))

                try:
                    page.goto(url, wait_until="load", timeout=5000)
                except Exception as e:  # noqa: BLE001
                    return ["Page failed to load or hung on load (infinite loop in startup code?): " + str(e)[:100]], True, ""
                page.wait_for_timeout(700)

                info = page.evaluate("""() => {
                    const b = document.body;
                    return {kids: b ? b.children.length : 0,
                            text: b ? (b.innerText || '').trim().length : 0,
                            visuals: document.querySelectorAll('canvas,svg,img').length};
                }""")
                if info["kids"] == 0 or (info["text"] == 0 and info["visuals"] == 0):
                    issues.append("Page renders blank on load: nothing visible in <body>.")
                _try(lambda: page.screenshot(path=str(shot_path)))

                # phone layout first, while the app is still in its clean initial state
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_timeout(300)
                overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
                if overflow > 2:
                    issues.append(f"Horizontal overflow at phone width (390px) by {overflow}px — content must fit without sideways scrolling.")
                page.set_viewport_size({"width": 1280, "height": 800})
                page.wait_for_timeout(200)

                _smoke_interact(page, url, issues, deadline)
                page.wait_for_timeout(400)
            finally:
                browser.close()
    except Exception as e:  # noqa: BLE001
        return issues, False, f"runtime test could not complete ({str(e)[:80]})"

    seen = []
    for e in errors:
        if e not in seen:
            seen.append(e)
    return issues + seen[:5], True, ""


def verify(html, html_path, shot_path):
    """Run every available check. Returns (issues, checks)."""
    scan = _scan(html)
    issues = list(scan.issues)
    js_issues, js_ran = _check_js(scan.scripts)
    issues += js_issues
    checks = {"syntax": js_ran, "runtime": False, "note": ""}
    if not issues:  # only spend browser time on pages that parse
        rt_issues, rt_ran, note = _runtime_check(html_path, shot_path)
        issues += rt_issues
        checks["runtime"], checks["note"] = rt_ran, note
    return issues, checks


# --------------------------------------------------------------------------------------
# PLAN / GENERATE / REVIEW / REPAIR
# --------------------------------------------------------------------------------------
def make_spec(client, request):
    return _llm(client, [{"role": "system", "content": SPEC_PROMPT},
                         {"role": "user", "content": request}], temperature=0.5, effort="low", max_tokens=1500).strip()


def generate(client, request, spec, prev_html=None):
    if prev_html:
        user = (f"BUILD SPEC:\n{spec}\n\nCURRENT HTML:\n{prev_html}\n\n"
                f"Apply the 'Requested change' from the spec and return the COMPLETE updated HTML file. "
                f"Keep everything else working exactly as before.")
    else:
        user = f"USER REQUEST: {request}\n\nBUILD SPEC (implement all of it):\n{spec}"
    return extract_html(_llm(client, [{"role": "system", "content": BUILDER_SYSTEM_PROMPT},
                                       {"role": "user", "content": user}], temperature=0.35))


def review(client, request, spec, html):
    """Returns a list of issues ([] if it passes). Never blocks on a malformed review."""
    try:
        raw = _llm(client, [{"role": "system", "content": REVIEW_PROMPT},
                            {"role": "user", "content": f"REQUEST: {request}\n\nSPEC:\n{spec}\n\nHTML:\n{html}"}],
                   temperature=0, effort="medium", max_tokens=2000)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        if data.get("pass", True):
            return []
        return [str(i)[:300] for i in data.get("issues", []) if str(i).strip()][:6]
    except Exception:  # noqa: BLE001
        return []


def repair(client, request, spec, html, issues):
    bullet = "\n".join(f"- {i}" for i in issues)
    user = (f"USER REQUEST: {request}\n\nSPEC:\n{spec}\n\nCURRENT HTML:\n{html}\n\n"
            f"AUTOMATED TESTING / REVIEW FOUND THESE PROBLEMS:\n{bullet}\n\n"
            f"Fix ALL of them without breaking anything else. Return the COMPLETE corrected HTML file only. "
            f"If the problem is truncation, rewrite MORE COMPACTLY (tighter CSS/JS, no comments).")
    return extract_html(_llm(client, [{"role": "system", "content": BUILDER_SYSTEM_PROMPT},
                                       {"role": "user", "content": user}], temperature=0.2))


# --------------------------------------------------------------------------------------
# State (so "make it blue" still works after a restart)
# --------------------------------------------------------------------------------------
def _load_last():
    try:
        return json.loads(LAST_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_last(folder, version):
    LAST_FILE.write_text(json.dumps({"slug": folder.name, "version": version}), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def build(client, request, update_existing=False, progress=print, open_browser=True):
    """Plan -> generate -> verify -> repair -> review -> open. Returns a plain-text result for Jarvis to relay."""
    BUILDS_DIR.mkdir(exist_ok=True)
    request = request.strip()

    prev = _load_last() if update_existing else None
    prev_html = None
    if prev and (BUILDS_DIR / prev["slug"] / "index.html").exists():
        folder = BUILDS_DIR / prev["slug"]
        version = prev["version"] + 1
        prev_html = (folder / "index.html").read_text(encoding="utf-8")
        old_spec = (folder / "spec.md").read_text(encoding="utf-8") if (folder / "spec.md").exists() else ""
        spec = f"{old_spec}\n\n## Requested change (v{version})\n{request}"
        progress(f"[build] updating {folder.name} -> v{version}")
    else:
        folder = BUILDS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slugify(request)}"
        folder.mkdir(parents=True, exist_ok=True)
        version = 1
        progress("[build] 1/4 planning the build...")
        spec = make_spec(client, request)

    progress("[build] 2/4 writing the code...")
    html = generate(client, request, spec, prev_html)

    reviewed = review_found = False
    issues, checks, repairs = [], {}, 0
    for attempt in range(MAX_REPAIRS + 1):
        html = _autofix(html, _slugify(request).replace("-", " ").title())
        path = folder / f"v{version}.html"
        path.write_text(html, encoding="utf-8")
        progress("[build] 3/4 testing (structure, JS syntax, real browser run)...")
        issues, checks = verify(html, path, folder / "preview.png")
        if not issues and not reviewed:
            reviewed = True
            progress("[build] 4/4 reviewing against the acceptance checklist...")
            issues = review(client, request, spec, html)
            review_found = bool(issues)
        if not issues or attempt == MAX_REPAIRS:
            break
        repairs += 1
        progress(f"[build] fixing {len(issues)} issue(s) (attempt {repairs}/{MAX_REPAIRS}): {issues[0][:90]}")
        html = repair(client, request, spec, html, issues)

    (folder / "index.html").write_text(html, encoding="utf-8")
    (folder / "spec.md").write_text(spec, encoding="utf-8")
    _save_last(folder, version)
    index = folder / "index.html"
    if open_browser:
        webbrowser.open(index.resolve().as_uri())
    progress(f"[build] done -> {index}")

    title = _title_of(html, request)
    if issues:
        listing = "; ".join(i[:140] for i in issues[:4])
        return (f"BUILT WITH KNOWN ISSUES: '{title}' (v{version}) is open in the browser at {index}, but still has problems "
                f"after {repairs} automatic repair attempts: {listing}. Tell the user honestly and offer to fix them.")
    ok = ["structure"]
    if checks.get("syntax"):
        ok.append("JS syntax")
    if checks.get("runtime"):
        ok.append("headless-browser run with every control clicked and no errors")
        ok.append("phone layout")
    if reviewed:
        ok.append("reviewer's findings fixed and re-tested" if review_found else "review against its acceptance checklist")
    note = f" (Not tested: {checks['note']}.)" if checks.get("note") else ""
    fixed = f" It needed {repairs} automatic fix(es) along the way." if repairs else ""
    return (f"BUILD SUCCESS: '{title}' (v{version}) is open in the user's browser at {index}. "
            f"Verified: {', '.join(ok)}.{fixed}{note}")


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from groq import Groq
    load_dotenv()
    if len(sys.argv) < 2:
        sys.exit('usage: python builder.py "what to build"  [--update]')
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print(build(groq_client, " ".join(a for a in sys.argv[1:] if a != "--update"), update_existing="--update" in sys.argv))