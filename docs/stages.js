// RedPhys website: the animated explainer for the Plan (the evolving red team).
// Each animation is a pure render(t) of a looping clock, so pause, replay and reduced motion are trivial.
(() => {
  const byId = (s) => document.getElementById(s);
  if (!byId("st-chart")) return;
  const NS = "http://www.w3.org/2000/svg";
  const clamp = (x) => Math.min(1, Math.max(0, x));
  const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - (2 - 2 * x) ** 2 / 2);
  const el = (tag, cls, html = "", parent) => {
    const e = tag.startsWith("svg:") ? document.createElementNS(NS, tag.slice(4)) : document.createElement(tag);
    if (cls) e.setAttribute("class", cls);
    e.innerHTML = html;
    parent?.append(e);
    return e;
  };
  const toggle = (e, cls, on) => e.classList.toggle(cls, !!on);
  const text = (e, s) => { if (e.textContent !== s) e.textContent = s; };

  // A looping clock that runs only while `root` is on screen and not paused.
  function player(root, { loop, end, render, play, replay }) {
    let t = 0, last = 0, raf = 0, seen = false, paused = false;
    const frame = (now) => {
      t += last ? Math.min(now - last, 100) / 1000 : 0;
      last = now;
      if (t >= loop) t = 0;
      render(t);
      raf = requestAnimationFrame(frame);
    };
    const sync = () => {
      cancelAnimationFrame(raf);
      last = 0;
      if (seen && !paused) raf = requestAnimationFrame(frame);
      play.textContent = paused ? "Play" : "Pause";
    };
    if (reduced) return render(end);
    render(0);
    const check = () => {  // on screen: scrolled in, resized, or its <details> opened (toggle does not bubble, so capture)
      const r = root.getBoundingClientRect(), v = r.height > 0 && r.bottom > 0 && r.top < innerHeight;
      if (v !== seen) { seen = v; sync(); }
    };
    for (const ev of ["scroll", "resize", "toggle"]) addEventListener(ev, check, { passive: true, capture: true });
    check();
    play.addEventListener("click", () => { paused = !paused; sync(); });
    replay.addEventListener("click", () => { t = 0; paused = false; render(0); sync(); });
  }

  /* ---------------- The Plan: an evolving red team ---------------- */
  const ROUNDS = [  // where each curve ends after the attack / defense half of the round; new: the attacker uses its agent-written method
    { hi: 0.66, lo: 0.38, ds: 0.3 },
    { hi: 0.6, lo: 0.34, ds: 0.4 },
    { hi: 0.63, lo: 0.36, ds: 0.47 },
    { hi: 0.55, lo: 0.31, ds: 0.54 },
    { hi: 0.8, lo: 0.42, ds: 0.61, new: true },
    { hi: 0.67, lo: 0.37, ds: 0.68, new: true },
    { hi: 0.62, lo: 0.34, ds: 0.73 },
    { hi: 0.59, lo: 0.32, ds: 0.77 },
  ];
  const WRITE_AT = 4, PH = 3.2, WR = 5.7;  // seconds per half-round and for writing a method
  const MODELS = { "Model A": [0.62, 0.58, 0.6, 0.55, 0.57, 0.5, 0.54, 0.56, 0.55], "Model B": [0.55, 0.6, 0.57, 0.59, 0.52, 0.56, 0.58, 0.53, 0.57],
    "Model C": [0.48, 0.5, 0.54, 0.57, 0.6, 0.58, 0.52, 0.55, 0.6], "Model D": [0.4, 0.45, 0.43, 0.48, 0.46, 0.52, 0.49, 0.51, 0.5] };
  const SEGS = [];
  let acc = 1;
  ROUNDS.forEach((_, r) => {
    if (r === WRITE_AT) SEGS.push({ r, k: "write", t0: acc, d: WR }), (acc += WR);
    for (const k of ["attack", "defense"]) SEGS.push({ r, k, t0: acc, d: PH }), (acc += PH);
  });
  const E_END = acc, E_LOOP = E_END + 6, W0 = SEGS.find((s) => s.k === "write").t0;
  const ASR = [0.3, ...ROUNDS.flatMap((r) => [r.hi, r.lo])];
  const DEF = [0.18, ...ROUNDS.flatMap((r, i) => [i ? ROUNDS[i - 1].ds : 0.18, r.ds])];

  // chart
  const W = 600, H = 235, L = 50, R = 14, T = 20, B = 30, N = ROUNDS.length;
  const X = (x) => L + (x / N) * (W - L - R), Y = (v) => T + (1 - v) * (H - T - B);
  const chart = byId("st-chart");
  chart.innerHTML = `<defs><linearGradient id="st-ga" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f472b6" stop-opacity=".28"/><stop offset="1" stop-color="#f472b6" stop-opacity="0"/></linearGradient></defs>
    <rect class="st-band" id="st-band" y="${T}" height="${H - T - B}" width="${X(1) - X(0)}"/>
    ${[0.25, 0.5, 0.75].map((v) => `<line class="st-grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/>`).join("")}
    <line class="st-axisl" x1="${L}" x2="${W - R}" y1="${Y(0)}" y2="${Y(0)}"/>
    <text class="st-yl" x="${L - 8}" y="${Y(0.95)}">high</text><text class="st-yl" x="${L - 8}" y="${Y(0.03)}">low</text>
    ${ROUNDS.map((_, r) => `<text class="st-xl" x="${X(r + 0.5)}" y="${H - 10}">R${r + 1}</text>`).join("")}
    <g class="st-mark" id="st-mark"><line x1="${X(WRITE_AT)}" x2="${X(WRITE_AT)}" y1="${T}" y2="${Y(0)}"/><text x="${X(WRITE_AT) + 6}" y="${T + 12}">new method added</text></g>
    <path class="st-area" id="st-area"/><path class="st-pd" id="st-pd"/><path class="st-pa" id="st-pa"/>
    <circle class="st-hd" id="st-hd" r="4.5"/><circle class="st-ha" id="st-ha" r="4.5"/>`;
  const at = (arr, x) => { const j = Math.min(Math.floor(x * 2), arr.length - 2), f = x * 2 - j; return arr[j] + (arr[j + 1] - arr[j]) * f; };
  const pts = (arr, x) => { const p = []; for (let j = 0; j / 2 < x; j++) p.push([X(j / 2), Y(arr[j])]); p.push([X(x), Y(at(arr, x))]); return p; };
  const line = (p) => "M" + p.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join("L");

  // the attack agent: the weakness it finds, the angle it takes, and the line it writes
  const IDEAS = [
    ["transfer depends on a single surrogate", "ensemble several CLIP models", "loss = sum(sim(lib.encode(adv, m), t[m]) for m in ens)"],
    ["FreqPure keeps low frequencies", "perturb the low frequencies", 'g = lowpass(lib.grad(loss, adv), p["cutoff"])'],
    ["PAD removes localized patches", "spread the change over the image", "g = g / g.abs().max()  # no hot spots"],
    ["the purifier smooths fine noise", "use larger structures", 'adv = lib.project(adv + p["lr"] * blur(g).sign(), x)'],
    ["no library method survives the denoiser", "optimize what the defense will see", 'loss = sim(lib.encode(smooth(adv), "B16"), t)'],
    ["the new method fails on text cues", "mix in the target's text", 'adv = overlay(adv, target.text, p["alpha"])'],
    ["the defenses adapted to smooth noise", "alternate smooth and sharp steps", 'lr = p["lr"] * (1 if i % 2 else 0.3)'],
    ["PAD, FreqPure and BlueSuffix are all on", "combine what still works", "loss = ens_loss + 0.5 * smooth_loss"],
  ];
  const agent = byId("st-agent"), agentState = byId("st-agent-state");
  let shown = "";
  function writeAgent(r, frac, state) {
    const [weak, angle, core] = IDEAS[r], lines = [`# round ${r + 1} · weakness: ${weak}`, `# angle: ${angle}`, "def attack(x, target, lib, p):", "    ...", `    ${core}`, "    ..."];
    let left = Math.floor(frac * lines.join("\n").length), key = `${r}:${left}`;
    if (key !== shown) {
      shown = key;
      const out = [];
      for (let n = 0; n < lines.length && left >= 0; n++) {  // lines are shown as they start: the caret sits at the end of the typed text
        const l = lines[n], part = l.slice(0, left);
        out.push(n < 2 ? `<span class="c">${esc(part)}</span>` : part.startsWith("def") ? `<span class="k">def</span>${esc(part.slice(3))}` : n === 4 && frac >= 1 ? `<span class="hl">${esc(part)}</span>` : esc(part));  // the finished idea is highlighted
        left -= l.length + 1;
      }
      agent.innerHTML = out.join("\n") + (frac < 1 ? '<span class="st-caret"></span>' : "");
    }
    text(agentState, state);
  }

  // leaderboard
  const rows = Object.keys(MODELS).map((m) => el("div", "st-row", `<span class="mono st-rank"></span><b>${m}</b><i><u></u></i><em></em>`, byId("st-rows")));
  const bar = (v) => 10 + 90 * clamp((v - 0.4) / 0.25);  // scores sit in 0.40-0.65; stretch so rank changes show
  const rank = (c) => { const s = Object.values(MODELS).map((v, i) => [v[c], i]).sort((a, b) => b[0] - a[0]); const r = []; s.forEach(([, i], k) => (r[i] = k)); return r; };
  let board = -1;

  const cap = byId("st-cap");
  const sideA = byId("st-side-a"), sideD = byId("st-side-d"), token = byId("st-token");
  function renderE(t) {
    const seg = SEGS.find((s) => t >= s.t0 && t < s.t0 + s.d);
    const p = seg ? ease(clamp((t - seg.t0) / seg.d)) : 0, k = seg?.k, r = seg ? seg.r : t >= E_END ? N - 1 : 0;
    const x = t >= E_END ? N : !seg ? 0 : k === "write" ? r : r + (k === "defense" ? 0.5 : 0) + p / 2;

    const ang = Math.PI * (k === "attack" ? 1 + p : k === "defense" ? p : 1);
    token.setAttribute("cx", 120 + 92 * Math.cos(ang));
    token.setAttribute("cy", 120 + 92 * Math.sin(ang));
    toggle(byId("st-arc-a"), "st-on", k === "attack");
    toggle(byId("st-arc-d"), "st-on", k === "defense");
    byId("st-arc-a").style.strokeDashoffset = k === "attack" ? 1 - p : 0;
    byId("st-arc-d").style.strokeDashoffset = k === "defense" ? 1 - p : 0;
    toggle(sideA, "st-on", k === "attack" || k === "write");
    toggle(sideD, "st-on", k === "defense");
    toggle(sideA, "st-write", k === "write");
    text(byId("st-round"), String(r + 1).padStart(2, "0"));
    const R0 = ROUNDS[r], newly = k === "attack" && R0.new;
    const [head, sub] = k === "write" ? ["library falls short → agent writes a new method", "checked against tests, then added"]
      : k === "attack" ? (newly ? ["attack agent → agent-written method", "the new method gets through the current defense"] : ["attack agent adapts", "finds what gets through the current defense"])
      : k === "defense" ? ["defense agent adapts", "adapts to the new attacks"]
      : t >= E_END ? ["8 rounds done → results go to the leaderboard", "attacks and defenses kept adapting to each other"]
      : ["round 1 · starting from the library", "attacks and defenses adapt to each other"];
    text(cap.firstChild, head);
    text(cap.lastChild, sub);

    const pa = pts(ASR, x), pd = pts(DEF, x), last = (p) => p[p.length - 1];
    byId("st-pa").setAttribute("d", line(pa));
    byId("st-pd").setAttribute("d", line(pd));
    byId("st-area").setAttribute("d", `${line(pa)}V${Y(0)}H${X(0)}Z`);
    for (const [id, p] of [["st-ha", pa], ["st-hd", pd]]) { byId(id).setAttribute("cx", last(p)[0]); byId(id).setAttribute("cy", last(p)[1]); }
    byId("st-band").setAttribute("x", X(r));
    byId("st-band").style.fill = k === "attack" ? "rgba(244,114,182,.10)" : k === "defense" ? "rgba(52,211,153,.10)" : k === "write" ? "rgba(124,92,255,.16)" : "";
    toggle(byId("st-mark"), "st-on", t >= W0);

    const u = seg ? clamp((t - seg.t0) / seg.d) : 0, w = k === "attack" ? clamp(u / 0.8) : k === "write" ? clamp(u / 0.85) : k === "defense" || t >= E_END ? 1 : 0;
    writeAgent(r, w, k === "defense" ? "testing against the defense" : k === "write" ? "writing a new method" : t >= E_END ? "done" : w < 0.4 ? "thinking" : "writing");

    const c = SEGS.filter((s) => s.k === "defense" && t >= s.t0 + s.d).length;
    if (c !== board) {
      const now = rank(c), prev = rank(Math.max(0, c - 1));
      rows.forEach((e, i) => {
        e.style.setProperty("--k", now[i]);
        e.querySelector(".st-rank").textContent = now[i] + 1;
        e.querySelector("u").style.width = `${bar(Object.values(MODELS)[i][c])}%`;
        const d = c ? prev[i] - now[i] : 0, m = e.querySelector("em");
        if (d) { e.classList.remove("moved"); void e.offsetWidth; e.classList.add("moved"); }
        m.textContent = d > 0 ? "▲" : d < 0 ? "▼" : "–";
        m.className = d > 0 ? "st-up" : d < 0 ? "st-down" : "";
      });
      board = c;
    }
  }
  player(byId("st-evo"), { loop: E_LOOP, end: E_LOOP - 0.01, render: renderE, play: byId("st-e-play"), replay: byId("st-e-replay") });
})();
