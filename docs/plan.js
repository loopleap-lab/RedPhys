// RedPhys website: the planned automatic weakness reports, one loop per input (a model, then a use case).
(() => {
  const root = document.getElementById("plan-anim");
  if (!root) return;
  const $ = (id) => document.getElementById(id);
  const MODES = [
    { key: "model", pick: "your model", bars: [["Scenario A", 0.74], ["Scenario B", 0.62], ["Scenario C", 0.5], ["Scenario D", 0.4], ["Scenario E", 0.3]] },
    { key: "case", pick: "driving", bars: [["Situation A", 0.7], ["Situation B", 0.58], ["Situation C", 0.46], ["Situation D", 0.38], ["Situation E", 0.3]] },
  ];
  const LOOP = 11.5, ease = (x) => 1 - (1 - Math.min(1, Math.max(0, x))) ** 3, seg = (t, a, b) => Math.min(1, Math.max(0, (t - a) / (b - a)));
  let mode = -1, t = 0, last = 0, raf = 0, seen = false, n = 0, bars = [];
  function build(m) {
    const M = MODES[m];
    $("pa-bars").innerHTML = M.bars.map(([name, v]) => `<div><b>${name}</b><span><u style="--w:${v * 100}%"></u></span></div>`).join("");
    bars = [...$("pa-bars").children];
    $("pa-model").querySelector("b").textContent = ""; $("pa-case").querySelector("b").textContent = "";
    mode = m;
  }
  function render(t, n) {
    const m = n % MODES.length, M = MODES[m];
    if (m !== mode) build(m);
    const picked = t > 0.5, ownEl = $(`pa-${M.key}`), otherEl = $(`pa-${M.key === "model" ? "case" : "model"}`);
    ownEl.classList.toggle("on", picked); ownEl.classList.remove("dim"); otherEl.classList.remove("on"); otherEl.classList.toggle("dim", picked);
    ownEl.querySelector("b").textContent = picked ? M.pick : "";
    const steps = root.querySelectorAll(".pa-steps li"), at = [1.4, 2.9, 4.4], done = [2.9, 4.4, 6.4];
    steps.forEach((li, i) => { li.classList.toggle("on", t >= at[i] && t < done[i]); li.classList.toggle("done", t >= done[i]); li.classList.toggle("seen", t >= at[i]); });
    $("pa-prog").style.width = `${ease(seg(t, 4.4, 6.4)) * 100}%`;
    bars.forEach((el, i) => { const k = ease(seg(t, 6.6 + i * 0.25, 7.6 + i * 0.25)); el.style.setProperty("--k", k); el.classList.toggle("weak", t >= 9.2 && i < 2); });
    root.classList.toggle("report", t >= 6.6);
    const sum = $("pa-sum"); sum.textContent = t >= 9.2 ? "Weak spots found" : ""; sum.classList.toggle("show", t >= 9.2);
  }
  const frame = (ts) => { t += last ? Math.min(ts - last, 100) / 1000 : 0; last = ts; if (t >= LOOP) { t = 0; n++; } render(t, n); raf = requestAnimationFrame(frame); };
  const sync = () => { cancelAnimationFrame(raf); last = 0; if (seen && !reduced) raf = requestAnimationFrame(frame); };
  const check = () => { const r = root.getBoundingClientRect(), v = r.height > 0 && r.bottom > 0 && r.top < innerHeight; if (v !== seen) { seen = v; if (v && t > 8) t = 0; sync(); } };
  for (const ev of ["scroll", "resize", "toggle"]) addEventListener(ev, check, { passive: true, capture: true });  // toggle: the <details> opens
  render(reduced ? 99 : 0, 0); check();
})();
