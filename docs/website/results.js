/* results: numbers from the paper (Tables 1 and 5, Section 5) */
const RESULTS = {"attacks": [["PromptInject", 64.8, "Injection"], ["CoA", 54.6, "Perturbation"], ["AnyAttack", 49.9, "Generation / editing"], ["FigStep", 49.4, "Injection"], ["FOA", 48.1, "Perturbation"], ["V-Attack", 47.9, "Perturbation"], ["AdvDiffVLM", 47.0, "Generation / editing"], ["M-Attack", 46.3, "Perturbation"], ["ImageMix", 44.8, "Injection"], ["PA-Attack", 43.3, "Perturbation"], ["AdvEDM-A", 42.7, "Generation / editing"], ["PhysPatch", 33.0, "Perturbation"]], "models": [["Gemini-3-Flash", 68.5, 45.2], ["Qwen3.6-Flash", 71.7, 46.9], ["Seed-2.0-Mini", 67.2, 46.1], ["GPT-4.1-Mini", 68.3, 46.4], ["Qwen3.5-122B-A10B", 72.8, 45.4], ["Qwen3.5-27B", 71.2, 45.2], ["Qwen3.5-9B", 68.9, 47.4], ["Kimi-K2.5", 68.3, 46.5], ["Qwen3-VL-32B", 66.9, 47.4], ["Qwen3-VL-30B-A3B", 64.1, 49.9], ["Qwen3-VL-8B", 61.9, 51.4], ["Cosmos-Reason2-8B", 63.2, 49.2], ["Cosmos-Reason1-7B", 59.6, 52.6]], "defense": {"cols": ["None", "PAD", "FreqPure", "BlueSuffix"], "rows": [["FOA", [48.1, 47.8, 42.4, 43.6]], ["AdvDiffVLM", [47.0, 46.8, 45.8, 45.6]], ["AnyAttack", [50.0, 49.8, 47.6, 48.8]], ["FigStep", [49.1, 48.9, 48.7, 50.8]], ["PromptInject", [65.6, 65.3, 65.1, 43.9]]]}};
const FINDS = [
  ["Text injection fails most", "PromptInject has the highest ASR of all 12 attacks, and FigStep also ranks high."],
  ["Multimodal co-optimization transfers best among visual attacks", "CoA leads the visual perturbations, ahead of FOA, V-Attack and M-Attack."],
  ["Amortized attacks keep up", "AnyAttack needs a single forward pass and still beats the iterative FOA."],
  ["Localized patches transfer less", "PhysPatch ranks last: an attack confined to a region transfers less well."],
  ["Scale does not fix it", "Qwen3.5 gets more accurate from 9B to 122B, yet PromptInject succeeds about as often."],
  ["Action domains fail most", "Manipulation and Driving have the highest ASR, Scene QA the lowest."],
];
const CH_COLOR = { "Perturbation": "var(--violet)", "Generation / editing": "var(--cyan)", "Injection": "var(--pink)" };
const legend = (id, items) => ($(id).innerHTML = items.map(([c, t]) => `<span><i style="background:${c}"></i>${t}</span>`).join(""));
legend("#rs-legend", Object.entries(CH_COLOR).map(([t, c]) => [c, t]));
$("#rs-bars").innerHTML = RESULTS.attacks.map(([n, v, ch]) =>
  `<div class="rs-row"><b>${n}</b><span class="rs-track"><u style="--w:${(v / 70 * 100).toFixed(1)}%;background:${CH_COLOR[ch]}"></u></span><em class="mono">${v.toFixed(1)}</em></div>`).join("");
$("#rs-finds").innerHTML = FINDS.map(([t, d]) => `<div class="rs-find"><h3>${t}</h3><p>${d}</p></div>`).join("");

(() => {  // clean accuracy vs average ASR, one dot per model
  const W = 520, H = 220, L = 48, R = 16, T = 14, B = 40, [x0, x1, y0, y1] = [58, 74, 44, 54];
  const X = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const pts = RESULTS.models.map(([n, c, a]) => [n, c, a]), n = pts.length;
  const mx = pts.reduce((s, p) => s + p[1], 0) / n, my = pts.reduce((s, p) => s + p[2], 0) / n;
  const k = pts.reduce((s, p) => s + (p[1] - mx) * (p[2] - my), 0) / pts.reduce((s, p) => s + (p[1] - mx) ** 2, 0);
  const named = { "Cosmos-Reason1-7B": [8, -8, "start"], "Qwen3-VL-8B": [8, -8, "start"], "Qwen3.5-122B-A10B": [8, 20, "end"], "Gemini-3-Flash": [-8, 16, "end"] };
  $("#rs-scatter").innerHTML =
    [46, 48, 50, 52].map((v) => `<line class="rs-g" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="rs-t" x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`).join("") +
    [60, 64, 68, 72].map((v) => `<text class="rs-t" x="${X(v)}" y="${H - 20}" text-anchor="middle">${v}</text>`).join("") +
    `<text class="rs-t rs-ax" x="${(L + W - R) / 2}" y="${H - 3}" text-anchor="middle">clean accuracy (%)</text>` +
    `<text class="rs-t rs-ax" transform="translate(11 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">average ASR (%)</text>` +
    `<line class="rs-fit" x1="${X(x0 + 1)}" y1="${Y(my + k * (x0 + 1 - mx))}" x2="${X(x1 - 1)}" y2="${Y(my + k * (x1 - 1 - mx))}"/>` +
    pts.map(([name, c, a]) => `<circle class="rs-dot" cx="${X(c)}" cy="${Y(a)}" r="5"><title>${name}: clean ${c}%, average ASR ${a}%</title></circle>` +
      (named[name] ? `<text class="rs-t rs-nm" x="${X(c) + named[name][0]}" y="${Y(a) + named[name][1]}" text-anchor="${named[name][2]}">${name.replace("-A10B", "")}</text>` : "")).join("");
})();

(() => {  // defenses: ASR per attack, shaded by the drop from the undefended model
  const { cols, rows } = RESULTS.defense;
  legend("#rs-dlegend", [["var(--green)", "ASR drops by 2 points or more"]]);
  $("#rs-def").innerHTML = `<table><thead><tr><th></th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
    rows.map(([a, v]) => `<tr><th>${a}</th>${v.map((x, i) => {
      const d = v[0] - x;
      return `<td class="${i === 0 ? "base" : d >= 2 ? "cut" : ""}"${d >= 2 ? ` style="--k:${Math.min(1, d / 22).toFixed(2)}"` : ""}>${x.toFixed(1)}</td>`;
    }).join("")}</tr>`).join("") + "</tbody></table>";
})();

(() => {  // per-model table: clean accuracy and average ASR, sortable
  const rows = RESULTS.models.map(([name, clean, asr]) => ({ name, clean, asr }));
  const cols = [["name", "Model"], ["clean", "Clean accuracy"], ["asr", "Average ASR"]];
  let key = "asr", dir = -1;
  const lo = Math.min(...rows.map((x) => x.asr)) - 5, hi = Math.max(...rows.map((x) => x.asr));
  const box = $("#rs-models");
  function draw() {
    const sorted = [...rows].sort((a, b) => (key === "name" ? a.name.localeCompare(b.name) : a[key] - b[key]) * dir);
    box.innerHTML = `<table><thead><tr>${cols.map(([k, t]) => `<th${k === key ? ` aria-sort="${dir < 0 ? "descending" : "ascending"}"` : ""}><button type="button" data-k="${k}">${t}${k === key ? (dir < 0 ? " ↓" : " ↑") : ""}</button></th>`).join("")}</tr></thead><tbody>` +
      sorted.map((x) => `<tr><th scope="row">${x.name}</th><td class="mono">${x.clean.toFixed(1)}</td><td><span class="rs-mini"><u style="width:${(((x.asr - lo) / (hi - lo)) * 100).toFixed(0)}%"></u></span><em class="mono">${x.asr.toFixed(1)}</em></td></tr>`).join("") + "</tbody></table>";
  }
  box.addEventListener("click", (e) => {
    const k = e.target.closest("button")?.dataset.k;
    if (!k) return;
    dir = k === key ? -dir : k === "name" ? 1 : -1; key = k; draw();
    box.querySelector(`button[data-k="${k}"]`).focus();
  });
  draw();
})();
