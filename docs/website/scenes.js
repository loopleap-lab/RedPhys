// RedPhys website: the physical-world scenes orbiting in two counter-rotating rings (only the near half is shown, nearer scenes larger; every image keeps its own brightness).
(() => {
  const root = document.getElementById("sc-ring");
  if (!root) return;
  const DOMAINS = { driving: ["Driving", "#22d3ee"], manipulation: ["Manipulation", "#f472b6"], grasping: ["Grasping", "#fbbf24"], physics: ["Physics", "#60a5fa"],
    egocentric: ["Egocentric", "#a78bfa"], diagnostics: ["Diagnostics", "#34d399"], scene_qa: ["Scene QA", "#fb923c"] };  // the paper's order
  const SCENES = [["driving_01","driving"],["manipulation_01","manipulation"],["grasping_01","grasping"],["physics_01","physics"],["egocentric_01","egocentric"],["diagnostics_01","diagnostics"],["scene_qa_01","scene_qa"],["driving_02","driving"],["manipulation_02","manipulation"],["grasping_02","grasping"],["physics_02","physics"],["egocentric_02","egocentric"],["diagnostics_02","diagnostics"],["scene_qa_02","scene_qa"],["driving_03","driving"],["manipulation_03","manipulation"],["grasping_03","grasping"],["physics_03","physics"],["egocentric_03","egocentric"],["diagnostics_03","diagnostics"],["scene_qa_03","scene_qa"],["driving_04","driving"],["manipulation_04","manipulation"],["grasping_04","grasping"],["physics_04","physics"],["egocentric_04","egocentric"],["diagnostics_04","diagnostics"],["scene_qa_04","scene_qa"],["driving_05","driving"],["manipulation_05","manipulation"],["grasping_05","grasping"],["physics_05","physics"],["egocentric_05","egocentric"],["diagnostics_05","diagnostics"],["scene_qa_05","scene_qa"],["driving_06","driving"],["manipulation_06","manipulation"],["grasping_06","grasping"],["physics_06","physics"],["egocentric_06","egocentric"],["diagnostics_06","diagnostics"],["scene_qa_06","scene_qa"],["driving_07","driving"],["manipulation_07","manipulation"],["diagnostics_07","diagnostics"],["scene_qa_07","scene_qa"],["driving_08","driving"],["scene_qa_08","scene_qa"],["driving_09","driving"],["driving_10","driving"]];  // [file, domain], domains interleaved
  const RINGS = [{ items: SCENES.filter((_, i) => i % 3 < 2), a: 0.62, b: 0.1, w: 0.14, speed: 0.04, dy: 0, z: 50 },  // front: slow
                 { items: SCENES.filter((_, i) => i % 3 === 2), a: 0.55, b: 0.06, w: 0.1, speed: -0.055, dy: -0.09, z: 0 }];  // behind it, the other way
  const els = RINGS.map((r) => r.items.map(([id, dom]) => {
    const f = document.createElement("figure");
    f.className = "sc"; f.dataset.d = dom; f.style.setProperty("--c", DOMAINS[dom][1]);
    f.innerHTML = `<img src="assets/scenes/${id}.jpg" alt="" decoding="async"><figcaption><i style="background:${DOMAINS[dom][1]}"></i>${DOMAINS[dom][0]}</figcaption>`;
    root.append(f);
    return f;
  }));
  document.getElementById("sc-legend").innerHTML = Object.values(DOMAINS).map(([n, c]) => `<li><i style="background:${c}"></i>${n}</li>`).join("");

  const figs = els.flat(), keys = Object.keys(DOMAINS);
  [...document.querySelectorAll("#sc-legend li")].forEach((li, i) => {  // hover or focus a domain: its scenes light up (nothing freezes)
    const set = (on) => figs.forEach((f) => f.classList.toggle("hl", on && f.dataset.d === keys[i]));
    li.tabIndex = 0; li.onpointerenter = li.onfocus = () => set(true); li.onpointerleave = li.onblur = () => set(false);
  });
  let t = 0, last = 0, raf = 0, seen = false, born = 0;
  function place(dt = 0) {
    const W = root.clientWidth, H = root.clientHeight, small = W < 600;  // phones: larger scenes on a taller ellipse
    const k0 = reduced ? 1 : born ? 1 - (1 - Math.min(1, (performance.now() - born) / 1400)) ** 3 : 0.4;  // the rings fan out once, the first time they are seen
    RINGS.forEach((r, k) => r.items.forEach((_, i) => {
      const th = (i / r.items.length) * Math.PI * 2 + k + t * r.speed, sv = Math.sin(th), f = els[k][i];
      f.style.visibility = sv < 0 ? "hidden" : "visible";  // only the near half of the ring: scenes slide in from one side and out the other
      if (sv < 0) return;
      const w = Math.max(small ? 88 : 64, r.w * W), s = 0.75 + 0.37 * sv;
      f.style.width = `${w}px`;
      f.classList.toggle("tiny", w * s < 104);
      f.style.zIndex = r.z + Math.round(sv * 40);
      f.style.transform = `translate(${W / 2 + Math.cos(th) * r.a * W * (0.4 + 0.6 * k0) - w / 2}px, ${H * 0.42 + (r.dy + sv * r.b * (small ? 1.3 : 1)) * W - w * 0.375}px) scale(${s})`;
    }));
  }
  const frame = (now) => { const dt = last ? Math.min(now - last, 100) / 1000 : 0; t += dt; last = now; place(dt); raf = requestAnimationFrame(frame); };
  const sync = () => { cancelAnimationFrame(raf); last = 0; if (seen && !born) born = performance.now(); if (seen && !reduced) raf = requestAnimationFrame(frame); };
  const check = () => {  // on screen: scrolled in or resized
    const r = root.getBoundingClientRect(), v = r.bottom > 0 && r.top < innerHeight;
    if (v !== seen) { seen = v; sync(); }
    if (reduced) place();
  };
  for (const ev of ["scroll", "resize"]) addEventListener(ev, check, { passive: true });
  place(); check();
})();
