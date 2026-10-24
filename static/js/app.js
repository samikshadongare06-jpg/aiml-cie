const $ = (s) => document.querySelector(s);
let loadedActual = null;

document.querySelectorAll("nav button").forEach(
  (b) =>
    (b.onclick = () => {
      document
        .querySelectorAll("main > section")
        .forEach((s) => (s.hidden = s.id !== b.dataset.tab));
      document
        .querySelectorAll("nav button")
        .forEach((x) => x.classList.toggle("on", x === b));
    }),
);

$("#ex-load").onclick = async () => {
  const r = await fetch(
    `/example?year=${$("#ex-year").value}&date=${$("#ex-date").value}`,
  );
  const d = await r.json();
  if (d.error) return;
  for (const [k, v] of Object.entries(d.values))
    $(`[name=${k}]`).value = +v.toFixed(2);
  loadedActual = d.actual;
  $("#form").requestSubmit();
};
$("#form").oninput = () => {
  loadedActual = null;
};

$("#form").onsubmit = async (e) => {
  e.preventDefault();
  const r = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(Object.fromEntries(new FormData(e.target))),
  });
  render(await r.json());
};

function render(d) {
  if (d.error) {
    $("#result").innerHTML = `<p class="empty">${d.error}</p>`;
    return;
  }
  const pct = Math.min(100, (d.rain / d.rule_mm) * 100);
  const rows = d.top
    .map(
      (t) =>
        `<tr><td>${t.label}</td><td>${t.value} ${t.unit}</td><td>${t.median}</td></tr>`,
    )
    .join("");
  $("#result").innerHTML = `
    <div class="verdict ${d.safe ? "yes" : "no"}">
      <h2>${d.safe ? "Safe to sow" : "Not safe yet"}</h2>
      <p>${d.safe ? "Conditions look right for sowing on this date." : "Wait, then check again in a week."}</p>
    </div>
    <h3>Rain rule</h3>
    <div class="bar"><i style="width:${pct}%"></i></div>
    <p class="note">Rain since 1 May: ${d.rain} of ${d.rule_mm} mm usually needed before sowing.</p>
    <h3>What the model looked at most</h3>
    <table><tr><th>Reading</th><th>Your value</th><th>Usual</th></tr>${rows}</table>
    <p class="note">Model score ${d.score} of 100. In our 1995 to 2025 tests, "safe" calls were right about 73% of the time. When wrong, poor rain followed in the next three weeks, which the model cannot foresee.</p>
    ${loadedActual === null ? "" : `<p class="actual">What actually happened on that date: ${loadedActual ? "safe to sow" : "not safe"}.</p>`}`;
}

const LADDER = [
  ["Yearly rows, one split", "Exp0 annual 80/20 split"],
  ["Yearly rows, 5-fold", "Exp1 annual 5-fold"],
  ["Weekly, always guess", "Baseline weekly (always majority)"],
  ["Weekly, tested by year", "Exp2 grouped by year (honest) accuracy"],
  ["One-line rain rule", "Rule-only baseline accuracy"],
  ["Final Random Forest", "Exp4 Random Forest accuracy"],
];
const CHARTS = [
  [
    "exp0_confusion",
    "Experiment 0: one yearly split. The model mostly guessed the common class.",
  ],
  [
    "exp1_folds",
    "Experiment 1: 5-fold test on yearly data. Scores swing and fall below guessing.",
  ],
  [
    "exp2_cv",
    "Experiment 2: weekly rows. Random and year-grouped tests give similar scores.",
  ],
  [
    "exp2_confusion",
    'Weekly model tested year by year: most safe dates found, some false "safe" calls.',
  ],
  [
    "exp2_vs_rule",
    "A one-line rule (75 mm of rain) scores about as well as the model.",
  ],
  [
    "exp3_correlation",
    "Several local features move together, so we kept one or two from each group.",
  ],
  [
    "exp3_importance",
    "Rain since 1 May matters most by far; climate indices add a little.",
  ],
  ["exp3_selection", "Choosing features raised the F1 score a little."],
  [
    "exp4_models",
    "Random Forest scored higher than a decision tree and logistic regression.",
  ],
  [
    "exp4_threshold",
    'A stricter cutoff did not make "safe" calls more reliable.',
  ],
  [
    "case_studies",
    "2015 started wet then dried out; 2025 had heavy rain in May.",
  ],
];
fetch("/static/results.json")
  .then((r) => r.json())
  .then((R) => {
    $("#ladder").innerHTML = LADDER.filter(([, k]) => R[k])
      .map(([l, k]) => {
        const v = Math.round(R[k].mean * 100);
        return `<div class="row"><span>${l}</span><div class="bar"><i style="width:${v}%"></i></div><b>${v}%</b></div>`;
      })
      .join("");
  })
  .catch(() => {});
$("#gallery").innerHTML = CHARTS.map(
  ([f, c]) =>
    `<figure><img src="/static/charts/${f}.png" alt="${c}" loading="lazy"><figcaption>${c}</figcaption></figure>`,
).join("");
