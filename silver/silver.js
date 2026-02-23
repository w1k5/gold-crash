async function loadJSON() {
  const res = await fetch("../silver.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load silver.json (${res.status})`);
  return await res.json();
}

function el(id) { return document.getElementById(id); }

function setText(id, v) { el(id).textContent = (v ?? "—"); }

function scoreToLabel(x) {
  if (x == null || Number.isNaN(Number(x))) return "—";
  return String(Math.round(Number(x)));
}

function renderList(ul, items) {
  ul.innerHTML = "";
  (items || []).forEach((t) => {
    const li = document.createElement("li");
    li.textContent = t;
    ul.appendChild(li);
  });
}

function renderPillars(container, pillars) {
  container.innerHTML = "";
  (pillars || []).forEach((p) => {
    const d = document.createElement("div");
    d.className = "card";
    d.innerHTML = `
      <div class="name">${p.name}</div>
      <div class="row">
        <div class="state">score</div>
        <div class="last">${scoreToLabel(p.score)}/100</div>
      </div>
      <div class="subrow" style="display:block;margin-top:10px;">
        ${(p.metrics || []).map((m) => `<div style="margin:4px 0;color:#9aa7b6;">${m}</div>`).join("")}
      </div>
    `;
    container.appendChild(d);
  });
}

function renderSignals(container, signals) {
  container.innerHTML = "";
  (signals || []).forEach((s) => {
    const d = document.createElement("div");
    d.className = "card";
    d.innerHTML = `
      <div class="name">${s.name}</div>
      <div class="row">
        <div class="state">${s.state}</div>
        <div class="last">${s.last}</div>
      </div>
      <div class="subrow">
        <div>${s.percentile}</div>
        <div class="spark">${s.sparkline || "—"}</div>
      </div>
    `;
    container.appendChild(d);
  });
}

function renderRules(tbody, rules) {
  tbody.innerHTML = "";
  (rules || []).forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="reg">${r.regime}</td>
      <td>${r.definition}</td>
      <td>${r.action_bias}</td>
      <td>${r.escalation}</td>
      <td>${r.deescalation}</td>
    `;
    tbody.appendChild(tr);
  });
}

(async function main() {
  try {
    const j = await loadJSON();

    setText("title", j.title);
    setText("subtitle", j.subtitle);
    setText("as_of", j.as_of);
    setText("data_dates", j.data_dates);
    setText("confidence", j.confidence);

    const pill = el("regime_pill");
    pill.textContent = j.regime || "—";
    pill.classList.remove("GREEN", "BLUE", "ORANGE", "RED");
    if (j.regime) pill.classList.add(j.regime);

    setText("action_bias", j.action_bias);
    const stability = j.stability || {};
    const stabilityLabel = String(stability.label || "unstable").replaceAll("_", " ");
    setText("stability_state", `${stabilityLabel} (${scoreToLabel(stability.score_0_100)}/100)`);
    const stabilityReasons = Array.isArray(stability.reasons) && stability.reasons.length
      ? `Reasons: ${stability.reasons.join(", ")}`
      : "Reasons: none";
    setText("stability_reasons", stabilityReasons);
    setText("regime_desc", j.regime_description);

    setText("event_risk", scoreToLabel(j.now_cards?.event_risk_1_5d));
    setText("fragility", scoreToLabel(j.now_cards?.fragility_1_3m));
    setText("liq_regime", j.now_cards?.liquidity_regime);

    renderList(el("drivers"), j.top_drivers);
    renderList(el("escalate"), j.what_changes_next?.escalate);
    renderList(el("deescalate"), j.what_changes_next?.deescalate);

    renderPillars(el("pillars"), j.pillars);
    renderSignals(el("signals"), j.signals);

    const triggered = (j.signals || []).filter((x) => x.state === "Triggered").length;
    setText("signal_count", `Signals today: ${triggered} triggered`);

    renderRules(el("rules").querySelector("tbody"), j.rules);
  } catch (e) {
    console.error(e);
    setText("data_dates", "Failed to load silver.json");
    setText("regime_desc", String(e));
  }
})();
