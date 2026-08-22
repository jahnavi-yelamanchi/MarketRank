const form = document.querySelector("#request-form");
const matches = document.querySelector("#matches");
const inspector = document.querySelector("#inspector");
const emptyState = document.querySelector("#empty-state");
const policy = document.querySelector("#policy");

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void findMatches();
});

async function findMatches() {
  const values = new FormData(form);
  const body = {
    request_id: `web-${Date.now()}`,
    user_id: "web-user",
    category: values.get("category").trim().toLowerCase(),
    latitude: -23.5505,
    longitude: -46.6333,
    budget: Number(values.get("budget")),
    max_distance_km: Number(values.get("radius")),
  };
  try {
    const response = await fetch("/v1/matches", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!response.ok) throw new Error("The request could not be matched.");
    renderMatches(await response.json(), body.category);
  } catch (error) {
    renderError(error.message);
  }
}

void findMatches();

function renderMatches(data, category) {
  matches.innerHTML = "";
  emptyState.hidden = data.matches.length > 0;
  emptyState.textContent = data.matches.length
    ? ""
    : `No feasible ${category} providers meet this request. Try a larger radius or budget.`;
  policy.textContent = data.policy.replace("_", " ");
  data.matches.forEach((match, index) => {
    const element = document.createElement("button");
    element.className = "match";
    element.innerHTML = `
      <span class="rank">0${index + 1}</span><span class="provider">${title(match.provider_id)}</span>
      <span class="data">${match.ranking_features.distance_km.toFixed(1)} km</span>
      <span class="data">$${match.ranking_features.price.toFixed(2)}</span>
      <span class="score">${match.predicted_score.toFixed(2)}<span class="bar"><i style="width:${Math.max(4, match.predicted_score * 100)}%"></i></span></span>`;
    element.addEventListener("click", () => selectMatch(match, data.constraints_applied, element));
    matches.append(element);
    if (index === 0) selectMatch(match, data.constraints_applied, element);
  });
}

function renderError(message) {
  matches.innerHTML = "";
  emptyState.hidden = false;
  emptyState.textContent = message;
  inspector.innerHTML = '<p class="inspector-empty">Adjust the request and try again.</p>';
}

function selectMatch(match, constraints, selected) {
  document.querySelectorAll(".match").forEach((row) => row.classList.remove("selected"));
  selected.classList.add("selected");
  const feature = match.ranking_features;
  inspector.innerHTML = `<div class="inspector-heading"><h3>Why this ranked first</h3><p class="reason">${title(match.provider_id)}. ${match.reasons.join(". ")}.</p></div>
    <div class="evidence-grid"><section><h4>Key feature values</h4><ul class="features">
      ${featureRow("Quality (0–1)", feature.quality)}${featureRow("Completion rate", feature.completion_rate)}
      ${featureRow("Capacity remaining", feature.capacity_remaining)}${featureRow("Price fit", feature.user_price_fit)}
      ${featureRow("Delivery ETA", `${feature.delivery_time_hours.toFixed(1)} hours`)}</ul></section>
    <section><h4>Applied constraints</h4><ul class="constraints">${constraints.map((item) => `<li><span>${item}</span><b>Pass</b></li>`).join("")}</ul></section></div>`;
}

function featureRow(label, value) { return `<li><span>${label}</span><b>${typeof value === "number" ? value.toFixed(2) : value}</b></li>`; }
function title(value) { return value.replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
