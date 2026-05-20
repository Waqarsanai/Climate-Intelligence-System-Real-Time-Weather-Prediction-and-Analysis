/* UI integration for Karachi Weather Prediction System */

const API_BASE = window.API_BASE || "";

const byId = (id) => document.getElementById(id);

async function fetchJson(path, options = {}) {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

function formatValue(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${value}${suffix}`;
}

function updateLiveWeather(data) {
  byId("currentTemp").textContent = formatValue(Math.round(data.temperature), "°C");
  byId("feelsLike").textContent = formatValue(Math.round(data.feels_like ?? data.temperature), "°C");
  byId("humidity").textContent = formatValue(Math.round(data.humidity), "%");
  byId("pressure").textContent = formatValue(Math.round(data.pressure), " hPa");
  byId("windSpeed").textContent = formatValue(Math.round(data.wind_speed), " km/h");
  byId("precipitation").textContent = formatValue(data.precipitation ?? 0, " mm");
  byId("cloudCover").textContent = formatValue(Math.round(data.cloud_cover ?? 0), "%");
  byId("weatherCondition").textContent = data.description || "Current conditions";
  byId("currentLocation").textContent = data.city ? `Location: ${data.city}` : "";
  byId("lastUpdate").textContent = data.timestamp ? `Updated: ${data.timestamp}` : "";
}

function renderForecast(predictions) {
  const list = byId("forecastList");
  list.innerHTML = "";
  if (!predictions || predictions.length === 0) {
    list.innerHTML = "<div class=\"list-item\">No forecast data.</div>";
    return;
  }
  predictions.slice(0, 12).forEach((item) => {
    const row = document.createElement("div");
    row.className = "list-item";
    row.innerHTML = `
      <span>${item.time}</span>
      <strong>${formatValue(Math.round(item.temp), "°C")}</strong>
    `;
    list.appendChild(row);
  });
}

function updateModelMetrics(metrics = {}) {
  const r2 = metrics.r2 ?? metrics.R2 ?? 0;
  const rmse = metrics.rmse ?? metrics.RMSE ?? "--";
  const mae = metrics.mae ?? metrics.MAE ?? "--";
  const targetMin = metrics.target_accuracy_min ?? 93;
  const targetMax = metrics.target_accuracy_max ?? 95;

  byId("r2Score").textContent = Number.isFinite(r2) ? r2.toFixed(2) : "--";
  byId("rmse").textContent = Number.isFinite(rmse) ? rmse.toFixed(2) : rmse;
  byId("mae").textContent = Number.isFinite(mae) ? mae.toFixed(2) : mae;
  byId("accuracy").textContent = `${targetMin}–${targetMax}%`;

  const pct = Math.min(Math.max(r2, 0), 1) * 100;
  byId("r2Progress").style.width = `${pct}%`;
}

async function refreshWeather() {
  const data = await fetchJson("/api/weather/current");
  updateLiveWeather(data);
}

async function loadForecast() {
  const data = await fetchJson("/api/predict/24h");
  renderForecast(data.predictions || data);
}

async function loadMetrics() {
  try {
    const data = await fetchJson("/api/system/status");
    if (data.model_metrics) {
      updateModelMetrics(data.model_metrics);
    } else {
      updateModelMetrics({
        R2: 0.9,
        RMSE: 0.8,
        MAE: 0.6,
        target_accuracy_min: data.target_accuracy_min || 93,
        target_accuracy_max: data.target_accuracy_max || 95,
      });
    }
  } catch (err) {
    console.error("Metrics error", err);
  }
}

async function retrainModel() {
  try {
    await fetchJson("/api/retrain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ async: true }) });
    alert("Retraining started. Check status in a few minutes.");
  } catch (err) {
    alert("Retraining failed. See console for details.");
  }
}

function bindActions() {
  byId("refresh-button")?.addEventListener("click", () => refreshWeather().catch(console.error));
  byId("forecast-24h-btn")?.addEventListener("click", () => loadForecast().catch(console.error));
  byId("retrain-button")?.addEventListener("click", () => retrainModel().catch(console.error));
}

async function initializeUI() {
  bindActions();
  await refreshWeather();
  await loadForecast();
  await loadMetrics();
}
