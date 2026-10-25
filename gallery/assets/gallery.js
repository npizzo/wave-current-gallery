"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const seeds = [1, 2, 3, 4, 6, 7, 8];
  const completeStates = new Set(["complete", "completed", "success", "passed"]);
  const states = {
    awaiting_sync: "Awaiting first sync", unknown: "Not confirmed", pending: "Awaiting output",
    queued: "Queued", running: "In progress", partial: "Partly complete", analysis_pending: "Analysis pending",
    complete: "Complete", completed: "Complete", success: "Complete", passed: "Passed",
    failed: "Failed", error: "Needs attention", cancelled: "Cancelled", stale: "Update overdue"
  };
  let lastData = null;

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function statusName(value) {
    return Object.prototype.hasOwnProperty.call(states, value) ? value : "unknown";
  }

  function badge(value) {
    const name = statusName(value);
    return node("span", "status " + name, states[name]);
  }

  function displayValue(value) {
    if (value === null || value === undefined) return "Not available";
    if (typeof value === "number") return Number.isFinite(value) ? String(value) : "Not available";
    if (Array.isArray(value)) return value.map(displayValue).join(" × ");
    if (typeof value === "object") {
      if (Object.prototype.hasOwnProperty.call(value, "value")) {
        return displayValue(value.value) + (value.unit ? " " + String(value.unit) : "");
      }
      return JSON.stringify(value);
    }
    return String(value);
  }

  // Imported assets must remain ordinary relative paths inside this gallery.
  function assetPath(value) {
    if (typeof value !== "string" || !value || /[\\\u0000-\u001f]/.test(value)) return null;
    let decoded;
    try { decoded = decodeURIComponent(value); } catch (_) { return null; }
    if (decoded.startsWith("/") || decoded.includes(":") || decoded.split("/").some((p) => p === "..")) return null;
    if (/[?#\\\u0000-\u001f]/.test(decoded)) return null;
    const root = new URL("./", window.location.href);
    const resolved = new URL(value, root);
    return resolved.origin === root.origin && resolved.pathname.startsWith(root.pathname) ? value : null;
  }

  function figure(item) {
    if (!item || typeof item !== "object") return null;
    const path = assetPath(item.src);
    if (!path) return null;
    const card = node("figure", "figure-card");
    const link = node("a", "figure-image");
    link.href = path;
    link.setAttribute("aria-label", "Open figure: " + (item.title || "Simulation result"));
    const image = node("img");
    image.src = path;
    image.alt = String(item.title || "Simulation result");
    image.loading = "lazy";
    image.decoding = "async";
    image.addEventListener("error", () => {
      const message = node("span", "image-failed", "This figure could not be loaded. Refresh the snapshot or open its link to check availability.");
      link.replaceChildren(message);
    }, {once: true});
    link.append(image);
    const caption = node("figcaption");
    caption.append(node("h3", "", item.title || "Simulation result"));
    if (item.caption) caption.append(node("p", "", item.caption));
    card.append(link, caption);
    return card;
  }

  function renderParameters(parameters) {
    if (!parameters || typeof parameters !== "object" || !Object.keys(parameters).length) return;
    const labels = {
      tank_m: "Tank, x × y × z", domain_m: "Tank, x × y × z", grid: "Grid", resolution: "Grid",
      k_rad_m: "Carrier wavenumber (rad m⁻¹)", k: "Carrier wavenumber (rad m⁻¹)",
      ak: "Wave steepness, ak", dt_s: "Time step (s)", tau_s: "Packet time scale, τ₀ (s)",
      tau0_s: "Packet time scale, τ₀ (s)", duration_s: "Duration (s)", tend_s: "Duration (s)",
      preconditioning_s: "Turbulence preconditioning (s)", precondition_s: "Turbulence preconditioning (s)",
      case: "Laboratory case", tank: "Tank, x × y × z", wave_steepness: "Wave steepness, ak",
      wave_number: "Carrier wavenumber", packet_width: "Packet time scale, τ₀", time_step: "Time step",
      numerics: "Numerical method", seeds: "Retained seeds"
    };
    const rows = [];
    for (const [key, value] of Object.entries(parameters)) {
      const row = node("tr");
      const heading = node("th", "", labels[key] || key.replaceAll("_", " "));
      heading.scope = "row";
      row.append(heading, node("td", "", displayValue(value)));
      rows.push(row);
    }
    $("parameters").replaceChildren(...rows);
  }

  function renderMembers(members, awaiting) {
    const cards = [];
    for (const member of members) {
      const card = node("article", "member-card");
      const content = node("div", "member-card-content");
      const top = node("div", "member-top");
      const title = member.label || (member.seed !== null && member.seed !== undefined ? "Seed " + member.seed : "Deterministic controls");
      top.append(node("h3", "", title), badge(awaiting ? "unknown" : member.status));
      content.append(top, node("p", "member-summary", member.summary || (awaiting ? "No Unity result has been imported yet." : "No verified completed output is available in this snapshot.")));
      if (Array.isArray(member.downloads) && member.downloads.length) {
        const downloads = node("div", "downloads");
        for (const item of member.downloads) {
          const path = assetPath(item.href);
          if (!path) continue;
          const link = node("a", "", item.label || "Download data");
          link.href = path;
          downloads.append(link);
        }
        if (downloads.children.length) content.append(downloads);
      }
      card.append(content);
      for (const item of member.figures || []) {
        const result = figure(item);
        if (result) card.append(result);
      }
      cards.push(card);
    }
    $("members").replaceChildren(...cards);
  }

  function render(data) {
    if (!data || data.schema_version !== 1 || !data.campaign || !Array.isArray(data.members)) {
      throw new Error("The published snapshot has an unsupported format.");
    }
    const campaign = data.campaign;
    const awaiting = campaign.status === "awaiting_sync";
    const state = statusName(campaign.status);
    const total = Number.isInteger(campaign.total_pairs) && campaign.total_pairs > 0 ? campaign.total_pairs : 7;
    const confirmed = Number.isInteger(campaign.completed_pairs) && campaign.completed_pairs >= 0 && campaign.completed_pairs <= total;
    $("campaign-title").textContent = campaign.title || "Wave-packet laboratory reproduction";
    $("campaign-status").className = "status " + state;
    $("campaign-status").textContent = states[state];
    $("pair-progress").replaceChildren(document.createTextNode(awaiting || !confirmed ? "— " : campaign.completed_pairs + " "), node("span", "", "/ " + total + " pairs"));
    $("progress-description").textContent = awaiting ? "Completed pairs have not yet been checked on Unity." : confirmed && campaign.completed_pairs === total ? "All paired runs have verified completion records." : "Verified wave / no-wave pairs in this snapshot.";
    const segments = seeds.map((seed) => {
      const member = data.members.find((item) => Number(item.seed) === seed);
      const value = awaiting ? "unknown" : member ? statusName(member.status) : "pending";
      const display = completeStates.has(value) ? "complete" : value;
      const segment = node("span", "pair-segment " + display);
      segment.title = "Seed " + seed + ": " + states[value];
      segment.setAttribute("aria-label", segment.title);
      segment.append(node("i"), node("span", "", String(seed)));
      return segment;
    });
    $("pair-segments").replaceChildren(...segments);
    const validation = campaign.validation || {};
    $("validation-status").textContent = states[statusName(validation.status)];
    $("validation-status").style.color = completeStates.has(validation.status) ? "var(--green)" : validation.status === "failed" ? "var(--red)" : "var(--muted)";
    $("validation-detail").textContent = validation.job_id ? (awaiting ? "Reported from job " : "Job ") + String(validation.job_id) : "No verification record available";
    const stamp = new Date(data.generated_at);
    const validStamp = data.generated_at && !Number.isNaN(stamp.getTime());
    const stale = !awaiting && validStamp && Date.now() - stamp.getTime() > 24 * 60 * 60 * 1000;
    const formatted = validStamp ? stamp.toLocaleString("en-GB", {year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "UTC"}) + " UTC" : "time unavailable";
    $("sync-status").textContent = awaiting ? "Awaiting the first Unity synchronization. Submission is confirmed; current run progress is not." : (stale ? "Snapshot more than 24 hours old · " : "Published snapshot · ") + formatted + (stale ? ". These are the last known results." : ". Updates arrive after synchronization.");
    $("sync-status").parentElement.classList.toggle("stale", Boolean(stale));
    renderParameters(campaign.parameters);
    if (typeof campaign.source_commit === "string" && /^[a-f0-9]{40}$/i.test(campaign.source_commit)) {
      $("source-commit").textContent = campaign.source_commit.slice(0, 12);
      $("source-commit").href = "https://github.com/glwagner/NumericalWaveTanks.jl/tree/" + campaign.source_commit;
    }
    const resultFigures = (Array.isArray(data.results) ? data.results : []).map(figure).filter(Boolean);
    $("result-figures").replaceChildren(...resultFigures);
    const resultDownloads = [];
    for (const item of Array.isArray(data.downloads) ? data.downloads : []) {
      const path = assetPath(item.href);
      if (!path) continue;
      const link = node("a", "", item.label || "Download data");
      link.href = path;
      resultDownloads.push(link);
    }
    $("result-downloads").replaceChildren(...resultDownloads);
    $("result-downloads").hidden = resultDownloads.length === 0;
    $("empty-results").hidden = resultFigures.length > 0;
    $("result-count").textContent = resultFigures.length ? resultFigures.length + " published figures" : state === "analysis_pending" ? "Ensemble analysis pending" : "Results pending";
    if (!resultFigures.length) {
      const message = $("empty-results").querySelector(".small");
      if (!awaiting) {
        message.replaceChildren(document.createTextNode("The final ensemble comparison is not available in this snapshot. Any verified individual results appear below."));
      }
    }
    const metrics = [];
    for (const item of Array.isArray(data.metrics) ? data.metrics : []) {
      const box = node("div", "metric");
      const value = node("div", "metric-value", displayValue(item.value));
      if (item.unit) value.append(node("span", "metric-unit", item.unit));
      box.append(node("div", "metric-label", item.label || "Diagnostic"), value);
      if (item.note) box.append(node("div", "metric-note", item.note));
      metrics.push(box);
    }
    $("metrics").replaceChildren(...metrics);
    $("metrics").hidden = metrics.length === 0;
    renderMembers(data.members, awaiting);
    $("notices").replaceChildren(...(Array.isArray(data.notices) ? data.notices : []).map((item) => node("li", "", item)));
    $("data-error").hidden = true;
    lastData = data;
  }

  async function refresh() {
    const button = $("refresh-button");
    button.disabled = true;
    try {
      const response = await fetch("data/campaign.json", {cache: "no-store"});
      if (!response.ok) throw new Error("The snapshot request returned HTTP " + response.status + ".");
      render(await response.json());
    } catch (error) {
      $("data-error").hidden = false;
      $("data-error").textContent = "The latest campaign snapshot could not be loaded. " + (lastData ? "The previous snapshot remains visible. " : "Current computation status is unavailable. ") + "Try Refresh snapshot, or open the campaign snapshot link below.";
      $("sync-status").textContent = lastData ? "Refresh failed. Showing the last successfully loaded snapshot." : "No campaign snapshot is available. This does not indicate a simulation failure.";
      if (!lastData) $("members").replaceChildren(node("p", "loading-text", "Run summaries will appear when the published snapshot is available."));
    } finally {
      button.disabled = false;
    }
  }

  $("refresh-button").addEventListener("click", refresh);
  refresh();
})();
