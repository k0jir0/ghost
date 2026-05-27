(function () {
  const state = {
    route: window.location.pathname,
    eventSource: null,
    dialogSubmit: null,
    liveSnapshot: null,
  };

  const elements = {
    content: null,
    liveStatus: null,
    liveStats: null,
    pageKicker: null,
    pageTitle: null,
    pageSubtitle: null,
    refresh: null,
    globalRun: null,
    dialog: null,
    dialogKicker: null,
    dialogTitle: null,
    dialogCopy: null,
    dialogBody: null,
    dialogSubmit: null,
  };

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    elements.content = document.getElementById("content");
    elements.liveStatus = document.getElementById("live-status");
    elements.liveStats = document.getElementById("live-stats");
    elements.pageKicker = document.getElementById("page-kicker");
    elements.pageTitle = document.getElementById("page-title");
    elements.pageSubtitle = document.getElementById("page-subtitle");
    elements.refresh = document.getElementById("global-refresh");
    elements.globalRun = document.getElementById("global-run");
    elements.dialog = document.getElementById("action-dialog");
    elements.dialogKicker = document.getElementById("dialog-kicker");
    elements.dialogTitle = document.getElementById("dialog-title");
    elements.dialogCopy = document.getElementById("dialog-copy");
    elements.dialogBody = document.getElementById("dialog-body");
    elements.dialogSubmit = document.getElementById("dialog-submit");

    document.body.addEventListener("click", handleClick);
    window.addEventListener("popstate", renderRoute);
    elements.refresh.addEventListener("click", renderRoute);
    elements.globalRun.addEventListener("click", () => openRunDialog());
    elements.dialog.addEventListener("close", () => {
      state.dialogSubmit = null;
      elements.dialogSubmit.disabled = false;
      elements.dialogBody.innerHTML = "";
    });
    elements.dialogSubmit.addEventListener("click", async (event) => {
      if (!state.dialogSubmit) {
        return;
      }
      event.preventDefault();
      const form = elements.dialogBody.querySelector("form");
      const payload = form ? new FormData(form) : new FormData();
      try {
        elements.dialogSubmit.disabled = true;
        await state.dialogSubmit(payload);
        elements.dialog.close();
      } catch (error) {
        elements.dialogSubmit.disabled = false;
        showToast(messageOf(error), "error");
      }
    });

    connectLiveStream();
    renderRoute();
  }

  function handleClick(event) {
    const link = event.target.closest("[data-link]");
    if (link) {
      event.preventDefault();
      navigate(link.getAttribute("href"));
      return;
    }

    const action = event.target.closest("[data-action]");
    if (!action) {
      return;
    }

    event.preventDefault();
    const name = action.dataset.action;
    const id = action.dataset.id;
    if (name === "refresh") {
      renderRoute();
    } else if (name === "open-run") {
      openRunDialog();
    } else if (name === "queue-task") {
      openTaskDialog();
    } else if (name === "resume-run") {
      resumeRun(id);
    } else if (name === "cancel-run") {
      cancelRun(id);
    } else if (name === "register-run") {
      openRegisterDialog(id);
    } else if (name === "promote-model") {
      openPromoteDialog(id);
    } else if (name === "reject-model") {
      openRejectDialog(id);
    } else if (name === "retrain-model") {
      openRetrainDialog(id);
    } else if (name === "toggle-task") {
      toggleTask(id, action.dataset.completed === "true");
    } else if (name === "delete-task") {
      deleteTask(id);
    } else if (name === "edit-task") {
      openEditTaskDialog(id, action.dataset.text || "");
    } else if (name === "predict-model") {
      openPlaygroundForModel(id);
    }
  }

  function navigate(path) {
    if (window.location.pathname !== path) {
      history.pushState({}, "", path);
    }
    renderRoute();
  }

  async function renderRoute() {
    syncNav();
    const path = window.location.pathname;

    if (path === "/") {
      setPage(
        "Control Plane",
        "Overview",
        "System health, active work, production posture, and the next move."
      );
      await renderOverviewPage();
      return;
    }

    if (path === "/runs") {
      setPage(
        "Training Ledger",
        "Runs",
        "Every orchestration and experiment record, with status, metrics, and action hooks."
      );
      await renderRunsPage();
      return;
    }

    if (path.startsWith("/runs/")) {
      setPage(
        "Run Inspection",
        "Run Detail",
        "Deep inspection of planning, metrics, artifacts, and runtime state."
      );
      await renderRunDetailPage(path.split("/")[2]);
      return;
    }

    if (path === "/registry") {
      setPage(
        "Governance",
        "Registry",
        "Candidates, staging, production, and the audit trail behind every promotion."
      );
      await renderRegistryPage();
      return;
    }

    if (path.startsWith("/models/")) {
      setPage(
        "Model Operations",
        "Model Detail",
        "Evaluation posture, observability, drift, serving readiness, and retraining pressure."
      );
      await renderModelDetailPage(path.split("/")[2]);
      return;
    }

    if (path === "/tasks") {
      setPage(
        "Agent Work Queue",
        "Tasks",
        "The queue Ghost watches, plus the runtime state the training agent last persisted."
      );
      await renderTasksPage();
      return;
    }

    if (path === "/datasets") {
      setPage(
        "Data Surface",
        "Datasets",
        "Governed manifests, validation reports, and the shape of what Ghost is training on."
      );
      await renderDatasetsPage();
      return;
    }

    if (path === "/playground") {
      setPage(
        "Serving Surface",
        "Playground",
        "Probe promoted models, inspect scores, and watch observability move with live inference."
      );
      await renderPlaygroundPage();
      return;
    }

    navigate("/");
  }

  function syncNav() {
    document.querySelectorAll(".nav-stack a").forEach((link) => {
      const href = link.getAttribute("href");
      link.classList.toggle(
        "is-active",
        href === "/" ? window.location.pathname === "/" : window.location.pathname.startsWith(href)
      );
    });
  }

  function setPage(kicker, title, subtitle) {
    elements.pageKicker.textContent = kicker;
    elements.pageTitle.textContent = title;
    elements.pageSubtitle.textContent = subtitle;
  }

  async function renderOverviewPage() {
    renderLoading("Loading Ghost overview...");
    try {
      const data = await api("/api/overview");
      elements.content.innerHTML = `
        <section class="stats-grid">
          ${statCard("Runs", totalCount(data.run_counts), "Across orchestration and experiment records")}
          ${statCard("Production Models", data.production_models.length, "Registry entries currently serving")}
          ${statCard("Pending Tasks", data.task_counts.pending, "Queue items waiting on the agent")}
        </section>

        <section class="hero-grid">
          <article class="hero-panel hero-panel--accent">
            <p class="eyebrow">Operational Posture</p>
            <h3>Ghost is ${escapeHtml(data.health.status)} with ${data.active_runs.length} active run(s).</h3>
            <p class="hero-copy">
              Current backend preference is <span class="mono">${escapeHtml(data.config.training_backend)}</span>,
              Ollama model <span class="mono">${escapeHtml(data.config.ollama_model)}</span>, and synthetic data is
              <strong>${data.config.allow_synthetic_data ? "enabled" : "disabled"}</strong>.
            </p>
            <div class="pill-row">
              ${badge(data.health.status)}
              ${badge(`${data.agent.running ? "running" : "idle"}`)}
              ${badge(`${data.task_counts.pending} pending tasks`, "draft")}
            </div>
            <div class="metric-line">
              ${sparklineSvg(data.recent_runs.map((run) => run.final_loss).filter((value) => typeof value === "number"))}
            </div>
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Quick Actions</p>
                <h3>Move the system</h3>
              </div>
            </div>
            <div class="stack">
              <button class="ghost-button" data-action="open-run">Launch a training run now</button>
              <button class="ghost-button ghost-button--quiet" data-action="queue-task">Queue work for the agent</button>
              <a class="ghost-button ghost-button--quiet" href="/registry" data-link>Inspect registry pressure</a>
            </div>
            <div class="subtle-separator"></div>
            <div class="stack">
              <div class="callout">
                <p class="list-card__label">Agent memory</p>
                <strong>${escapeHtml(data.agent.recent_task || "No recent task")}</strong>
                <p class="muted-copy">Iterations: ${escapeHtml(String(data.agent.iterations || 0))}</p>
              </div>
              <div class="callout">
                <p class="list-card__label">Storage</p>
                <strong class="mono">${escapeHtml(data.config.data_cache_dir)}</strong>
                <p class="muted-copy">Models at ${escapeHtml(data.config.model_cache_dir)}</p>
              </div>
            </div>
          </article>
        </section>

        <section class="two-column">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Active Runs</p>
                <h3>What is moving right now</h3>
              </div>
              <button class="ghost-button ghost-button--quiet" data-action="refresh">Refresh</button>
            </div>
            ${data.active_runs.length ? runList(data.active_runs) : emptyState("No active runs", "Ghost has no queued or running orchestration at the moment.")}
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Recent Alerts</p>
                <h3>Operational pressure</h3>
              </div>
            </div>
            ${data.recent_alerts.length ? alertList(data.recent_alerts) : emptyState("No current alerts", "Latency, error, and drift alerts will appear here once the serving surface has signal.")}
          </article>
        </section>

        <section class="two-column">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Recent Runs</p>
                <h3>Newest activity</h3>
              </div>
            </div>
            ${tableRuns(data.recent_runs)}
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Production Models</p>
                <h3>Serving candidates</h3>
              </div>
            </div>
            ${data.production_models.length ? modelList(data.production_models) : emptyState("No production models", "Promoted registry entries will show up here after evaluation and approval.")}
          </article>
        </section>

        <section class="triple-grid">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Health</p>
                <h3>Runtime thresholds</h3>
              </div>
            </div>
            <div class="key-grid">
              <div><dt>Status</dt><dd>${badge(data.health.status)}</dd></div>
              <div><dt>System Memory</dt><dd>${displayPercent(data.health.system_memory_percent)}</dd></div>
              <div><dt>GPU Memory</dt><dd>${displayPercent(data.health.gpu_memory_percent)}</dd></div>
              <div><dt>Contexts</dt><dd>${escapeHtml(String(data.context_count))}</dd></div>
            </div>
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Environments</p>
                <h3>Directory profiles</h3>
              </div>
            </div>
            <div class="stack">
              ${data.environments.map(environmentCard).join("")}
            </div>
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Workflows</p>
                <h3>Recent automation</h3>
              </div>
            </div>
            ${data.recent_workflows.length ? workflowList(data.recent_workflows) : emptyState("No workflows queued", "Retraining workflows will appear when drift policies trigger or operators request them.")}
          </article>
        </section>
      `;
    } catch (error) {
      renderError(error);
    }
  }

  async function renderRunsPage() {
    renderLoading("Loading run ledger...");
    try {
      const data = await api("/api/runs");
      elements.content.innerHTML = `
        <section class="stats-grid">
          ${statCard("Total Runs", data.count, "Combined orchestration and experiment rows")}
          ${statCard("Running", countStatus(data.runs, "running"), "Includes queued, planned, and resumed work")}
          ${statCard("Completed", countStatus(data.runs, "completed"), "Runs eligible for registry admission")}
        </section>

        <section class="panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Run Ledger</p>
              <h3>Training activity and current state</h3>
              <p>Use this surface to inspect details, resume interrupted runs, register completed work, or stop active training.</p>
            </div>
            <div class="inline-actions">
              <button class="ghost-button ghost-button--quiet" data-action="refresh">Refresh</button>
              <button class="ghost-button" data-action="open-run">Launch Run</button>
            </div>
          </div>
          ${runsTable(data.runs)}
        </section>
      `;
    } catch (error) {
      renderError(error);
    }
  }

  async function renderRunDetailPage(runId) {
    renderLoading(`Loading run ${runId}...`);
    try {
      const detail = await api(`/api/runs/${runId}`);
      const summary = detail.summary;
      const lossValues = detail.metrics_history.map((metric) => metric.loss);
      const lastMetrics = detail.metrics_history.slice(-8).reverse();

      elements.content.innerHTML = `
        <section class="detail-card">
          <div class="detail-head">
            <div>
              <p class="eyebrow">Run ${escapeHtml(summary.run_id)}</p>
              <h3>${escapeHtml(summary.model_id)}</h3>
              <p class="page-subtitle">
                ${escapeHtml(summary.backend || "unknown backend")} /
                ${escapeHtml(summary.architecture || "unknown architecture")} /
                ${escapeHtml(summary.dataset_id || "no dataset")}
              </p>
            </div>
            <div class="inline-actions">
              ${summary.can_resume ? `<button class="ghost-button ghost-button--quiet" data-action="resume-run" data-id="${escapeHtml(summary.run_id)}">Resume</button>` : ""}
              ${summary.can_cancel ? `<button class="ghost-button ghost-button--danger" data-action="cancel-run" data-id="${escapeHtml(summary.run_id)}">Cancel</button>` : ""}
              ${summary.can_register ? `<button class="ghost-button" data-action="register-run" data-id="${escapeHtml(summary.run_id)}">Register Model</button>` : ""}
            </div>
          </div>
          <div class="pill-row">
            ${badge(summary.status)}
            ${badge(summary.runtime_state || "unknown")}
            ${summary.health_status ? badge(summary.health_status) : ""}
            ${summary.has_checkpoint ? badge("checkpointed", "completed") : ""}
          </div>
        </section>

        <section class="stats-grid">
          ${statCard("Final Loss", displayMetric(summary.final_loss), "Best recorded loss")}
          ${statCard("Final Accuracy", displayPercent(summary.final_accuracy, 100), "Last recorded accuracy")}
          ${statCard("Epochs", summary.epochs_completed || 0, "Completed epochs")}
        </section>

        <section class="detail-grid">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Metrics</p>
                <h3>Training curve</h3>
              </div>
            </div>
            ${lossValues.length ? `<div class="metric-line">${sparklineSvg(lossValues)}</div>` : emptyState("No metric history", "This run has not produced metric checkpoints yet.")}
            ${lastMetrics.length ? metricsTable(lastMetrics) : ""}
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Plan</p>
                <h3>Execution envelope</h3>
              </div>
            </div>
            ${keyValueGrid({
              status: summary.status,
              source: summary.source,
              dataset: summary.dataset_id || "n/a",
              version: summary.dataset_version || "n/a",
              backend: summary.backend || "n/a",
              architecture: summary.architecture || "n/a",
            })}
          </article>
        </section>

        <section class="detail-grid">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Events</p>
                <h3>Lifecycle timeline</h3>
              </div>
            </div>
            ${detail.events.length ? timeline(detail.events) : emptyState("No lifecycle events", "Ghost will persist orchestration events here as the run moves through planning and training.")}
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Artifacts + Registry</p>
                <h3>Downstream outputs</h3>
              </div>
            </div>
            <div class="stack">
              <div class="callout">
                <p class="list-card__label">Artifacts</p>
                ${detail.artifacts.length ? detail.artifacts.map((artifact) => `<p class="mono">${escapeHtml(artifact.artifact_type)} -> ${escapeHtml(artifact.uri)}</p>`).join("") : "<p class='muted-copy'>No persisted artifacts yet.</p>"}
              </div>
              <div class="callout">
                <p class="list-card__label">Registry entries</p>
                ${detail.registry_entries.length ? detail.registry_entries.map((entry) => `<p><a class="data-link" href="/models/${escapeHtml(entry.registry_id)}" data-link>${escapeHtml(entry.registry_id)}</a> ${badge(entry.stage)}</p>`).join("") : "<p class='muted-copy'>This run has not been registered yet.</p>"}
              </div>
            </div>
          </article>
        </section>

        <section class="two-column">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Request</p>
                <h3>Input</h3>
              </div>
            </div>
            ${jsonBlock(detail.request)}
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Analysis</p>
                <h3>Outcome</h3>
              </div>
            </div>
            ${jsonBlock(detail.analysis || {})}
          </article>
        </section>
      `;
    } catch (error) {
      renderError(error);
    }
  }

  async function renderRegistryPage() {
    renderLoading("Loading model registry...");
    try {
      const data = await api("/api/models");
      elements.content.innerHTML = `
        <section class="stats-grid">
          ${statCard("Registry Entries", data.count, "All versioned records")}
          ${statCard("Production", countStatus(data.models, "production", "stage"), "Serving models")}
          ${statCard("Rejected", countStatus(data.models, "rejected", "stage"), "Candidates blocked by governance")}
        </section>

        <section class="panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Registry Surface</p>
              <h3>Promotion and rejection control</h3>
            </div>
            <button class="ghost-button ghost-button--quiet" data-action="refresh">Refresh</button>
          </div>
          ${registryTable(data.models)}
        </section>
      `;
    } catch (error) {
      renderError(error);
    }
  }

  async function renderModelDetailPage(registryId) {
    renderLoading(`Loading model ${registryId}...`);
    try {
      const data = await api(`/api/models/${registryId}`);
      const model = data.model;
      elements.content.innerHTML = `
        <section class="detail-card">
          <div class="detail-head">
            <div>
              <p class="eyebrow">Registry ${escapeHtml(model.registry_id)}</p>
              <h3>${escapeHtml(model.model_id)}</h3>
              <p class="page-subtitle">
                ${escapeHtml(model.backend || "unknown backend")} /
                ${escapeHtml(model.architecture || "unknown architecture")} /
                ${escapeHtml(model.dataset_id || "no dataset")}
              </p>
            </div>
            <div class="inline-actions">
              ${data.can_promote ? `<button class="ghost-button" data-action="promote-model" data-id="${escapeHtml(model.registry_id)}">Promote</button>` : ""}
              <button class="ghost-button ghost-button--quiet" data-action="retrain-model" data-id="${escapeHtml(model.registry_id)}">Queue Retraining</button>
              ${model.stage !== "rejected" ? `<button class="ghost-button ghost-button--danger" data-action="reject-model" data-id="${escapeHtml(model.registry_id)}">Reject</button>` : ""}
            </div>
          </div>
          <div class="pill-row">
            ${badge(model.stage)}
            ${badge(model.evaluation_status)}
            ${data.serving_ready ? badge("serving-ready", "completed") : badge("not-serving", "draft")}
          </div>
        </section>

        <section class="stats-grid">
          ${statCard("Accuracy", displayPercent(model.metrics.final_accuracy, 100), "Recorded at registration time")}
          ${statCard("Loss", displayMetric(model.metrics.final_loss), "Recorded at registration time")}
          ${statCard("Error Rate", displayPercent(data.observability.error_rate, 100), "From served traffic")}
        </section>

        <section class="detail-grid">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Observability</p>
                <h3>Serving behavior</h3>
              </div>
            </div>
            ${keyValueGrid({
              requests: data.observability.request_count,
              errors: data.observability.error_count,
              "avg latency ms": roundNumber(data.observability.avg_latency_ms),
              "p95 latency ms": roundNumber(data.observability.p95_latency_ms),
              "last event": data.observability.last_event_at || "n/a",
              stage: model.stage,
            })}
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Drift</p>
                <h3>Input stability</h3>
              </div>
            </div>
            ${keyValueGrid({
              status: data.drift_report.status,
              samples: data.drift_report.sample_count,
              baseline: roundNumber(data.drift_report.baseline_input_mean),
              current: roundNumber(data.drift_report.current_input_mean),
              shift: roundNumber(data.drift_report.mean_shift),
              issues: (data.drift_report.issues || []).join(", ") || "none",
            })}
          </article>
        </section>

        <section class="two-column">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Evaluation</p>
                <h3>Gate result</h3>
              </div>
            </div>
            ${data.evaluation ? jsonBlock(data.evaluation) : emptyState("No evaluation record", "This registry entry does not have a persisted evaluation payload.")}
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Alerts</p>
                <h3>Derived pressure</h3>
              </div>
            </div>
            ${data.alerts.length ? alertList(data.alerts) : emptyState("No active alerts", "Latency, error, and drift thresholds are currently quiet for this model.")}
          </article>
        </section>

        <section class="two-column">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Audit</p>
                <h3>Decision trail</h3>
              </div>
            </div>
            ${data.audit_entries.length ? auditList(data.audit_entries) : emptyState("No audit entries", "Registry actions will create an audit trail here.")}
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Source Run</p>
                <h3>Lineage</h3>
              </div>
            </div>
            ${data.run ? `<p><a class="data-link" href="/runs/${escapeHtml(data.run.summary.run_id)}" data-link>${escapeHtml(data.run.summary.run_id)}</a></p>${jsonBlock(data.run.summary)}` : emptyState("No linked run", "The originating run detail could not be loaded.")}
          </article>
        </section>

        <section class="panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Inline Prediction</p>
              <h3>Probe this model</h3>
            </div>
          </div>
          ${predictionWorkbench(model.registry_id)}
        </section>
      `;

      wirePredictionWorkbench(model.registry_id);
    } catch (error) {
      renderError(error);
    }
  }

  async function renderTasksPage() {
    renderLoading("Loading task queue...");
    try {
      const data = await api("/api/tasks");
      elements.content.innerHTML = `
        <section class="stats-grid">
          ${statCard("Pending", data.tasks.filter((task) => !task.completed).length, "Tasks not yet marked complete")}
          ${statCard("Completed", data.tasks.filter((task) => task.completed).length, "Historical queue items")}
          ${statCard("Agent Iterations", data.agent.iterations || 0, "Persisted in AGENT.json")}
        </section>

        <section class="two-column">
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Agent State</p>
                <h3>What Ghost last remembered</h3>
              </div>
            </div>
            ${jsonBlock(data.agent)}
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Queue Source</p>
                <h3>Task storage</h3>
              </div>
            </div>
            ${keyValueGrid({
              path: data.source_path,
              format: data.source_format,
              "recent task": data.agent.recent_task || "n/a",
              running: String(Boolean(data.agent.running)),
            })}
            <div class="inline-actions">
              <button class="ghost-button" data-action="queue-task">Create Task</button>
            </div>
          </article>
        </section>

        <section class="panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Tasks</p>
              <h3>Queue contents</h3>
            </div>
          </div>
          ${tasksTable(data.tasks)}
        </section>
      `;
    } catch (error) {
      renderError(error);
    }
  }

  async function renderDatasetsPage() {
    renderLoading("Loading dataset manifests...");
    try {
      const data = await api("/api/datasets");
      elements.content.innerHTML = `
        <section class="stats-grid">
          ${statCard("Manifests", data.datasets.length, "Persisted dataset definitions")}
          ${statCard("Validated", countValidated(data.datasets, "passed"), "Reports with passing validation")}
          ${statCard("Failed Validation", countValidated(data.datasets, "failed"), "Datasets needing operator attention")}
        </section>

        <section class="panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Dataset Catalog</p>
              <h3>Manifest and validation surface</h3>
            </div>
            <button class="ghost-button ghost-button--quiet" data-action="refresh">Refresh</button>
          </div>
          <div class="grid-list grid-list--two">
            ${data.datasets.length ? data.datasets.map(datasetCard).join("") : emptyState("No dataset manifests", "Training on a governed dataset will materialize a manifest here.")}
          </div>
        </section>
      `;
    } catch (error) {
      renderError(error);
    }
  }

  async function renderPlaygroundPage() {
    renderLoading("Loading serving playground...");
    try {
      const data = await api("/api/models");
      const servingModels = data.models.filter((model) =>
        ["staging", "production"].includes(model.stage)
      );
      const requestedModel = new URLSearchParams(window.location.search).get("model");
      elements.content.innerHTML = `
        <section class="hero-grid">
          <article class="hero-panel hero-panel--accent">
            <p class="eyebrow">Serving Surface</p>
            <h3>Send one input and inspect how a promoted model responds.</h3>
            <p class="hero-copy">
              This uses the same Ghost inference surface that writes observability
              events and contributes to drift detection.
            </p>
            <div class="pill-row">
              ${badge(`${servingModels.length} serving models`, servingModels.length ? "completed" : "draft")}
            </div>
          </article>

          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Model Selection</p>
                <h3>Choose a promoted target</h3>
              </div>
            </div>
            ${servingModels.length ? servingModels.map((model) => `
              <div class="callout">
                <p class="list-card__label">${escapeHtml(model.registry_id)}</p>
                <strong>${escapeHtml(model.model_id)}</strong>
                <p class="muted-copy">${escapeHtml(model.dataset_id || "no dataset")} / ${badge(model.stage)}</p>
              </div>
            `).join("") : "<p class='muted-copy'>Promote a model to staging or production to enable the playground.</p>"}
          </article>
        </section>

        <section class="panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Prediction Playground</p>
              <h3>Interactive inference</h3>
            </div>
          </div>
          <form id="playground-form" class="form-stack">
            <div class="field-grid">
              <div class="field">
                <label for="playground-model">Registry Model</label>
                <select id="playground-model" name="registry_id">
                  ${servingModels.map((model) => `
                    <option value="${escapeHtml(model.registry_id)}" ${requestedModel === model.registry_id ? "selected" : ""}>
                      ${escapeHtml(model.registry_id)} (${escapeHtml(model.stage)})
                    </option>
                  `).join("")}
                </select>
              </div>
              <div class="field">
                <label for="playground-template">Input Template</label>
                <input id="playground-template" value="[0.0, 0.1, 0.2, 0.3]" readonly />
              </div>
            </div>
            <div class="field">
              <label for="playground-features">Features (JSON array)</label>
              <textarea id="playground-features" name="features">[0.0, 0.1, 0.2, 0.3]</textarea>
            </div>
            <div class="inline-actions">
              <button class="ghost-button" type="submit">Predict</button>
            </div>
          </form>
          <div id="playground-result" class="stack"></div>
        </section>
      `;
      wirePlaygroundForm();
    } catch (error) {
      renderError(error);
    }
  }

  function renderLoading(message) {
    elements.content.innerHTML = `
      <div class="loading-state">
        <div class="spinner"></div>
        <p>${escapeHtml(message)}</p>
      </div>
    `;
  }

  function renderError(error) {
    elements.content.innerHTML = `
      <div class="error-state">
        <p class="eyebrow">Request Failed</p>
        <h3>Ghost could not complete that page load.</h3>
        <p>${escapeHtml(messageOf(error))}</p>
        <button class="ghost-button" data-action="refresh">Try again</button>
      </div>
    `;
  }

  function statCard(label, value, meta) {
    return `
      <article class="stat-card">
        <p class="stat-card__label">${escapeHtml(String(label))}</p>
        <p class="stat-card__value">${escapeHtml(String(value))}</p>
        <p class="stat-card__meta">${escapeHtml(String(meta))}</p>
      </article>
    `;
  }

  function badge(label, override) {
    const raw = String(label || "unknown");
    const key = (override || raw).toLowerCase().replace(/\s+/g, "-");
    return `<span class="badge badge--${escapeHtml(key)}">${escapeHtml(raw)}</span>`;
  }

  function keyValueGrid(entries) {
    const keys = Object.entries(entries || {});
    return `
      <dl class="key-grid">
        ${keys
          .map(
            ([key, value]) => `
              <div>
                <dt>${escapeHtml(key)}</dt>
                <dd>${formatValue(value)}</dd>
              </div>
            `
          )
          .join("")}
      </dl>
    `;
  }

  function jsonBlock(payload) {
    return `<pre class="json-block mono">${escapeHtml(JSON.stringify(payload || {}, null, 2))}</pre>`;
  }

  function tableRuns(runs) {
    if (!runs.length) {
      return emptyState("No recent runs", "Launch Ghost training work and this table will populate with run records.");
    }

    return `
      <div class="table-shell">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Dataset</th>
              <th>Loss</th>
            </tr>
          </thead>
          <tbody>
            ${runs
              .map(
                (run) => `
                  <tr>
                    <td>
                      <a class="data-link mono" href="/runs/${escapeHtml(run.run_id)}" data-link>${escapeHtml(run.run_id)}</a>
                      <div class="muted-copy">${escapeHtml(run.model_id)}</div>
                    </td>
                    <td>${badge(run.status)}</td>
                    <td>${escapeHtml(run.dataset_id || "n/a")}</td>
                    <td>${escapeHtml(displayMetric(run.final_loss))}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function runsTable(runs) {
    if (!runs.length) {
      return emptyState("No runs yet", "Ghost has not persisted any orchestration or experiment rows yet.");
    }

    return `
      <div class="table-shell">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Data</th>
              <th>Metrics</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            ${runs
              .map(
                (run) => `
                  <tr>
                    <td>
                      <a class="data-link mono" href="/runs/${escapeHtml(run.run_id)}" data-link>${escapeHtml(run.run_id)}</a>
                      <div>${escapeHtml(run.model_id)}</div>
                      <div class="muted-copy">${escapeHtml(run.backend || "unknown")} / ${escapeHtml(run.architecture || "unknown")}</div>
                    </td>
                    <td>
                      ${badge(run.status)}
                      ${run.runtime_state ? `<div class="muted-copy">${escapeHtml(run.runtime_state)}</div>` : ""}
                    </td>
                    <td>
                      <strong>${escapeHtml(run.dataset_id || "n/a")}</strong>
                      <div class="muted-copy">${escapeHtml(run.dataset_version || "n/a")}</div>
                    </td>
                    <td>
                      <div>loss ${escapeHtml(displayMetric(run.final_loss))}</div>
                      <div>acc ${escapeHtml(displayPercent(run.final_accuracy, 100))}</div>
                      <div class="muted-copy">${escapeHtml(String(run.epochs_completed || 0))} epoch(s)</div>
                    </td>
                    <td>
                      <div class="inline-actions">
                        <a class="ghost-button ghost-button--quiet" href="/runs/${escapeHtml(run.run_id)}" data-link>Inspect</a>
                        ${run.can_resume ? `<button class="ghost-button ghost-button--quiet" data-action="resume-run" data-id="${escapeHtml(run.run_id)}">Resume</button>` : ""}
                        ${run.can_cancel ? `<button class="ghost-button ghost-button--danger" data-action="cancel-run" data-id="${escapeHtml(run.run_id)}">Cancel</button>` : ""}
                        ${run.can_register ? `<button class="ghost-button" data-action="register-run" data-id="${escapeHtml(run.run_id)}">Register</button>` : ""}
                      </div>
                    </td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function registryTable(models) {
    if (!models.length) {
      return emptyState("No registry entries", "Register a completed run to create a candidate model record.");
    }

    return `
      <div class="table-shell">
        <table>
          <thead>
            <tr>
              <th>Registry</th>
              <th>Stage</th>
              <th>Evaluation</th>
              <th>Dataset</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            ${models
              .map(
                (model) => `
                  <tr>
                    <td>
                      <a class="data-link mono" href="/models/${escapeHtml(model.registry_id)}" data-link>${escapeHtml(model.registry_id)}</a>
                      <div>${escapeHtml(model.model_id)}</div>
                    </td>
                    <td>${badge(model.stage)}</td>
                    <td>
                      ${badge(model.evaluation_status)}
                      ${(model.evaluation_issues || []).length ? `<div class="muted-copy">${escapeHtml(model.evaluation_issues[0])}</div>` : ""}
                    </td>
                    <td>${escapeHtml(model.dataset_id || "n/a")}</td>
                    <td>
                      <div class="inline-actions">
                        <a class="ghost-button ghost-button--quiet" href="/models/${escapeHtml(model.registry_id)}" data-link>Inspect</a>
                        ${model.evaluation_status === "passed" ? `<button class="ghost-button" data-action="promote-model" data-id="${escapeHtml(model.registry_id)}">Promote</button>` : ""}
                        <button class="ghost-button ghost-button--quiet" data-action="retrain-model" data-id="${escapeHtml(model.registry_id)}">Retrain</button>
                        ${model.stage !== "rejected" ? `<button class="ghost-button ghost-button--danger" data-action="reject-model" data-id="${escapeHtml(model.registry_id)}">Reject</button>` : ""}
                      </div>
                    </td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function tasksTable(tasks) {
    if (!tasks.length) {
      return emptyState("No queued tasks", "Use the queue action to give the agent new work.");
    }

    return `
      <div class="table-shell">
        <table>
          <thead>
            <tr>
              <th>Task</th>
              <th>Status</th>
              <th>Updated</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            ${tasks
              .map(
                (task) => `
                  <tr>
                    <td>
                      <div>${escapeHtml(task.text)}</div>
                      <div class="muted-copy mono">${escapeHtml(task.task_id || "n/a")}</div>
                    </td>
                    <td>${badge(task.completed ? "completed" : "pending", task.completed ? "completed" : "running")}</td>
                    <td>${escapeHtml(formatDate(task.updated_at || task.created_at))}</td>
                    <td>
                      <div class="inline-actions">
                        <button class="ghost-button ghost-button--quiet" data-action="toggle-task" data-id="${escapeHtml(task.task_id || "")}" data-completed="${task.completed ? "true" : "false"}">${task.completed ? "Reopen" : "Complete"}</button>
                        <button class="ghost-button ghost-button--quiet" data-action="edit-task" data-id="${escapeHtml(task.task_id || "")}" data-text="${escapeAttr(task.text)}">Edit</button>
                        <button class="ghost-button ghost-button--danger" data-action="delete-task" data-id="${escapeHtml(task.task_id || "")}">Delete</button>
                      </div>
                    </td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function datasetCard(dataset) {
    const report = dataset.validation_report;
    const stats = report && report.stats ? report.stats : {};
    return `
      <article class="list-card">
        <div class="split-header">
          <div>
            <p class="list-card__label">${escapeHtml(dataset.dataset_id)}</p>
            <strong>${escapeHtml(dataset.version)}</strong>
          </div>
          ${badge(dataset.validation_status || "unknown")}
        </div>
        <p class="muted-copy mono">${escapeHtml(dataset.source_uri || "n/a")}</p>
        <div class="subtle-separator"></div>
        ${keyValueGrid({
          source: dataset.metadata && dataset.metadata.source ? dataset.metadata.source : "n/a",
          synthetic: dataset.metadata && dataset.metadata.synthetic ? "true" : "false",
          "train samples": stats.train_samples ?? "n/a",
          "eval samples": stats.eval_samples ?? "n/a",
          classes: stats.observed_num_classes ?? dataset.schema.num_classes ?? "n/a",
          shape: Array.isArray(stats.train_feature_shape) ? stats.train_feature_shape.join(" x ") : "n/a",
        })}
      </article>
    `;
  }

  function modelList(models) {
    return `
      <div class="stack">
        ${models
          .map(
            (model) => `
              <div class="callout">
                <p class="list-card__label">${escapeHtml(model.registry_id)}</p>
                <strong>${escapeHtml(model.model_id)}</strong>
                <p class="muted-copy">${escapeHtml(model.dataset_id || "n/a")} / ${badge(model.stage)}</p>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function runList(runs) {
    return `
      <div class="stack">
        ${runs
          .map(
            (run) => `
              <div class="callout">
                <div class="split-header">
                  <div>
                    <p class="list-card__label">${escapeHtml(run.run_id)}</p>
                    <strong>${escapeHtml(run.model_id)}</strong>
                  </div>
                  ${badge(run.status)}
                </div>
                <p class="muted-copy">${escapeHtml(run.backend || "unknown")} / ${escapeHtml(run.dataset_id || "n/a")}</p>
                <a class="data-link" href="/runs/${escapeHtml(run.run_id)}" data-link>Open run detail</a>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function workflowList(workflows) {
    return `
      <div class="stack">
        ${workflows
          .map(
            (workflow) => `
              <div class="callout">
                <p class="list-card__label">${escapeHtml(workflow.workflow_type || "workflow")}</p>
                <strong>${escapeHtml(workflow.registry_id || "n/a")}</strong>
                <p class="muted-copy">${escapeHtml(workflow.status || "queued")}</p>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function alertList(alerts) {
    return `
      <div class="stack">
        ${alerts
          .map(
            (alert) => `
              <div class="callout">
                <div class="split-header">
                  <strong>${escapeHtml(alert.registry_id || alert.type || "alert")}</strong>
                  ${badge(alert.type || "alert", "warning")}
                </div>
                <p class="muted-copy">${escapeHtml(alert.message || "")}</p>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function auditList(entries) {
    return `
      <div class="timeline">
        ${entries
          .map(
            (entry) => `
              <div class="timeline-item">
                <strong>${escapeHtml(entry.action)}</strong>
                <div class="timeline-item__meta">
                  <span>${escapeHtml(entry.actor)}</span>
                  <span>${escapeHtml(formatDate(entry.created_at))}</span>
                </div>
                ${entry.details ? jsonBlock(entry.details) : ""}
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function metricsTable(metrics) {
    return `
      <div class="table-shell">
        <table>
          <thead>
            <tr>
              <th>Epoch</th>
              <th>Step</th>
              <th>Loss</th>
              <th>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            ${metrics
              .map(
                (metric) => `
                  <tr>
                    <td>${escapeHtml(String(metric.epoch))}</td>
                    <td>${escapeHtml(String(metric.step))}</td>
                    <td>${escapeHtml(displayMetric(metric.loss))}</td>
                    <td>${escapeHtml(displayPercent(metric.accuracy, 100))}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function timeline(events) {
    return `
      <div class="timeline">
        ${events
          .map(
            (event) => `
              <div class="timeline-item">
                <strong>${escapeHtml(event.event || "event")}</strong>
                <div class="timeline-item__meta">
                  <span>${escapeHtml(formatDate(event.created_at || event.timestamp))}</span>
                  ${event.task ? `<span>${escapeHtml(event.task)}</span>` : ""}
                  ${event.error ? `<span>${escapeHtml(event.error)}</span>` : ""}
                </div>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function predictionWorkbench(registryId) {
    return `
      <form id="model-predict-form" class="form-stack">
        <div class="field">
          <label for="model-predict-input">Features (JSON array)</label>
          <textarea id="model-predict-input" name="features">[0.0, 0.1, 0.2, 0.3]</textarea>
        </div>
        <div class="inline-actions">
          <button class="ghost-button" type="submit">Predict with ${escapeHtml(registryId)}</button>
          <a class="ghost-button ghost-button--quiet" href="/playground?model=${escapeHtml(registryId)}" data-link>Open playground</a>
        </div>
      </form>
      <div id="model-predict-result" class="stack"></div>
    `;
  }

  function wirePredictionWorkbench(registryId) {
    const form = document.getElementById("model-predict-form");
    const output = document.getElementById("model-predict-result");
    if (!form || !output) {
      return;
    }
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      output.innerHTML = `<div class="loading-state"><div class="spinner"></div><p>Running prediction...</p></div>`;
      try {
        const payload = parseJson(form.elements.features.value);
        const result = await api(`/api/models/${registryId}/predict`, {
          method: "POST",
          body: JSON.stringify({ features: payload }),
        });
        output.innerHTML = `
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Prediction Result</p>
                <h3>Inference payload</h3>
              </div>
            </div>
            ${jsonBlock(result)}
          </article>
        `;
      } catch (error) {
        output.innerHTML = `<div class="error-state"><p>${escapeHtml(messageOf(error))}</p></div>`;
      }
    });
  }

  function wirePlaygroundForm() {
    const form = document.getElementById("playground-form");
    const output = document.getElementById("playground-result");
    if (!form || !output) {
      return;
    }
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const registryId = form.elements.registry_id.value;
      if (!registryId) {
        showToast("Select a promoted model first.", "error");
        return;
      }
      output.innerHTML = `<div class="loading-state"><div class="spinner"></div><p>Scoring payload...</p></div>`;
      try {
        const payload = parseJson(form.elements.features.value);
        const result = await api(`/api/models/${registryId}/predict`, {
          method: "POST",
          body: JSON.stringify({ features: payload }),
        });
        output.innerHTML = `
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Prediction</p>
                <h3>Model response</h3>
              </div>
            </div>
            ${jsonBlock(result.prediction)}
          </article>
          <article class="panel">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Observability</p>
                <h3>After this request</h3>
              </div>
            </div>
            ${jsonBlock(result.observability)}
          </article>
        `;
      } catch (error) {
        output.innerHTML = `<div class="error-state"><p>${escapeHtml(messageOf(error))}</p></div>`;
      }
    });
  }

  function connectLiveStream() {
    if (state.eventSource) {
      state.eventSource.close();
    }
    const eventSource = new EventSource("/api/events");
    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        state.liveSnapshot = payload;
        updateLiveChrome(payload);
      } catch (error) {
        console.error(error);
      }
    };
    eventSource.onerror = () => {
      updateLiveChrome(null, true);
    };
    state.eventSource = eventSource;
  }

  function updateLiveChrome(snapshot, isError) {
    const dot = elements.liveStatus.querySelector(".status-dot");
    if (isError || !snapshot) {
      dot.className = "status-dot is-danger";
      elements.liveStatus.lastElementChild.textContent = "Live stream interrupted";
      elements.liveStats.innerHTML = "";
      return;
    }

    const healthStatus = snapshot.health && snapshot.health.status ? snapshot.health.status : "warning";
    dot.className = `status-dot ${healthStatus === "healthy" ? "is-healthy" : healthStatus === "degraded" ? "is-danger" : ""}`;
    elements.liveStatus.lastElementChild.textContent = `Health ${healthStatus} / ${snapshot.active_runs.length} active run(s)`;
    elements.liveStats.innerHTML = `
      <div><dt>Active</dt><dd>${escapeHtml(String(snapshot.active_runs.length))}</dd></div>
      <div><dt>Pending</dt><dd>${escapeHtml(String(snapshot.task_counts.pending))}</dd></div>
      <div><dt>Completed</dt><dd>${escapeHtml(String(snapshot.run_counts.completed || 0))}</dd></div>
      <div><dt>Updated</dt><dd>${escapeHtml(timeOnly(snapshot.generated_at))}</dd></div>
    `;
  }

  function openDialog({ kicker, title, copy, submitLabel, formHtml, onSubmit }) {
    elements.dialogKicker.textContent = kicker || "Action";
    elements.dialogTitle.textContent = title || "Ghost Action";
    elements.dialogCopy.textContent = copy || "";
    elements.dialogBody.innerHTML = formHtml || "";
    elements.dialogSubmit.textContent = submitLabel || "Save";
    elements.dialogSubmit.disabled = false;
    state.dialogSubmit = onSubmit;
    elements.dialog.showModal();
  }

  function openRunDialog() {
    openDialog({
      kicker: "Training",
      title: "Launch Run",
      copy: "Prepare a run record now, then let Ghost execute it in the background.",
      submitLabel: "Launch Run",
      formHtml: `
        <form class="form-stack">
          <div class="field">
            <label for="run-task">Task</label>
            <textarea id="run-task" name="task">Train a classifier on mnist</textarea>
          </div>
          <div class="field-grid">
            <div class="field">
              <label for="run-dataset">Dataset</label>
              <input id="run-dataset" name="dataset_ref" placeholder="mnist" />
            </div>
            <div class="field">
              <label for="run-model-name">Model Name</label>
              <input id="run-model-name" name="model_name" placeholder="Ghost Console Run" />
            </div>
          </div>
          <label class="checkbox-row">
            <input type="checkbox" name="allow_synthetic" />
            Allow synthetic fallback if the dataset runtime needs demo-mode behavior
          </label>
        </form>
      `,
      onSubmit: async (formData) => {
        const payload = {
          task: formData.get("task"),
          dataset_ref: formData.get("dataset_ref"),
          model_name: formData.get("model_name"),
          allow_synthetic: formData.get("allow_synthetic") === "on",
        };
        const result = await api("/api/runs", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        showToast(`Run ${result.summary.run_id} launched.`, "success");
        navigate(`/runs/${result.summary.run_id}`);
      },
    });
  }

  function openTaskDialog() {
    openDialog({
      kicker: "Queue",
      title: "Create Task",
      copy: "Add work for the autonomous training agent to pick up from TASKS.json.",
      submitLabel: "Create Task",
      formHtml: `
        <form class="form-stack">
          <div class="field">
            <label for="task-text">Task Text</label>
            <textarea id="task-text" name="text">Train a model on cifar10 with a balanced validation split</textarea>
          </div>
        </form>
      `,
      onSubmit: async (formData) => {
        await api("/api/tasks", {
          method: "POST",
          body: JSON.stringify({ text: formData.get("text") }),
        });
        showToast("Task queued for the agent.", "success");
        if (window.location.pathname === "/tasks") {
          await renderTasksPage();
        }
      },
    });
  }

  function openEditTaskDialog(taskId, currentText) {
    openDialog({
      kicker: "Queue",
      title: "Edit Task",
      copy: "Update the queued text while preserving the same task identity.",
      submitLabel: "Save Task",
      formHtml: `
        <form class="form-stack">
          <div class="field">
            <label for="task-edit-text">Task Text</label>
            <textarea id="task-edit-text" name="text">${escapeHtml(currentText)}</textarea>
          </div>
        </form>
      `,
      onSubmit: async (formData) => {
        await api(`/api/tasks/${taskId}`, {
          method: "PATCH",
          body: JSON.stringify({ text: formData.get("text") }),
        });
        showToast("Task updated.", "success");
        await renderTasksPage();
      },
    });
  }

  function openRegisterDialog(runId) {
    openDialog({
      kicker: "Registry",
      title: "Register Run",
      copy: "Evaluate this completed run and create a draft registry candidate if it passes the configured gates.",
      submitLabel: "Register Model",
      formHtml: `
        <form class="form-stack">
          <div class="field-grid">
            <div class="field">
              <label for="register-actor">Actor</label>
              <input id="register-actor" name="actor" value="web-console" />
            </div>
            <div class="field">
              <label for="register-baseline">Baseline Registry ID</label>
              <input id="register-baseline" name="baseline_registry_id" placeholder="optional" />
            </div>
          </div>
          <div class="field-grid">
            <div class="field">
              <label for="register-min-accuracy">Min Accuracy</label>
              <input id="register-min-accuracy" name="min_accuracy" type="number" min="0" max="1" step="0.01" />
            </div>
            <div class="field">
              <label for="register-max-loss">Max Loss</label>
              <input id="register-max-loss" name="max_loss" type="number" min="0" step="0.01" />
            </div>
          </div>
        </form>
      `,
      onSubmit: async (formData) => {
        const payload = {
          actor: formData.get("actor") || "web-console",
          baseline_registry_id: formData.get("baseline_registry_id") || null,
          min_accuracy: parseMaybeNumber(formData.get("min_accuracy")),
          max_loss: parseMaybeNumber(formData.get("max_loss")),
        };
        const result = await api(`/api/runs/${runId}/register`, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        showToast(`Run registered as ${result.model.registry_id}.`, "success");
        navigate(`/models/${result.model.registry_id}`);
      },
    });
  }

  function openPromoteDialog(registryId) {
    openDialog({
      kicker: "Registry",
      title: "Promote Model",
      copy: "Move this candidate to staging, production, or archive it after review.",
      submitLabel: "Promote",
      formHtml: `
        <form class="form-stack">
          <div class="field-grid">
            <div class="field">
              <label for="promote-stage">Stage</label>
              <select id="promote-stage" name="stage">
                <option value="staging">staging</option>
                <option value="production">production</option>
                <option value="archived">archived</option>
              </select>
            </div>
            <div class="field">
              <label for="promote-alias">Alias</label>
              <input id="promote-alias" name="alias" placeholder="optional alias" />
            </div>
          </div>
          <div class="field">
            <label for="promote-actor">Approved By</label>
            <input id="promote-actor" name="approved_by" value="web-console" />
          </div>
        </form>
      `,
      onSubmit: async (formData) => {
        const payload = {
          stage: formData.get("stage"),
          approved_by: formData.get("approved_by") || "web-console",
          alias: formData.get("alias") || null,
        };
        await api(`/api/models/${registryId}/promote`, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        showToast(`Model ${registryId} promoted.`, "success");
        if (window.location.pathname === "/registry") {
          await renderRegistryPage();
        } else {
          await renderModelDetailPage(registryId);
        }
      },
    });
  }

  function openRejectDialog(registryId) {
    openDialog({
      kicker: "Registry",
      title: "Reject Model",
      copy: "Record why this candidate should not progress further in the registry.",
      submitLabel: "Reject",
      formHtml: `
        <form class="form-stack">
          <div class="field">
            <label for="reject-reason">Reason</label>
            <textarea id="reject-reason" name="reason">Evaluation or operational risk prevents promotion.</textarea>
          </div>
          <div class="field">
            <label for="reject-actor">Rejected By</label>
            <input id="reject-actor" name="rejected_by" value="web-console" />
          </div>
        </form>
      `,
      onSubmit: async (formData) => {
        await api(`/api/models/${registryId}/reject`, {
          method: "POST",
          body: JSON.stringify({
            reason: formData.get("reason"),
            rejected_by: formData.get("rejected_by") || "web-console",
          }),
        });
        showToast(`Model ${registryId} rejected.`, "success");
        if (window.location.pathname === "/registry") {
          await renderRegistryPage();
        } else {
          await renderModelDetailPage(registryId);
        }
      },
    });
  }

  function openRetrainDialog(registryId) {
    openDialog({
      kicker: "Workflow",
      title: "Queue Retraining",
      copy: "Create or reopen a retraining task for this registered model.",
      submitLabel: "Queue Retraining",
      formHtml: `
        <form class="form-stack">
          <div class="field">
            <label for="retrain-reason">Reason</label>
            <textarea id="retrain-reason" name="reason">Drift signal or operator request</textarea>
          </div>
        </form>
      `,
      onSubmit: async (formData) => {
        await api(`/api/models/${registryId}/retrain`, {
          method: "POST",
          body: JSON.stringify({ reason: formData.get("reason") }),
        });
        showToast(`Retraining requested for ${registryId}.`, "success");
      },
    });
  }

  async function resumeRun(runId) {
    try {
      await api(`/api/runs/${runId}/resume`, { method: "POST" });
      showToast(`Run ${runId} resumed.`, "success");
      navigate(`/runs/${runId}`);
    } catch (error) {
      showToast(messageOf(error), "error");
    }
  }

  async function cancelRun(runId) {
    try {
      await api(`/api/runs/${runId}/cancel`, { method: "POST" });
      showToast(`Stop signal sent for ${runId}.`, "success");
      if (window.location.pathname === `/runs/${runId}`) {
        await renderRunDetailPage(runId);
      } else {
        await renderRunsPage();
      }
    } catch (error) {
      showToast(messageOf(error), "error");
    }
  }

  async function toggleTask(taskId, isCompleted) {
    try {
      await api(`/api/tasks/${taskId}`, {
        method: "PATCH",
        body: JSON.stringify({ completed: !isCompleted }),
      });
      showToast("Task updated.", "success");
      await renderTasksPage();
    } catch (error) {
      showToast(messageOf(error), "error");
    }
  }

  async function deleteTask(taskId) {
    try {
      await api(`/api/tasks/${taskId}`, { method: "DELETE" });
      showToast("Task removed.", "success");
      await renderTasksPage();
    } catch (error) {
      showToast(messageOf(error), "error");
    }
  }

  function openPlaygroundForModel(registryId) {
    navigate(`/playground?model=${encodeURIComponent(registryId)}`);
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      ...options,
    });

    const raw = await response.text();
    const payload = raw ? JSON.parse(raw) : {};
    if (!response.ok) {
      throw new Error(
        payload.detail || payload.error || `Request failed with status ${response.status}`
      );
    }
    return payload;
  }

  function sparklineSvg(values) {
    if (!values.length) {
      return "";
    }
    const numericValues = values.map(Number).filter((value) => !Number.isNaN(value));
    if (!numericValues.length) {
      return "";
    }
    const width = 480;
    const height = 84;
    const min = Math.min(...numericValues);
    const max = Math.max(...numericValues);
    const spread = max - min || 1;
    const step = numericValues.length > 1 ? width / (numericValues.length - 1) : width;
    const points = numericValues
      .map((value, index) => {
        const x = index * step;
        const y = height - ((value - min) / spread) * (height - 8) - 4;
        return `${x},${y}`;
      })
      .join(" ");
    return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><path d="M ${points.replace(/ /g, " L ")}"></path></svg>`;
  }

  function emptyState(title, message) {
    return `
      <div class="empty-state">
        <p class="eyebrow">Empty</p>
        <h3>${escapeHtml(title)}</h3>
        <p>${escapeHtml(message)}</p>
      </div>
    `;
  }

  function environmentCard(environment) {
    return `
      <div class="callout">
        <p class="list-card__label">${escapeHtml(environment.name)}</p>
        <strong class="mono">${escapeHtml(environment.metadata_dir)}</strong>
        <p class="muted-copy">data ${escapeHtml(environment.data_dir)}</p>
      </div>
    `;
  }

  function countStatus(records, status, key = "status") {
    return records.filter((record) => record[key] === status).length;
  }

  function countValidated(datasets, status) {
    return datasets.filter((dataset) => dataset.validation_status === status).length;
  }

  function totalCount(counts) {
    return Object.values(counts || {}).reduce((sum, value) => sum + Number(value || 0), 0);
  }

  function formatValue(value) {
    if (typeof value === "string") {
      return escapeHtml(value);
    }
    if (typeof value === "number") {
      return escapeHtml(String(roundNumber(value)));
    }
    if (value === null || typeof value === "undefined") {
      return "n/a";
    }
    return escapeHtml(JSON.stringify(value));
  }

  function displayMetric(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "n/a";
    }
    return roundNumber(value).toFixed(4);
  }

  function displayPercent(value, scale = 1) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "n/a";
    }
    const percent = scale === 100 ? value : value * 100;
    return `${roundNumber(percent).toFixed(2)}%`;
  }

  function roundNumber(value) {
    return Math.round(Number(value) * 1000) / 1000;
  }

  function formatDate(value) {
    if (!value) {
      return "n/a";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return String(value);
    }
    return date.toLocaleString();
  }

  function timeOnly(value) {
    if (!value) {
      return "n/a";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return String(value);
    }
    return date.toLocaleTimeString();
  }

  function parseJson(source) {
    try {
      return JSON.parse(source);
    } catch (error) {
      throw new Error("Features must be valid JSON.");
    }
  }

  function parseMaybeNumber(value) {
    if (value === null || value === "") {
      return null;
    }
    const parsed = Number(value);
    return Number.isNaN(parsed) ? null : parsed;
  }

  function messageOf(error) {
    return error && error.message ? error.message : String(error);
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).replaceAll("`", "&#96;");
  }

  function showToast(message, type) {
    const existing = document.querySelector(".toast");
    if (existing) {
      existing.remove();
    }
    const toast = document.createElement("div");
    toast.className = `toast toast--${type || "success"}`;
    toast.innerHTML = `<strong>${escapeHtml(type === "error" ? "Request failed" : "Ghost updated")}</strong><p>${escapeHtml(message)}</p>`;
    document.body.appendChild(toast);
    window.setTimeout(() => toast.remove(), 3200);
  }
})();
