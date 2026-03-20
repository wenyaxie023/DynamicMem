(function () {
  const state = {
    payload: null,
    visibleStates: [],
    selectedStateKey: "",
    selectedCheckpointId: "",
  };

  const elements = {
    meta: document.getElementById("dataset-meta"),
    matrix: document.getElementById("matrix-container"),
    stateHeader: document.getElementById("state-header"),
    timelineCards: document.getElementById("timeline-cards"),
    categoryFilter: document.getElementById("category-filter"),
    taskFilter: document.getElementById("task-filter"),
    matrixViewFilter: document.getElementById("matrix-view-filter"),
    changedOnly: document.getElementById("changed-only-filter"),
    taskcOnly: document.getElementById("taskc-only-filter"),
    search: document.getElementById("search-filter"),
    dataPathInput: document.getElementById("data-path-input"),
    reloadButton: document.getElementById("reload-button"),
    jsonTemplate: document.getElementById("json-block-template"),
  };

  function init() {
    const url = new URL(window.location.href);
    const dataPath = url.searchParams.get("data") || "./state_timeline_viewer_data.json";
    elements.dataPathInput.value = dataPath;
    elements.reloadButton.addEventListener("click", () => loadDataset(elements.dataPathInput.value.trim()));
    [
      elements.categoryFilter,
      elements.taskFilter,
      elements.matrixViewFilter,
      elements.changedOnly,
      elements.taskcOnly,
      elements.search,
    ].forEach((node) => node.addEventListener("input", render));
    loadDataset(dataPath);
  }

  async function loadDataset(path) {
    elements.meta.textContent = `Loading ${path}…`;
    try {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      state.payload = await response.json();
      state.selectedStateKey = "";
      state.selectedCheckpointId = "";
      populateFilters();
      render();
    } catch (error) {
      elements.meta.textContent = `Failed to load ${path}: ${error.message}`;
      elements.matrix.innerHTML = "";
      elements.timelineCards.innerHTML = "";
    }
  }

  function populateFilters() {
    if (!state.payload) return;
    const categories = Array.from(new Set((state.payload.states || []).map((item) => item.state_category))).sort();
    elements.categoryFilter.innerHTML = ['<option value="all">all</option>']
      .concat(categories.map((value) => `<option value="${escapeAttr(value)}">${escapeHtml(value)}</option>`))
      .join("");
  }

  function render() {
    if (!state.payload) return;
    const checkpointIds = state.payload.meta.checkpoint_ids || [];
    const states = filterStates(state.payload.states || []);
    state.visibleStates = states;
    if (!state.selectedStateKey && states.length) {
      state.selectedStateKey = states[0].state_key;
    }
    if (state.selectedStateKey && !states.find((item) => item.state_key === state.selectedStateKey)) {
      state.selectedStateKey = states.length ? states[0].state_key : "";
    }
    if (!state.selectedCheckpointId && checkpointIds.length) {
      state.selectedCheckpointId = checkpointIds[0];
    }
    if (state.selectedCheckpointId && !checkpointIds.includes(state.selectedCheckpointId)) {
      state.selectedCheckpointId = checkpointIds.length ? checkpointIds[0] : "";
    }
    renderMatrix(states, checkpointIds);
    renderDetails(state.selectedStateKey, checkpointIds);
    elements.meta.textContent = buildMetaSummary(states.length, checkpointIds.length);
  }

  function filterStates(states) {
    const category = elements.categoryFilter.value;
    const task = elements.taskFilter.value;
    const changedOnly = elements.changedOnly.checked;
    const taskcOnly = elements.taskcOnly.checked;
    const search = elements.search.value.trim().toLowerCase();
    return states.filter((item) => {
      if (category !== "all" && item.state_category !== category) return false;
      if (task !== "all" && !(item.tasks_present || []).includes(task)) return false;
      if (changedOnly && !(item.changed_checkpoints || []).length) return false;
      if (taskcOnly && !(item.tasks_present || []).includes("C")) return false;
      if (search && !item.state_key.toLowerCase().includes(search)) return false;
      return true;
    });
  }

  function renderMatrix(states, checkpointIds) {
    if (!states.length) {
      elements.matrix.innerHTML = '<p class="empty">No states match current filters.</p>';
      return;
    }
    const matrixMode = elements.matrixViewFilter.value || "task_a";
    const table = document.createElement("table");
    table.className = "matrix-table";
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    headRow.innerHTML = `<th class="state-label">State</th>${checkpointIds
      .map((id) => `<th>${escapeHtml(shortCheckpointLabel(id))}</th>`)
      .join("")}`;
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    states.forEach((item) => {
      const row = document.createElement("tr");
      const labelCell = document.createElement("td");
      labelCell.className = "state-label";
      const rowButton = document.createElement("button");
      rowButton.className = "state-row-button";
      rowButton.innerHTML = `${escapeHtml(item.display_name)}<span class="minor">${escapeHtml(item.state_key)}</span>`;
      rowButton.addEventListener("click", () => {
        state.selectedStateKey = item.state_key;
        if (!state.selectedCheckpointId && checkpointIds.length) {
          state.selectedCheckpointId = checkpointIds[0];
        }
        render();
      });
      labelCell.appendChild(rowButton);
      row.appendChild(labelCell);

      checkpointIds.forEach((checkpointId) => {
        const cell = document.createElement("td");
        const cellData = (((state.payload.timeline || {})[item.state_key] || {})[checkpointId]) || {};
        const button = document.createElement("button");
        const matrixScore = matrixScoreValue(cellData, matrixMode);
        button.className = buildCellClassNames(
          cellData,
          item.state_key === state.selectedStateKey && checkpointId === state.selectedCheckpointId,
          matrixScore
        );
        const statusText = cellStatusText(cellData, matrixMode);
        const scoreText = statusText === "not present"
          ? ""
          : (matrixScore === null ? "n/a" : matrixScore.toFixed(2));
        const evidenceText = matrixEvidenceText(cellData, matrixMode);
        button.innerHTML = `
          <span class="cell-score">${escapeHtml(scoreText)}</span>
          <span class="cell-status">${escapeHtml(statusText)}</span>
          ${evidenceText ? `<span class="cell-evidence">${escapeHtml(evidenceText)}</span>` : ""}
        `;
        button.addEventListener("click", () => {
          state.selectedStateKey = item.state_key;
          state.selectedCheckpointId = checkpointId;
          render();
          scrollToSelectedCheckpoint();
        });
        cell.appendChild(button);
        row.appendChild(cell);
      });
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    elements.matrix.innerHTML = "";
    elements.matrix.appendChild(table);
  }

  function renderDetails(stateKey, checkpointIds) {
    if (!stateKey) {
      elements.stateHeader.innerHTML = '<h2>Select a state</h2><p class="subtle">Click a row or cell on the left.</p>';
      elements.timelineCards.innerHTML = "";
      return;
    }
    const stateMeta = (state.payload.states || []).find((item) => item.state_key === stateKey);
    const stateTimeline = ((state.payload.timeline || {})[stateKey]) || {};
    elements.stateHeader.innerHTML = `
      <h2>${escapeHtml(stateMeta.display_name)}</h2>
      <p><strong>${escapeHtml(stateMeta.state_key)}</strong> · ${escapeHtml(stateMeta.state_category)}</p>
      <p class="subtle">Appears: ${escapeHtml((stateMeta.appears_in_checkpoints || []).join(", ") || "none")}<br/>Changed: ${escapeHtml((stateMeta.changed_checkpoints || []).join(", ") || "none")}</p>
    `;
    elements.timelineCards.innerHTML = "";
    checkpointIds.forEach((checkpointId) => {
      const cell = stateTimeline[checkpointId] || {};
      const card = document.createElement("article");
      card.className = "checkpoint-card";
      card.dataset.checkpointId = checkpointId;
      if (checkpointId === state.selectedCheckpointId) {
        card.classList.add("selected-checkpoint");
      }
      const taskCCount = (cell.task_c_items || []).length;
      card.innerHTML = `
        <header>
          <div>
            <h3>${escapeHtml(checkpointId)}</h3>
            <div class="subtle">${escapeHtml(cell.exists_in_validated ? "validated view" : "not validated / not present")}</div>
            <div class="subtle">${escapeHtml(transitionSummary(cell))}</div>
          </div>
          <div class="checkpoint-flags">
            ${cell.is_changed ? '<span class="flag changed">changed</span>' : ""}
            ${taskCCount ? `<span class="flag taskc">Task C x ${taskCCount}</span>` : ""}
          </div>
        </header>
      `;
      const grid = document.createElement("div");
      grid.className = "section-grid";

      grid.appendChild(renderStateSnapshotsBlock(cell));
      grid.appendChild(renderTaskABlock(cell));
      grid.appendChild(renderTaskBBlock(cell));
      grid.appendChild(renderTaskCBlock(cell));

      card.appendChild(grid);
      elements.timelineCards.appendChild(card);
    });
    scrollToSelectedCheckpoint();
  }

  function scrollToSelectedCheckpoint() {
    if (!state.selectedCheckpointId) return;
    const selector = `.checkpoint-card[data-checkpoint-id="${cssEscape(state.selectedCheckpointId)}"]`;
    const card = elements.timelineCards.querySelector(selector);
    if (!card) return;
    requestAnimationFrame(() => {
      card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }

  function renderStateSnapshotsBlock(cell) {
    const block = createBlock("State snapshots");
    block.appendChild(renderMetricsList({
      "presence_status": cell.presence_status || "n/a",
      "transition_status": ((cell.transition_from_previous || {}).status) || "n/a",
    }));
    if (cell.transition_from_previous) {
      block.appendChild(jsonDetails("transition_from_previous", cell.transition_from_previous));
    }
    if (cell.state_questionability) {
      block.appendChild(jsonDetails("state_questionability", cell.state_questionability));
    }
    block.appendChild(jsonDetails("expected", cell.expected_state_value));
    block.appendChild(jsonDetails("validated", cell.validated_state_value));
    block.appendChild(jsonDetails("predicted Task A", cell.predicted_task_a_value));
    return block;
  }

  function renderTaskABlock(cell) {
    const block = createBlock("Task A");
    const taskA = (cell.model_inputs || {}).task_a || null;
    const evalPayload = cell.task_a_eval || {};
    const slotEval = evalPayload.slot_eval || {};
    block.appendChild(renderMetricsList({
      "point_score": formatNumber(taskAScoreValue(evalPayload)),
      "value_f1": formatNumber(evalPayload.value_f1),
      "exact_match": String(Boolean(evalPayload.exact_match)),
      "point_judgment_count": String((slotEval.judgments || []).length || 0),
      "evidence_f1": formatNumber((((evalPayload.evidence_id_metrics || {}).f1) || 0)),
      "evidence_recall": formatNumber((((evalPayload.evidence_id_metrics || {}).recall) || 0)),
      "evidence_precision": formatNumber((((evalPayload.evidence_id_metrics || {}).precision) || 0)),
      "evidence_ids_nonempty_rate": formatNumber((((evalPayload.evidence_structure || {}).app_log_id_nonempty_rate) || 0)),
    }));
    if (taskA) {
      block.appendChild(jsonDetails("retrieval_query", taskA.retrieval_query));
      block.appendChild(collapsibleNode(`retrieved user memory (${(taskA.user_memory_logs || []).length} logs)`, renderLogList(taskA.user_memory_logs || [])));
      block.appendChild(jsonDetails("prompt", taskA.prompt));
      block.appendChild(jsonDetails("raw_model_output", taskA.raw_model_output));
    } else {
      block.appendChild(emptyNode("not applicable"));
    }
    block.appendChild(
      renderSlotJudgmentSection("Scoring points", slotEval.slot_context || [], slotEval.judgments || [], {
        emptyText: "no scoring points",
      })
    );
    block.appendChild(jsonDetails("evaluation", evalPayload));
    return block;
  }

  function renderTaskBBlock(cell) {
    const block = createBlock("Task B");
    const payload = cell.task_b_payload || {};
    const slotEval = ((payload || {}).eval || {}).slot_eval || {};
    if (!payload.applicable) {
      block.appendChild(emptyNode("not applicable"));
      return block;
    }
    block.appendChild(renderMetricsList({
      "state_predict_score": formatNumber(taskBScoreValue(payload)),
      "reason_score": formatNumber((slotEval.change_reason || {}).score_0_1),
      "before_point_score": formatNumber((slotEval.before || {}).score_0_1),
      "after_point_score": formatNumber((slotEval.after || {}).score_0_1),
      "before_after_f1": formatNumber((((payload.eval || {}).before_after_f1) || 0)),
      "before_after_exact": String(Boolean(((payload.eval || {}).before_after_exact))),
      "evidence_f1": formatNumber(((((payload.eval || {}).evidence_id_metrics || {}).f1) || 0)),
    }));
    block.appendChild(jsonDetails("predicted before/after/reason", {
      before: payload.before,
      after: payload.after,
      change_reason: payload.change_reason,
      evidence: payload.evidence,
    }));
    block.appendChild(jsonDetails("expected before/after/reason", {
      before: payload.expected_before,
      after: payload.expected_after,
      change_reason: payload.expected_change_reason,
    }));
    if (payload.model_inputs) {
      if ((payload.model_inputs.user_memory_logs || []).length) {
        block.appendChild(
          collapsibleNode(
            `retrieved user memory (${(payload.model_inputs.user_memory_logs || []).length} logs)`,
            renderLogList(payload.model_inputs.user_memory_logs)
          )
        );
      }
      block.appendChild(jsonDetails("prompt", payload.model_inputs.prompt));
      block.appendChild(jsonDetails("raw_model_output", payload.model_inputs.raw_model_output));
    }
    block.appendChild(
      renderGroupedSlotJudgmentSections([
        { label: "Before scoring points", slotContext: (slotEval.before || {}).slot_context || [], judgments: (slotEval.before || {}).judgments || [] },
        { label: "After scoring points", slotContext: (slotEval.after || {}).slot_context || [], judgments: (slotEval.after || {}).judgments || [] },
        { label: "Change-reason scoring points", slotContext: (slotEval.change_reason || {}).slot_context || [], judgments: (slotEval.change_reason || {}).judgments || [] },
      ])
    );
    block.appendChild(jsonDetails("evaluation", payload.eval));
    return block;
  }

  function renderTaskCBlock(cell) {
    const block = createBlock("Task C");
    const items = cell.task_c_items || [];
    if (!items.length) {
      block.appendChild(emptyNode("not applicable"));
      return block;
    }
    items.forEach((item) => {
      const wrapper = document.createElement("div");
      wrapper.className = "qa-item";
      const evidenceMetrics = ((item.eval || {}).evidence_id_metrics) || {};
      const slotEval = ((item.eval || {}).slot_eval) || {};
      const judgments = slotEval.judgments || [];
      wrapper.appendChild(renderMetricsList({
        "qa_id": item.qa_id,
        "answer_point_score": formatNumber(slotEval.score_0_1),
        "point_judgment_count": String(judgments.length || 0),
        "evidence_f1": formatNumber(evidenceMetrics.f1 || 0),
        "evidence_recall": formatNumber(evidenceMetrics.recall || 0),
        "evidence_precision": formatNumber(evidenceMetrics.precision || 0),
      }));
      wrapper.appendChild(renderTaskCGroundTruth(item));
      wrapper.appendChild(renderTaskCPrediction(item));
      wrapper.appendChild(
        renderSlotJudgmentSection("Scoring points", slotEval.slot_context || [], judgments, {
          emptyText: "no scoring points",
        })
      );
      if (item.model_inputs) {
        wrapper.appendChild(jsonDetails("retrieval_query", item.model_inputs.retrieval_query));
        wrapper.appendChild(
          collapsibleNode(
            `retrieved user memory (${(item.model_inputs.user_memory_logs || []).length} logs)`,
            renderLogList(item.model_inputs.user_memory_logs || [])
          )
        );
        wrapper.appendChild(jsonDetails("raw_model_output", item.model_inputs.raw_model_output, { open: true }));
        wrapper.appendChild(jsonDetails("prompt", item.model_inputs.prompt));
      }
      wrapper.appendChild(jsonDetails("evaluation", item.eval));
      block.appendChild(wrapper);
    });
    return block;
  }

  function createBlock(title) {
    const block = document.createElement("section");
    block.className = "panel-block";
    const heading = document.createElement("h4");
    heading.textContent = title;
    block.appendChild(heading);
    return block;
  }

  function renderMetricsList(entries) {
    const dl = document.createElement("dl");
    dl.className = "metrics-list";
    Object.entries(entries).forEach(([key, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = key;
      const dd = document.createElement("dd");
      dd.textContent = value;
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
    return dl;
  }

  function renderTaskCGroundTruth(item) {
    const section = document.createElement("section");
    section.className = "judgment-section";
    const title = document.createElement("h5");
    title.textContent = "Ground truth";
    section.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "judgment-card";
    meta.innerHTML = `
      <div class="judgment-text"><strong>service_category</strong>: ${escapeHtml(item.service_category || "n/a")}</div>
      <div class="judgment-text"><strong>scenario</strong>: ${escapeHtml(item.scenario || "")}</div>
      <div class="judgment-text"><strong>question</strong>: ${escapeHtml(item.question || "")}</div>
      <div class="judgment-text"><strong>reference_answer</strong>: ${escapeHtml(item.reference_answer || "")}</div>
      <div class="judgment-text"><strong>gold_memory_evidence_app_log_ids</strong>: ${escapeHtml(JSON.stringify(item.gold_memory_evidence_app_log_ids || []))}</div>
    `;
    section.appendChild(meta);
    return section;
  }

  function renderTaskCPrediction(item) {
    const section = document.createElement("section");
    section.className = "judgment-section";
    const title = document.createElement("h5");
    title.textContent = "Model prediction";
    section.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "judgment-card";
    meta.innerHTML = `
      <div class="judgment-text"><strong>predicted_answer</strong>: ${escapeHtml(item.predicted_answer || "")}</div>
      <div class="judgment-text"><strong>evidence</strong>: ${escapeHtml(JSON.stringify(item.evidence || []))}</div>
    `;
    section.appendChild(meta);
    return section;
  }

  function renderGroupedSlotJudgmentSections(groups) {
    const wrapper = document.createElement("div");
    wrapper.className = "judgment-groups";
    let hasAny = false;
    groups.forEach((group) => {
      const judgments = group.judgments || [];
      const slotContext = group.slotContext || [];
      if (!judgments.length && !slotContext.length) return;
      hasAny = true;
      wrapper.appendChild(
        renderSlotJudgmentSection(group.label, slotContext, judgments, {
          emptyText: "no scoring points",
        })
      );
    });
    if (!hasAny) return emptyNode("no scoring points");
    return wrapper;
  }

  function renderSlotJudgmentSection(titleText, slotContext, judgments, options = {}) {
    const section = document.createElement("section");
    section.className = "judgment-section";
    const title = document.createElement("h5");
    title.textContent = titleText;
    section.appendChild(title);

    let effectiveSlotContext = Array.isArray(slotContext) ? slotContext : [];
    let effectiveJudgments = Array.isArray(judgments) ? judgments : [];
    if (!effectiveSlotContext.length && effectiveJudgments.length) {
      effectiveSlotContext = effectiveJudgments.map((judgment) => {
        const slot = { ...judgment };
        delete slot.analysis;
        delete slot.correct;
        return slot;
      });
      effectiveJudgments = effectiveJudgments.map((judgment) => ({
        point_id: judgment.point_id,
        analysis: judgment.analysis,
        correct: judgment.correct,
      }));
    }

    if (!effectiveSlotContext.length && !effectiveJudgments.length) {
      section.appendChild(emptyNode(options.emptyText || "no scoring points"));
      return section;
    }

    const list = document.createElement("div");
    list.className = "judgment-list";
    const judgmentById = new Map(
      effectiveJudgments.map((judgment) => [String((judgment || {}).point_id || ""), judgment || {}])
    );
    effectiveSlotContext.forEach((slot) => {
      const card = document.createElement("div");
      const pointId = slot && slot.point_id ? String(slot.point_id) : "unknown_point";
      const judgment = judgmentById.get(pointId) || {};
      const correct = Boolean(judgment && judgment.correct);
      card.className = `judgment-card ${correct ? "correct" : "incorrect"}`;
      const pointType = slot && slot.point_type ? String(slot.point_type) : "slot";
      const referenceValue = slot && slot.reference_value !== undefined
        ? JSON.stringify(slot.reference_value)
        : "n/a";
      const predictedValue = slot && slot.predicted_value !== undefined
        ? JSON.stringify(slot.predicted_value)
        : "n/a";
      const analysis = judgment && judgment.analysis ? String(judgment.analysis) : "";
      const pointText = slot && slot.point_text ? String(slot.point_text) : "";
      const slotLine = pointText
        ? `<div class="judgment-text"><strong>slot</strong>: ${escapeHtml(pointText)}</div>`
        : "";
      const groundTruthLine = (pointType === "field" || pointType === "list_item") && slot && slot.reference_value !== undefined
        ? `<div class="judgment-text"><strong>ground truth</strong>: ${escapeHtml(referenceValue)}</div>`
        : "";
      card.innerHTML = `
        <div class="judgment-head">
          <span class="judgment-id">${escapeHtml(pointId)}</span>
        </div>
        <div class="judgment-meta">${escapeHtml(pointType)}</div>
        ${slotLine}
        ${groundTruthLine}
        <div class="judgment-text"><strong>model prediction</strong>: ${escapeHtml(predictedValue)}</div>
        <div class="judgment-text"><strong>analysis</strong>: ${escapeHtml(analysis)}</div>
        <div class="judgment-text"><strong>verdict</strong>: <span class="judgment-badge ${correct ? "correct" : "incorrect"}">${correct ? "correct" : "incorrect"}</span></div>
      `;
      list.appendChild(card);
    });
    section.appendChild(list);
    return section;
  }

  function renderLogList(logs) {
    if (!logs.length) return emptyNode("no retrieved logs");
    const wrapper = document.createElement("div");
    wrapper.className = "log-list";
    logs.forEach((log) => {
      const card = document.createElement("div");
      card.className = "log-card";
      card.innerHTML = `
        <h5>${escapeHtml(log.app_log_id || "")}</h5>
        <div class="log-meta">${escapeHtml(log.timestamp || "")} · ${escapeHtml(log.app_name || "")}${log.api_name ? " · " + escapeHtml(log.api_name) : ""}</div>
        <p>${escapeHtml(log.summary_text || "")}</p>
      `;
      card.appendChild(jsonDetails("request", log.request));
      card.appendChild(jsonDetails("response", log.response));
      wrapper.appendChild(card);
    });
    return wrapper;
  }

  function collapsibleNode(label, innerNode) {
    const details = document.createElement("details");
    details.className = "json-block memory-block";
    const summary = document.createElement("summary");
    summary.textContent = label;
    details.appendChild(summary);
    details.appendChild(innerNode);
    return details;
  }

  function jsonDetails(label, payload, options = {}) {
    const node = elements.jsonTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector("summary").textContent = label;
    node.querySelector("pre").textContent = payload === undefined ? "undefined" : JSON.stringify(payload, null, 2);
    if (options.open) node.open = true;
    return node;
  }

  function emptyNode(text) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = text;
    return p;
  }

  function taskAScoreValue(evalPayload) {
    const slotEval = evalPayload && evalPayload.slot_eval;
    if (slotEval && typeof slotEval.score_0_1 === "number") return slotEval.score_0_1;
    return null;
  }

  function taskAEvidenceMetrics(evalPayload) {
    const metrics = (evalPayload && evalPayload.evidence_id_metrics) || {};
    return {
      f1: typeof metrics.f1 === "number" ? metrics.f1 : null,
      recall: typeof metrics.recall === "number" ? metrics.recall : null,
      precision: typeof metrics.precision === "number" ? metrics.precision : null,
    };
  }

  function taskCEvidenceMetrics(items) {
    if (!Array.isArray(items) || !items.length) {
      return { f1: null, recall: null, precision: null };
    }
    const meanMetric = (field) => {
      const values = items
        .map((item) => Number((((item || {}).eval || {}).evidence_id_metrics || {})[field]))
        .filter((value) => Number.isFinite(value));
      if (!values.length) return null;
      const total = values.reduce((sum, value) => sum + value, 0);
      return total / values.length;
    };
    return {
      f1: meanMetric("f1"),
      recall: meanMetric("recall"),
      precision: meanMetric("precision"),
    };
  }

  function matrixEvidenceText(cell, matrixMode) {
    let evidence = null;
    if (matrixMode === "task_a") {
      const evalPayload = cell && cell.task_a_eval;
      if (!evalPayload) return "";
      evidence = taskAEvidenceMetrics(evalPayload);
    } else if (matrixMode === "task_c") {
      evidence = taskCEvidenceMetrics((cell && cell.task_c_items) || []);
    } else {
      return "";
    }
    const hasAny = [evidence.f1, evidence.recall].some((value) => typeof value === "number");
    if (!hasAny) return "";
    return `ev ${shortNumber(evidence.f1)} · r ${shortNumber(evidence.recall)}`;
  }

  function taskBScoreValue(payload) {
    if (!payload || !payload.applicable) return null;
    const evalPayload = payload && payload.eval;
    const slotEval = evalPayload && evalPayload.slot_eval;
    if (slotEval && slotEval.state_predict && typeof slotEval.state_predict.score_0_1 === "number") {
      return slotEval.state_predict.score_0_1;
    }
    return null;
  }

  function taskCScoreValue(items) {
    if (!Array.isArray(items) || !items.length) return null;
    const scored = items
      .map((item) => Number((((item.eval || {}).slot_eval) || {}).score_0_1))
      .filter((value) => Number.isFinite(value));
    if (!scored.length) return null;
    const total = scored.reduce((sum, value) => sum + value, 0);
    return total / scored.length;
  }

  function matrixScoreValue(cell, matrixMode) {
    if (!cell.exists_in_expected && !cell.exists_in_validated) return null;
    if (matrixMode === "task_c") return taskCScoreValue(cell.task_c_items);
    if (matrixMode === "task_b") return taskBScoreValue(cell.task_b_payload);
    return taskAScoreValue(cell.task_a_eval);
  }

  function cellStatusText(cell, matrixMode) {
    if (!cell.exists_in_expected && !cell.exists_in_validated) return "not present";
    if (cell.exists_in_expected && !cell.exists_in_validated) return "filtered";
    if (matrixMode === "task_c") {
      const items = cell.task_c_items || [];
      if (!items.length) return "not applicable";
      const score = taskCScoreValue(items);
      if (score === null) return "unscored";
      if (score >= 0.9) return "strong";
      if (score >= 0.5) return "partial";
      return "weak";
    }
    if (matrixMode === "task_b") {
      if ((cell.task_b_payload || {}).applicable) return "changed";
      return "not changed";
    }
    if (cell.exists_in_validated && cell.predicted_task_a_value === undefined) return "validated only";
    if (cell.is_changed) return "changed";
    return "present";
  }

  function buildCellClassNames(cell, selected, score) {
    const classes = ["cell-button"];
    if (selected) classes.push("selected");
    if (score === null) classes.push("validated");
    if (cell.is_changed) classes.push("changed");
    if ((cell.task_c_items || []).length) classes.push("has-taskc");
    const taskAEval = cell.task_a_eval || {};
    const taskAEvidence = taskAEvidenceMetrics(taskAEval);
    if (typeof score === "number" && typeof taskAEvidence.f1 === "number" && score >= 0.15 && taskAEvidence.f1 <= 0.05) {
      classes.push("evidence-gap");
    }
    if (score !== null) {
      if (score < 0.5) classes.push("score-low");
      else if (score < 0.9) classes.push("score-mid");
      else classes.push("score-high");
    }
    return classes.join(" ");
  }

  function buildMetaSummary(stateCount, checkpointCount) {
    const meta = state.payload.meta || {};
    return `${meta.user_id || "user"} · ${checkpointCount} checkpoints · ${stateCount} visible states · ${meta.eval_protocol_version || "eval"}`;
  }

  function shortCheckpointLabel(checkpointId) {
    return checkpointId.replace("cal_quarterly_", "cp");
  }

  function transitionSummary(cell) {
    const transition = cell.transition_from_previous || {};
    if (transition.summary) return transition.summary;
    return cell.presence_status || "no transition info";
  }

  function formatNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "n/a";
  }

  function shortNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "n/a";
  }

  function formatRubric(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(1) : "n/a";
  }

  function escapeHtml(value) {
    return String(value)
      .split("&").join("&amp;")
      .split("<").join("&lt;")
      .split(">").join("&gt;")
      .split('"').join("&quot;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).split("'").join("&#39;");
  }

  function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === "function") {
      return window.CSS.escape(String(value));
    }
    return String(value).split('"').join('\\"');
  }

  init();
})();
