(function () {
  const appState = {
    payload: null,
    selectedCheckpointId: "",
    selectedEntryKey: "",
  };

  const elements = {
    meta: document.getElementById("dataset-meta"),
    dataPathInput: document.getElementById("data-path-input"),
    reloadButton: document.getElementById("reload-button"),
    checkpointFilter: document.getElementById("checkpoint-filter"),
    focusFilter: document.getElementById("focus-filter"),
    categoryFilter: document.getElementById("category-filter"),
    taskcStatusFilter: document.getElementById("taskc-status-filter"),
    searchFilter: document.getElementById("search-filter"),
    checkpointSummary: document.getElementById("checkpoint-summary"),
    entryList: document.getElementById("entry-list"),
    detailHeader: document.getElementById("detail-header"),
    detailBody: document.getElementById("detail-body"),
    jsonTemplate: document.getElementById("json-block-template"),
  };

  function init() {
    const url = new URL(window.location.href);
    const dataPath = url.searchParams.get("data") || "./task_pack.json";
    elements.dataPathInput.value = dataPath;
    elements.reloadButton.addEventListener("click", () => loadDataset(elements.dataPathInput.value.trim()));
    [
      elements.checkpointFilter,
      elements.focusFilter,
      elements.categoryFilter,
      elements.taskcStatusFilter,
      elements.searchFilter,
    ].forEach((node) => node.addEventListener("input", render));
    loadDataset(dataPath);
  }

  async function loadDataset(path) {
    elements.meta.textContent = `Loading ${path}…`;
    try {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      appState.payload = await response.json();
      appState.selectedCheckpointId = "";
      appState.selectedEntryKey = "";
      populateFilters();
      render();
    } catch (error) {
      elements.meta.textContent = `Failed to load ${path}: ${error.message}`;
      elements.checkpointSummary.innerHTML = "";
      elements.entryList.innerHTML = "";
      elements.detailBody.innerHTML = "";
    }
  }

  function populateFilters() {
    const checkpoints = getCheckpoints();
    elements.checkpointFilter.innerHTML = checkpoints
      .map((cp) => `<option value="${escapeAttr(cp.checkpoint_id)}">${escapeHtml(cp.checkpoint_id)}</option>`)
      .join("");
    const categories = Array.from(new Set(checkpoints.flatMap((cp) => checkpointEntries(cp).map((entry) => entry.category)))).sort();
    elements.categoryFilter.innerHTML = ['<option value="all">all</option>']
      .concat(categories.map((category) => `<option value="${escapeAttr(category)}">${escapeHtml(category)}</option>`))
      .join("");
  }

  function render() {
    if (!appState.payload) return;
    const checkpoints = getCheckpoints();
    if (!checkpoints.length) {
      elements.meta.textContent = "No checkpoints found in loaded JSON.";
      return;
    }

    if (!appState.selectedCheckpointId || !checkpoints.find((cp) => cp.checkpoint_id === appState.selectedCheckpointId)) {
      appState.selectedCheckpointId = checkpoints[0].checkpoint_id;
      elements.checkpointFilter.value = appState.selectedCheckpointId;
    } else {
      elements.checkpointFilter.value = appState.selectedCheckpointId;
    }

    const checkpoint = checkpoints.find((cp) => cp.checkpoint_id === elements.checkpointFilter.value) || checkpoints[0];
    appState.selectedCheckpointId = checkpoint.checkpoint_id;

    const entries = filterEntries(checkpointEntries(checkpoint));
    if (!appState.selectedEntryKey || !entries.find((entry) => entry.entryKey === appState.selectedEntryKey)) {
      appState.selectedEntryKey = entries.length ? entries[0].entryKey : "";
    }

    renderSummary(checkpoint);
    renderEntryList(entries);
    renderDetail(checkpoint, entries.find((entry) => entry.entryKey === appState.selectedEntryKey) || null);
    elements.meta.textContent = buildMetaSummary(checkpoint, entries.length);
  }

  function getCheckpoints() {
    return Array.isArray(appState.payload && appState.payload.checkpoints)
      ? appState.payload.checkpoints.filter((cp) => cp && cp.checkpoint_id)
      : [];
  }

  function checkpointEntries(checkpoint) {
    const categoryMap = buildCategoryMap(checkpoint);
    const focus = elements.focusFilter.value;
    const entries = [];
    const taskAKeys = objectEntries(checkpoint.state_completion_pack && checkpoint.state_completion_pack.keys);
    const taskBKeys = objectEntries(checkpoint.change_tracking_pack && checkpoint.change_tracking_pack.keys);
    const taskCKeys = objectEntries(checkpoint.rq3_apply_service_qa && checkpoint.rq3_apply_service_qa.keys);
    const seen = new Set();

    if (focus === "task_a" || focus === "overview") {
      taskAKeys.forEach(([stateKey, node]) => {
        entries.push(buildEntry("task_a", stateKey, node, categoryMap));
        seen.add(stateKey);
      });
    }

    if (focus === "task_b" || focus === "overview") {
      taskBKeys.forEach(([stateKey, node]) => {
        entries.push(buildEntry("task_b", stateKey, node, categoryMap));
        seen.add(stateKey);
      });
    }

    if (focus === "task_c" || focus === "overview") {
      taskCKeys.forEach(([stateKey, node]) => {
        entries.push(buildEntry("task_c", stateKey, node, categoryMap));
        seen.add(stateKey);
      });
    }

    if (focus === "overview") {
      objectEntries(checkpoint.state_questionability).forEach(([stateKey]) => {
        if (!seen.has(stateKey)) {
          entries.push(buildEntry("state_only", stateKey, null, categoryMap));
        }
      });
    }

    return entries.sort((a, b) => a.stateKey.localeCompare(b.stateKey));
  }

  function buildCategoryMap(checkpoint) {
    const map = {};
    const candidates = [
      checkpoint.state_questionability,
      checkpoint.state_completion_pack && checkpoint.state_completion_pack.keys,
      checkpoint.change_tracking_pack && checkpoint.change_tracking_pack.keys,
      checkpoint.rq3_apply_service_qa && checkpoint.rq3_apply_service_qa.keys,
    ];
    candidates.forEach((obj) => {
      objectEntries(obj).forEach(([stateKey]) => {
        map[stateKey] = stateKey.split(":")[0] || "unknown";
      });
    });
    return map;
  }

  function buildEntry(taskType, stateKey, node, categoryMap) {
    const taskCNode = taskType === "task_c" ? node : null;
    const acceptedCount = taskCNode ? ((taskCNode.items || []).length) : 0;
    const discardedCount = taskCNode ? ((taskCNode.discarded_items || []).length) : 0;
    const noItem = taskCNode ? (acceptedCount === 0 && discardedCount === 0) : false;
    return {
      entryKey: `${taskType}::${stateKey}`,
      taskType,
      stateKey,
      displayName: stateKey.split(":")[1] || stateKey,
      category: categoryMap[stateKey] || "unknown",
      node,
      acceptedCount,
      discardedCount,
      noItem,
    };
  }

  function filterEntries(entries) {
    const category = elements.categoryFilter.value;
    const search = elements.searchFilter.value.trim().toLowerCase();
    const taskcStatus = elements.taskcStatusFilter.value;
    return entries.filter((entry) => {
      if (category !== "all" && entry.category !== category) return false;
      if (search && !entry.stateKey.toLowerCase().includes(search)) return false;
      if (taskcStatus !== "all" && entry.taskType === "task_c") {
        if (taskcStatus === "accepted" && entry.acceptedCount === 0) return false;
        if (taskcStatus === "discarded" && entry.discardedCount === 0) return false;
        if (taskcStatus === "no_item" && !entry.noItem) return false;
      } else if (taskcStatus !== "all" && elements.focusFilter.value === "task_c") {
        return false;
      }
      return true;
    });
  }

  function renderSummary(checkpoint) {
    const validationSummary = checkpoint.state_validation_summary || {};
    const taskAKeys = objectEntries(checkpoint.state_completion_pack && checkpoint.state_completion_pack.keys).length;
    const taskBKeys = objectEntries(checkpoint.change_tracking_pack && checkpoint.change_tracking_pack.keys).length;
    const taskCNodes = objectEntries(checkpoint.rq3_apply_service_qa && checkpoint.rq3_apply_service_qa.keys).map(([, node]) => node || {});
    const accepted = sum(taskCNodes.map((node) => (node.items || []).length));
    const discarded = sum(taskCNodes.map((node) => (node.discarded_items || []).length));
    const noItem = taskCNodes.filter((node) => !(node.items || []).length && !(node.discarded_items || []).length).length;
    const cards = [
      summaryCard("Validated states", validationSummary.after_l1_l2_count ?? "n/a", `pre ${validationSummary.pre_validate_count ?? "n/a"}`),
      summaryCard("Task A keys", taskAKeys, "state completion"),
      summaryCard("Task B keys", taskBKeys, "change tracking"),
      summaryCard("Task C accepted", accepted, `${discarded} discarded · ${noItem} no-item`),
      summaryCard("Stage1 reuse", validationSummary.reused_count ?? "n/a", `${validationSummary.computed_count ?? "n/a"} computed`),
      summaryCard("Checkpoint", checkpoint.checkpoint_id || "n/a", ((checkpoint.as_of || {}).timestamp) || ""),
    ];
    elements.checkpointSummary.innerHTML = cards.join("");
  }

  function renderEntryList(entries) {
    if (!entries.length) {
      elements.entryList.innerHTML = '<p class="empty">No entries match current filters.</p>';
      return;
    }
    elements.entryList.innerHTML = "";
    entries.forEach((entry) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `entry-button${entry.entryKey === appState.selectedEntryKey ? " selected" : ""}`;
      button.innerHTML = `
        <span class="entry-title">${escapeHtml(entry.displayName)}</span>
        <span class="entry-key">${escapeHtml(entry.stateKey)}</span>
        <div class="badge-row">
          <span class="badge">${escapeHtml(entry.taskTypeLabel || taskTypeLabel(entry.taskType))}</span>
          <span class="badge">${escapeHtml(entry.category)}</span>
          ${taskCBadges(entry)}
        </div>
      `;
      button.addEventListener("click", () => {
        appState.selectedEntryKey = entry.entryKey;
        render();
      });
      elements.entryList.appendChild(button);
    });
  }

  function renderDetail(checkpoint, entry) {
    if (!entry) {
      elements.detailHeader.innerHTML = '<h2>No entry selected</h2><p class="subtle">Pick a state from the left.</p>';
      elements.detailBody.innerHTML = "";
      return;
    }
    const questionability = (checkpoint.state_questionability || {})[entry.stateKey] || null;
    const validatedValue = lookupStateValue(checkpoint.validated_snapshot_state, entry.stateKey);
    const observable = lookupStateValue(checkpoint.state_observability, entry.stateKey);
    elements.detailHeader.innerHTML = `
      <h2>${escapeHtml(entry.displayName)}</h2>
      <p><strong>${escapeHtml(entry.stateKey)}</strong> · ${escapeHtml(taskTypeLabel(entry.taskType))} · ${escapeHtml(checkpoint.checkpoint_id)}</p>
      <p class="subtle">${escapeHtml(entry.category)}</p>
    `;

    const blocks = [];
    blocks.push(renderStateBlock(questionability, validatedValue, observable));
    if (entry.taskType === "task_a" || entry.taskType === "overview") {
      const taskANode = (checkpoint.state_completion_pack && checkpoint.state_completion_pack.keys && checkpoint.state_completion_pack.keys[entry.stateKey]) || null;
      if (taskANode) blocks.push(renderTaskABlock(taskANode));
    }
    if (entry.taskType === "task_b" || entry.taskType === "overview") {
      const taskBNode = (checkpoint.change_tracking_pack && checkpoint.change_tracking_pack.keys && checkpoint.change_tracking_pack.keys[entry.stateKey]) || null;
      if (taskBNode) blocks.push(renderTaskBBlock(taskBNode, checkpoint.change_tracking_pack || {}));
    }
    if (entry.taskType === "task_c" || entry.taskType === "overview") {
      const taskCNode = (checkpoint.rq3_apply_service_qa && checkpoint.rq3_apply_service_qa.keys && checkpoint.rq3_apply_service_qa.keys[entry.stateKey]) || null;
      if (taskCNode) blocks.push(renderTaskCBlock(taskCNode));
    }

    elements.detailBody.innerHTML = "";
    blocks.forEach((block) => elements.detailBody.appendChild(block));
  }

  function renderStateBlock(questionability, validatedValue, observable) {
    const block = createBlock("State Context");
    block.appendChild(renderMetricsList({
      is_questionable: String(Boolean(questionability && questionability.is_questionable)),
      l1_is_questionable: String(Boolean(questionability && questionability.l1_is_questionable)),
      l2_is_questionable: String(Boolean(questionability && questionability.l2_is_questionable)),
      validated_fields: String(((questionability && questionability.validated_field_paths) || []).length),
    }));
    if (questionability) {
      block.appendChild(jsonDetails("state_questionability", questionability));
    }
    block.appendChild(jsonDetails("validated_state_value", validatedValue));
    block.appendChild(jsonDetails("state_observability", observable));
    return block;
  }

  function renderTaskABlock(node) {
    const block = createBlock("Task A Pack");
    block.appendChild(renderMetricsList({
      scoring_points: String((node.scoring_points || []).length),
      point_types: summarizePointTypes(node.scoring_points || []),
      pack_source: String(node.pack_source || "n/a"),
      item_id: String(node.item_id || "n/a"),
    }));
    block.appendChild(jsonDetails("question_text", node.question_text));
    block.appendChild(jsonDetails("answer_template", node.answer_template));
    block.appendChild(jsonDetails("retrieval_query", node.retrieval_query));
    block.appendChild(renderScoringPoints(node.scoring_points || [], "Task A scoring points"));
    block.appendChild(jsonDetails("pack_identity", node.pack_identity));
    return block;
  }

  function renderTaskBBlock(node, packMeta) {
    const block = createBlock("Task B Pack");
    block.appendChild(renderMetricsList({
      before_points: String((node.before_scoring_points || []).length),
      after_points: String((node.after_scoring_points || []).length),
      reason_points: String((node.change_reason_scoring_points || []).length),
      previous_checkpoint_id: String(packMeta.previous_checkpoint_id || "n/a"),
    }));
    block.appendChild(jsonDetails("question_text", node.question_text));
    block.appendChild(jsonDetails("retrieval_query", node.retrieval_query));
    block.appendChild(jsonDetails("before_value", node.before_value));
    block.appendChild(jsonDetails("after_value", node.after_value));
    block.appendChild(jsonDetails("reference_change_reason", node.reference_change_reason));
    block.appendChild(renderScoringPoints(node.before_scoring_points || [], "Before scoring points"));
    block.appendChild(renderScoringPoints(node.after_scoring_points || [], "After scoring points"));
    block.appendChild(renderScoringPoints(node.change_reason_scoring_points || [], "Reason scoring points"));
    block.appendChild(jsonDetails("pack_identity", node.pack_identity));
    return block;
  }

  function renderTaskCBlock(node) {
    const block = createBlock("Task C Pack");
    const accepted = node.items || [];
    const discarded = node.discarded_items || [];
    block.appendChild(renderMetricsList({
      accepted_items: String(accepted.length),
      discarded_items: String(discarded.length),
      no_item: String(accepted.length === 0 && discarded.length === 0),
      pack_source: String(node.pack_source || "n/a"),
    }));

    if (accepted.length) {
      const acceptedWrap = document.createElement("div");
      acceptedWrap.innerHTML = "<h4>Accepted items</h4>";
      accepted.forEach((item) => acceptedWrap.appendChild(renderTaskCItem(item, true)));
      block.appendChild(acceptedWrap);
    }

    if (discarded.length) {
      const discardedWrap = document.createElement("div");
      discardedWrap.innerHTML = "<h4>Discarded items</h4>";
      discarded.forEach((item) => discardedWrap.appendChild(renderTaskCItem(item, false)));
      block.appendChild(discardedWrap);
    }

    if (!accepted.length && !discarded.length) {
      const p = document.createElement("p");
      p.className = "empty";
      p.textContent = "No Task C item was saved for this key.";
      block.appendChild(p);
      block.appendChild(jsonDetails("pack_identity", node.pack_identity));
    }
    return block;
  }

  function renderTaskCItem(item, isAccepted) {
    const wrapper = document.createElement("section");
    wrapper.className = "item-card";
    const qaValidation = item.qa_validation || null;
    const atomicFactValidation = item.atomic_fact_validation || null;
    const validation = mergedTaskCValidation(item);
    wrapper.innerHTML = `
      <h4>${escapeHtml(item.qa_id || "item")} ${isAccepted ? '<span class="badge good">accepted</span>' : '<span class="badge bad">discarded</span>'}</h4>
      <div class="badge-row">
        ${failedRuleBadges(validation)}
      </div>
    `;
    wrapper.appendChild(renderMetricsList({
      qa_valid: String(qaValidation ? Boolean(qaValidation.is_valid) : Boolean(validation.is_valid)),
      qa_rewrite_attempts: String(qaValidation ? (qaValidation.rewrite_attempts ?? 0) : (validation.rewrite_attempts ?? 0)),
      atomic_facts_valid: String(atomicFactValidation ? Boolean(atomicFactValidation.is_valid) : Boolean(validation.is_valid)),
      atomic_fact_rewrite_attempts: String(atomicFactValidation ? (atomicFactValidation.rewrite_attempts ?? 0) : (validation.rewrite_attempts ?? 0)),
      used_safe_fallback: String(Boolean(atomicFactValidation && atomicFactValidation.used_safe_fallback)),
      gold_evidence_ids: String((item.gold_memory_evidence_app_log_ids || []).length),
    }));
    wrapper.appendChild(jsonDetails("service_category", item.service_category));
    wrapper.appendChild(jsonDetails("question", item.question || item.apply_question));
    wrapper.appendChild(jsonDetails("reference_answer", item.reference_answer || item.apply_reference_answer));
    wrapper.appendChild(jsonDetails("rubric", item.rubric || []));
    if (item.apply_scenario) {
      wrapper.appendChild(jsonDetails("legacy_scenario", item.apply_scenario));
    }
    wrapper.appendChild(jsonDetails("retrieval_query", item.retrieval_query));
    wrapper.appendChild(jsonDetails("gold_memory_evidence_app_log_ids", item.gold_memory_evidence_app_log_ids || []));
    wrapper.appendChild(renderTaskCValidationBlock(item));
    wrapper.appendChild(renderPairedRubrics(item.answer_scoring_points || []));
    wrapper.appendChild(jsonDetails("raw_item_json", item));
    return wrapper;
  }

  function mergedTaskCValidation(item) {
    if (item.qa_validation || item.atomic_fact_validation) {
      const qaValidation = item.qa_validation || {};
      const atomicFactValidation = item.atomic_fact_validation || {};
      const failedRules = [];
      (qaValidation.failed_rules || []).forEach((rule) => {
        if (rule && !failedRules.includes(rule)) failedRules.push(rule);
      });
      (atomicFactValidation.failed_rules || []).forEach((rule) => {
        if (rule && !failedRules.includes(rule)) failedRules.push(rule);
      });
      return {
        is_valid: Boolean(qaValidation.is_valid) && (atomicFactValidation ? Boolean(atomicFactValidation.is_valid) : true),
        rewrite_attempts: Math.max(qaValidation.rewrite_attempts || 0, atomicFactValidation.rewrite_attempts || 0),
        manual_review_required: Boolean(qaValidation.manual_review_required) || Boolean(atomicFactValidation.manual_review_required),
        failed_rules: failedRules,
        semantic_criteria: qaValidation.semantic_criteria || [],
      };
    }
    return item.validation || {};
  }

  function renderTaskCValidationBlock(item) {
    if (!(item.qa_validation || item.atomic_fact_validation)) {
      return renderValidationBlock(item.validation || {});
    }
    const block = document.createElement("div");
    block.className = "panel-block";
    const heading = document.createElement("h3");
    heading.textContent = "Validation";
    block.appendChild(heading);
    block.appendChild(renderNamedValidationSubblock("QA validation", item.qa_validation || {}, true));
    if (item.atomic_fact_validation) {
      block.appendChild(renderNamedValidationSubblock("Atomic-fact validation", item.atomic_fact_validation || {}, false));
    } else {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "No atomic-fact validation recorded.";
      block.appendChild(empty);
    }
    return block;
  }

  function renderNamedValidationSubblock(title, validation, includeCriteria) {
    const section = document.createElement("section");
    section.className = "panel-block";
    const heading = document.createElement("h4");
    heading.textContent = title;
    section.appendChild(heading);
    const split = document.createElement("div");
    split.className = "split-grid";
    const left = document.createElement("div");
    const failedRules = validation.failed_rules || [];
    left.appendChild(renderMetricsList({
      is_valid: String(Boolean(validation.is_valid)),
      rewrite_attempts: String(validation.rewrite_attempts ?? 0),
      used_safe_fallback: String(Boolean(validation.used_safe_fallback)),
      manual_review_required: String(Boolean(validation.manual_review_required)),
      failed_rules: failedRules.join(", ") || "none",
    }));
    const right = document.createElement("div");
    right.appendChild(jsonDetails("failed_rules", failedRules));
    if (!includeCriteria) {
      right.appendChild(jsonDetails("failed_point_ids", validation.failed_point_ids || []));
      right.appendChild(jsonDetails("set_failures", validation.set_failures || []));
    }
    split.appendChild(left);
    split.appendChild(right);
    section.appendChild(split);

    if (includeCriteria) {
      const criteria = validation.semantic_criteria || [];
      if (criteria.length) {
        const criteriaWrap = document.createElement("div");
        criteriaWrap.className = "criteria-list";
        criteria.forEach((criterion) => {
          const card = document.createElement("div");
          const passed = criterion.pass !== undefined ? criterion.pass : criterion.passed;
          card.className = `criterion-card ${passed ? "pass" : "fail"}`;
          card.innerHTML = `
            <h5>${escapeHtml(criterion.criterion || "criterion")} · ${passed ? "pass" : "fail"}</h5>
            <div>${escapeHtml(criterion.analysis || "")}</div>
          `;
          criteriaWrap.appendChild(card);
        });
        section.appendChild(criteriaWrap);
      }
    } else {
      section.appendChild(jsonDetails("point_results", validation.point_results || []));
    }
    return section;
  }

  function renderValidationBlock(validation) {
    const block = document.createElement("div");
    block.className = "panel-block";
    const heading = document.createElement("h3");
    heading.textContent = "Validation";
    block.appendChild(heading);

    const split = document.createElement("div");
    split.className = "split-grid";
    const left = document.createElement("div");
    const failedRules = validation.failed_rules || [];
    left.appendChild(renderMetricsList({
      failed_rules: failedRules.join(", ") || "none",
      rubric_schema_valid: formatBoolean(!failedRules.includes("rubric_invalid")),
      gold_evidence_present: formatBoolean(!failedRules.includes("missing_gold_memory_evidence")),
    }));
    const right = document.createElement("div");
    right.appendChild(jsonDetails("failed_rules", failedRules));
    split.appendChild(left);
    split.appendChild(right);
    block.appendChild(split);

    const criteria = validation.semantic_criteria || [];
    if (criteria.length) {
      const criteriaWrap = document.createElement("div");
      criteriaWrap.className = "criteria-list";
      criteria.forEach((criterion) => {
        const card = document.createElement("div");
        const passed = criterion.pass !== undefined ? criterion.pass : criterion.passed;
        card.className = `criterion-card ${passed ? "pass" : "fail"}`;
        card.innerHTML = `
          <h5>${escapeHtml(criterion.criterion || "criterion")} · ${passed ? "pass" : "fail"}</h5>
          <div>${escapeHtml(criterion.analysis || "")}</div>
        `;
        criteriaWrap.appendChild(card);
      });
      block.appendChild(criteriaWrap);
    } else {
      const p = document.createElement("p");
      p.className = "empty";
      p.textContent = "No semantic criteria recorded.";
      block.appendChild(p);
    }
    return block;
  }

  function renderPairedRubrics(points) {
    const block = document.createElement("div");
    block.className = "panel-block";
    const heading = document.createElement("h3");
    heading.textContent = "Paired Rubric Points";
    block.appendChild(heading);
    const positive = points.filter((point) => point && point.polarity === "positive");
    const negative = points.filter((point) => point && point.polarity === "negative");
    block.appendChild(renderMetricsList({
      total_points: String(points.length),
      positive_points: String(positive.length),
      negative_points: String(negative.length),
      point_types: summarizePointTypes(points),
    }));

    const grid = document.createElement("div");
    grid.className = "rubric-grid";
    grid.appendChild(renderRubricColumn("Positive points", positive, "positive"));
    grid.appendChild(renderRubricColumn("Negative points", negative, "negative"));
    block.appendChild(grid);
    return block;
  }

  function renderScoringPoints(points, title) {
    const block = document.createElement("div");
    block.className = "panel-block";
    const heading = document.createElement("h3");
    heading.textContent = title;
    block.appendChild(heading);
    block.appendChild(renderMetricsList({
      total_points: String(points.length),
      point_types: summarizePointTypes(points),
    }));
    block.appendChild(renderRubricColumn("All points", points, "positive"));
    return block;
  }

  function renderRubricColumn(title, points, kind) {
    const column = document.createElement("div");
    column.className = `rubric-column ${kind}`;
    column.innerHTML = `<h5>${escapeHtml(title)}</h5>`;
    if (!points.length) {
      const p = document.createElement("p");
      p.className = "empty";
      p.textContent = "none";
      column.appendChild(p);
      return column;
    }
    const list = document.createElement("div");
    list.className = "rubric-list";
    points.forEach((point) => {
      const item = document.createElement("div");
      item.className = "rubric-point";
      const refLine = point.reference_value !== undefined && point.reference_value !== null && String(point.reference_value) !== ""
        ? `<span class="ref">reference: ${escapeHtml(formatValueInline(point.reference_value))}</span>`
        : "";
      item.innerHTML = `
        <span class="point-id">${escapeHtml(point.point_id || "")}</span>
        <div><strong>${escapeHtml(point.point_type || "")}</strong>${point.target_path ? ` · ${escapeHtml(point.target_path)}` : ""}</div>
        <div>${escapeHtml(point.point_text || "")}</div>
        ${refLine}
      `;
      list.appendChild(item);
    });
    column.appendChild(list);
    return column;
  }

  function createBlock(title) {
    const section = document.createElement("section");
    section.className = "panel-block";
    const heading = document.createElement("h3");
    heading.textContent = title;
    section.appendChild(heading);
    return section;
  }

  function renderMetricsList(values) {
    const dl = document.createElement("dl");
    dl.className = "metrics-list";
    Object.entries(values).forEach(([key, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = key;
      const dd = document.createElement("dd");
      dd.textContent = String(value);
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
    return dl;
  }

  function jsonDetails(label, payload) {
    const node = elements.jsonTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector("summary").textContent = label;
    node.querySelector("pre").textContent = payload === undefined ? "undefined" : JSON.stringify(payload, null, 2);
    return node;
  }

  function taskCBadges(entry) {
    if (entry.taskType !== "task_c") return "";
    const badges = [];
    if (entry.acceptedCount) badges.push(`<span class="badge good">accepted ${entry.acceptedCount}</span>`);
    if (entry.discardedCount) badges.push(`<span class="badge bad">discarded ${entry.discardedCount}</span>`);
    if (entry.noItem) badges.push('<span class="badge warn">no-item</span>');
    return badges.join("");
  }

  function failedRuleBadges(validation) {
    const rules = validation.failed_rules || [];
    if (!rules.length) return '<span class="badge good">no failed rules</span>';
    return rules.map((rule) => `<span class="badge bad">${escapeHtml(rule)}</span>`).join("");
  }

  function taskTypeLabel(taskType) {
    if (taskType === "task_a") return "Task A";
    if (taskType === "task_b") return "Task B";
    if (taskType === "task_c") return "Task C";
    return "State";
  }

  function summaryCard(label, value, subtitle) {
    return `
      <article class="summary-card">
        <h3>${escapeHtml(label)}</h3>
        <div class="value">${escapeHtml(String(value))}</div>
        <div class="subtle">${escapeHtml(String(subtitle || ""))}</div>
      </article>
    `;
  }

  function lookupStateValue(snapshot, stateKey) {
    if (!snapshot || !stateKey) return null;
    const [category, rest] = stateKey.split(":");
    if (!category || !rest) return null;
    const parts = rest.split(".");
    let current = snapshot[category];
    for (const part of parts) {
      if (!current || typeof current !== "object") return null;
      current = current[part];
    }
    return current === undefined ? null : current;
  }

  function objectEntries(value) {
    return value && typeof value === "object" ? Object.entries(value) : [];
  }

  function summarizePointTypes(points) {
    const counts = {};
    points.forEach((point) => {
      const key = String((point && point.point_type) || "unknown");
      counts[key] = (counts[key] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([key, count]) => `${key}:${count}`)
      .join(", ") || "none";
  }

  function sum(values) {
    return values.reduce((acc, value) => acc + Number(value || 0), 0);
  }

  function buildMetaSummary(checkpoint, visibleCount) {
    const totalCheckpoints = getCheckpoints().length;
    const userId = String((appState.payload && appState.payload.user_id) || "user");
    return `${userId} · ${totalCheckpoints} checkpoints loaded · reviewing ${checkpoint.checkpoint_id} · ${visibleCount} visible entries`;
  }

  function formatValueInline(value) {
    if (value === null || value === undefined) return "null";
    if (typeof value === "string") return value;
    return JSON.stringify(value);
  }

  function formatBoolean(value) {
    return value === true ? "true" : value === false ? "false" : "n/a";
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).replaceAll("'", "&#39;");
  }

  init();
})();
