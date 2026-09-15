const output = document.querySelector("#output");
const outputStatus = document.querySelector("#outputStatus");
const predictionSummary = document.querySelector("#predictionSummary");
const skuSummary = document.querySelector("#skuSummary");
const cameraPreview = document.querySelector("#cameraPreview");
const cameraStatus = document.querySelector("#cameraStatus");
const cameraPill = document.querySelector("#cameraPill");
const capturePredictStatus = document.querySelector("#capturePredictStatus");

let lastLookup = null;
let predictionItems = [];
let predictionJobId = null;
let requestInProgress = false;
let predictionCapturedViews = [];
let predictionReady = false;
let predictionCompleted = false;

function field(id) {
  return document.querySelector(`#${id}`);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch (error) {
    data = {ok: false, error: "Response was not valid JSON.", details: text};
  }
  if (!response.ok) {
    throw {
      ok: false,
      http_status: response.status,
      error: data.error || "Request failed.",
      stage: data.stage,
      details: data.details,
      response: data
    };
  }
  return data;
}

function showStatus(data, label = "Response received") {
  outputStatus.textContent = label;
  output.classList.remove("error");
  output.textContent = JSON.stringify(data, null, 2);
  renderPredictionSummary(data, false);
}

function showError(error, label = "Error") {
  outputStatus.textContent = label;
  output.classList.add("error");
  output.textContent = JSON.stringify(error, null, 2);
  renderPredictionSummary(error, true);
}

function resultValue(value, fallback = "-") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(", ") : "none";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(2) : fallback;
  }
  return String(value);
}

function addSummaryRow(container, label, value, badgeClass = "") {
  const row = document.createElement("div");
  row.className = "result-row";

  const key = document.createElement("span");
  key.className = "result-key";
  key.textContent = label;
  row.appendChild(key);

  const val = document.createElement("span");
  val.className = badgeClass ? `result-value result-badge ${badgeClass}` : "result-value";
  val.textContent = resultValue(value);
  row.appendChild(val);
  container.appendChild(row);
}

function boxLabel(box) {
  if (!box || typeof box !== "object") {
    return null;
  }
  return [
    resultValue(box.length_in),
    resultValue(box.width_in),
    resultValue(box.height_in)
  ].join(" x ");
}

function metadataFromLookupResponse(data) {
  if (!data || typeof data !== "object") {
    return null;
  }
  if (data.item && data.item.metadata) {
    return data.item.metadata;
  }
  return data.metadata || data.item || null;
}

function renderSkuSummary(data, isError = false) {
  skuSummary.textContent = "";
  const grid = document.createElement("div");
  grid.className = "result-grid";
  if (isError) {
    addSummaryRow(grid, "SKU status", data.error || "Lookup failed", "fail");
    skuSummary.appendChild(grid);
    return;
  }
  const metadata = metadataFromLookupResponse(data);
  const found = data && (data.status === "found" || data.known);
  addSummaryRow(grid, "SKU status", found ? "found" : "not found", found ? "pass" : "warn");
  addSummaryRow(grid, "SKU", data && data.sku);
  addSummaryRow(grid, "sku_category", metadata && metadata.category);
  addSummaryRow(grid, "sku_common_name", metadata && metadata.common_name);
  addSummaryRow(grid, "sku_spec", metadata && metadata.spec);
  skuSummary.appendChild(grid);
}

function itemPayloadFromSku() {
  const sku = field("sku-input").value.trim();
  const quantity = Number(field("quantity-input").value || "1");
  if (!Number.isInteger(quantity) || quantity <= 0) {
    throw {error: "Quantity must be a positive integer."};
  }
  if (!sku) {
    return [{quantity}];
  }
  return [{sku, quantity}];
}

function normalizeSku(sku) {
  return String(sku || "").trim().toUpperCase();
}

function totalQuantity() {
  return predictionItems.reduce((total, item) => total + item.quantity, 0);
}

function renderItems() {
  const body = field("item-list");
  body.textContent = "";
  predictionItems.forEach((item, index) => {
    const row = document.createElement("tr");
    [item.sku, item.quantity].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    });
    const action = document.createElement("td");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger outline";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      predictionItems.splice(index, 1);
      resetPredictionJob();
      renderItems();
    });
    action.appendChild(remove);
    row.appendChild(action);
    body.appendChild(row);
  });
  field("quantity-summary").textContent = `Total quantity: ${totalQuantity()}`;
}

function addItem() {
  try {
    const [next] = itemPayloadFromSku();
    if (!next.sku) {
      throw {error: "SKU is required for group items."};
    }
    const key = normalizeSku(next.sku);
    const existing = predictionItems.find((item) => normalizeSku(item.sku) === key);
    if (existing) {
      existing.quantity += next.quantity;
    } else {
      predictionItems.push({sku: next.sku, quantity: next.quantity});
    }
    field("sku-input").value = "";
    field("quantity-input").value = "1";
    resetPredictionJob();
    renderItems();
  } catch (error) {
    showError(error, "Item error");
  }
}

function resetPredictionJob() {
  predictionJobId = null;
  predictionCapturedViews = [];
  predictionReady = false;
  predictionCompleted = false;
  field("capture-view-01").hidden = false;
  field("capture-view-01").disabled = false;
  field("capture-view-02").hidden = true;
  field("capture-view-02").disabled = false;
  field("start-new-prediction").hidden = true;
  field("start-new-prediction").disabled = false;
  predictionSummary.textContent = "";
  const emptyMessage = document.createElement("p");
  emptyMessage.className = "status-text";
  emptyMessage.textContent = "No prediction yet.";
  predictionSummary.appendChild(emptyMessage);
  output.textContent = "{}";
  output.classList.remove("error");
  outputStatus.textContent = "Waiting for action";
  capturePredictStatus.textContent = "Ready to capture View 01.";
}

function activeItems() {
  if (field("job-type").value === "group") {
    if (!predictionItems.length) {
      throw {error: "Add at least one group item before capture."};
    }
    return predictionItems.map((item) => ({sku: item.sku, quantity: item.quantity}));
  }
  return itemPayloadFromSku().map((item) => ({...item, quantity: 1}));
}

function renderMode() {
  const group = field("job-type").value === "group";
  field("shape-field").hidden = group;
  field("arrangement-field").hidden = !group;
  field("item-table-wrap").hidden = !group;
  field("add-item").hidden = !group;
  field("quantity-input").disabled = !group;
  if (!group) {
    field("quantity-input").value = "1";
  }
  resetPredictionJob();
}

function renderPredictionSummary(data, isError) {
  predictionSummary.textContent = "";
  const grid = document.createElement("div");
  grid.className = "result-grid";

  if (isError) {
    addSummaryRow(grid, "Status", data.error || "Prediction failed", "fail");
    if (data.stage) {
      addSummaryRow(grid, "Stage", data.stage, "warn");
    }
    predictionSummary.appendChild(grid);
    return;
  }

  if (data.ready_for_prediction === false) {
    addSummaryRow(grid, "Status", "view captured", "pass");
    addSummaryRow(grid, "Captured views", data.captured_views);
    addSummaryRow(grid, "Remaining views", data.remaining_views, "warn");
    addSummaryRow(grid, "Next action", data.operator_instruction);
    predictionSummary.appendChild(grid);
    return;
  }

  if (!data.model_available) {
    addSummaryRow(grid, "Status", "model missing", "warn");
    addSummaryRow(grid, "Warning", data.warnings);
    addSummaryRow(grid, "Model path", data.model && data.model.path);
    predictionSummary.appendChild(grid);
    return;
  }

  const prediction = data.prediction || {};
  const dimensions = prediction.dimensions || {};
  addSummaryRow(grid, "Status", "predicted", "pass");
  if (data.prediction_job_id) {
    addSummaryRow(grid, "Prediction job", data.prediction_job_id);
  }
  const quality = data.quality_summary || (data.capture && data.capture.quality_summary);
  addSummaryRow(grid, "Geometry quality", quality && (quality.status_label || quality.decision));
  addSummaryRow(grid, "Raw AI2 dimensions", boxLabel(prediction.predicted_box));
  addSummaryRow(grid, "Confidence", prediction.confidence);
  addSummaryRow(grid, "Quality", prediction.quality);
  addSummaryRow(grid, "Object dims", boxLabel({
    length_in: dimensions.object_length_in,
    width_in: dimensions.object_width_in,
    height_in: dimensions.object_height_in
  }));
  addSummaryRow(grid, "Warnings", prediction.warnings);

  const gt = data.ground_truth || {};
  addSummaryRow(grid, "GT present", gt.present, gt.present ? "pass" : "warn");
  if (gt.present) {
    addSummaryRow(grid, "GT box", boxLabel(gt.actual_box));
    addSummaryRow(grid, "GT fit", gt.fit);
  }
  if (data.comparison) {
    addSummaryRow(grid, "Match", data.comparison.match, data.comparison.match ? "pass" : "warn");
    addSummaryRow(grid, "Diff", boxLabel(data.comparison.diff_in));
  }
  predictionSummary.appendChild(grid);
}

async function lookupSku() {
  try {
    const sku = field("sku-input").value.trim();
    if (!sku) {
      throw {error: "SKU is required for lookup."};
    }
    const response = await postJson("/api/sku/lookup", {sku});
    lastLookup = response;
    renderSkuSummary(response);
    outputStatus.textContent = response.status === "found" ? "SKU found" : "SKU not found";
  } catch (error) {
    renderSkuSummary(error, true);
    showError(error, "SKU lookup failed");
  }
}

async function captureAndPredict(viewId) {
  if (requestInProgress || predictionCompleted) {
    return;
  }
  const activeButton = field(viewId === "view_02" ? "capture-view-02" : "capture-view-01");
  const buttons = [field("capture-view-01"), field("capture-view-02"), field("start-new-prediction")];
  try {
    requestInProgress = true;
    activeButton.disabled = true;
    capturePredictStatus.textContent = `${viewId} capture and processing running.`;
    if (!field("sku-input").value.trim()) {
      renderSkuSummary({
        status: "not_provided",
        known: false,
        sku: null,
        metadata: {
          category: null,
          common_name: null,
          spec: null
        }
      });
    }
    const response = await postJson("/api/ai2/capture-predict", {
      job_type: field("job-type").value,
      shape_mode: field("shape-mode").value,
      arrangement_type: field("job-type").value === "group"
        ? field("arrangement-type").value.trim()
        : "1x1",
      items: activeItems(),
      prediction_job_id: predictionJobId,
      view_id: viewId
    });
    predictionJobId = response.prediction_job_id || predictionJobId;
    predictionCapturedViews = response.captured_views || [];
    predictionReady = response.ready_for_prediction === true;
    predictionCompleted = predictionReady;
    showStatus(response, response.ready_for_prediction ? "Prediction complete" : "View captured");
    if (!response.ready_for_prediction) {
      capturePredictStatus.textContent = response.operator_instruction || "Capture the remaining view.";
      field("capture-view-01").hidden = true;
      field("capture-view-02").hidden = !(response.remaining_views || []).includes("view_02");
      field("start-new-prediction").hidden = true;
    } else {
      capturePredictStatus.textContent = response.status === "model_unavailable"
        ? "Capture complete; the selected model is unavailable."
        : "Capture and prediction complete.";
      field("capture-view-01").hidden = true;
      field("capture-view-02").hidden = true;
      field("start-new-prediction").hidden = false;
    }
  } catch (error) {
    capturePredictStatus.textContent = error.error ? `Capture & Predict failed: ${error.error}` : "Capture & Predict failed.";
    showError(error, "Capture & Predict failed");
  } finally {
    requestInProgress = false;
    buttons.forEach((button) => { button.disabled = false; });
  }
}

async function predict() {
  try {
    const jobId = field("predict-job-id").value.trim();
    if (!jobId) {
      throw {error: "Manual Job ID is required. Use dataset_2 or another datasets/single/jobs ID."};
    }
    const response = await postJson("/api/ai2/predict", {
      job_type: field("predict-job-type").value,
      job_id: jobId
    });
    showStatus(response, "Prediction complete");
  } catch (error) {
    showError(error, "Prediction failed");
  }
}

function reloadLiveFeed() {
  cameraStatus.textContent = "Reloading live feed...";
  cameraPill.textContent = "Camera Preview";
  cameraPill.classList.remove("error-pill");
  cameraPreview.src = `/video_feed?ts=${Date.now()}`;
}

cameraPreview.addEventListener("load", () => {
  cameraStatus.textContent = "Live feed active.";
});

cameraPreview.addEventListener("error", () => {
  cameraStatus.textContent = "Camera preview unavailable.";
  cameraPill.textContent = "Camera Error";
  cameraPill.classList.add("error-pill");
});

field("lookup-sku").addEventListener("click", lookupSku);
field("add-item").addEventListener("click", addItem);
field("capture-view-01").addEventListener("click", () => captureAndPredict("view_01"));
field("capture-view-02").addEventListener("click", () => captureAndPredict("view_02"));
field("start-new-prediction").addEventListener("click", resetPredictionJob);
field("job-type").addEventListener("change", renderMode);
field("shape-mode").addEventListener("change", resetPredictionJob);
field("arrangement-type").addEventListener("change", resetPredictionJob);
field("sku-input").addEventListener("input", resetPredictionJob);
field("predict").addEventListener("click", predict);
field("reload-feed").addEventListener("click", reloadLiveFeed);
renderItems();
renderMode();
