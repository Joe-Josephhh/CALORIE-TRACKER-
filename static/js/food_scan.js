/**
 * food_scan.js
 * Feature D: Food prediction upload progress, XMLHttpRequest genuine byte progress tracking,
 * 4-stage lifecycle display, in-place prediction rendering, and retry handling.
 */

document.addEventListener("DOMContentLoaded", function () {
    const scanForm = document.getElementById("food_scan_form");
    const imageInput = document.getElementById("image_input");
    const submitBtn = document.getElementById("predict_submit_btn");
    const uploadFormCard = document.getElementById("upload_form_card");

    const previewBox = document.getElementById("image_preview_box");
    const previewImg = document.getElementById("image_preview_img");
    const fileMeta = document.getElementById("image_file_meta");

    const progressContainer = document.getElementById("prediction_progress_container");
    const statusText = document.getElementById("progress_status_text");
    const statusDesc = document.getElementById("progress_status_desc");
    const percentageText = document.getElementById("progress_percentage_text");
    const progressBarFill = document.getElementById("progress_bar_fill");

    const stepUpload = document.getElementById("step_upload");
    const stepProcess = document.getElementById("step_process");
    const stepInfer = document.getElementById("step_infer");
    const stepDone = document.getElementById("step_done");

    const errorBox = document.getElementById("prediction_error_box");
    const errorMsg = document.getElementById("prediction_error_message");
    const retryBtn = document.getElementById("retry_scan_btn");

    const resultsContainer = document.getElementById("prediction_results_container");
    const bestGuessTitle = document.getElementById("best_guess_title");
    const bestGuessConf = document.getElementById("best_guess_confidence");
    const acceptBtn = document.getElementById("accept_prediction_btn");
    const topPredictionsList = document.getElementById("top_predictions_list");
    const scanAgainBtn = document.getElementById("scan_again_btn");

    if (!scanForm || !imageInput) return;

    // Client-side instant preview upon selecting image
    imageInput.addEventListener("change", function () {
        const file = this.files[0];
        if (!file) {
            if (previewBox) previewBox.style.display = "none";
            return;
        }

        // Validate file size (10 MB max)
        const maxBytes = 10 * 1024 * 1024;
        if (file.size > maxBytes) {
            alert("⚠️ Selected file exceeds 10 MB limit. Please select a smaller photo.");
            this.value = "";
            if (previewBox) previewBox.style.display = "none";
            return;
        }

        const sizeKb = (file.size / 1024).toFixed(1);
        if (fileMeta) fileMeta.textContent = `${file.name} (${sizeKb} KB)`;

        const reader = new FileReader();
        reader.onload = function (e) {
            if (previewImg) previewImg.src = e.target.result;
            if (previewBox) previewBox.style.display = "block";
        };
        reader.readAsDataURL(file);
    });

    function setStepState(stepEl, state) {
        if (!stepEl) return;
        stepEl.classList.remove("active", "completed");
        const icon = stepEl.querySelector(".step-icon-circle");
        if (state === "completed") {
            stepEl.classList.add("completed");
            if (icon) icon.textContent = "✓";
        } else if (state === "active") {
            stepEl.classList.add("active");
        }
    }

    function resetToUploadForm() {
        if (errorBox) errorBox.style.display = "none";
        if (progressContainer) progressContainer.style.display = "none";
        if (resultsContainer) resultsContainer.style.display = "none";
        if (uploadFormCard) uploadFormCard.style.display = "block";

        imageInput.disabled = false;
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = "Predict Food";
        }

        if (progressBarFill) {
            progressBarFill.classList.remove("indeterminate");
            progressBarFill.style.width = "0%";
        }
    }

    if (retryBtn) {
        retryBtn.addEventListener("click", resetToUploadForm);
    }

    if (scanAgainBtn) {
        scanAgainBtn.addEventListener("click", () => {
            imageInput.value = "";
            if (previewBox) previewBox.style.display = "none";
            resetToUploadForm();
        });
    }

    // Intercept form submission with asynchronous XMLHttpRequest for genuine upload progress tracking
    scanForm.addEventListener("submit", function (e) {
        e.preventDefault();

        const file = imageInput.files[0];
        if (!file) {
            alert("Please select an image file first.");
            return;
        }

        // Disable submit button to prevent duplicate submissions
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = "Analyzing...";
        }
        imageInput.disabled = true;

        if (errorBox) errorBox.style.display = "none";
        if (resultsContainer) resultsContainer.style.display = "none";
        if (progressContainer) progressContainer.style.display = "block";

        // Reset progress steps
        setStepState(stepUpload, "active");
        setStepState(stepProcess, "");
        setStepState(stepInfer, "");
        setStepState(stepDone, "");

        if (statusText) statusText.textContent = "Uploading image...";
        if (statusDesc) statusDesc.textContent = `Uploading ${file.name} (${(file.size / 1024).toFixed(0)} KB) to server...`;
        if (percentageText) percentageText.textContent = "0%";
        if (progressBarFill) {
            progressBarFill.classList.remove("indeterminate");
            progressBarFill.style.width = "0%";
        }

        const formData = new FormData(scanForm);
        const xhr = new XMLHttpRequest();

        // 1. Genuine upload progress tracking
        xhr.upload.onprogress = function (event) {
            if (event.lengthComputable) {
                const percentComplete = Math.round((event.loaded / event.total) * 100);
                if (progressBarFill) progressBarFill.style.width = percentComplete + "%";
                if (percentageText) percentageText.textContent = percentComplete + "%";

                if (percentComplete >= 100) {
                    // Upload completed! Move truthfully to Python processing state
                    setStepState(stepUpload, "completed");
                    setStepState(stepProcess, "active");
                    if (statusText) statusText.textContent = "Processing image...";
                    if (statusDesc) statusDesc.textContent = "Decoding tensor and preparing for model evaluation...";
                    if (percentageText) percentageText.textContent = "Processing";
                    if (progressBarFill) progressBarFill.classList.add("indeterminate");

                    // Brief transition to inference step while awaiting server response
                    setTimeout(() => {
                        if (xhr.readyState < 4) {
                            setStepState(stepProcess, "completed");
                            setStepState(stepInfer, "active");
                            if (statusText) statusText.textContent = "Generating prediction...";
                            if (statusDesc) statusDesc.textContent = "Running fine-tuned EfficientNet-B0 inference on Python backend...";
                            if (percentageText) percentageText.textContent = "Classifying";
                        }
                    }, 400);
                }
            }
        };

        // 2. Response received from backend
        xhr.onload = function () {
            if (progressBarFill) progressBarFill.classList.remove("indeterminate");

            if (xhr.status >= 200 && xhr.status < 300) {
                let data = null;
                try {
                    data = JSON.parse(xhr.responseText);
                } catch (parseErr) {
                    console.error("JSON parse error:", parseErr);
                }

                if (data && data.success && data.predictions) {
                    // Mark progress completed
                    setStepState(stepProcess, "completed");
                    setStepState(stepInfer, "completed");
                    setStepState(stepDone, "completed");
                    if (progressBarFill) progressBarFill.style.width = "100%";
                    if (percentageText) percentageText.textContent = "100%";
                    if (statusText) statusText.textContent = "Prediction complete!";
                    if (statusDesc) statusDesc.textContent = "Model successfully classified food dish.";

                    // Smooth transition to results display
                    setTimeout(() => {
                        displayPredictionResults(data);
                    }, 350);
                    return;
                }
            }

            // Failure handling
            handleFailure(xhr);
        };

        // 3. Network or connection failure
        xhr.onerror = function () {
            handleFailure(xhr, "Network error occurred. Could not reach server. Please check your connection.");
        };

        xhr.ontimeout = function () {
            handleFailure(xhr, "Request timed out while waiting for model inference. Please try again.");
        };

        const targetUrl = scanForm.getAttribute("action") || "/predict";
        xhr.open("POST", targetUrl, true);
        xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");
        xhr.setRequestHeader("Accept", "application/json");
        xhr.send(formData);
    });

    function handleFailure(xhr, customMsg) {
        if (progressContainer) progressContainer.style.display = "none";
        let message = customMsg || "Unable to process the image. Please try another image.";

        try {
            if (xhr.responseText) {
                const parsed = JSON.parse(xhr.responseText);
                if (parsed.error) message = parsed.error;
            }
        } catch (_) {}

        if (errorMsg) errorMsg.textContent = message;
        if (errorBox) errorBox.style.display = "block";
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = "Predict Food";
        }
        imageInput.disabled = false;
    }

    function displayPredictionResults(data) {
        if (progressContainer) progressContainer.style.display = "none";
        if (uploadFormCard) uploadFormCard.style.display = "none";

        const best = data.best_prediction || (data.predictions && data.predictions[0]);
        if (best) {
            if (bestGuessTitle) bestGuessTitle.textContent = best.food;
            if (bestGuessConf) bestGuessConf.textContent = Number(best.confidence).toFixed(2) + "%";
            if (acceptBtn) acceptBtn.href = data.accept_url || `/log-food?food=${encodeURIComponent(best.food)}`;
        }

        // Render Top 5 Predictions with confidence bars safely
        if (topPredictionsList) {
            topPredictionsList.textContent = "";

            (data.predictions || []).forEach(pred => {
                const row = document.createElement("div");
                row.className = "prediction-row";
                row.style.flexDirection = "column";
                row.style.gap = "4px";

                const topLine = document.createElement("div");
                topLine.style.display = "flex";
                topLine.style.justifyContent = "space-between";
                topLine.style.alignItems = "center";

                const nameStrong = document.createElement("strong");
                nameStrong.textContent = `${pred.rank}. ${pred.food}`;
                topLine.appendChild(nameStrong);

                const confSpan = document.createElement("span");
                confSpan.style.fontWeight = "600";
                confSpan.style.color = pred.rank === 1 ? "#2563eb" : "#64748b";
                confSpan.textContent = `${Number(pred.confidence).toFixed(2)}%`;
                topLine.appendChild(confSpan);

                // Visual confidence bar
                const barBg = document.createElement("div");
                barBg.className = "confidence-bar-bg";

                const barFill = document.createElement("div");
                barFill.className = "confidence-bar-fill";
                barFill.style.width = `${Math.min(100, Math.max(0, pred.confidence))}%`;
                if (pred.rank === 1) {
                    barFill.style.background = "#2563eb";
                } else {
                    barFill.style.background = "#94a3b8";
                }

                barBg.appendChild(barFill);

                row.appendChild(topLine);
                row.appendChild(barBg);
                topPredictionsList.appendChild(row);
            });
        }

        if (resultsContainer) {
            resultsContainer.style.display = "block";
            resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }
});
