/**
 * log_food.js
 * Feature A: Asynchronous food logging & live calorie/macro updates without page reload.
 * Feature B (part): Debounced food name autocomplete against /api/foods.
 */

// Global presets from server config or fallback
const GENERAL_PRESETS = window.GENERAL_PRESETS || [];
const LOG_FOOD_URL = window.LOG_FOOD_URL || "/log-food";

function setWeight() {
    const select = document.getElementById("weight");
    const input = document.getElementById("weight_input");
    if (!select || !input) return;

    if (select.value !== "custom" && select.value !== "") {
        input.value = select.value;
    }
}

function toggleCustomNutrition() {
    const checkbox = document.getElementById("custom_food");
    const section = document.getElementById("custom-nutrition");
    if (!checkbox || !section) return;

    if (checkbox.checked) {
        section.style.display = "block";
    } else {
        section.style.display = "none";
    }
}

function openCustomWithPresets() {
    const checkbox = document.getElementById("custom_food");
    const section = document.getElementById("custom-nutrition");
    if (!checkbox || !section) return;

    checkbox.checked = true;
    section.style.display = "block";

    const filterInput = document.getElementById("preset_search_input");
    if (filterInput) {
        filterInput.focus();
        filterInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function applyPreset(btn) {
    if (!btn) return;
    const name = btn.dataset.name;
    const calories = btn.dataset.calories;
    const protein = btn.dataset.protein;
    const carbs = btn.dataset.carbs;
    const fats = btn.dataset.fats;
    const fibre = btn.dataset.fibre;

    // Set food name
    document.getElementById("dish_name").value = name;

    // Ensure custom checkbox is checked and custom nutrition is visible
    const checkbox = document.getElementById("custom_food");
    checkbox.checked = true;
    document.getElementById("custom-nutrition").style.display = "block";

    // Populate macro values
    document.getElementById("custom_calories").value = calories;
    document.getElementById("custom_protein").value = protein;
    document.getElementById("custom_carbs").value = carbs;
    document.getElementById("custom_fats").value = fats;
    document.getElementById("custom_fibre").value = fibre;

    // Reset optional micronutrients
    const micros = ["custom_free_sugar", "custom_sodium", "custom_calcium", "custom_iron", "custom_vitamin_c", "custom_folate"];
    micros.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = "0";
    });

    // Highlight active chip
    document.querySelectorAll(".preset-chip-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    // Show alert banner
    const alertBox = document.getElementById("preset_applied_alert");
    if (alertBox) {
        document.getElementById("applied_preset_name").textContent = name;
        document.getElementById("applied_preset_summary").textContent = 
            `${calories} kcal | ${protein}g P | ${carbs}g C | ${fats}g F | ${fibre}g Fibre`;
        alertBox.style.display = "flex";
    }

    // Close any open food autocomplete suggestions
    closeSuggestions();
}

function applyPresetByName(presetName) {
    if (!presetName) return false;
    const match = GENERAL_PRESETS.find(p => p.name.toLowerCase() === presetName.trim().toLowerCase());
    if (match) {
        const btn = document.querySelector(`.preset-chip-btn[data-name="${match.name}"]`);
        if (btn) {
            applyPreset(btn);
            return true;
        } else {
            // Apply directly from object
            document.getElementById("dish_name").value = match.name;
            const customBox = document.getElementById("custom_food");
            if (customBox) customBox.checked = true;
            const customSect = document.getElementById("custom-nutrition");
            if (customSect) customSect.style.display = "block";

            document.getElementById("custom_calories").value = match.calories;
            document.getElementById("custom_protein").value = match.protein;
            document.getElementById("custom_carbs").value = match.carbohydrates;
            document.getElementById("custom_fats").value = match.fats;
            document.getElementById("custom_fibre").value = match.fibre;

            const alertBox = document.getElementById("preset_applied_alert");
            if (alertBox) {
                document.getElementById("applied_preset_name").textContent = match.name;
                document.getElementById("applied_preset_summary").textContent = 
                    `${match.calories} kcal | ${match.protein}g P | ${match.carbohydrates}g C | ${match.fats}g F | ${match.fibre}g Fibre`;
                alertBox.style.display = "flex";
            }
            return true;
        }
    }
    return false;
}

function dismissPresetAlert() {
    const alertBox = document.getElementById("preset_applied_alert");
    if (alertBox) alertBox.style.display = "none";
}

function filterPresetChips(query) {
    const term = (query || "").trim().toLowerCase();
    const buttons = document.querySelectorAll(".preset-chip-btn");

    buttons.forEach(btn => {
        const name = (btn.dataset.name || "").toLowerCase();
        if (!term || name.includes(term)) {
            btn.style.display = "inline-flex";
        } else {
            btn.style.display = "none";
        }
    });
}

// --- Debounced Live Search for Food Input (Feature B) ---
let searchDebounceTimer = null;
let currentSearchController = null;

function handleFoodInput(val) {
    const term = (val || "").trim();

    if (!term || term.length < 1) {
        closeSuggestions();
        return;
    }

    if (searchDebounceTimer) {
        clearTimeout(searchDebounceTimer);
    }

    // Step 1: Immediate local match from presets for instant UI feedback
    const localMatches = GENERAL_PRESETS.filter(p => p.name.toLowerCase().includes(term.toLowerCase())).slice(0, 4);
    if (localMatches.length > 0) {
        renderFoodSuggestions(localMatches, term);
    }

    // Step 2: Debounced backend search across full nutrition dataset (250ms)
    searchDebounceTimer = setTimeout(async () => {
        if (currentSearchController) {
            currentSearchController.abort();
        }
        currentSearchController = new AbortController();

        try {
            const res = await fetch(`/api/foods?q=${encodeURIComponent(term)}&limit=10`, {
                signal: currentSearchController.signal,
                headers: { "Accept": "application/json" }
            });
            if (!res.ok) return;
            const data = await res.json();
            if (data.success && data.foods) {
                renderFoodSuggestions(data.foods, term);
            }
        } catch (e) {
            if (e.name !== 'AbortError') {
                console.error("Food autocomplete search error:", e);
            }
        }
    }, 250);
}

function renderFoodSuggestions(items, term) {
    const box = document.getElementById("food_suggestions");
    if (!box) return;

    if (!items || items.length === 0) {
        box.style.display = "none";
        box.textContent = "";
        return;
    }

    // Safe DOM construction preventing unsafe innerHTML injection
    box.textContent = "";

    items.forEach(p => {
        const itemDiv = document.createElement("div");
        itemDiv.className = "food-suggestion-item";
        itemDiv.tabIndex = 0;

        const nameSpan = document.createElement("span");
        nameSpan.className = "food-suggestion-name";
        const isPreset = GENERAL_PRESETS.some(g => g.name.toLowerCase() === p.name.toLowerCase());
        nameSpan.textContent = (isPreset ? "⚡ " : "📖 ") + p.name;

        const metaSpan = document.createElement("span");
        metaSpan.className = "food-suggestion-meta";
        const cals = Math.round(p.calories || 0);
        const prot = p.protein !== undefined ? p.protein : 0;
        const carbs = p.carbohydrates !== undefined ? p.carbohydrates : 0;
        metaSpan.textContent = `${cals} kcal · ${prot}g P · ${carbs}g C`;

        itemDiv.appendChild(nameSpan);
        itemDiv.appendChild(metaSpan);

        const selectThis = () => {
            selectFoodSuggestion(p);
        };

        itemDiv.addEventListener("click", selectThis);
        itemDiv.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                selectThis();
            }
        });

        box.appendChild(itemDiv);
    });

    box.style.display = "block";
}

function selectFoodSuggestion(item) {
    document.getElementById("dish_name").value = item.name;
    closeSuggestions();

    const isPreset = applyPresetByName(item.name);
    if (!isPreset) {
        // If from full dataset, ensure standard weight is set
        const weightInput = document.getElementById("weight_input");
        if (weightInput && !weightInput.value) {
            const weightSelect = document.getElementById("weight");
            if (weightSelect) weightSelect.value = "100";
            setWeight();
        }
    }
}

function closeSuggestions() {
    const box = document.getElementById("food_suggestions");
    if (box) {
        box.style.display = "none";
        box.textContent = "";
    }
}

// Close suggestions on outside click
document.addEventListener("click", function(e) {
    if (!e.target.closest(".food-input-wrapper")) {
        closeSuggestions();
    }
});

// --- Asynchronous Food Log Submission (Feature A) ---
document.addEventListener("DOMContentLoaded", function () {
    const foodLogForm = document.getElementById("food_log_form");
    const submitLogBtn = document.getElementById("submit_log_btn");
    const asyncLogAlert = document.getElementById("async_log_alert");

    function showLogAlert(message, type) {
        if (!asyncLogAlert) return;
        asyncLogAlert.className = `alert ${type}`;
        asyncLogAlert.textContent = message;
        asyncLogAlert.style.display = "flex";
        asyncLogAlert.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function pulseElement(el) {
        if (!el) return;
        el.classList.remove("value-pulse");
        void el.offsetWidth;
        el.classList.add("value-pulse");
    }

    if (foodLogForm) {
        foodLogForm.addEventListener("submit", async function(e) {
            e.preventDefault();

            const dishName = (document.getElementById("dish_name").value || "").trim();
            const weight = parseFloat(document.getElementById("weight_input").value);

            if (!dishName) {
                showLogAlert("Please enter a food name.", "error");
                return;
            }

            if (isNaN(weight) || weight <= 0) {
                showLogAlert("Please enter a valid weight greater than 0g.", "error");
                return;
            }

            // Prevent accidental duplicate submissions
            submitLogBtn.disabled = true;
            const originalBtnText = submitLogBtn.textContent;
            submitLogBtn.textContent = "⏳ Adding to Diary...";

            const formData = new FormData(foodLogForm);

            try {
                const response = await fetch(LOG_FOOD_URL, {
                    method: "POST",
                    headers: {
                        "X-Requested-With": "XMLHttpRequest",
                        "Accept": "application/json"
                    },
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    // Display success message
                    showLogAlert("✓ " + data.message, "success");

                    // Update displayed calorie and macronutrient totals authoritative from backend
                    if (data.totals) {
                        const calEl = document.getElementById("live_today_calories");
                        const protEl = document.getElementById("live_today_protein");
                        const carbsEl = document.getElementById("live_today_carbs");
                        const fatsEl = document.getElementById("live_today_fats");

                        if (calEl && data.totals["Calories (kcal)"] !== undefined) {
                            calEl.textContent = Number(data.totals["Calories (kcal)"]).toFixed(1);
                            pulseElement(calEl);
                        }
                        if (protEl && data.totals["Protein (g)"] !== undefined) {
                            protEl.textContent = Number(data.totals["Protein (g)"]).toFixed(1);
                            pulseElement(protEl);
                        }
                        if (carbsEl && data.totals["Carbohydrates (g)"] !== undefined) {
                            carbsEl.textContent = Number(data.totals["Carbohydrates (g)"]).toFixed(1);
                            pulseElement(carbsEl);
                        }
                        if (fatsEl && data.totals["Fats (g)"] !== undefined) {
                            fatsEl.textContent = Number(data.totals["Fats (g)"]).toFixed(1);
                            pulseElement(fatsEl);
                        }
                    }

                    // Update the food diary list on the page
                    const mealsList = document.getElementById("today_meals_list");
                    const emptyMsg = document.getElementById("empty_meals_msg");
                    if (emptyMsg) {
                        emptyMsg.style.display = "none";
                    }

                    if (mealsList && data.entry) {
                        const entry = data.entry;
                        const nutr = entry.nutrition || {};
                        const entryDiv = document.createElement("div");
                        entryDiv.className = "diary-entry";
                        if (entry.id) entryDiv.id = `entry-${entry.id}`;

                        const leftDiv = document.createElement("div");
                        const titleEl = document.createElement("strong");
                        titleEl.style.color = "#1e293b";
                        titleEl.style.fontSize = "15px";
                        titleEl.textContent = entry.dish_name;
                        leftDiv.appendChild(titleEl);

                        const metaP = document.createElement("p");
                        metaP.style.margin = "2px 0 0";
                        metaP.style.color = "#64748b";
                        metaP.style.fontSize = "13px";
                        const mealTypeCap = (entry.meal_type || "other").charAt(0).toUpperCase() + (entry.meal_type || "other").slice(1);
                        metaP.textContent = `${entry.time || 'Just now'} • ${mealTypeCap} • ${entry.weight} g`;
                        if (entry.notes) {
                            const noteSpan = document.createElement("em");
                            noteSpan.textContent = ` • ${entry.notes}`;
                            metaP.appendChild(noteSpan);
                        }
                        leftDiv.appendChild(metaP);

                        const rightDiv = document.createElement("div");
                        rightDiv.style.textAlign = "right";
                        rightDiv.style.minWidth = "90px";

                        const kcalEl = document.createElement("strong");
                        kcalEl.style.color = "#2563eb";
                        kcalEl.style.fontSize = "15px";
                        kcalEl.textContent = `${Number(nutr["Calories (kcal)"] || 0).toFixed(1)} kcal`;
                        rightDiv.appendChild(kcalEl);

                        const macrosSmall = document.createElement("small");
                        macrosSmall.style.display = "block";
                        macrosSmall.style.color = "#64748b";
                        macrosSmall.style.fontSize = "12px";
                        macrosSmall.textContent = `P: ${Number(nutr["Protein (g)"] || 0).toFixed(1)}g • C: ${Number(nutr["Carbohydrates (g)"] || 0).toFixed(1)}g • F: ${Number(nutr["Fats (g)"] || 0).toFixed(1)}g`;
                        rightDiv.appendChild(macrosSmall);

                        entryDiv.appendChild(leftDiv);
                        entryDiv.appendChild(rightDiv);

                        mealsList.insertBefore(entryDiv, mealsList.firstChild);

                        const countEl = document.getElementById("today_meals_count");
                        if (countEl) {
                            countEl.textContent = parseInt(countEl.textContent || "0", 10) + 1;
                        }
                    }

                    // Reset form inputs for next entry
                    document.getElementById("dish_name").value = "";
                    document.getElementById("weight_input").value = "";
                    document.getElementById("notes").value = "";
                    document.getElementById("weight").value = "";
                    dismissPresetAlert();

                    // Uncheck custom food if it was open
                    const customCheckbox = document.getElementById("custom_food");
                    if (customCheckbox && customCheckbox.checked) {
                        customCheckbox.checked = false;
                        toggleCustomNutrition();
                    }

                    // Remove chip active class
                    document.querySelectorAll(".preset-chip-btn").forEach(b => b.classList.remove("active"));

                    document.getElementById("dish_name").focus();

                } else {
                    showLogAlert("⚠️ " + (data.error || "Failed to log food entry."), "error");
                }
            } catch (networkError) {
                console.error("Async food log network error:", networkError);
                showLogAlert("⚠️ Network failure. Could not connect to server. Please check your network and try again.", "error");
            } finally {
                submitLogBtn.disabled = false;
                submitLogBtn.textContent = originalBtnText;
            }
        });
    }

    // Auto-select on initial page load if matching food is passed in URL query
    const initialFood = document.getElementById("dish_name") ? document.getElementById("dish_name").value : "";
    if (initialFood) {
        applyPresetByName(initialFood);
    }
});
