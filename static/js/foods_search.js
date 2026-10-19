/**
 * foods_search.js
 * Feature B: Debounced food database search with AbortController, safe DOM rendering, and instant clear.
 */

document.addEventListener("DOMContentLoaded", function () {
    const searchForm = document.getElementById("food_search_form");
    const searchInput = document.getElementById("food_search_input");
    const clearBtn = document.getElementById("clear_search_btn");
    const loadingIndicator = document.getElementById("search_loading");
    const errorAlert = document.getElementById("search_error");
    const resultsTitle = document.getElementById("results_title");
    const resultsCountBadge = document.getElementById("results_count_badge");
    const resultsList = document.getElementById("foods_results_list");
    const noResultsBox = document.getElementById("no_results_box");
    const noResultsMsg = document.getElementById("no_results_message");
    const createCustomBtn = document.getElementById("create_custom_btn");

    if (!searchInput || !resultsList) return;

    let debounceTimer = null;
    let abortController = null;

    // Cache initial server-rendered HTML for instant reset
    const initialTitle = resultsTitle ? resultsTitle.textContent : "";
    const initialBadge = resultsCountBadge ? resultsCountBadge.textContent : "";
    const initialListHTML = resultsList.innerHTML;
    const initialNoResultsDisplay = noResultsBox ? noResultsBox.style.display : "none";

    function toggleClearBtn() {
        if (!clearBtn) return;
        if (searchInput.value.trim().length > 0) {
            clearBtn.style.display = "block";
        } else {
            clearBtn.style.display = "none";
        }
    }

    function resetSearchToInitial() {
        if (resultsTitle) resultsTitle.textContent = initialTitle;
        if (resultsCountBadge) resultsCountBadge.textContent = initialBadge;
        resultsList.innerHTML = initialListHTML;
        resultsList.style.display = "flex";
        if (noResultsBox) noResultsBox.style.display = initialNoResultsDisplay;
        if (errorAlert) errorAlert.style.display = "none";
        if (loadingIndicator) loadingIndicator.style.display = "none";
        toggleClearBtn();
    }

    async function performSearch(query) {
        const term = (query || "").trim();

        if (term.length === 0) {
            resetSearchToInitial();
            return;
        }

        // Cancel previous pending request
        if (abortController) {
            abortController.abort();
        }
        abortController = new AbortController();

        if (loadingIndicator) loadingIndicator.style.display = "flex";
        if (errorAlert) errorAlert.style.display = "none";

        try {
            const url = `/api/foods?q=${encodeURIComponent(term)}&limit=50`;
            const response = await fetch(url, {
                signal: abortController.signal,
                headers: { "Accept": "application/json" }
            });

            if (!response.ok) {
                throw new Error(`Search failed with status: ${response.status}`);
            }

            const data = await response.json();
            renderSearchResults(data, term);

        } catch (err) {
            if (err.name === "AbortError") {
                return; // Ignored as user typed a new character
            }
            console.error("Food search error:", err);
            if (errorAlert) {
                errorAlert.textContent = "⚠️ Could not search foods at this moment. Please check your network and retry.";
                errorAlert.style.display = "flex";
            }
        } finally {
            if (loadingIndicator) loadingIndicator.style.display = "none";
        }
    }

    function renderSearchResults(data, query) {
        const count = data.count || 0;
        const foods = data.foods || [];

        // Update headers safely
        if (resultsTitle) resultsTitle.textContent = `Results for "${query}"`;
        if (resultsCountBadge) resultsCountBadge.textContent = `${count} ${count === 1 ? 'dish' : 'dishes'} found`;

        // Clear existing list safely
        resultsList.textContent = "";

        if (foods.length === 0) {
            resultsList.style.display = "none";
            if (noResultsMsg) noResultsMsg.textContent = `No matching foods found for "${query}".`;
            if (createCustomBtn) {
                createCustomBtn.href = `/log-food?food=${encodeURIComponent(query)}`;
                createCustomBtn.textContent = `+ Add "${query}" as Custom Food`;
            }
            if (noResultsBox) noResultsBox.style.display = "block";
            return;
        }

        if (noResultsBox) noResultsBox.style.display = "none";
        resultsList.style.display = "flex";

        // Safely create DOM elements to avoid unsafe innerHTML insertion
        foods.forEach(food => {
            const itemDiv = document.createElement("div");
            itemDiv.className = "food-item";

            const infoDiv = document.createElement("div");

            const nameSpan = document.createElement("span");
            nameSpan.style.fontWeight = "600";
            nameSpan.style.color = "#1e293b";
            nameSpan.style.display = "block";
            nameSpan.textContent = food.name;
            infoDiv.appendChild(nameSpan);

            // Display per 100g nutrition chips
            const macroSpan = document.createElement("small");
            macroSpan.style.display = "block";
            macroSpan.style.color = "#64748b";
            macroSpan.style.marginTop = "3px";
            macroSpan.style.fontSize = "12px";

            const calText = `${food.calories} kcal`;
            const macroText = ` • P: ${food.protein}g • C: ${food.carbohydrates}g • F: ${food.fats}g (per 100g)`;
            macroSpan.textContent = calText + macroText;
            infoDiv.appendChild(macroSpan);

            const logBtn = document.createElement("a");
            logBtn.href = `/log-food?food=${encodeURIComponent(food.name)}`;
            logBtn.className = "button small";
            logBtn.style.margin = "0";
            logBtn.textContent = "Log";

            itemDiv.appendChild(infoDiv);
            itemDiv.appendChild(logBtn);

            resultsList.appendChild(itemDiv);
        });
    }

    // Debounced input event listener (300ms)
    searchInput.addEventListener("input", function () {
        toggleClearBtn();
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            performSearch(searchInput.value);
        }, 300);
    });

    // Clear search button
    if (clearBtn) {
        clearBtn.addEventListener("click", function () {
            searchInput.value = "";
            toggleClearBtn();
            if (abortController) abortController.abort();
            resetSearchToInitial();
            searchInput.focus();
        });
    }

    // Intercept form submission with JS for responsive instant search
    if (searchForm) {
        searchForm.addEventListener("submit", function (e) {
            e.preventDefault();
            clearTimeout(debounceTimer);
            performSearch(searchInput.value);
        });
    }
});
