const API_BASE = "";

const el = {
    q: document.getElementById("q"),
    go: document.getElementById("go"),

    participant: document.getElementById("participant"),
    topK: document.getElementById("topK"),

    startDate: document.getElementById("startDate"),
    endDate: document.getElementById("endDate"),

    detected: document.getElementById("detected"),
    results: document.getElementById("results"),
    resultCount: document.getElementById("resultCount"),

    statusCard: document.getElementById("statusCard"),
    statusText: document.getElementById("statusText"),
    statusMeta: document.getElementById("statusMeta"),

    messageCount: document.getElementById("messageCount")
};


/* ---------------------------------------------------------
   HTML escaping
--------------------------------------------------------- */

function escapeHtml(value) {

    return String(value).replace(/[&<>"']/g, (character) => {

        const map = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        };

        return map[character];

    });
}


/* ---------------------------------------------------------
   Backend health
--------------------------------------------------------- */

async function loadHealth() {

    try {

        const response = await fetch(`${API_BASE}/health`);

        if (!response.ok) {
            throw new Error("Backend unavailable");
        }

        const data = await response.json();

        el.statusCard.classList.add("online");

        el.statusText.textContent = "Backend online";

        el.statusMeta.textContent =
            `${Number(data.messages_indexed).toLocaleString()} messages · ${data.embedding_backend}`;

        if (el.messageCount) {

            el.messageCount.textContent =
                Number(data.messages_indexed).toLocaleString();

        }

    } catch (error) {

        el.statusCard.classList.add("offline");

        el.statusText.textContent = "Backend offline";

        el.statusMeta.textContent =
            "Start FastAPI on port 8000";

    }
}


/* ---------------------------------------------------------
   Participants
--------------------------------------------------------- */

async function loadParticipants() {

    try {

        const response =
            await fetch(`${API_BASE}/participants`);

        if (!response.ok) {
            throw new Error("Could not load participants");
        }

        const data = await response.json();

        if (!el.participant) {
            return;
        }

        data.participants.forEach((person) => {

            const option =
                document.createElement("option");

            option.value = person;
            option.textContent = person;

            el.participant.appendChild(option);

        });

    } catch (error) {

        console.warn(
            "Participant list could not be loaded:",
            error
        );

    }
}


/* ---------------------------------------------------------
   Loading state
--------------------------------------------------------- */

function setLoading(query) {

    el.go.disabled = true;

    el.go.innerHTML =
        `Searching <span class="button-spinner"></span>`;

    el.resultCount.textContent = "";

    el.detected.innerHTML =
        `<span class="query-chip">
            Searching for: ${escapeHtml(query)}
        </span>`;

    el.results.innerHTML = `

        <article class="result-card skeleton-card">

            <div class="skeleton-line short"></div>

            <div class="skeleton-line medium"></div>

            <div class="skeleton-line long"></div>

            <div class="skeleton-line medium"></div>

        </article>

        <article class="result-card skeleton-card">

            <div class="skeleton-line short"></div>

            <div class="skeleton-line long"></div>

            <div class="skeleton-line medium"></div>

        </article>

    `;
}


function clearLoading() {

    el.go.disabled = false;

    el.go.innerHTML =
        `Search <span>→</span>`;

}


/* ---------------------------------------------------------
   Detected filters / metadata
--------------------------------------------------------- */

function renderDetected(output) {

    const chips = [];

    if (output.search_time_ms !== undefined) {

        chips.push(`
            <span class="info-chip">
                ${escapeHtml(output.search_time_ms)} ms
            </span>
        `);

    }

    if (output.detected_person) {

        chips.push(`
            <span class="info-chip">
                Person:
                ${escapeHtml(output.detected_person)}
            </span>
        `);

    }

    if (output.detected_time_range) {

        chips.push(`
            <span class="info-chip">
                ${escapeHtml(output.detected_time_range[0])}
                →
                ${escapeHtml(output.detected_time_range[1])}
            </span>
        `);

    }

    if (output.filters_relaxed) {

        chips.push(`
            <span class="warning-chip">
                Filters relaxed
            </span>
        `);

    }

    el.detected.innerHTML = chips.join("");

}


/* ---------------------------------------------------------
   Context renderer
--------------------------------------------------------- */

function renderContext(context, message) {

    if (!Array.isArray(context) || context.length === 0) {

        return `
            <div class="no-context">
                No surrounding context available.
            </div>
        `;

    }

    return context.map((item) => {

        const isCenter =
            item.id === message.id;

        return `

            <div class="context-line ${isCenter ? "center" : ""}">

                <span class="context-sender">
                    ${escapeHtml(item.sender)}
                </span>

                <span class="context-text">
                    ${escapeHtml(item.text)}
                </span>

            </div>

        `;

    }).join("");

}


/* ---------------------------------------------------------
   Result renderer
--------------------------------------------------------- */

function renderResults(output) {

    el.results.innerHTML = "";

    if (output.no_relevant_message) {

        el.resultCount.textContent =
            "0 results";

        el.results.innerHTML = `

            <div class="empty-state">

                <div class="empty-icon">
                    ⌕
                </div>

                <h3>
                    No sufficiently relevant message found
                </h3>

                <p>
                    Try changing your wording or removing a filter.
                </p>

            </div>

        `;

        return;
    }


    const resultList =
        Array.isArray(output.results)
            ? output.results
            : [];


    el.resultCount.textContent =
        `${resultList.length} result${resultList.length === 1 ? "" : "s"}`;


    resultList.forEach((result, index) => {

        const message = result.message;

        const card =
            document.createElement("article");

        card.className = "result-card";


        const similarity =
            typeof result.score === "number"
                ? `${(result.score * 100).toFixed(0)}%`
                : "—";


        card.innerHTML = `

            <div class="result-rank">
                #${index + 1}
            </div>


            <div class="result-top">

                <div class="message-meta">

                    <span class="sender-name">
                        ${escapeHtml(message.sender)}
                    </span>

                    <span>
                        ${escapeHtml(message.timestamp)}
                    </span>

                    <span>
                        ${escapeHtml(
                            message.conversation_id || "general"
                        )}
                    </span>

                </div>

            </div>


            <div class="message-text">
                ${escapeHtml(message.text)}
            </div>


            <div class="result-tags">

                <span class="similarity-tag">
                    ${similarity} similarity
                </span>

                <span class="type-tag">
                    ${escapeHtml(message.type)}
                </span>

            </div>


            <details class="context-wrapper" open>

                <summary>
                    Conversation context
                    <span>⌄</span>
                </summary>

                <div class="context-list">

                    ${renderContext(
                        result.context,
                        message
                    )}

                </div>

            </details>

        `;

        el.results.appendChild(card);

    });

}


/* ---------------------------------------------------------
   Error renderer
--------------------------------------------------------- */

function renderError(message) {

    el.resultCount.textContent = "";

    el.detected.innerHTML = "";

    el.results.innerHTML = `

        <div class="empty-state error-state">

            <div class="empty-icon">
                !
            </div>

            <h3>
                Search failed
            </h3>

            <p>
                ${escapeHtml(message)}
            </p>

        </div>

    `;

}


/* ---------------------------------------------------------
   Search
--------------------------------------------------------- */

async function runSearch() {

    const query =
        el.q.value.trim();


    if (!query) {

        renderError(
            "Please enter a search query."
        );

        return;
    }


    document
        .querySelectorAll(".suggestion")
        .forEach((button) => {

            button.classList.toggle(
                "active",
                button.dataset.query === query
            );

        });


    const params =
        new URLSearchParams({

            q: query,

            top_k:
                el.topK.value || "5"

        });


    if (el.participant.value) {

        params.set(
            "participant",
            el.participant.value
        );

    }


    if (el.startDate.value) {

        params.set(
            "start_date",
            el.startDate.value
        );

    }


    if (el.endDate.value) {

        params.set(
            "end_date",
            el.endDate.value
        );

    }


    setLoading(query);


    try {

        const response =
            await fetch(
                `${API_BASE}/search?${params.toString()}`
            );


        if (!response.ok) {

            throw new Error(
                `Search failed with HTTP ${response.status}`
            );

        }


        const output =
            await response.json();


        renderDetected(output);

        renderResults(output);


    } catch (error) {

        renderError(
            error.message || "Search failed."
        );

    } finally {

        clearLoading();

    }

}


/* ---------------------------------------------------------
   Suggestion buttons
--------------------------------------------------------- */

document
    .querySelectorAll(".suggestion")
    .forEach((button) => {

        button.addEventListener(
            "click",
            () => {

                el.q.value =
                    button.dataset.query;

                runSearch();

            }
        );

    });


/* ---------------------------------------------------------
   Search button
--------------------------------------------------------- */

el.go.addEventListener(
    "click",
    runSearch
);


/* ---------------------------------------------------------
   Enter key
--------------------------------------------------------- */

el.q.addEventListener(
    "keydown",
    (event) => {

        if (event.key === "Enter") {

            runSearch();

        }

    }
);


/* ---------------------------------------------------------
   Home-page query support
--------------------------------------------------------- */

const urlParams =
    new URLSearchParams(window.location.search);

const initialQuery =
    urlParams.get("q");


if (initialQuery && el.q) {

    el.q.value =
        initialQuery;

}


/* ---------------------------------------------------------
   Start
--------------------------------------------------------- */

loadHealth();

loadParticipants();

runSearch();