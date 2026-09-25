// Front end for the PyWebView vocab review screen (Milestone 1 of
// .claude/FLASHCARD_WEB_UPGRADE_PLAN.md). Mirrors the old gui/vocab_review.py's behavior:
// JS holds {cards, index, detailCache, filters} and Prev/Next just move `index` and re-render, no
// round trip to Python - the same "stable list, not a destructively-popped queue" design the old
// screen's own docstring called out. Python (gui/vocab_review_api.py) holds no session state;
// every Api call is a fresh, independent DB call.

const state = {
  cards: [],
  index: 0,
  detailCache: {},
  filters: {
    language: "Japanese",
    track: "reading",
    showAll: false,
    missingMeaningOnly: false,
    ambiguousOnly: false,
    shuffle: false,
  },
};

// Fisher-Yates - reorders whatever cards listQueue() returned (respecting every existing filter,
// so shuffle "just works" on any current or future range/filter combination, e.g. a later
// struggling-words filter) without needing any backend change.
function shuffleArray(arr) {
  const copy = arr.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function el(id) {
  return document.getElementById(id);
}

function showError(message) {
  const banner = el("error-banner");
  banner.textContent = message;
  banner.hidden = false;
}

function clearError() {
  el("error-banner").hidden = true;
}

async function callApi(name, ...args) {
  const res = await window.pywebview.api[name](...args);
  if (!res.ok) {
    showError(res.error);
    throw new Error(res.error);
  }
  clearError();
  return res.data;
}

function currentCard() {
  return state.cards[state.index] || null;
}

function hasNoMeaning(card) {
  const meaning = card.meaning || {};
  return meaning.status !== "found";
}

function formatBody(card, detail) {
  const lines = [];
  const meaning = card.meaning || {};
  const language = state.filters.language;

  if (language === "Japanese") {
    lines.push(`Reading: ${card.lemmaReading || ""}`);
    if (meaning.status === "found") {
      lines.push(`Meaning: ${(meaning.gloss || []).slice(0, 3).join("; ")}`);
    }
    if (card.cognateForm) {
      lines.push(`Chinese cognate: ${card.cognateForm} (${card.cognatePinyin || ""})`);
    }
  } else {
    if (meaning.status === "found") {
      lines.push(`Meaning: ${(meaning.gloss || []).slice(0, 3).join("; ")}`);
    }
    for (const h of card.hanjaCandidates || []) {
      const gloss = (h.gloss || []).slice(0, 2).join("; ");
      lines.push(`Hanja: ${h.hanja} (${h.pinyin || ""}) — ${gloss}`);
    }
  }

  if (detail && detail.linked && detail.linked.length) {
    const other = language === "Japanese" ? "Korean" : "Japanese";
    lines.push("");
    lines.push(`${other} cognate cousins: ` + detail.linked.map((l) => l.lemma).join(", "));
  }

  return lines.join("\n");
}

async function getDetail(card) {
  const cacheKey = `${state.filters.language}:${card.vocabId}`;
  if (state.detailCache[cacheKey]) return state.detailCache[cacheKey];
  const detail = await callApi("getCardDetail", card.vocabId, state.filters.language);
  state.detailCache[cacheKey] = detail;
  return detail;
}

function renderHanjaPicker(card) {
  const container = el("hanjaPicker");
  container.innerHTML = "";
  if (state.filters.language !== "Korean") return;

  const candidates = card.hanjaCandidates || [];
  if (!candidates.length) return;

  if (candidates.length >= 2) {
    const note = document.createElement("div");
    note.textContent = "Multiple Hanja candidates - keep only the right one:";
    container.appendChild(note);

    for (const c of candidates) {
      const row = document.createElement("div");
      row.className = "hanja-candidate-row";

      const label = document.createElement("span");
      const gloss = (c.gloss || []).slice(0, 1).join("; ");
      label.textContent = `${c.hanja} (${c.pinyin || ""}) — ${gloss}`;

      const btn = document.createElement("button");
      btn.textContent = "Keep only this";
      btn.addEventListener("click", () => keepOnlyHanja(c.hanjaId));

      row.appendChild(label);
      row.appendChild(btn);
      container.appendChild(row);
    }
  }

  // Shown even for a single candidate - matches the old Tkinter screen's reasoning: a word with
  // only one (wrong) candidate isn't "ambiguous", it's just wrong, and "keep only this" has
  // nothing to contrast against for that case.
  const removeBtn = document.createElement("button");
  removeBtn.textContent = "Remove Hanja entirely (native word)";
  removeBtn.style.marginTop = "6px";
  removeBtn.addEventListener("click", clearHanja);
  container.appendChild(removeBtn);
}

async function render() {
  const cards = state.cards;
  const prevBtn = el("prevBtn");
  const nextBtn = el("nextBtn");
  const randomBtn = el("randomCardBtn");

  if (!cards.length) {
    el("prompt").textContent = "No words match these filters";
    el("warning").textContent = "";
    el("answerBody").textContent = "";
    el("positionLabel").textContent = "0 / 0";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    randomBtn.disabled = true;
    el("hanjaPicker").innerHTML = "";
    return;
  }

  const card = currentCard();
  el("prompt").textContent = card.surface || card.lemma;
  el("warning").textContent = hasNoMeaning(card)
    ? "⚠ No meaning found - fill one in with Edit Meaning"
    : "";
  el("positionLabel").textContent = `${state.index + 1} / ${cards.length}`;
  prevBtn.disabled = state.index <= 0;
  nextBtn.disabled = state.index >= cards.length - 1;
  randomBtn.disabled = cards.length <= 1;

  renderHanjaPicker(card);

  const detail = await getDetail(card);
  el("answerBody").textContent = formatBody(card, detail);
  renderOccurrenceRow(detail);
}

function renderOccurrenceRow(detail) {
  const row = el("occurrenceRow");
  const occ = detail && detail.occurrence;
  if (!occ) {
    row.hidden = true;
    return;
  }
  row.hidden = false;
  el("occurrenceLine").textContent = `From ${occ.group} — ${occ.song}: ${occ.lyricLine}`;
}

async function playOccurrenceAudio() {
  const card = currentCard();
  if (!card) return;
  const detail = await getDetail(card);
  const occ = detail && detail.occurrence;
  if (!occ) return;
  await callApi("playOccurrenceAudio", occ.group, occ.song, occ.startChunk, occ.endChunk);
}

async function loadQueue() {
  const f = state.filters;
  let cards = await callApi(
    "listQueue", f.language, f.track, f.showAll, f.missingMeaningOnly, f.ambiguousOnly
  );
  if (f.shuffle) {
    cards = shuffleArray(cards);
  }
  state.cards = cards;
  state.index = 0;
  state.detailCache = {};
  await render();
}

function goPrev() {
  if (state.index > 0) {
    state.index -= 1;
    render();
  }
}

function goNext() {
  if (state.index < state.cards.length - 1) {
    state.index += 1;
    render();
  }
}

function goRandom() {
  const count = state.cards.length;
  if (count <= 1) return;
  // Avoid landing back on the same card twice in a row - a same-index reroll would look broken.
  let next;
  do {
    next = Math.floor(Math.random() * count);
  } while (next === state.index);
  state.index = next;
  render();
}

async function rate(rating) {
  const card = currentCard();
  if (!card) return;
  await callApi("rate", card.vocabId, state.filters.language, state.filters.track, rating);
  goNext();
}

async function keepOnlyHanja(hanjaId) {
  const card = currentCard();
  await callApi("keepOnlyHanja", card.vocabId, hanjaId);
  const kept = (card.hanjaCandidates || []).find((c) => c.hanjaId === hanjaId);
  card.hanjaCandidates = kept ? [kept] : [];
  render();
}

async function clearHanja() {
  const card = currentCard();
  await callApi("clearHanja", card.vocabId);
  card.hanjaCandidates = [];
  render();
}

function openEditMeaningDialog() {
  const card = currentCard();
  if (!card) return;
  const meaning = card.meaning || {};
  el("editMeaningInput").value = (meaning.gloss || []).join("; ");
  el("editMeaningDialog").showModal();
}

async function saveEditMeaning() {
  const card = currentCard();
  if (!card) return;
  const raw = el("editMeaningInput").value;
  const glossList = raw.split(";").map((s) => s.trim()).filter(Boolean);
  el("editMeaningDialog").close();
  if (!glossList.length) return;

  await callApi("updateMeaning", card.vocabId, state.filters.language, glossList);
  card.meaning = { status: "found", pos: (card.meaning || {}).pos, gloss: glossList };
  delete state.detailCache[`${state.filters.language}:${card.vocabId}`];
  render();
}

function openDeleteDialog() {
  const card = currentCard();
  if (!card) return;
  el("deleteConfirmText").textContent = `Permanently delete "${card.surface || card.lemma}"?`;
  el("deleteConfirmDialog").showModal();
}

async function confirmDelete() {
  const card = currentCard();
  el("deleteConfirmDialog").close();
  if (!card) return;

  await callApi("deleteWord", card.vocabId, state.filters.language);
  state.cards.splice(state.index, 1);
  if (state.index >= state.cards.length) {
    state.index = Math.max(0, state.cards.length - 1);
  }
  render();
}

function updateAmbiguousFilterVisibility() {
  el("ambiguousFilterRow").dataset.hidden = state.filters.language !== "Korean";
}

function wireControls() {
  document.querySelectorAll('input[name="language"]').forEach((r) =>
    r.addEventListener("change", (e) => {
      state.filters.language = e.target.value;
      updateAmbiguousFilterVisibility();
      loadQueue();
    })
  );
  document.querySelectorAll('input[name="track"]').forEach((r) =>
    r.addEventListener("change", (e) => {
      state.filters.track = e.target.value;
      loadQueue();
    })
  );
  el("showAll").addEventListener("change", (e) => {
    state.filters.showAll = e.target.checked;
    loadQueue();
  });
  el("missingMeaningOnly").addEventListener("change", (e) => {
    state.filters.missingMeaningOnly = e.target.checked;
    loadQueue();
  });
  el("ambiguousOnly").addEventListener("change", (e) => {
    state.filters.ambiguousOnly = e.target.checked;
    loadQueue();
  });
  el("shuffleOrder").addEventListener("change", (e) => {
    state.filters.shuffle = e.target.checked;
    loadQueue();
  });

  el("prevBtn").addEventListener("click", goPrev);
  el("nextBtn").addEventListener("click", goNext);
  el("randomCardBtn").addEventListener("click", goRandom);
  document.addEventListener("keydown", (e) => {
    const tag = e.target.tagName;
    if (tag === "TEXTAREA" || tag === "INPUT") return;
    if (e.key === "ArrowLeft") goPrev();
    if (e.key === "ArrowRight") goNext();
  });

  document.querySelectorAll(".rate-btn").forEach((btn) =>
    btn.addEventListener("click", () => rate(btn.dataset.rating))
  );

  el("editMeaningBtn").addEventListener("click", openEditMeaningDialog);
  el("editMeaningCancel").addEventListener("click", () => el("editMeaningDialog").close());
  el("editMeaningSave").addEventListener("click", saveEditMeaning);

  el("deleteBtn").addEventListener("click", openDeleteDialog);
  el("deleteCancel").addEventListener("click", () => el("deleteConfirmDialog").close());
  el("deleteConfirm").addEventListener("click", confirmDelete);

  el("reloadBtn").addEventListener("click", loadQueue);

  el("playAudioBtn").addEventListener("click", playOccurrenceAudio);
}

window.addEventListener("pywebviewready", () => {
  wireControls();
  updateAmbiguousFilterVisibility();
  loadQueue();
});
