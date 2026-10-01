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
  // "study" = FSRS queue, one card at a time from Python (getNextCard) and re-fetched after every
  // rating so an Again card can return this session. "browse" = the old stable list with
  // prev/next/random/show-all/filters, for editing and looking things up, not for scheduling.
  mode: "study",
  filters: {
    language: "Japanese",
    track: "reading",
    showAll: false,
    missingMeaningOnly: false,
    ambiguousOnly: false,
    shuffle: false,
  },
  // Cloze mode (Milestone 5) only - the currently displayed blanked-line card, and whether its
  // answer has been revealed yet. Deliberately not part of `detailCache` (see renderCloze()'s own
  // comment: a fresh random occurrence every time, never cached/pinned).
  currentCloze: null,
  clozeRevealed: false,
  // Selected {group, song} (null = all songs). Per language - songs differ between Japanese and
  // Korean, so switching language clears it. songs = the picker's data for the current language.
  song: null,
  songs: [],
  pickerGroup: null,
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
  const detail = await callApi("getCardDetail", card.vocabId, state.filters.language, state.song);
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

function setRatingRowVisible(visible) {
  document.querySelector(".rating-row").hidden = !visible;
}

// Labels each rating button with its real outcome ("Again <1m", "Good 10m", "Easy 8d") - what
// makes Easy visibly mean something. Fetched for the current card only; a stale response (card
// changed mid-flight) is dropped.
async function refreshIntervalLabels(card) {
  document.querySelectorAll(".rate-interval").forEach((s) => (s.textContent = ""));
  const preview = await callApi(
    "previewIntervals", card.vocabId, state.filters.language, state.filters.track
  );
  if (currentCard() !== card) return;
  document.querySelectorAll(".rate-btn").forEach((btn) => {
    btn.querySelector(".rate-interval").textContent = preview.labels[btn.dataset.rating] || "";
  });
}

// The clip belongs to the card it was started on - stop it the moment a different card (or the
// empty "done" screen) is shown. Keyed by language/vocabId so re-rendering the SAME card (after an
// Edit Meaning, say) doesn't cut its audio off. Fire-and-forget: a failed stop isn't worth a banner.
let lastRenderedCardKey = null;
function stopAudioIfCardChanged(card) {
  const key = card ? `${state.filters.language}:${card.vocabId}` : null;
  if (key !== lastRenderedCardKey) {
    lastRenderedCardKey = key;
    window.pywebview.api.stopAudio();
  }
}

async function render() {
  const cards = state.cards;
  stopAudioIfCardChanged(cards[state.index] || null);
  const prevBtn = el("prevBtn");
  const nextBtn = el("nextBtn");
  const randomBtn = el("randomCardBtn");

  if (!cards.length) {
    el("standardView").hidden = false;
    el("clozePanel").hidden = true;
    el("prompt").textContent = state.mode === "study"
      ? "All done for now 2014 nothing left to study."
      : "No words match these filters";
    el("warning").textContent = "";
    el("answerBody").textContent = "";
    el("positionLabel").textContent = "0 / 0";
    el("queueBadge").textContent = "";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    randomBtn.disabled = true;
    el("hanjaPicker").innerHTML = "";
    setRatingRowVisible(true);
    return;
  }

  const card = currentCard();
  updateSuspendButton();
  highlightSongWord();
  el("positionLabel").textContent = `${state.index + 1} / ${cards.length}`;
  el("queueBadge").textContent = state.mode === "study" ? (card.queueType || "").toUpperCase() : "";
  prevBtn.disabled = state.index <= 0;
  nextBtn.disabled = state.index >= cards.length - 1;
  randomBtn.disabled = cards.length <= 1;

  if (state.filters.track === "cloze") {
    el("standardView").hidden = true;
    el("clozePanel").hidden = false;
    await renderCloze(card);
  } else {
    el("clozePanel").hidden = true;
    el("standardView").hidden = false;
    setRatingRowVisible(true);
    await renderStandard(card);
  }
}

async function renderStandard(card) {
  el("prompt").textContent = card.surface || card.lemma;
  el("warning").textContent = hasNoMeaning(card)
    ? "⚠ No meaning found - fill one in with Edit Meaning"
    : "";

  renderHanjaPicker(card);
  refreshIntervalLabels(card);

  const detail = await getDetail(card);
  el("answerBody").textContent = formatBody(card, detail);
  renderOccurrenceRow(detail);

  const bridge = await callApi("getCognateBridge", card.vocabId, state.filters.language);
  renderCognateBridge(bridge);
}

// Cloze (Milestone 5): unlike getCardDetail(), deliberately NOT cached - "a random real
// occurrence each review" (confirmed with the user) means even re-visiting the same card via
// Prev/Next should be free to show a different real line next time, not a pinned one.
async function renderCloze(card) {
  state.clozeRevealed = false;
  state.currentCloze = null;
  el("clozeAnswer").hidden = true;
  el("clozeUnavailable").hidden = true;
  el("clozeLine").textContent = "...";
  setRatingRowVisible(false);

  const clozeCard = await callApi(
    "getClozeCardDetail", card.vocabId, state.filters.language, card.lemma, state.song
  );
  // The card may have changed (Prev/Next clicked again) while this await was in flight.
  if (currentCard() !== card) return;

  state.currentCloze = clozeCard;
  if (!clozeCard) {
    el("clozeLine").textContent = "";
    el("clozeUnavailable").hidden = false;
    el("revealBtn").hidden = true;
    return;
  }
  el("revealBtn").hidden = false;
  el("clozeLine").textContent = clozeCard.blankedLine;
}

function revealCloze() {
  const c = state.currentCloze;
  if (!c || state.clozeRevealed) return;
  state.clozeRevealed = true;
  el("revealBtn").hidden = true;

  // Japanese only - Korean's Hangul answer already shows its own pronunciation directly, no
  // separate reading needed. The queue card already carries the word's dictionary-form reading
  // (card.lemmaReading, same field the standard Reading/Meaning cards already show) - reused
  // as-is rather than fetched again. Skipped when it's identical to the answer's own text (a
  // kana-only word has nothing extra to add).
  const card = currentCard();
  const reading = card && card.lemmaReading && card.lemmaReading !== c.answerSurface
    ? ` (${card.lemmaReading})`
    : "";
  const parts = [`Answer: ${c.answerSurface}${reading}`];
  if (c.gloss) parts.push(c.gloss);
  if (c.group && c.song) parts.push(`From ${c.group} — ${c.song}`);
  el("clozeAnswer").innerHTML = parts.map((p) => `<div>${p}</div>`).join("");
  el("clozeAnswer").hidden = false;

  setRatingRowVisible(true);
  refreshIntervalLabels(card);
}

function _legHtml(title, bodyHtml) {
  if (!bodyHtml) {
    return `<div class="cognate-leg empty"><h4>${title}</h4>(none)</div>`;
  }
  return `<div class="cognate-leg"><h4>${title}</h4>${bodyHtml}</div>`;
}

function renderCognateBridge(bridge) {
  const container = el("cognateBridge");
  if (!bridge || !bridge.cognateForm) {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }
  container.hidden = false;

  const ja = bridge.japanese;
  const jaHtml = ja ? `${ja.lemma} <span class="control-label">(${ja.lemmaReading || ""})</span>` : null;

  const ko = bridge.korean;
  let koHtml = null;
  if (ko) {
    const rows = (ko.candidates || []).map((c) => {
      const gloss = (c.gloss || []).slice(0, 2).join("; ");
      const mark = c.linked ? " &#9733;" : "";
      return `${c.hanjaForm} (${c.pinyin || ""})${mark} — ${gloss}`;
    });
    koHtml = `${ko.lemma}<br>` + rows.join("<br>");
  } else if (bridge.predictedKorean) {
    // No real Korean word from the user's own vocab is linked - show the predicted hangul,
    // labelled as such. "plausible" = passed the checks but isn't in Korean Wiktionary, so also
    // flagged unconfirmed. The Korean gloss lets you spot meaning drift (大丈夫 -> 대장부).
    const p = bridge.predictedKorean;
    const gloss = (p.gloss || []).slice(0, 1).join("; ");
    const note = p.tier === "plausible" ? "predicted, unconfirmed" : "predicted";
    koHtml = `${p.hangul} <span class="control-label">(${note} · ${p.hanja})</span>`
      + (gloss ? `<br>${gloss}` : "");
  }

  const zh = bridge.chinese;
  const zhHtml = zh
    ? `${(zh.gloss || []).slice(0, 2).join("; ")} <span class="control-label">(${zh.pinyin || ""})</span>`
      + (zh.status === "not_attested" ? " <em>(not attested - Japan-coined)</em>" : "")
    : null;

  container.innerHTML =
    _legHtml("Japanese", jaHtml) + _legHtml("Chinese source", zhHtml) + _legHtml("Korean", koHtml);
}

let lastShownOccurrence = null;   // guards the async timing redraw against a card change

function renderOccurrenceRow(detail) {
  const row = el("occurrenceRow");
  const occ = detail && detail.occurrence;
  if (!occ) {
    lastShownOccurrence = null;
    row.hidden = true;
    return;
  }
  row.hidden = false;
  el("occurrenceSource").textContent = `From ${occ.group} — ${occ.song}`;

  // Full lyric as clickable words. Drawn at once from the plain character-count estimate
  // (karaoke.js), then redrawn from the pause-aware timing (core/karaoke_timing.py: label-row
  // pauses + reading-based beats) as soon as the backend answers. Pieces reassemble to the exact
  // text either way, so line breaks survive (CSS white-space: pre-wrap) and nothing is truncated.
  const locale = state.filters.language === "Korean" ? "ko" : "ja";
  drawLyricLine(Karaoke.segmentLine(occ.lyricLine, locale));

  const requested = occ;
  callApi("getLineTiming", occ.group, occ.song, occ.startChunk, occ.endChunk, occ.lyricLine,
    occ.singers || [], state.filters.language, occ.lyricId || null).then((timing) => {
    if (timing && timing.pieces && requested === lastShownOccurrence) drawLyricLine(timing.pieces);
  }).catch(() => { /* keep the estimate drawn above; callApi already surfaced the error */ });
  lastShownOccurrence = occ;
}

function drawLyricLine(pieces) {
  const line = el("occurrenceLine");
  line.textContent = "";
  let lastFraction = 0;
  for (const seg of pieces) {
    if (!seg.isWord) {
      line.appendChild(document.createTextNode(seg.text));
      continue;
    }
    const span = document.createElement("span");
    span.className = "tok";
    span.textContent = seg.text;
    span.title = "Play from here (estimated position)";
    span.addEventListener("click", () => playOccurrenceAudio(seg.fraction));
    line.appendChild(span);
    lastFraction = seg.fraction;
  }
  el("playLastWordBtn").dataset.fraction = String(lastFraction);
}

// Pause editor: add/remove the author's pause markers on the lyric being shown, saved straight to that
// song's lyrics file (core/lyric_file.py) - works for any song, not only the one the Tk app has open. The
// visible stand-in below must match core.lyric_text.EDITOR_PAUSE_GLYPH (the stored marker is invisible).
const PAUSE_GLYPH = String.fromCharCode(0x25BE);

function showPauseEditError(message) {
  const box = el("pauseEditError");
  box.textContent = message || "";
  box.hidden = !message;
}

async function openPauseEditor() {
  const occ = lastShownOccurrence;
  if (!occ) return;
  // Called directly (not callApi): a failure belongs in the dialog, not the page banner behind it.
  const res = await window.pywebview.api.getLyricForPauseEdit(
    occ.group, occ.song, occ.lyricId || null, occ.lyricLine);
  if (!res.ok) {
    showError(res.error);
    return;
  }
  el("pauseEditSource").textContent = `${occ.group} — ${occ.song}`;
  el("pauseEditInput").value = res.data.text;
  showPauseEditError("");
  el("pauseEditDialog").showModal();
  el("pauseEditInput").focus();
}

function insertPauseGlyph() {
  const box = el("pauseEditInput");
  box.setRangeText(PAUSE_GLYPH, box.selectionStart, box.selectionEnd, "end");
  box.focus();
}

async function savePauseEditor() {
  const occ = lastShownOccurrence;
  if (!occ) return;
  const res = await window.pywebview.api.savePauseMarkers(
    occ.group, occ.song, occ.lyricId || null, occ.lyricLine, el("pauseEditInput").value);
  if (!res.ok) {
    showPauseEditError(res.error);       // keep the dialog open so the edit isn't lost
    return;
  }
  el("pauseEditDialog").close();
  const card = currentCard();
  const detail = card ? await getDetail(card) : null;
  renderOccurrenceRow(detail);           // re-fetches the timing, which now sees the new markers
}

async function playOccurrenceAudio(startFraction = 0) {
  const card = currentCard();
  if (!card) return;
  const detail = await getDetail(card);
  const occ = detail && detail.occurrence;
  if (!occ) return;
  await callApi("playOccurrenceAudio", occ.group, occ.song, occ.startChunk, occ.endChunk,
    startFraction);
}

async function loadNextStudyCard() {
  const f = state.filters;
  const card = await callApi("getNextCard", f.language, f.track, state.song, f.shuffle);
  state.cards = card ? [card] : [];
  state.index = 0;
  state.detailCache = {};
  await render();
}

function applyModeVisibility() {
  const browse = state.mode === "browse";
  const triage = state.mode === "triage";
  el("browseFilters").hidden = !browse;
  el("browseNav").hidden = !browse;
  el("reshuffleBtn").hidden = !browse;
  if (!browse) el("songWords").hidden = true;
  el("triagePanel").hidden = !triage;
  // The study-card chrome is meaningless on the triage list.
  el("queueBadge").hidden = triage;
  el("studyActions").hidden = triage;
  if (triage) {
    el("standardView").hidden = true;
    el("clozePanel").hidden = true;
    setRatingRowVisible(false);
  }
}

// ---- Triage: bulk "I already know this" / suspend for the most frequent untouched words ----
async function loadTriage() {
  const rows = await callApi("listTriageCandidates", state.filters.language, 150);
  const list = el("triageList");
  list.innerHTML = "";
  for (const r of rows) {
    const row = document.createElement("label");
    row.className = "triage-row";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.dataset.vocabId = r.vocabId;
    row.appendChild(box);
    const cells = [r.surface || r.lemma, r.reading || "", r.gloss || "", `${r.occurrences}\u00d7`];
    ["lemma", "reading", "gloss", "count"].forEach((cls, i) => {
      const span = document.createElement("span");
      span.className = cls;
      span.textContent = cells[i];
      row.appendChild(span);
    });
    list.appendChild(row);
  }
  el("triageCount").textContent = rows.length ? `${rows.length} words` : "Nothing left to triage.";
}

function triageSelectedIds() {
  return Array.from(document.querySelectorAll("#triageList input:checked"))
    .map((b) => Number(b.dataset.vocabId));
}

async function triageApply(apiName, ...extra) {
  const ids = triageSelectedIds();
  if (!ids.length) return;
  await callApi(apiName, ids, state.filters.language, ...extra);
  await loadTriage();
}

async function markCurrentKnown() {
  const card = currentCard();
  if (!card) return;
  await callApi("markKnown", [card.vocabId], state.filters.language);
  await afterCardHandled(card, false);
}

async function toggleSuspendCurrent() {
  const card = currentCard();
  if (!card) return;
  const suspend = !card.suspended;
  await callApi("setSuspended", [card.vocabId], state.filters.language, suspend);
  await afterCardHandled(card, suspend);
}

// Study: the card is out of the queue either way, so fetch the next one. Browse: keep it on
// screen and just reflect the new flag.
async function afterCardHandled(card, suspended) {
  if (state.mode === "study") {
    await loadNextStudyCard();
  } else {
    card.suspended = suspended;
    updateSuspendButton();
  }
}

function updateSuspendButton() {
  const card = currentCard();
  el("suspendBtn").textContent = card && card.suspended ? "Unsuspend" : "Suspend";
}

async function loadQueue() {
  applyModeVisibility();
  if (state.mode === "triage") return loadTriage();
  if (state.mode === "study") return loadNextStudyCard();
  const f = state.filters;
  let cards = await callApi(
    "listQueue", f.language, f.track, f.showAll, f.missingMeaningOnly, f.ambiguousOnly, state.song
  );
  if (f.shuffle) {
    cards = shuffleArray(cards);
  }
  state.cards = cards;
  state.index = 0;
  state.detailCache = {};
  renderSongWords();
  await render();
}

// ---- Song picker: Group -> Song ----
function songLabel(s) {
  return `${s.group} — ${s.song}`;
}

function updateSongBar() {
  const s = state.song;
  el("songPickBtn").textContent = s ? `${songLabel(s)} ▾` : "All songs ▾";
  el("songClearBtn").hidden = !s;
}

async function loadSongs() {
  state.songs = await callApi("listSongs", state.filters.language);
  const stillThere = state.song && state.songs.some(
    (s) => s.group === state.song.group && s.song === state.song.song);
  if (!stillThere) state.song = null;
  updateSongBar();
}

function songItem(main, sub, count, active, onClick) {
  const row = document.createElement("div");
  row.className = "song-item" + (active ? " active" : "");
  const left = document.createElement("span");
  left.textContent = main;
  if (sub) {
    const subEl = document.createElement("span");
    subEl.className = "sub";
    subEl.textContent = ` ${sub}`;
    left.appendChild(subEl);
  }
  const cnt = document.createElement("span");
  cnt.className = "count";
  cnt.textContent = count;
  row.append(left, cnt);
  row.addEventListener("click", onClick);
  return row;
}

function renderSongPicker() {
  const query = el("songSearch").value.trim().toLowerCase();
  const groupsEl = el("songGroups");
  const listEl = el("songList");
  groupsEl.textContent = "";
  listEl.textContent = "";

  const groups = new Map();
  for (const s of state.songs) {
    groups.set(s.group, (groups.get(s.group) || 0) + s.wordCount);
  }
  const selectGroup = (g) => {
    state.pickerGroup = g;
    el("songSearch").value = "";
    renderSongPicker();
  };

  groupsEl.appendChild(songItem("All songs", "", "", !state.song && !state.pickerGroup,
    () => chooseSong(null)));
  for (const [g, n] of groups) {
    groupsEl.appendChild(songItem(g, "", `${n}`, !query && state.pickerGroup === g,
      () => selectGroup(g)));
  }

  // Searching spans every group (flat "Group - Song" results); otherwise show the chosen group.
  const matches = query
    ? state.songs.filter((s) => songLabel(s).toLowerCase().includes(query))
    : state.songs.filter((s) => s.group === state.pickerGroup);
  if (!query && !state.pickerGroup) {
    const hint = document.createElement("div");
    hint.className = "song-item sub";
    hint.textContent = "Pick a group, or type to search.";
    listEl.appendChild(hint);
  }
  for (const s of matches) {
    const active = state.song && state.song.group === s.group && state.song.song === s.song;
    listEl.appendChild(songItem(s.song, query ? s.group : "", `${s.wordCount} words`, active,
      () => chooseSong({ group: s.group, song: s.song })));
  }
}

function openSongPicker() {
  state.pickerGroup = state.song ? state.song.group : (state.pickerGroup || null);
  el("songSearch").value = "";
  renderSongPicker();
  el("songDialog").showModal();
  el("songSearch").focus();
}

function chooseSong(song) {
  state.song = song;
  el("songDialog").close();
  updateSongBar();
  loadQueue();
}

// Word index for the selected song (Browse): every word in lyric order; click to jump to it.
function renderSongWords() {
  const box = el("songWords");
  const show = state.mode === "browse" && state.song;
  box.hidden = !show;
  if (!show) return;
  el("songWordsSummary").textContent = `Words in ${state.song.song} (${state.cards.length})`;
  const list = el("songWordsList");
  list.textContent = "";
  state.cards.forEach((card, i) => {
    const chip = document.createElement("span");
    chip.className = "song-chip";
    chip.textContent = card.surface || card.lemma;
    chip.title = card.lemma;
    chip.dataset.index = String(i);
    chip.addEventListener("click", () => { state.index = i; render(); });
    list.appendChild(chip);
  });
}

function highlightSongWord() {
  document.querySelectorAll(".song-chip").forEach((chip) => {
    const current = Number(chip.dataset.index) === state.index;
    chip.classList.toggle("current", current);
    if (current) chip.scrollIntoView({ block: "nearest" });
  });
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
  window.pywebview.api.stopAudio();  // even if the same card comes straight back (Again)
  await callApi("rate", card.vocabId, state.filters.language, state.filters.track, rating);
  if (state.mode === "study") await loadNextStudyCard();
  else goNext();
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

function openEditSpellingDialog() {
  const card = currentCard();
  if (!card) return;
  el("editSpellingInput").value = card.surface || card.lemma;
  el("editSpellingDialog").showModal();
}

async function saveEditSpelling() {
  const card = currentCard();
  if (!card) return;
  const value = el("editSpellingInput").value.trim();
  el("editSpellingDialog").close();
  await callApi("updateSurface", card.vocabId, state.filters.language, value);
  card.surface = value || card.lemma;
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
  if (state.mode === "study") {
    await loadNextStudyCard();
    return;
  }
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
      state.song = null;
      state.pickerGroup = null;
      updateAmbiguousFilterVisibility();
      loadSongs().then(loadQueue);
    })
  );
  document.querySelectorAll('input[name="mode"]').forEach((r) =>
    r.addEventListener("change", (e) => {
      state.mode = e.target.value;
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

  el("songPickBtn").addEventListener("click", openSongPicker);
  el("songClearBtn").addEventListener("click", () => chooseSong(null));
  el("songDialogClose").addEventListener("click", () => el("songDialog").close());
  el("songSearch").addEventListener("input", renderSongPicker);
  el("songSearch").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const first = document.querySelector("#songList .song-item:not(.sub)");
    if (first) first.click();
  });
  el("reshuffleBtn").addEventListener("click", loadQueue);

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
  el("editSpellingBtn").addEventListener("click", openEditSpellingDialog);
  el("editSpellingCancel").addEventListener("click", () => el("editSpellingDialog").close());
  el("editSpellingSave").addEventListener("click", saveEditSpelling);

  el("deleteBtn").addEventListener("click", openDeleteDialog);
  el("deleteCancel").addEventListener("click", () => el("deleteConfirmDialog").close());
  el("deleteConfirm").addEventListener("click", confirmDelete);

  el("reloadBtn").addEventListener("click", loadQueue);

  el("playAudioBtn").addEventListener("click", () => playOccurrenceAudio(0));
  el("editPausesBtn").addEventListener("click", openPauseEditor);
  el("pauseEditInsert").addEventListener("click", insertPauseGlyph);
  el("pauseEditPlay").addEventListener("click", () => playOccurrenceAudio(0));
  el("pauseEditCancel").addEventListener("click", () => el("pauseEditDialog").close());
  el("pauseEditSave").addEventListener("click", savePauseEditor);
  el("pauseEditInput").addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.code === "Space") {   // same hotkey as the Tk Lyric Editor
      event.preventDefault();
      insertPauseGlyph();
    }
  });
  el("playLastWordBtn").addEventListener("click",
    () => playOccurrenceAudio(Number(el("playLastWordBtn").dataset.fraction || 0)));

  el("knownBtn").addEventListener("click", markCurrentKnown);
  el("suspendBtn").addEventListener("click", toggleSuspendCurrent);
  el("triageKnown").addEventListener("click", () => triageApply("markKnown"));
  el("triageSuspend").addEventListener("click", () => triageApply("setSuspended", true));
  const setAllTriage = (v) =>
    document.querySelectorAll("#triageList input").forEach((b) => (b.checked = v));
  el("triageSelectAll").addEventListener("click", () => setAllTriage(true));
  el("triageSelectNone").addEventListener("click", () => setAllTriage(false));

  el("revealBtn").addEventListener("click", revealCloze);
}

window.addEventListener("pywebviewready", () => {
  wireControls();
  updateAmbiguousFilterVisibility();
  loadSongs().then(loadQueue);
});
