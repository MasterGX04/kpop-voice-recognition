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
    stopPlayback();
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

  // Lit words take the first singer's colour from groups.json (the CSS default blue when unknown).
  el("occurrenceLine").style.removeProperty("--singer-color");
  callApi("getMemberColor", occ.group, occ.singers || []).then((color) => {
    if (color && requested === lastShownOccurrence) el("occurrenceLine").style.setProperty("--singer-color", color);
  }).catch(() => {});
}

function drawLyricLine(pieces) {
  const line = el("occurrenceLine");
  line.textContent = "";
  karaokeWords = [];
  let lastFraction = 0;
  let lastExact = false;
  for (const seg of pieces) {
    if (!seg.isWord) {
      line.appendChild(document.createTextNode(seg.text));
      continue;
    }
    const span = document.createElement("span");
    span.className = "tok";
    span.textContent = seg.text;
    span.title = "Play from here (estimated position)";
    // playFraction (pause-aware timing) already includes a lead-in that never reaches back into the
    // previous line; the plain-estimate fallback has none, so it gets the default lead.
    const exact = seg.playFraction !== undefined;
    const from = exact ? seg.playFraction : seg.fraction;
    span.addEventListener("click", () => {
      if (String(window.getSelection()).length) return;   // the user is selecting text to copy, not asking to play
      playOccurrenceAudio(from, exact);
    });
    line.appendChild(span);
    // Only words with real chunk info (core/karaoke_timing) can light up; the plain-estimate fallback has none.
    karaokeWords.push({ span, startChunk: seg.startChunk, lit: false });
    lastFraction = from;
    lastExact = exact;
  }
  el("playLastWordBtn").dataset.fraction = String(lastFraction);
  el("playLastWordBtn").dataset.exact = lastExact ? "1" : "";
  if (karaokePlay) paintKaraoke(currentKaraokeChunk());   // a redraw mid-clip (timing arrived late) keeps the lit state
}

// Karaoke highlight (K1 + K2, .claude/KARAOKE_PLAN.md). The audio plays in the Python process, so the page runs its
// own clock from what playOccurrenceAudio returns (Karaoke.chunkAt). Each word lights when the clock reaches its
// start chunk and stays lit until the clip ends. Row starts are exact (hand-timed labels); a word inside a row is
// an even-tempo estimate - that is what the "Edit pauses" route is for. `?debug` shows the live chunk readout.
const KARAOKE_DEBUG = /[?&]debug\b/.test(location.search);
let karaokeWords = [];         // [{span, startChunk, lit}] for the line on screen
let karaokePlay = null;        // {play, startedAt} while a clip is running
let karaokeFrame = 0;
let karaokeOnFrame = null;     // called with the clock's chunk (null when stopped/over) on every repaint
let karaokeOnEnd = null;       // called once when a clip ends on its own (never when it is stopped)

function currentKaraokeChunk() {
  return karaokePlay ? Karaoke.chunkAt(karaokePlay.play, performance.now() - karaokePlay.startedAt) : null;
}

function paintKaraoke(chunk) {
  for (const w of karaokeWords) {
    const lit = Karaoke.isLit(w.startChunk, chunk);
    if (lit !== w.lit) {                      // toggle only on a state change
      w.lit = lit;
      w.span.classList.toggle("lit", lit);
    }
  }
  if (karaokeOnFrame) karaokeOnFrame(chunk);
  el("occurrenceLine").classList.toggle("playing", chunk !== null);   // unsung words dim only while a clip runs
  const readout = el("karaokeDebug");
  if (readout && KARAOKE_DEBUG) {
    readout.hidden = false;
    readout.textContent = chunk === null ? "chunk -" : `chunk ${chunk.toFixed(1)}`;
  }
}

function karaokeTick() {
  const chunk = currentKaraokeChunk();
  paintKaraoke(chunk);
  if (chunk === null) {                       // clip over
    karaokePlay = null;
    karaokeFrame = 0;
    const done = karaokeOnEnd;                // the clip ran out by itself (not stopped): tell the tap tool
    karaokeOnEnd = null;
    if (done) done();
    return;
  }
  karaokeFrame = requestAnimationFrame(karaokeTick);
}

function startKaraoke(play, roundTripMs) {
  stopKaraoke();
  // When did the audio really start? The server stamped the moment play() ran (same machine, same epoch as
  // Date.now()), so the clock starts exactly there. Without a stamp, assume half the bridge round trip ago.
  const sinceStart = play.startedAtMs ? Math.max(0, Date.now() - play.startedAtMs) : roundTripMs / 2;
  karaokePlay = { play, startedAt: performance.now() - sinceStart };
  karaokeFrame = requestAnimationFrame(karaokeTick);
}

function stopKaraoke() {
  karaokeOnEnd = null;
  if (karaokeFrame) cancelAnimationFrame(karaokeFrame);
  karaokeFrame = 0;
  karaokePlay = null;
  paintKaraoke(null);
}

// Every way a clip can end early (new card, rating, dialog) goes through here.
function stopPlayback() {
  stopKaraoke();
  window.pywebview.api.stopAudio();
}

// Pause editor: add/remove the author's pause markers on the lyric being shown, saved straight to that
// song's lyrics file (core/lyric_file.py) - works for any song, not only the one the Tk app has open. The
// visible stand-in below must match core.lyric_text.EDITOR_PAUSE_GLYPH (the stored marker is invisible).
const PAUSE_GLYPH = String.fromCharCode(0x25BE);

// Live preview: which words land in which label row for the text as currently edited (not saved). A wrong
// guess shows up at once - a word in the wrong row, an empty row - and a ▾ where the singer pauses fixes it.
let pausePreviewTimer = null;

function schedulePausePreview() {
  clearTimeout(pausePreviewTimer);
  pausePreviewTimer = setTimeout(refreshPausePreview, 250);
}

async function refreshPausePreview() {
  const occ = lastShownOccurrence;
  const summary = el("pausePreviewSummary");
  const list = el("pausePreview");
  if (!occ) return;
  const res = await window.pywebview.api.previewPauseEdit(
    occ.group, occ.song, occ.startChunk, occ.endChunk, occ.singers || [], state.filters.language,
    el("pauseEditInput").value);
  list.textContent = "";
  if (!res.ok || !res.data) {
    summary.textContent = "";
    return;
  }
  const { rows, markers, rowCount } = res.data;
  const needed = rowCount - 1;
  const complete = markers === needed;
  summary.className = "pause-summary " + (complete ? "ok" : "guess");
  summary.textContent = complete
    ? `${rowCount} label rows · ${markers} pauses marked - every pause is marked, so these rows are exact.`
    : `${rowCount} label rows · ${markers} of ${needed} pauses marked - unmarked pauses are GUESSED. `
      + "Add a ▾ wherever the singer pauses (or a row boundary the guess missed).";
  rows.forEach((row, i) => {
    const li = document.createElement("li");
    const when = document.createElement("span");
    when.className = "when";
    when.textContent = `${row.start}–${row.end} (${row.seconds}s)  `;
    li.appendChild(when);
    li.appendChild(document.createTextNode(row.words || "(no words in this row)"));
    if (!row.words) li.className = "empty";
    list.appendChild(li);
  });
}

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
  stopPlayback();
  el("pauseEditSource").textContent = `${occ.group} — ${occ.song}`;
  el("pauseEditInput").value = res.data.text;
  showPauseEditError("");
  el("pauseEditDialog").showModal();
  el("pauseEditInput").focus();
  refreshPausePreview();
}

function insertPauseGlyph() {
  const box = el("pauseEditInput");
  box.setRangeText(PAUSE_GLYPH, box.selectionStart, box.selectionEnd, "end");
  box.focus();
  schedulePausePreview();
}

// Tap along (.claude/KARAOKE_PLAN.md Part 5): hit Space as each word is sung. A tap reads the same page clock the
// highlight uses (currentKaraokeChunk), so any constant clock-vs-audio offset cancels out here; the server measures
// and removes the human reaction lag (saveTaps) from the taps on exact row-start words.
const tap = { timing: null, take: null, chips: [], phase: "idle", countTimer: 0, token: 0, unit: "syllable", rate: 1,
              preview: null, replayStarts: null, edit: null };
const LATIN_UNIT = /^[A-Za-z]+$/;
const HELD_BEAT = /^ー$/;

// With "tap ー っ ん" off, those beats are never asked for: the take steps over them (they keep their estimate).
// Beats the take steps over by itself: っ ん unless "tap っ ん" is ticked, and held vowels (ー, the う of じょう, the い of
// せい) while "held vowels" is ticked - they are sung as the syllable before, not tapped again.
function tapHolds() { return tapEl("tapHolds").checked; }
function foldHeld() { return tapEl("tapHeld").checked; }
function isHoldChip(i) {
  const chip = tap.chips[i];
  return chip !== undefined && ((chip.classList.contains("hold") && !tapHolds()) || (chip.classList.contains("held") && foldHeld()));
}
function autoSkipHolds() {
  while (tap.take && !tap.take.done && isHoldChip(tap.take.next)) tap.take.skip();
}

function tapEl(id) { return el(id); }

function setTapError(message) {
  const box = tapEl("tapError");
  box.textContent = message || "";
  box.hidden = !message;
}

// One chip per tap index, laid out like the lyric (line breaks kept). A split kanji gets a chip per kana part.
function renderTapChips(timing) {
  const line = tapEl("tapLine");
  line.textContent = "";
  tap.chips = [];
  for (const seg of timing.pieces) {
    if (!seg.isWord) {
      line.appendChild(document.createTextNode(seg.text));
      continue;
    }
    if (timing.unit === "syllable" && seg.syllables) {
      // One chip per sung beat (ー and っ count like ひ・と・つ). The written word sits small above its kana when it
      // differs (恋 over こ い); an English word stays ONE wide pill.
      const group = document.createElement("span");
      group.className = "tap-word";
      const joined = seg.syllables.map((x) => x.label).join("");
      if (seg.text && seg.text !== joined) {
        const caption = document.createElement("span");
        caption.className = "tap-kanji";
        caption.textContent = seg.text;
        group.appendChild(caption);
      }
      for (const syl of seg.syllables) {
        const chip = document.createElement("span");
        chip.className = "tap-chip" + (LATIN_UNIT.test(syl.label) ? " word-chip" : "") + (syl.hold || syl.held ? (HELD_BEAT.test(syl.label) || syl.held ? " held" : " hold") : "");
        chip.textContent = syl.label;
        group.appendChild(chip);
        tap.chips.push(chip);
      }
      line.appendChild(group);
      continue;
    }
    const parts = seg.parts && seg.parts.length > 1 ? seg.parts : [null];
    parts.forEach((part, k) => {
      const chip = document.createElement("span");
      chip.className = "tap-chip" + (k > 0 ? " sub" : "");
      chip.textContent = k === 0 ? seg.text : `(${part.reading})`;
      line.appendChild(chip);
      tap.chips.push(chip);
    });
  }
}

function paintTapChips() {
  const entries = tap.take ? tap.take.entries() : [];
  tapEl("tapProgress").textContent = tap.take && tap.phase !== "idle"
    ? `${tap.take.tappedCount()} tapped of ${tap.chips.length}` : `${tap.chips.length} taps in this line`;
  tap.chips.forEach((chip, i) => {
    const entry = entries[i];
    chip.classList.toggle("tapped", !!entry && entry.chunk !== null);
    chip.classList.toggle("skipped", !!entry && entry.chunk === null);
    chip.classList.toggle("next", tap.phase === "record" && i === entries.length);
    const slot = editSlots() ? tap.edit.timing.slots[i] : null;       // the fine-tune view: where each beat comes from
    if (slot) chip.dataset.source = slot.source;
    else chip.removeAttribute("data-source");
    chip.classList.toggle("editable", !!slot);
    chip.classList.toggle("selected", !!slot && tap.edit.selected === i);
  });
}

function setTapPhase(phase) {
  tap.phase = phase;
  const recording = phase === "count" || phase === "record";
  tapEl("tapStart").hidden = recording;
  tapEl("tapStart").textContent = phase === "idle" ? "Start take" : "Retake";
  tapEl("tapReplay").hidden = recording || !tap.edit;
  tapEl("tapClear").hidden = !(tap.timing && tap.timing.tapStatus === "ok" && tap.timing.tapUnit === tap.unit) || recording;
  document.querySelectorAll("#tapDialog input[type=radio], #tapDialog input[type=checkbox]").forEach((box) => { box.disabled = recording; });
  if (recording) tapEl("tapDialog").focus();      // so Space taps instead of clicking a button
  updateTapButtons();
  updateNudgeBar();
  paintTapChips();
}

function tapRecording() { return tap.phase === "count" || tap.phase === "record"; }

// The fine-tune pass works on `tap.edit` while a take is finished or a saved take is showing; never while recording.
function editSlots() {
  return tap.edit && !tapRecording() && tap.edit.timing.slots && tap.edit.timing.slots.length === tap.chips.length;
}

function updateTapButtons() {
  tapEl("tapSave").hidden = tapRecording() || !(tap.phase === "done" || (tap.edit && tap.edit.dirty));
  tapEl("tapSave").textContent = tap.phase === "done" ? "Keep & save take" : "Save nudges";
}

// A saved take as an editable state: its corrected anchors, with the raw taps it came from (null for takes saved
// before raw taps were kept). Null when no saved take applies to the unit being viewed.
function savedTakeEdit(timing) {
  if (!timing || timing.tapStatus !== "ok" || timing.tapUnit !== tap.unit || timing.unit !== tap.unit || !timing.anchors) return null;
  return { anchors: { ...timing.anchors }, timing, raw: timing.raw || null, dirty: false, selected: null, start0: 0 };
}

function updateNudgeBar() {
  const show = !!editSlots();
  tapEl("tapNudgeHint").hidden = !show;
  const picked = show && tap.edit.selected !== null;
  tapEl("tapNudge").hidden = !picked;
  if (!picked) return;
  const slot = tap.edit.timing.slots[tap.edit.selected];
  const occ = lastShownOccurrence;
  const locked = slot.source === "row";
  tapEl("tapNudgeBeat").textContent = slot.label;
  tapEl("tapNudge").querySelectorAll("button[data-nudge]").forEach((b) => { b.disabled = locked; });
  tapEl("tapNudgeReset").disabled = slot.source !== "tapped";
  const secs = ((slot.startChunk - occ.startChunk) * 40 / 1000).toFixed(2);
  tapEl("tapNudgeInfo").textContent = locked
    ? `locked to the label row start (${secs} s into the clip) - your labels are exact`
    : `${slot.source === "tapped" ? "tapped" : "estimated"}, ${secs} s into the clip`;
}

function describeSavedTake(timing) {
  if (timing.tapStatus === "ok") {
    const other = timing.tapUnit !== tap.unit ? ` (it is a ${timing.tapUnit} take; you are viewing ${tap.unit}s)` : "";
    return `A saved ${timing.tapUnit} take is applied to this card${other} - ${Math.round(timing.tapLag * 40)} ms of tap lag was removed.`;
  }
  if (timing.tapStatus === "stale") return "A saved take exists but the words changed since (pause or kanji split edited), so it is NOT applied. Tap again to replace it.";
  return "No saved take: word times inside a row are an even-tempo estimate.";
}

// (Re)load the card's timing for the chosen tap unit and redraw the chips, pace hint and legend.
async function loadTapTiming() {
  const occ = lastShownOccurrence;
  const res = await window.pywebview.api.getLineTiming(occ.group, occ.song, occ.startChunk, occ.endChunk,
    occ.lyricLine, occ.singers || [], state.filters.language, occ.lyricId || null, tap.unit);
  if (!res.ok || !res.data) {
    showError(res.ok ? "This card has no audio span to tap along to." : res.error);
    return false;
  }
  tap.timing = res.data;
  tap.take = null;
  tap.preview = null;
  tap.edit = savedTakeEdit(res.data);
  renderTapChips(res.data);
  const { fastest, suggestedRate } = res.data.pace;
  tapEl("tapPace").textContent = `Busiest stretch: about ${fastest} taps per second at normal speed (an average - real bursts are faster).`
    + (suggestedRate < 1 ? ` Suggested speed: ${suggestedRate}x.` : " Normal speed should be fine.");
  tapEl("tapLegend").textContent = res.data.unit === "syllable"
    ? "Round chips = one syllable each. Small chips are held vowels (ー, the う of じょう) and っ / ん: they are skipped for you unless you tick 'tap っ ん' / untick 'held vowels count as one'. A wide dashed pill = a whole English word, ONE tap."
    : "One tap per word (a kana part of a split kanji counts as its own).";
  tapEl("tapCount").textContent = "";
  tapEl("tapStatus").textContent = describeSavedTake(res.data);
  tapEl("tapResult").textContent = "";
  setTapError("");
  setTapPhase("idle");
  return true;
}

async function openTapAlong() {
  const occ = lastShownOccurrence;
  if (!occ) return;
  stopPlayback();
  tap.rate = 1;
  document.querySelector('#tapDialog input[name=tapRate][value="1"]').checked = true;
  document.querySelector(`#tapDialog input[name=tapUnit][value="${tap.unit}"]`).checked = true;
  tapEl("tapSource").textContent = `${occ.group} — ${occ.song}`;
  tapEl("tapDialog").style.setProperty("--singer-color", el("occurrenceLine").style.getPropertyValue("--singer-color") || "#2e86de");
  if (await loadTapTiming()) tapEl("tapDialog").showModal();
}

function cancelTapTimers() {
  tap.token += 1;                                  // invalidates a pending count-in
  clearTimeout(tap.countTimer);
  clearTimeout(auditionTimer);
  endReplay();
}

// Replay: the chips light syllable by syllable at the NEW times (the take with your lag removed), while the clip
// plays at the speed you tapped at. Chips that were tapped show how far they moved from the old estimate.
function endReplay() {
  tap.replayStarts = null;
  karaokeOnFrame = null;
  tap.chips.forEach((chip) => chip.classList.remove("replay-lit"));
}

function showTapShifts() {
  tap.chips.forEach((chip) => { chip.removeAttribute("data-delta"); chip.classList.remove("big-shift"); });
  if (!tap.preview || !tap.edit || !tap.edit.timing.slots) return;
  const now = Karaoke.unitStarts(tap.edit.timing);
  const before = Karaoke.unitStarts(tap.preview.estimate);
  tap.edit.timing.slots.forEach((slot, i) => {
    if (slot.source !== "tapped" || now[i] === undefined || !tap.chips[i]) return;   // a row start is exact; an estimate is unchanged
    const ms = Math.round((now[i] - before[i]) * 40);
    if (Math.abs(ms) < 5) return;                   // a tap that agrees with the estimate
    tap.chips[i].dataset.delta = (ms > 0 ? "+" : "") + ms;
    tap.chips[i].classList.toggle("big-shift", Math.abs(ms) >= 200);
  });
}

// Plays the line with the chips lighting at the take's CURRENT times (nudges included), from `fraction` of the clip.
async function playTapView(fraction = 0) {
  if (!tap.edit || tapRecording()) return;
  endReplay();
  const token = tap.token;
  tap.replayStarts = Karaoke.unitStarts(tap.edit.timing);
  karaokeOnFrame = (chunk) => {
    tap.chips.forEach((chip, i) => chip.classList.toggle("replay-lit", chunk !== null && Karaoke.isLit(tap.replayStarts[i], chunk)));
  };
  try {
    await playOccurrenceAudio(fraction, fraction > 0, tap.rate);
  } catch (e) {
    endReplay();
    return;
  }
  if (token !== tap.token) endReplay();
}

function replayTake() { return playTapView(0); }

// ---- Fine-tune pass (T6): click a beat, nudge it by 40 ms steps, hear it. Nudges edit the lag-CORRECTED chunks, so
// saving them goes through saveCorrectedTake (no second lag subtraction).
const NUDGE_STEP_CHUNKS = 1;                          // 40 ms
const NUDGE_BIG_CHUNKS = 5;                           // 200 ms
const AUDITION_LEAD_MS = 300;                         // real time before the beat that the audition starts
const AUDITION_DELAY_MS = 300;                        // a held arrow key auditions once, when it settles
let auditionTimer = 0;

function selectBeat(i) {
  if (!editSlots() || i === null || i === undefined || i < 0) return;
  const edit = tap.edit;
  edit.selected = edit.selected === i ? null : i;
  edit.start0 = edit.timing.slots[i].startChunk;
  updateNudgeBar();
  paintTapChips();
  tapEl("tapDialog").focus();                         // keys go to the dialog, never to a button
  if (edit.selected !== null) scheduleAudition(i);
}

function scheduleAudition(index) {
  clearTimeout(auditionTimer);
  auditionTimer = setTimeout(() => auditionBeat(index), AUDITION_DELAY_MS);
}

function auditionBeat(index) {
  const edit = tap.edit;
  const occ = lastShownOccurrence;
  if (!edit || !occ || !editSlots() || !edit.timing.slots[index]) return Promise.resolve();
  const spanLength = Math.max(1, occ.endChunk - occ.startChunk);
  const leadChunks = (AUDITION_LEAD_MS * tap.rate) / 40;
  const fraction = Math.min(1, Math.max(0, (edit.timing.slots[index].startChunk - leadChunks - occ.startChunk) / spanLength));
  return playTapView(fraction);
}

// One nudge at a time, in order: a held arrow key queues them instead of racing the server.
function nudgeSelected(deltaChunks) {
  const edit = tap.edit;
  if (!edit || edit.selected === null) return Promise.resolve();
  const index = edit.selected;
  edit.chain = (edit.chain || Promise.resolve()).then(() => applyNudge(edit, index, deltaChunks)).catch((e) => setTapError(String(e)));
  return edit.chain;
}

function showNudgeNote(text) {
  tapEl("tapNudgeInfo").textContent = text;
}

async function applyNudge(edit, index, deltaChunks) {
  const occ = lastShownOccurrence;
  if (tap.edit !== edit || !occ) return;
  const res = await window.pywebview.api.nudgeTake(occ.group, occ.song, occ.startChunk, occ.endChunk, occ.lyricLine,
    occ.singers || [], state.filters.language, occ.lyricId || null, edit.anchors, tap.unit, index, deltaChunks);
  if (tap.edit !== edit) return;
  if (!res.ok) {
    setTapError(res.error);
    return;
  }
  setTapError("");
  const { status, anchors, timing } = res.data;
  if (status === "moved" || status === "reset") {
    edit.anchors = anchors;
    edit.timing = timing;
    edit.dirty = true;
    showTapShifts();
    updateTapButtons();
    updateNudgeBar();
    paintTapChips();
    const slot = timing.slots[index];
    const moved = Math.round((slot.startChunk - edit.start0) * 40);
    showNudgeNote(status === "reset" ? "back to the estimate"
      : `${moved > 0 ? "+" : ""}${moved} ms from where it was when you picked it, ${((slot.startChunk - occ.startChunk) * 40 / 1000).toFixed(2)} s into the clip`);
    scheduleAudition(index);
  } else if (status === "locked") {
    showNudgeNote("locked to the label row start - your labels are exact, so this beat does not move");
  } else {
    showNudgeNote(deltaChunks === null ? "nothing to reset - this beat is already the estimate"
      : "at its limit: a beat stays inside its own label row and never passes the tapped beat next to it");
  }
}

// Keys for the fine-tune pass. Returns true when the key was used.
function handleNudgeKey(event) {
  const edit = tap.edit;
  if (!edit || edit.selected === null || !editSlots()) return false;
  const big = event.shiftKey ? NUDGE_BIG_CHUNKS : NUDGE_STEP_CHUNKS;
  if (event.code === "ArrowLeft") nudgeSelected(-big);
  else if (event.code === "ArrowRight") nudgeSelected(big);
  else if (event.code === "Backspace" || event.code === "Delete") nudgeSelected(null);
  else if (event.code === "Space") auditionBeat(edit.selected);
  else if (event.code === "Escape") selectBeat(edit.selected);
  else return false;
  event.preventDefault();
  event.stopImmediatePropagation();
  return true;
}

// Count-in metronome: 3 low clicks, then a higher one on GO, which is when the clip starts. Scheduled on the audio
// clock up front (a setTimeout per click would wobble by tens of ms); the popup numbers follow on setTimeout.
const COUNT_BEAT_MS = 650;
let clickAudio = null;

function scheduleCountInClicks() {
  if (!tapEl("tapClicks").checked) return;
  try {
    clickAudio = clickAudio || new AudioContext();
    if (clickAudio.state === "suspended") clickAudio.resume();
    const t0 = clickAudio.currentTime + 0.05;
    [880, 880, 880, 1320].forEach((freq, k) => {
      const osc = clickAudio.createOscillator();
      const gain = clickAudio.createGain();
      const at = t0 + (k * COUNT_BEAT_MS) / 1000;
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, at);
      gain.gain.exponentialRampToValueAtTime(0.6, at + 0.005);
      gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.09);
      osc.connect(gain).connect(clickAudio.destination);
      osc.start(at);
      osc.stop(at + 0.1);
    });
  } catch (e) {
    // No audio device or the browser refused: the visual count-in still works.
  }
}

function showCountBeat(text) {
  const box = tapEl("tapCount");
  box.textContent = text;
  box.classList.remove("beat");
  void box.offsetWidth;                            // restart the pop animation
  box.classList.add("beat");
}

async function beginTake() {
  cancelTapTimers();
  stopPlayback();
  const token = tap.token;
  tap.take = Karaoke.newTake(tap.chips.length);
  tap.preview = null;
  tap.edit = null;                                   // a new take replaces whatever was being fine-tuned
  showTapShifts();                                   // clears the last take's shift marks
  autoSkipHolds();
  tapEl("tapResult").textContent = "";
  setTapError("");
  setTapPhase("count");
  if (tap.rate < 1) {                                // build the slowed clip now, not after the count-in
    const occ = lastShownOccurrence;
    tapEl("tapCount").textContent = "…";
    const prepared = await window.pywebview.api.prepareSlowClip(occ.group, occ.song, occ.startChunk, occ.endChunk, tap.rate);
    if (token !== tap.token) return;
    if (!prepared.ok) {
      setTapError(prepared.error);
      setTapPhase("idle");
      tapEl("tapCount").textContent = "";
      return;
    }
  }
  scheduleCountInClicks();
  for (const n of [3, 2, 1]) {
    showCountBeat(String(n));
    await new Promise((resolve) => { tap.countTimer = setTimeout(resolve, COUNT_BEAT_MS); });
    if (token !== tap.token) return;               // restarted or closed during the count-in
  }
  showCountBeat("GO");
  setTapPhase("record");
  try {
    await playOccurrenceAudio(0, false, tap.rate);
  } catch (e) {
    setTapPhase("idle");
    return;
  }
  if (token !== tap.token) return;
  karaokeOnEnd = () => finishTake();               // the clip ran out: whatever was not tapped stays estimated
  setTimeout(() => { if (token === tap.token && tap.phase === "record") tapEl("tapCount").textContent = ""; }, 500);
}

// Does the take agree with the label rows? Each row's first beat should land on the row start (gap ~0 once your
// reaction lag is removed). The red +/- numbers on the chips are something else: how far a beat moved from the ESTIMATE.
function describeRowCheck(gaps) {
  if (!gaps || gaps.length < 2) return "";
  const worstMs = Math.round(Math.max(...gaps.map((g) => Math.abs(g))) * 40);
  const list = gaps.map((g) => `${Math.round(g * 40)}`).join(", ");
  return `Row check: your taps agree with the labelled row starts to within ${worstMs} ms (per row: ${list} ms).`
    + (worstMs > 400 ? " That is more than a reaction time - the taps and the labels disagree." : "");
}

async function finishTake() {
  if (tap.phase !== "record") return;
  cancelTapTimers();
  stopPlayback();
  tapEl("tapCount").textContent = "";
  const tapped = tap.take.tappedCount();
  tap.preview = null;
  tapEl("tapResult").textContent = `${tapped} of ${tap.chips.length} tapped.` +
    (tapped ? " Replaying it now - keep it, or Retake." : " Nothing was tapped - Retake.");
  setTapPhase(tapped ? "done" : "idle");
  tapEl("tapStart").textContent = "Retake";
  if (!tapped) return;
  const occ = lastShownOccurrence;
  const token = tap.token;
  const res = await window.pywebview.api.previewTaps(occ.group, occ.song, occ.startChunk, occ.endChunk,
    occ.lyricLine, occ.singers || [], state.filters.language, occ.lyricId || null, tap.take.taps(), tap.unit, tap.rate);
  if (token !== tap.token || tap.phase !== "done") return;      // retaken or closed meanwhile
  if (!res.ok) {
    setTapError(res.error);
    return;
  }
  tap.preview = res.data;
  tap.edit = { anchors: res.data.anchors, timing: res.data.timing, raw: tap.take.taps(), lag: res.data.lag, dirty: false,
               selected: null, start0: 0 };
  setTapPhase("done");
  tapEl("tapResult").textContent += " " + describeRowCheck(res.data.rowCheck);
  const moved = res.data.timing.reassigned;
  if (moved) tapEl("tapResult").textContent += ` Your taps put ${moved} beat${moved === 1 ? "" : "s"} in a different label row than your ▾ markers say - the taps win.`;
  showTapShifts();
  replayTake();
}

function onTapKey(event) {
  if (tapEl("tapDialog").open && !tapRecording() && handleNudgeKey(event)) return;
  if (!tapEl("tapDialog").open || (tap.phase !== "record" && tap.phase !== "count")) return;
  if (event.code === "Escape") return;             // the dialog's own close
  event.preventDefault();
  if (event.repeat || tap.phase !== "record") return;
  if (event.code === "Space") {
    if (tap.take.tap(currentKaraokeChunk())) {
      autoSkipHolds();
      paintTapChips();
      if (tap.take.done) finishTake();
    }
  } else if (event.code === "Tab") {
    if (tap.take.skip()) {
      autoSkipHolds();
      paintTapChips();
      if (tap.take.done) finishTake();
    }
  } else if (event.code === "Backspace") {
    let popped = tap.take.undo();
    // Beats skipped for you are not something to undo: keep stepping back until a real tap is removed.
    while (popped && popped.chunk === null && isHoldChip(popped.index)) popped = tap.take.undo();
    paintTapChips();
  } else if (event.code === "KeyR") {
    beginTake();
  }
}

async function saveNudgedTake() {
  const occ = lastShownOccurrence;
  const edit = tap.edit;
  const fresh = tap.phase === "done" && tap.take && tap.preview;     // a new take keeps ITS raw taps; a saved one keeps its own
  const res = await window.pywebview.api.saveCorrectedTake(occ.group, occ.song, occ.startChunk, occ.endChunk,
    occ.lyricLine, occ.singers || [], state.filters.language, occ.lyricId || null, edit.anchors, tap.unit,
    fresh ? tap.rate : null, fresh ? tap.take.taps() : null, fresh ? tap.preview.lag : null);
  if (!res.ok) {
    setTapError(res.error);
    return;
  }
  setTapError("");
  tap.timing = res.data.timing;
  tap.edit = savedTakeEdit(res.data.timing);
  tapEl("tapStatus").textContent = describeSavedTake(res.data.timing);
  tapEl("tapResult").textContent = "Saved your fine-tuned take (the raw taps are kept beside it).";
  setTapPhase("idle");
  tapEl("tapStart").textContent = "Retake";
  showTapShifts();
  const card = currentCard();
  renderOccurrenceRow(card ? await getDetail(card) : null);
}

async function saveTapTake() {
  const occ = lastShownOccurrence;
  if (!occ) return;
  if (tap.edit && tap.edit.dirty) return saveNudgedTake();
  if (!tap.take) return;
  const res = await window.pywebview.api.saveTaps(occ.group, occ.song, occ.startChunk, occ.endChunk,
    occ.lyricLine, occ.singers || [], state.filters.language, occ.lyricId || null, tap.take.taps(), tap.unit, tap.rate);
  if (!res.ok) {
    setTapError(res.error);
    return;
  }
  const { lag, spread, measured, timing } = res.data;
  const lagMs = Math.round(lag * 40 / tap.rate);      // real reaction time: song-time lag / playback speed
  const rowStartTaps = Object.keys(tap.timing.rowStarts || {}).filter((i) => tap.take.taps()[i] !== undefined).length;
  tapEl("tapResult").textContent = measured
    ? `Saved. Your tap lag was ${lagMs} ms (measured on ${rowStartTaps} row-start words, steady to within ${Math.round(spread * 40)} ms) and has been removed.`
    : `Saved. Fewer than two row-start words were tapped, so a typical ${lagMs} ms lag was assumed - tap the first word of every row to measure yours.`;
  tap.timing = timing;
  tap.edit = savedTakeEdit(timing);
  tapEl("tapStatus").textContent = describeSavedTake(timing);
  setTapPhase("idle");
  showTapShifts();
  tapEl("tapSave").hidden = true;
  tapEl("tapStart").textContent = "Retake";
  const card = currentCard();
  const detail = card ? await getDetail(card) : null;
  renderOccurrenceRow(detail);                     // the card behind now highlights from the saved take
}

async function clearTapTake() {
  const occ = lastShownOccurrence;
  if (!occ) return;
  const res = await window.pywebview.api.saveTaps(occ.group, occ.song, occ.startChunk, occ.endChunk,
    occ.lyricLine, occ.singers || [], state.filters.language, occ.lyricId || null, {}, tap.unit, tap.rate);
  if (!res.ok) {
    setTapError(res.error);
    return;
  }
  tap.timing = res.data.timing;
  tap.edit = savedTakeEdit(tap.timing);
  tapEl("tapStatus").textContent = describeSavedTake(tap.timing);
  tapEl("tapResult").textContent = "Saved take cleared.";
  setTapPhase("idle");
  const card = currentCard();
  const detail = card ? await getDetail(card) : null;
  renderOccurrenceRow(detail);
}

// "Split kanji to hiragana": a pause can fall INSIDE a kanji word (Doughnut: Sana sings 恋 as こ ... い across two
// label rows). Select the kanji (or put the cursor in/next to it) and press the button: it appends the kana
// reading as 《こい》 right after the kanji - the lyric text stays 恋, the 《》 part is timing-only (core/lyric_text).
// Then put the cursor between こ and い and insert a pause: 恋《こ▾い》. Saved with the lyric, so it travels with it.
const KANJI_CHARS = "\\u4e00-\\u9fff\\u3005\\u3006\\u30f6";
const KANJI_RUN = new RegExp(`[${KANJI_CHARS}]+`, "g");
const KANJI_ONLY = new RegExp(`^[${KANJI_CHARS}]+$`);
const READING_OPEN_GLYPH = String.fromCharCode(0x300A);
const READING_CLOSE_GLYPH = String.fromCharCode(0x300B);

function kanjiRunToSplit(box) {
  const { selectionStart: a, selectionEnd: b, value } = box;
  if (a === b) {                                   // no selection: the kanji run touching the cursor
    for (const m of value.matchAll(KANJI_RUN)) {
      if (m.index <= a && a <= m.index + m[0].length) return [m.index, m.index + m[0].length];
    }
    return null;
  }
  return KANJI_ONLY.test(value.slice(a, b)) ? [a, b] : null;
}

async function splitKanjiAtCursor() {
  const box = el("pauseEditInput");
  const span = kanjiRunToSplit(box);
  if (!span) {
    showPauseEditError("Select a kanji word (or put the cursor on it) first.");
    return;
  }
  const [a, b] = span;
  if (box.value[b] === READING_OPEN_GLYPH) {
    showPauseEditError("That kanji already has a reading - edit the " + READING_OPEN_GLYPH + "..." + READING_CLOSE_GLYPH + " part directly.");
    return;
  }
  const res = await window.pywebview.api.getKanjiReading(box.value.slice(a, b));
  if (!res.ok || !res.data) {
    showPauseEditError(res.ok ? "No kana reading found for that word." : res.error);
    return;
  }
  showPauseEditError("");
  box.setRangeText(READING_OPEN_GLYPH + res.data + READING_CLOSE_GLYPH, b, b, "end");
  // Park the cursor after the first kana so one Ctrl+Space puts the pause there.
  box.setSelectionRange(b + 2, b + 2);
  box.focus();
  schedulePausePreview();
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

async function playOccurrenceAudio(startFraction = 0, exact = false, rate = 1) {
  const card = currentCard();
  if (!card) return;
  const detail = await getDetail(card);
  const occ = detail && detail.occurrence;
  if (!occ) return;
  stopKaraoke();
  const sent = performance.now();
  const play = await callApi("playOccurrenceAudio", occ.group, occ.song, occ.startChunk, occ.endChunk,
    startFraction, exact, rate);
  if (play) startKaraoke(play, performance.now() - sent);
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
  stopPlayback();  // even if the same card comes straight back (Again)
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
    if (document.querySelector("dialog[open]")) return;         // arrows inside a dialog must not change the card behind it
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
  el("tapAlongBtn").addEventListener("click", openTapAlong);
  el("tapStart").addEventListener("click", beginTake);
  el("tapSave").addEventListener("click", saveTapTake);
  el("tapReplay").addEventListener("click", replayTake);
  el("tapClear").addEventListener("click", clearTapTake);
  el("tapClose").addEventListener("click", () => el("tapDialog").close());
  el("tapLine").addEventListener("click", (event) => {
    const chip = event.target.closest(".tap-chip");
    if (chip) selectBeat(tap.chips.indexOf(chip));
  });
  el("tapNudge").querySelectorAll("button[data-nudge]").forEach((btn) =>
    btn.addEventListener("click", () => { nudgeSelected(Number(btn.dataset.nudge)); el("tapDialog").focus(); }));
  el("tapNudgeReset").addEventListener("click", () => { nudgeSelected(null); el("tapDialog").focus(); });
  el("tapDialog").addEventListener("cancel", (event) => {     // Esc deselects a picked beat before it closes the dialog
    if (tap.edit && tap.edit.selected !== null) {
      event.preventDefault();
      selectBeat(tap.edit.selected);
    }
  });
  document.querySelectorAll('#tapDialog input[name="tapUnit"]').forEach((box) => box.addEventListener("change", () => {
    tap.unit = box.value;
    loadTapTiming();
  }));
  document.querySelectorAll('#tapDialog input[name="tapRate"]').forEach((box) => box.addEventListener("change", () => {
    tap.rate = Number(box.value);
  }));
  el("tapDialog").addEventListener("close", () => { cancelTapTimers(); stopPlayback(); tap.phase = "idle"; });
  document.addEventListener("keydown", onTapKey, true);
  el("pauseEditInsert").addEventListener("click", insertPauseGlyph);
  el("pauseEditSplitKanji").addEventListener("click", splitKanjiAtCursor);
  el("pauseEditPlay").addEventListener("click", () => playOccurrenceAudio(0));
  el("pauseEditCancel").addEventListener("click", () => el("pauseEditDialog").close());
  el("pauseEditSave").addEventListener("click", savePauseEditor);
  el("pauseEditInput").addEventListener("input", schedulePausePreview);
  el("pauseEditInput").addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.code === "Space") {   // same hotkey as the Tk Lyric Editor
      event.preventDefault();
      insertPauseGlyph();
    }
  });
  el("playLastWordBtn").addEventListener("click",
    () => playOccurrenceAudio(Number(el("playLastWordBtn").dataset.fraction || 0),
                              !!el("playLastWordBtn").dataset.exact));

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
