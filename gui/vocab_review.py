"""
Review/browse/edit screen for the vocab SRS store (core.vocab_store_ja / core.vocab_store_ko) -
see .claude/ plan doc, Chunk 7, plus follow-up fixes:
  - Prev/Next navigation over a stable list (not a destructively-popped queue), so you can flip
    back and forth instead of cards only ever advancing forward.
  - Switching the Language/Track dropdown immediately reloads - previously the old card stayed on
    screen while reveal()/rate() silently started using the new language's store against the old
    card's vocabId, which happened to hit an unrelated word with the same row id in that table.
  - Words with no meaning are flagged, with an "Edit Meaning" action to fill one in by hand.
  - "Delete Word" removes a glitched/unwanted entry entirely.
  - Korean words with 2+ ambiguous Hanja candidates get a "Keep only this" button per candidate,
    to resolve the ambiguity and stop the rest from cluttering the list.

No audio playback here (the occurrence's start_chunk/end_chunk data is captured and available for
a later "play this line" enhancement, per Study_tool_ideas.txt's Idea 2, but that's out of scope
for this first pass).
"""

import tkinter as tk
from tkinter import messagebox, simpledialog

from core import vocab_store_ja, vocab_store_ko, vocab_link

_STORES = {"Japanese": vocab_store_ja, "Korean": vocab_store_ko}


def openVocabReviewWindow(parent):
    win = tk.Toplevel(parent)
    win.title("Vocab Review")
    win.geometry("620x520")
    win.minsize(560, 420)
    # Without transient(), this Toplevel's only tie to `parent` is the widget-cleanup hierarchy,
    # not window-manager stacking - it could render behind or on the wrong monitor/window (e.g.
    # behind the main app window instead of the Audio Tester window it was opened from).
    win.transient(parent)

    controlsFrame = tk.Frame(win)
    controlsFrame.pack(fill="x", padx=10, pady=(10, 4))

    languageVar = tk.StringVar(value="Japanese")
    trackVar = tk.StringVar(value="reading")
    showAllVar = tk.BooleanVar(value=False)
    missingMeaningOnlyVar = tk.BooleanVar(value=False)
    ambiguousOnlyVar = tk.BooleanVar(value=False)

    # Plain tk.Radiobutton, not ttk.Combobox - a themed ttk combobox was observed to render
    # completely invisible (no box, no dropdown arrow, just blank space) on at least one real
    # system, most likely a ttk-theme rendering quirk. Radiobuttons are the older, unthemed Tk
    # widget set (same family as the Checkbuttons on the row below, which did render fine), so
    # they sidestep that failure mode entirely - and a pair of buttons is arguably a more literal
    # "toggle" than a dropdown anyway. `command=loadQueue` reloads immediately on change (fixes
    # the same "old card stays on screen after switching language" bug outright, no separate
    # event binding needed).
    tk.Label(controlsFrame, text="Language:").pack(side="left")
    for value in ("Japanese", "Korean"):
        tk.Radiobutton(
            controlsFrame, text=value, variable=languageVar, value=value,
            command=lambda: loadQueue(),
        ).pack(side="left")

    tk.Label(controlsFrame, text="   Track:").pack(side="left")
    for value in ("reading", "meaning"):
        tk.Radiobutton(
            controlsFrame, text=value.capitalize(), variable=trackVar, value=value,
            command=lambda: loadQueue(),
        ).pack(side="left")

    filtersFrame = tk.Frame(win)
    filtersFrame.pack(fill="x", padx=10)
    tk.Checkbutton(
        filtersFrame, text="Show all words (ignore due date)", variable=showAllVar,
        command=lambda: loadQueue(),
    ).pack(side="left")
    tk.Checkbutton(
        filtersFrame, text="Only missing meaning", variable=missingMeaningOnlyVar,
        command=lambda: loadQueue(),
    ).pack(side="left", padx=(10, 0))
    ambiguousCheck = tk.Checkbutton(
        filtersFrame, text="Only ambiguous Hanja (2+)", variable=ambiguousOnlyVar,
        command=lambda: loadQueue(),
    )
    ambiguousCheck.pack(side="left", padx=(10, 0))

    navFrame = tk.Frame(win)
    navFrame.pack(fill="x", padx=10, pady=(8, 0))
    prevBtn = tk.Button(navFrame, text="◀ Prev")
    prevBtn.pack(side="left")
    positionLabel = tk.Label(navFrame, text="")
    positionLabel.pack(side="left", padx=10)
    nextBtn = tk.Button(navFrame, text="Next ▶")
    nextBtn.pack(side="left")

    promptLabel = tk.Label(win, text="", font=("Segoe UI", 20))
    promptLabel.pack(pady=(14, 4))

    warningLabel = tk.Label(win, text="", fg="#b03a2e")
    warningLabel.pack()

    answerText = tk.Text(win, height=8, wrap="word", state="disabled")
    answerText.pack(fill="both", expand=True, padx=10, pady=(4, 0))

    hanjaPickerFrame = tk.Frame(win)
    hanjaPickerFrame.pack(fill="x", padx=10, pady=(4, 0))

    actionsFrame = tk.Frame(win)
    actionsFrame.pack(pady=(8, 0))

    ratingFrame = tk.Frame(win)
    ratingFrame.pack(pady=(6, 10))

    state = {"cards": [], "index": 0}

    def _currentStore():
        return _STORES[languageVar.get()]

    def _setAnswerText(text):
        answerText.config(state="normal")
        answerText.delete("1.0", "end")
        answerText.insert("1.0", text)
        answerText.config(state="disabled")

    def _clearChildren(frame):
        for child in frame.winfo_children():
            child.destroy()

    def _formatBody(card, language):
        lines = []
        meaning = card.get("meaning") or {}

        if language == "Japanese":
            lines.append(f"Reading: {card.get('lemmaReading') or ''}")
            if meaning.get("status") == "found":
                lines.append(f"Meaning: {'; '.join(meaning['gloss'][:3])}")
            if card.get("cognateForm"):
                lines.append(f"Chinese cognate: {card['cognateForm']} ({card.get('cognatePinyin') or ''})")
            linked = vocab_link.getLinkedWords("ja", card["vocabId"])
            other = "Korean"
        else:
            if meaning.get("status") == "found":
                lines.append(f"Meaning: {'; '.join(meaning['gloss'][:3])}")
            for h in card.get("hanjaCandidates") or []:
                gloss = "; ".join((h.get("gloss") or [])[:2])
                lines.append(f"Hanja: {h['hanja']} ({h.get('pinyin') or ''}) — {gloss}")
            linked = vocab_link.getLinkedWords("ko", card["vocabId"])
            other = "Japanese"

        store = _STORES[language]
        occurrences = store.getOccurrences(card["vocabId"], limit=1)
        if occurrences:
            occ = occurrences[0]
            lines.append(f"\nFrom {occ['group']} — {occ['song']}: {occ['lyricLine']}")

        if linked:
            lines.append(f"\n{other} cognate cousins: " + ", ".join(l["lemma"] for l in linked))

        return "\n".join(lines)

    def _hasNoMeaning(card):
        meaning = card.get("meaning") or {}
        return meaning.get("status") != "found"

    def render():
        _clearChildren(hanjaPickerFrame)
        _clearChildren(ratingFrame)

        cards = state["cards"]
        if not cards:
            promptLabel.config(text="No words match these filters")
            warningLabel.config(text="")
            _setAnswerText("")
            positionLabel.config(text="0 / 0")
            prevBtn.config(state="disabled")
            nextBtn.config(state="disabled")
            return

        index = state["index"]
        card = cards[index]
        language = languageVar.get()

        promptLabel.config(text=card.get("surface") or card.get("lemma"))
        warningLabel.config(text="⚠ No meaning found - fill one in with Edit Meaning" if _hasNoMeaning(card) else "")
        _setAnswerText(_formatBody(card, language))
        positionLabel.config(text=f"{index + 1} / {len(cards)}")
        prevBtn.config(state=("normal" if index > 0 else "disabled"))
        nextBtn.config(state=("normal" if index < len(cards) - 1 else "disabled"))

        if language == "Korean":
            candidates = card.get("hanjaCandidates") or []
            if candidates:
                if len(candidates) >= 2:
                    tk.Label(hanjaPickerFrame, text="Multiple Hanja candidates - keep only the right one:").pack(anchor="w")
                    for c in candidates:
                        rowFrame = tk.Frame(hanjaPickerFrame)
                        rowFrame.pack(fill="x", anchor="w")
                        label = f"{c['hanja']} ({c.get('pinyin') or ''}) — {'; '.join((c.get('gloss') or [])[:1])}"
                        tk.Label(rowFrame, text=label).pack(side="left")
                        tk.Button(
                            rowFrame, text="Keep only this", command=lambda hid=c["hanjaId"]: keepOnlyHanja(hid)
                        ).pack(side="right")
                # Shown even for a single candidate - e.g. 해 ("sun/day", native) getting matched
                # against 害 ("harm", a real but unrelated Sino-Korean homophone) isn't "ambiguous"
                # (only one candidate), it's just wrong for this word - "keep only this" has
                # nothing to contrast against, so this is the only way to fix that case.
                tk.Button(
                    hanjaPickerFrame, text="Remove Hanja entirely (native word)", command=clearHanja
                ).pack(anchor="w", pady=(4, 0))

        for label, rating in (("Again", "again"), ("Good", "good"), ("Easy", "easy")):
            tk.Button(ratingFrame, text=label, width=10, command=lambda r=rating: rate(r)).pack(side="left", padx=4)

    def loadQueue():
        language = languageVar.get()
        track = trackVar.get()
        store = _STORES[language]

        cards = store.listAllVocab() if showAllVar.get() else store.getDueCards(track, limit=200)

        if missingMeaningOnlyVar.get():
            cards = [c for c in cards if _hasNoMeaning(c)]
        if ambiguousOnlyVar.get() and language == "Korean":
            cards = [c for c in cards if len(c.get("hanjaCandidates") or []) >= 2]

        state["cards"] = cards
        state["index"] = 0
        render()

    def goPrev():
        if state["index"] > 0:
            state["index"] -= 1
            render()

    def goNext():
        if state["index"] < len(state["cards"]) - 1:
            state["index"] += 1
            render()

    def rate(rating):
        cards = state["cards"]
        if not cards:
            return
        card = cards[state["index"]]
        _currentStore().submitReview(card["vocabId"], trackVar.get(), rating)
        goNext()

    def editMeaning():
        cards = state["cards"]
        if not cards:
            return
        card = cards[state["index"]]
        meaning = card.get("meaning") or {}
        existingGloss = "; ".join(meaning.get("gloss") or [])
        newGloss = simpledialog.askstring(
            "Edit Meaning", "Gloss (separate multiple senses with ;):",
            initialvalue=existingGloss, parent=win,
        )
        if newGloss is None:
            return
        glossList = [g.strip() for g in newGloss.split(";") if g.strip()]
        if not glossList:
            return
        _currentStore().updateMeaning(card["vocabId"], glossList)
        card["meaning"] = {"status": "found", "pos": meaning.get("pos"), "gloss": glossList}
        render()

    def deleteWord():
        cards = state["cards"]
        if not cards:
            return
        card = cards[state["index"]]
        if not messagebox.askyesno(
            "Delete Word", f"Permanently delete \"{card.get('surface') or card.get('lemma')}\"?", parent=win
        ):
            return
        _currentStore().deleteVocab(card["vocabId"])
        del cards[state["index"]]
        if state["index"] >= len(cards):
            state["index"] = max(0, len(cards) - 1)
        render()

    def keepOnlyHanja(hanjaId):
        cards = state["cards"]
        card = cards[state["index"]]
        vocab_store_ko.keepOnlyHanjaCandidate(card["vocabId"], hanjaId)
        kept = next(c for c in card["hanjaCandidates"] if c["hanjaId"] == hanjaId)
        card["hanjaCandidates"] = [kept]
        render()

    def clearHanja():
        cards = state["cards"]
        card = cards[state["index"]]
        vocab_store_ko.clearHanjaCandidates(card["vocabId"])
        card["hanjaCandidates"] = []
        render()

    prevBtn.config(command=goPrev)
    nextBtn.config(command=goNext)
    win.bind("<Left>", lambda e: goPrev())
    win.bind("<Right>", lambda e: goNext())

    tk.Button(actionsFrame, text="Edit Meaning", command=editMeaning).pack(side="left", padx=4)
    tk.Button(actionsFrame, text="Delete Word", command=deleteWord).pack(side="left", padx=4)
    tk.Button(actionsFrame, text="Reload", command=loadQueue).pack(side="left", padx=4)

    loadQueue()
    return win
