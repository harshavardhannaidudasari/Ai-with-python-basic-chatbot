const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const ragToggle = document.getElementById("rag-toggle");
const ingestBtn = document.getElementById("ingest-btn");
const resetBtn = document.getElementById("reset-btn");
const docInput = document.getElementById("doc-input");
const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const imagePreviewImg = document.getElementById("image-preview-img");
const imageRemoveBtn = document.getElementById("image-remove-btn");
const suggestions = document.getElementById("suggestions");

const hero = document.getElementById("hero");
const heroChatBtn = document.getElementById("hero-chat-btn");
const heroVoiceBtn = document.getElementById("hero-voice-btn");
const chatArea = document.getElementById("chat-area");
const talkBtn = document.getElementById("talk-btn");
const voicePanel = document.getElementById("voice-panel");
const voiceBackBtn = document.getElementById("voice-back-btn");
const voiceWave = document.getElementById("voice-wave");
const voiceStatus = document.getElementById("voice-status");
const voiceTranscript = document.getElementById("voice-transcript");
const voiceMuteBtn = document.getElementById("voice-mute-btn");
const voiceStopBtn = document.getElementById("voice-stop-btn");
const voiceReplayBtn = document.getElementById("voice-replay-btn");
const voiceMicBtn = document.getElementById("voice-mic-btn");

let pendingImage = null; // data URL of the currently attached image, if any
let turns = []; // ordered list of {userText, image, userWrapper, botWrapper, botBubble, botText, sources, sourceMode}

const SOURCE_LABELS = {
  rag: (sources) => `\u{1F4DA} Sources: ${sources.join(", ")}`,
  no_match: () => `\u{1F9E0} No matching documents — answered from the model's own knowledge`,
  no_rag: () => `\u{1F9E0} Knowledge base off — answered from the model's own knowledge`,
  image: () => `\u{1F441}️ Analyzed directly from the image (vision model)`,
};

function iconButton(label, title) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "msg-action";
  btn.textContent = label;
  btn.title = title;
  return btn;
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function clearPendingImage() {
  pendingImage = null;
  imageInput.value = "";
  imagePreview.hidden = true;
  imagePreviewImg.src = "";
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    pendingImage = reader.result;
    imagePreviewImg.src = pendingImage;
    imagePreview.hidden = false;
  };
  reader.readAsDataURL(file);
});

imageRemoveBtn.addEventListener("click", clearPendingImage);

suggestions.addEventListener("click", (event) => {
  const chip = event.target.closest(".suggestion-chip");
  if (!chip) return;
  chatInput.value = chip.dataset.prompt || "";
  chatInput.focus();
  chatInput.setSelectionRange(chatInput.value.length, chatInput.value.length);
});

// ---- Views: hero (onboarding) / chat / voice ----

function showView(name) {
  hero.hidden = name !== "hero";
  chatArea.hidden = name !== "chat";
  voicePanel.hidden = name !== "voice";
  if (name === "chat") {
    chatInput.focus();
  }
  if (name !== "voice" && recognitionActive) {
    stopListening();
  }
}

heroChatBtn.addEventListener("click", () => showView("chat"));
heroVoiceBtn.addEventListener("click", () => showView("voice"));
talkBtn.addEventListener("click", () => showView("voice"));
voiceBackBtn.addEventListener("click", () => showView("chat"));

// ---- Rendering ----

function renderUserTurn(turn) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";

  const col = document.createElement("div");
  col.className = "message-col";

  if (turn.image) {
    const imgBubble = document.createElement("div");
    imgBubble.className = "bubble";
    const img = document.createElement("img");
    img.src = turn.image;
    img.className = "chat-image";
    imgBubble.appendChild(img);
    col.appendChild(imgBubble);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = turn.userText;
  col.appendChild(bubble);
  turn.userTextBubble = bubble;

  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const editBtn = iconButton("✎ Edit", "Edit and resend this message");
  editBtn.addEventListener("click", () => beginEdit(turn));
  actions.appendChild(editBtn);
  col.appendChild(actions);

  wrapper.appendChild(col);
  turn.userWrapper = wrapper;
  return wrapper;
}

function renderBotPlaceholder(turn) {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot";

  const col = document.createElement("div");
  col.className = "message-col";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = "Thinking…";
  col.appendChild(bubble);

  wrapper.appendChild(col);
  turn.botWrapper = wrapper;
  turn.botBubble = bubble;
  turn.botCol = col;
  return wrapper;
}

function finalizeBotTurn(turn) {
  const actions = document.createElement("div");
  actions.className = "msg-actions";

  const copyBtn = iconButton("⎘ Copy", "Copy response");
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(turn.botText);
      const original = copyBtn.textContent;
      copyBtn.textContent = "✓ Copied";
      setTimeout(() => (copyBtn.textContent = original), 1200);
    } catch (err) {
      copyBtn.textContent = "Copy failed";
    }
  });
  actions.appendChild(copyBtn);

  const retryBtn = iconButton("↻ Retry", "Regenerate this response");
  retryBtn.addEventListener("click", () => retryTurn(turn));
  actions.appendChild(retryBtn);

  turn.botCol.appendChild(actions);
}

function removeTurnsFrom(index) {
  for (let i = index; i < turns.length; i++) {
    turns[i].userWrapper?.remove();
    turns[i].botWrapper?.remove();
  }
  turns = turns.slice(0, index);
}

// ---- Edit / Retry ----

function beginEdit(turn) {
  const bubble = turn.userTextBubble;
  const original = turn.userText;

  bubble.textContent = "";
  const textarea = document.createElement("textarea");
  textarea.className = "edit-textarea";
  textarea.value = original;
  bubble.appendChild(textarea);

  const controls = document.createElement("div");
  controls.className = "edit-controls";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.textContent = "Save & resend";
  saveBtn.className = "edit-save";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.textContent = "Cancel";
  cancelBtn.className = "edit-cancel";
  controls.appendChild(saveBtn);
  controls.appendChild(cancelBtn);
  bubble.appendChild(controls);

  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);

  cancelBtn.addEventListener("click", () => {
    bubble.textContent = original;
  });

  saveBtn.addEventListener("click", () => {
    const newText = textarea.value.trim();
    if (!newText) return;
    const index = turns.indexOf(turn);
    const image = turn.image;
    removeTurnsFrom(index);
    sendTurn(newText, image);
  });
}

function retryTurn(turn) {
  const index = turns.indexOf(turn);
  if (index === -1) return;
  const userText = turn.userText;
  const image = turn.image;
  removeTurnsFrom(index);
  sendTurn(userText, image);
}

// ---- Core send/stream ----

function textTurnsBefore(index) {
  return turns.slice(0, index).filter((t) => !t.image).length;
}

async function sendTurn(message, image, onToken, signal) {
  const index = turns.length;
  const turn = { userText: message, image, botText: "", sources: [], sourceMode: null };
  turns.push(turn);

  chatWindow.appendChild(renderUserTurn(turn));
  chatWindow.appendChild(renderBotPlaceholder(turn));
  scrollToBottom();

  const body = { message, use_rag: ragToggle.checked, image };
  const truncateTo = textTurnsBefore(index);
  // Only send truncate_to when rewinding is meaningful (i.e. this isn't simply the next new turn).
  if (truncateTo < countPriorTextTurnsOnServer) {
    body.truncate_to = truncateTo;
  }

  await streamChat(body, turn, onToken, signal);
  return turn;
}

// Tracks how many text (non-image) turns the server-side memory currently holds,
// so we only send `truncate_to` when we're actually rewinding history.
let countPriorTextTurnsOnServer = 0;

async function streamChat(body, turn, onToken, signal) {
  let response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    // Deliberately cancelled (voice mode interrupting an abandoned turn to
    // start a new question) — the caller already knows and is moving on,
    // not waiting on this turn's bubble/statusText.
    if (err.name === "AbortError") return;
    turn.botBubble.textContent = `Network error: ${err.message}`;
    turn.statusText = turn.botBubble.textContent;
    finalizeBotTurn(turn);
    return;
  }

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    turn.botBubble.textContent = `Error: ${data.error || "something went wrong"}`;
    turn.statusText = turn.botBubble.textContent;
    finalizeBotTurn(turn);
    return;
  }

  if (body.truncate_to !== undefined) {
    countPriorTextTurnsOnServer = body.truncate_to;
  }
  if (!body.image) {
    countPriorTextTurnsOnServer += 1;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let started = false;
  let streamError = null;
  let sseError = null;

  // The connection can drop mid-stream — a mobile tab getting backgrounded
  // (a call comes in, the user switches apps) is enough to kill it. Without
  // this try/catch, `reader.read()` throwing here left the bubble stuck on
  // "Thinking…" forever with no way to tell the conversation had stalled.
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sepIndex;
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);

        const eventLine = rawEvent.split("\n").find((l) => l.startsWith("event:"));
        const dataLine = rawEvent.split("\n").find((l) => l.startsWith("data:"));
        if (!eventLine || !dataLine) continue;

        const event = eventLine.slice(6).trim();
        const data = JSON.parse(dataLine.slice(5).trim());

        if (event === "sources") {
          turn.sources = data.sources || [];
          turn.sourceMode = data.source_mode || null;
        } else if (event === "token") {
          if (!started) {
            turn.botBubble.textContent = "";
            started = true;
          }
          text += data.text;
          turn.botBubble.textContent = text;
          scrollToBottom();
          onToken?.(data.text, text);
        } else if (event === "error") {
          sseError = data.error;
          turn.botBubble.textContent = `Error: ${data.error}`;
        }
      }
    }
  } catch (err) {
    // Deliberately cancelled mid-stream, not a real connection problem —
    // leave the bubble showing whatever text had streamed in so far
    // rather than an alarming "interrupted" message for something the
    // user themselves chose to move on from.
    if (err.name === "AbortError") {
      turn.botText = text;
      turn.botBubble.textContent = text || "[Cancelled]";
      turn.statusText = turn.botBubble.textContent;
      finalizeBotTurn(turn);
      return;
    }
    streamError = err;
  }

  turn.botText = text;
  if (streamError) {
    turn.botBubble.textContent = text
      ? `${text}\n\n[Connection interrupted — tap Retry to continue.]`
      : "Connection interrupted before a response arrived. Tap Retry.";
  } else if (sseError) {
    // Keep the real backend error on screen — don't let the "no tokens
    // arrived" fallback below clobber it with a generic, unhelpful message.
    turn.botBubble.textContent = `Error: ${sseError}`;
  } else if (!started && !text) {
    turn.botBubble.textContent = "(no response)";
  }
  // Captured before the sources <div> below is appended, so voice mode's
  // fallback (turn.botText is empty) reads this plain message instead of
  // botBubble.textContent, which would otherwise run the sources caption
  // straight onto the end of it with no separator — both on screen and,
  // worse, read aloud verbatim by the TTS queue.
  turn.statusText = turn.botBubble.textContent;

  if (turn.sourceMode && SOURCE_LABELS[turn.sourceMode]) {
    const src = document.createElement("div");
    src.className = "sources";
    src.textContent = SOURCE_LABELS[turn.sourceMode](turn.sources);
    turn.botBubble.appendChild(src);
  }

  finalizeBotTurn(turn);
  scrollToBottom();
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  const image = pendingImage;
  if (!message && !image) return;

  chatInput.value = "";
  chatInput.disabled = true;
  clearPendingImage();

  try {
    await sendTurn(message, image);
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
  }
});

// ---- Docs / reset (non-turn system messages) ----

function addSystemMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot";
  const col = document.createElement("div");
  col.className = "message-col";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  col.appendChild(bubble);
  wrapper.appendChild(col);
  chatWindow.appendChild(wrapper);
  scrollToBottom();
}

ingestBtn.addEventListener("click", async () => {
  ingestBtn.disabled = true;
  ingestBtn.textContent = "Indexing…";
  try {
    const response = await fetch("/api/ingest", { method: "POST" });
    const data = await response.json();
    addSystemMessage(
      data.chunks_indexed
        ? `Indexed ${data.chunks_indexed} chunks from data/docs/.`
        : "No documents found in data/docs/."
    );
  } finally {
    ingestBtn.disabled = false;
    ingestBtn.textContent = "Reindex docs";
  }
});

docInput.addEventListener("change", async () => {
  const file = docInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  addSystemMessage(`Uploading ${file.name}...`);
  try {
    const response = await fetch("/api/upload_doc", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      addSystemMessage(`Error: ${data.error || "upload failed"}`);
    } else if (data.warning) {
      addSystemMessage(`⚠️ ${data.warning}`);
    } else {
      addSystemMessage(
        `Uploaded ${data.filename} — indexed ${data.file_chunks} chunk(s) from it ` +
        `(${data.chunks_indexed} total in the knowledge base).`
      );
    }
  } catch (err) {
    addSystemMessage(`Network error: ${err.message}`);
  } finally {
    docInput.value = "";
  }
});

resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  chatWindow.innerHTML = "";
  turns = [];
  countPriorTextTurnsOnServer = 0;
  clearPendingImage();
  addSystemMessage("Conversation cleared. Ask me anything!");
});

// ---- Voice conversation ----

const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
const speechSupported = !!SpeechRecognitionCtor;

let recognition = null;
let recognitionActive = false;
let ttsMuted = false;
let voiceBusy = false;
// Bumped every time a new voice question starts. A turn captures its own
// value at the start and checks it before touching shared UI state (status
// text, mic enabled) or speaking its reply — if the user has since started
// a newer question, an older turn's late-arriving result is a no-op instead
// of stomping on the newer turn already in progress.
let voiceTurnToken = 0;
// Cancels the current turn's /api/chat request — set for the duration of
// one turn so a mic tap that interrupts it (see interruptVoiceTurn) can
// actually stop the abandoned request instead of just ignoring its result.
let voiceTurnAbortController = null;
// The full text of the last reply voice mode spoke (successful or not), so
// "Speak again" has something to replay after Stop or after it finishes.
let lastSpokenText = "";
// Sentences of the current/last reply, in order, each tracking whether it
// actually finished playing. Lets "Speak again" resume from the first
// sentence Stop cut off instead of always restarting from the top — see
// stopSpeaking() and the voiceReplayBtn click handler.
let speechSentences = [];
// True from a Stop-button press until the user resumes (voiceReplayBtn) or
// starts a new question. Deliberately a plain flag rather than reusing
// speechGeneration for this: speechGeneration is frozen per-turn at the
// moment a reply starts speaking, so gating on it would leave sentences
// that stream in *after* a Resume click permanently stranded (their frozen
// value can never match again). This flag is live, so enqueueSpeech checks
// it fresh for every sentence as it arrives — including ones the model is
// still streaming in after the user has already hit Resume.
let voiceSpeechPaused = false;
// Rebuilt on every onresult from the full (growing) results list — see
// createRecognition() — rather than only from isFinal chunks, so whatever
// the user was mid-word on is still captured if listening ends abruptly.
let voiceCurrentText = "";
let voiceSilenceTimer = null;
// True only when the current stop was "user is done talking" (debounce
// timeout or a manual mic tap) — false for a background/cleanup stop, so
// onend knows whether to actually send voiceCurrentText or just discard it.
let voiceFinalizePending = false;

if (!speechSupported) {
  voiceMicBtn.disabled = true;
  voiceMicBtn.title = "Voice input isn't supported in this browser";
  voiceStatus.textContent = "Voice input isn't supported in this browser — try Chrome or Edge.";
}

// Chrome loads its voice list asynchronously and the very first speak()
// call can be silently dropped while it's still empty. Poking getVoices()
// up front kicks that loading off well before the first reply needs it.
if (window.speechSynthesis) {
  window.speechSynthesis.getVoices();
}

// A fresh SpeechRecognition instance every time, rather than one cached
// singleton reused across turns — reusing a single instance's start/stop
// cycle is flaky across browsers (confirmed live: the first voice question
// in a session worked fine, but every question after it just sat in
// "Listening…" forever with no result, on both desktop and mobile).
function createRecognition() {
  const rec = new SpeechRecognitionCtor();
  rec.lang = navigator.language || "en-US";
  // Continuous, not a single-utterance session: continuous=false let
  // Chrome's own silence detector end the *whole* session — finalizing and
  // sending whatever had been captured so far — after the very first brief
  // pause, cutting real questions off mid-sentence (confirmed live: asking
  // a normal-length question with any natural pause in it only sent the
  // first half). Continuous mode keeps the mic open across pauses;
  // finishVoiceInput()'s own longer trailing-silence timer below decides
  // when the user has actually stopped talking instead.
  rec.continuous = true;
  rec.interimResults = true;

  rec.onresult = (event) => {
    // Rebuilt from the full results list (not just the newly-changed
    // range) every time, so this always holds the best current guess at
    // the complete utterance — including whatever's still interim — no
    // matter when listening ends.
    let confirmedFinal = "";
    let interim = "";
    for (let i = 0; i < event.results.length; i++) {
      const chunk = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        confirmedFinal += chunk;
      } else {
        interim += chunk;
      }
    }
    voiceCurrentText = (confirmedFinal + interim).trim();
    voiceTranscript.textContent = voiceCurrentText;
    // Any new speech — final or still-interim — means the user isn't done
    // yet: push the "are they finished talking" deadline back out.
    clearTimeout(voiceSilenceTimer);
    voiceSilenceTimer = setTimeout(finishVoiceInput, 1500);
  };

  rec.onerror = (event) => {
    recognitionActive = false;
    voiceWave.classList.remove("listening");
    voiceMicBtn.classList.remove("active");
    clearTimeout(voiceSilenceTimer);
    voiceSilenceTimer = null;
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      voiceStatus.textContent = "Microphone access was denied.";
    } else if (event.error === "no-speech") {
      voiceStatus.textContent = "Didn't catch that — tap the mic and try again.";
    } else {
      voiceStatus.textContent = "Voice input error — tap the mic to try again.";
    }
  };

  rec.onend = () => {
    recognitionActive = false;
    voiceWave.classList.remove("listening");
    voiceMicBtn.classList.remove("active");
    clearTimeout(voiceSilenceTimer);
    voiceSilenceTimer = null;
    const finalText = voiceCurrentText.trim();
    voiceCurrentText = "";
    const shouldSend = voiceFinalizePending && finalText;
    voiceFinalizePending = false;
    if (shouldSend) {
      handleVoiceFinalTranscript(finalText);
    } else if (!voiceBusy) {
      voiceStatus.textContent = "Tap the mic to talk";
    }
  };

  return rec;
}

function startListening() {
  if (!speechSupported || voiceBusy) return;
  recognition = createRecognition();
  try {
    recognition.start();
  } catch (err) {
    return; // already running
  }
  recognitionActive = true;
  voiceCurrentText = "";
  voiceFinalizePending = false;
  voiceWave.classList.add("listening");
  voiceMicBtn.classList.add("active");
  voiceStatus.textContent = "Listening…";
  voiceTranscript.textContent = "";
}

// The user is done talking — either finishVoiceInput's own trailing-silence
// timeout fired, or they tapped the mic to end their turn manually. Either
// way, stop and send whatever's been captured.
function finishVoiceInput() {
  clearTimeout(voiceSilenceTimer);
  voiceSilenceTimer = null;
  if (recognition && recognitionActive) {
    voiceFinalizePending = true;
    recognition.stop(); // onend below does the actual send
  }
}

// Cancel listening without sending — used for background/visibility
// cleanup, where the mic going quiet isn't the user signaling "I'm done."
function stopListening() {
  clearTimeout(voiceSilenceTimer);
  voiceSilenceTimer = null;
  voiceFinalizePending = false;
  if (recognition && recognitionActive) {
    recognition.stop();
  }
  recognitionActive = false;
  voiceWave.classList.remove("listening");
  voiceMicBtn.classList.remove("active");
}

// Backgrounding the tab/app (switching apps, locking the screen) doesn't
// change which in-page view is showing, so showView()'s own cleanup never
// runs — without this, the mic stays hot and listening in the background
// until the whole app is force-killed. visibilitychange covers switching
// apps or locking the screen; pagehide covers navigating away entirely.
// Both are cheap to call when nothing is actually active.
function stopVoiceForBackground() {
  if (recognitionActive) stopListening();
  stopSpeaking();
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopVoiceForBackground();
});
window.addEventListener("pagehide", stopVoiceForBackground);

// Tapping the mic while the current turn is still being generated or
// spoken is "never mind, here's my real question" — abandon it (cancel its
// request, stop any speech) and start listening right away, rather than
// forcing the user to wait out the whole reply before they can ask
// something else.
function interruptVoiceTurn() {
  voiceTurnToken++; // the abandoned turn's own completion becomes a no-op
  if (voiceTurnAbortController) voiceTurnAbortController.abort();
  stopSpeaking();
  // Drop the interrupted turn entirely — from the local turns[] array (so
  // the next turn's truncate_to rewinds the server's memory past it too)
  // and from the chat log's DOM — rather than leaving a half-finished
  // reply sitting in conversation history. Confirmed live: without this,
  // interrupting a dragon story mid-sentence and then asking "what is 2
  // plus 2" got a continuation of the dragon story back, because the
  // truncated reply was still in memory as context for the next question.
  const abandoned = turns.pop();
  if (abandoned) {
    abandoned.userWrapper?.remove();
    abandoned.botWrapper?.remove();
  }
  // The abandoned reply's text is gone from the chat log — nothing left to
  // resume or replay.
  speechSentences = [];
  lastSpokenText = "";
  updateReplayButtonVisibility();
  voiceBusy = false;
  voiceMicBtn.disabled = false;
  startListening();
}

voiceMicBtn.addEventListener("click", () => {
  // Must run synchronously inside this click handler — see
  // unlockSpeechSynthesisOnce()'s comment for why.
  unlockSpeechSynthesisOnce();
  if (recognitionActive) {
    finishVoiceInput();
  } else if (voiceBusy) {
    interruptVoiceTurn();
  } else {
    // Idle — nothing being generated or spoken. stopSpeaking() here is just
    // a safety net for any straggling audio (e.g. a Speak Again replay).
    stopSpeaking();
    startListening();
  }
});

voiceMuteBtn.addEventListener("click", () => {
  ttsMuted = !ttsMuted;
  voiceMuteBtn.textContent = ttsMuted ? "🔇" : "🔊";
  voiceMuteBtn.title = ttsMuted ? "Spoken replies are off" : "Toggle spoken replies";
  if (ttsMuted) stopSpeaking();
});

// Chrome bug: calling speechSynthesis.cancel() (done below, at the start of
// every voice turn, to stop the previous reply's speech) can leave the
// engine stuck in a "paused" state where speak() silently queues audio that
// never actually plays — text shows up but nothing is heard. resume() is a
// harmless no-op when it's already speaking, and unsticks it when it isn't.
// A second, unrelated Chrome bug auto-pauses long utterances (~15s+) mid-
// sentence; nudging resume() on an interval while speech is pending/active
// works around that too.
let ttsResumeTimer = null;
function ensureTtsResumeGuard() {
  const synth = window.speechSynthesis;
  if (ttsResumeTimer || !synth) return;
  ttsResumeTimer = setInterval(() => {
    if (!synth.speaking && !synth.pending) {
      clearInterval(ttsResumeTimer);
      ttsResumeTimer = null;
      return;
    }
    synth.resume();
  }, 4000);
}

// Safari has a bug where a SpeechSynthesisUtterance with no other
// reference can be garbage-collected mid-speech, killing the audio
// silently. Holding it here for as long as it's in flight prevents that.
const pendingUtterances = new Set();

function speakUtterance(utterance) {
  pendingUtterances.add(utterance);
  const release = () => {
    pendingUtterances.delete(utterance);
    updateStopButtonVisibility();
  };
  utterance.addEventListener("end", release);
  utterance.addEventListener("error", release);
  window.speechSynthesis.speak(utterance);
  updateStopButtonVisibility();
}

// Picks a female-sounding system voice for the browser TTS fallback, so it
// stays consistent with the server-side voice (Hannah, Orpheus) rather than
// whatever the browser's platform default happens to be (often male).
// Matched by name since the Web Speech API exposes no gender field.
const FEMALE_VOICE_NAME = /female|zira|aria|jenny|samantha|susan|victoria|karen|moira|tessa|fiona|kate|serena|allison|ava|hazel|salli|joanna|kendra|kimberly|ivy/i;
function pickFemaleBrowserVoice(lang) {
  const synth = window.speechSynthesis;
  if (!synth) return null;
  const voices = synth.getVoices();
  const inLang = voices.filter((v) => v.lang?.toLowerCase().startsWith(lang.slice(0, 2).toLowerCase()));
  const pool = inLang.length ? inLang : voices;
  return pool.find((v) => FEMALE_VOICE_NAME.test(v.name)) || null;
}

// Fallback voice — the browser's own built-in (robotic-sounding) TTS.
// Used only when server-side synthesis (below) isn't available or fails,
// so voice mode never goes completely silent. Returns a promise that
// resolves only once the utterance actually finishes (or errors) — callers
// chain sentences through this, so if it resolved immediately instead, the
// next (server-side, human-voiced) sentence would start playing on top of
// this one still talking, producing two overlapping voices.
function speakWithBrowserVoice(text) {
  return new Promise((resolve) => {
    if (!window.speechSynthesis || !text) {
      resolve();
      return;
    }
    const lang = navigator.language || "en-US";
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    const femaleVoice = pickFemaleBrowserVoice(lang);
    if (femaleVoice) utterance.voice = femaleVoice;
    utterance.addEventListener("end", resolve);
    utterance.addEventListener("error", resolve);
    speakUtterance(utterance);
    window.speechSynthesis.resume();
    ensureTtsResumeGuard();
  });
}

// ---- Server-side TTS (Groq Orpheus) — a natural, human-sounding voice
// instead of the browser's built-in one. ----
const ttsAudioEl = new Audio();
ttsAudioEl.preload = "auto";

// Bumped by stopSpeaking() — every in-flight synthesis/playback captures
// the generation it started under and checks it before doing anything
// further, so once Stop is pressed nothing already in flight (including
// sentences from a reply that's still streaming in after the click) can
// still end up audible.
let speechGeneration = 0;

// One token per sentence currently synthesizing/queued/playing — the stop
// button should be visible for the whole span from "asked for" to "done
// speaking", not just while audio happens to be playing. A Set of tokens
// (rather than a plain counter) so a late-arriving decrement from a
// request that stopSpeaking() already cleared can't wrongly cancel a
// later turn's count.
const pendingSpeechTokens = new Set();

// All /api/speak fetches currently in flight, so stopSpeaking() can abort
// every one of them at once — there can be more than one, since synthesis
// for sentence N+1 now starts as soon as it's enqueued, in parallel with
// sentence N still playing (see enqueueSpeech), not only after.
const activeSpeakControllers = new Set();

// Sentences must still be *played* in the order they were spoken even
// though synthesis for each one races ahead independently — this promise
// chain enforces playback order without making synthesis wait its turn.
let playbackQueue = Promise.resolve();

// stopSpeaking() dispatches this so a playServerAudio() call that's
// mid-playback can resolve its promise immediately instead of hanging —
// ttsAudioEl.pause() stops the sound but fires neither "ended" nor
// "error", so without this the playback queue would otherwise stall
// forever on the sentence that was playing when Stop was pressed.
const speechStopEmitter = new EventTarget();

function updateStopButtonVisibility() {
  voiceStopBtn.hidden = pendingSpeechTokens.size === 0;
}

// Reflects speechSentences into the replay button. Hidden while there's
// nothing spoken yet, or while speech is actively playing (the Stop button
// covers that state — showing "Resume" mid-playback would be misleading).
// Once playback isn't active — either Stop was pressed, or the reply
// finished naturally — it shows "Resume" if that left an unplayed sentence
// behind, or "Speak again" once every recorded sentence actually played.
function updateReplayButtonVisibility() {
  voiceReplayBtn.hidden = speechSentences.length === 0 || pendingSpeechTokens.size > 0;
  if (voiceReplayBtn.hidden) return;
  const hasUnplayed = speechSentences.some((s) => !s.played);
  voiceReplayBtn.textContent = hasUnplayed ? "▶ Resume" : "🔁 Speak again";
  voiceReplayBtn.title = hasUnplayed
    ? "Resume speaking from where it stopped"
    : "Speak the last reply again";
}

function stopSpeaking() {
  speechGeneration++;
  for (const controller of activeSpeakControllers) controller.abort();
  activeSpeakControllers.clear();
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  ttsAudioEl.pause();
  ttsAudioEl.currentTime = 0;
  speechStopEmitter.dispatchEvent(new Event("stop"));
  playbackQueue = Promise.resolve(); // drop any of the previous reply's still-queued sentences
  pendingSpeechTokens.clear();
  updateStopButtonVisibility();
  // Whatever sentence was cut off is still marked unplayed (its own
  // in-flight .then never got to set played=true because the generation it
  // captured no longer matches), so the replay button can flip to "Resume"
  // immediately — it doesn't need to wait for the reply to finish streaming.
  updateReplayButtonVisibility();
}

voiceStopBtn.addEventListener("click", () => {
  voiceSpeechPaused = true;
  stopSpeaking();
});

// Strips markdown/formatting the model may emit — bold/italic markers,
// headings, bullets, links, inline/fenced code, emoji — so voice replies
// read as natural speech instead of literally saying "asterisk asterisk"
// or reciting a URL. Text mode still shows the raw markdown; this only
// touches what gets spoken.
const EMOJI_PATTERN = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2190}-\u{21FF}\u{2B00}-\u{2BFF}]/gu;
function sanitizeForSpeech(text) {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_~]{1,3}([^*_~]+)[*_~]{1,3}/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(EMOJI_PATTERN, "")
    .replace(/\s+/g, " ")
    .trim();
}

function playServerAudio(blob, generation) {
  return new Promise((resolve) => {
    if (generation !== speechGeneration) {
      resolve();
      return;
    }
    const url = URL.createObjectURL(blob);
    const finish = () => {
      URL.revokeObjectURL(url);
      ttsAudioEl.removeEventListener("ended", finish);
      ttsAudioEl.removeEventListener("error", finish);
      speechStopEmitter.removeEventListener("stop", finish);
      updateStopButtonVisibility();
      resolve();
    };
    ttsAudioEl.addEventListener("ended", finish);
    ttsAudioEl.addEventListener("error", finish);
    speechStopEmitter.addEventListener("stop", finish, { once: true });
    ttsAudioEl.src = url;
    const playPromise = ttsAudioEl.play();
    if (playPromise && playPromise.catch) playPromise.catch(finish);
    updateStopButtonVisibility();
  });
}

// Synthesizes one sentence and returns a descriptor for playSynthesized()
// to play later — deliberately doesn't play it itself, so synthesis for
// several sentences can be in flight at once (see enqueueSpeech) instead
// of each one waiting for the previous sentence to finish *playing*
// before its own network request even starts.
function speakServerSide(text, generation) {
  const controller = new AbortController();
  activeSpeakControllers.add(controller);
  return fetch("/api/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal: controller.signal,
  })
    .then((response) => {
      if (!response.ok) throw new Error("tts request failed");
      return response.blob();
    })
    .then((blob) => ({ blob }))
    .catch((err) => ({ err }))
    .finally(() => activeSpeakControllers.delete(controller));
}

async function playSynthesized(result, text, generation) {
  if (generation !== speechGeneration) return;
  if (result.err) {
    // Deliberately cancelled via the Stop button — don't fall back to the
    // browser voice for text the user just asked to stop hearing.
    if (result.err.name === "AbortError") return;
    // Server TTS isn't configured (e.g. no GROQ_API_KEY locally) or the
    // request failed — fall back rather than going silent.
    await speakWithBrowserVoice(text);
    return;
  }
  await playServerAudio(result.blob, generation);
}

// iOS Safari gates both speechSynthesis and <audio>.play() the same way:
// a play() call must originate from inside a user-gesture handler at
// least once per page session before later *async* play() calls (ones
// triggered from a fetch response, not a tap) are allowed to produce
// sound. Playing (and immediately stopping) something inaudible here,
// synchronously on the first mic tap, unlocks both for the rest of the
// session. Desktop Chrome doesn't need this, but it's a harmless no-op
// there.
let speechUnlocked = false;
function unlockSpeechSynthesisOnce() {
  if (speechUnlocked) return;
  speechUnlocked = true;
  if (window.speechSynthesis) speakUtterance(new SpeechSynthesisUtterance(" "));
  ttsAudioEl.muted = true;
  const p = ttsAudioEl.play();
  if (p && p.catch) p.catch(() => {});
  ttsAudioEl.pause();
  ttsAudioEl.currentTime = 0;
  ttsAudioEl.muted = false;
}

// Actually synthesizes+plays a batch of already-recorded sentence entries,
// chaining each through the shared playbackQueue so they play in order.
// Used both for freshly-streamed sentences (one entry at a time, from
// enqueueSpeech) and for replaying/resuming a batch at once (from the
// voiceReplayBtn handler). An entry is marked played=true only if `generation`
// still matches speechGeneration when its playback promise settles — a
// forced stop resolves that promise early (see stopSpeaking's stop event),
// which leaves it correctly marked unplayed so "Speak again" knows to
// resume from it.
function speakSentenceEntries(entries, generation) {
  for (const entry of entries) {
    if (ttsMuted) {
      entry.played = true;
      continue;
    }
    const token = Symbol("speech");
    pendingSpeechTokens.add(token);
    updateStopButtonVisibility();
    // Synthesis starts immediately (in parallel with whatever's currently
    // playing); only the actual playback is serialized through the queue,
    // so audio keeps pace with the streaming text instead of the delay of
    // each sentence's synthesis stacking on top of the previous one's.
    const synthesis = speakServerSide(entry.text, generation);
    playbackQueue = playbackQueue
      .then(() => synthesis)
      .then((result) => playSynthesized(result, entry.text, generation))
      .then(() => {
        if (generation === speechGeneration) entry.played = true;
      })
      .finally(() => {
        pendingSpeechTokens.delete(token);
        updateStopButtonVisibility();
        updateReplayButtonVisibility();
      });
  }
  // Reflects the tokens just added above — keeps the replay button hidden
  // (in favor of Stop) for the span this batch is actively playing, rather
  // than only updating once the first entry's own .finally fires.
  updateReplayButtonVisibility();
}

// Records one sentence as it streams in and, unless the user has paused
// playback (Stop button, not yet resumed), speaks it right away — checked
// live via voiceSpeechPaused rather than a generation snapshot frozen at
// turn start, so this keeps working correctly for sentences that stream in
// *after* a Resume click too (see voiceSpeechPaused's own comment). If
// paused, the sentence is still recorded (unplayed) — so "Speak again" can
// resume it later — but synthesis/playback is skipped for now.
function enqueueSpeech(text) {
  const spoken = sanitizeForSpeech(text);
  if (!spoken) return;
  const entry = { text: spoken, played: false };
  speechSentences.push(entry);
  if (voiceSpeechPaused) {
    // Nothing is (or will be) playing for it yet — reflect that immediately.
    updateReplayButtonVisibility();
    return;
  }
  // speakSentenceEntries adds a pending token and calls
  // updateReplayButtonVisibility itself once that's reflected — calling it
  // again here first would show the button for one tick using the stale
  // (pre-token) pending count.
  speakSentenceEntries([entry], speechGeneration);
}

// Matches a sentence boundary once trailing whitespace/newlines follow it,
// so we don't speak "Mr." as a full sentence. Used against a *complete*,
// already-known text (the `$` branch treats the real end of that text as a
// boundary too — correct there, since there's nothing more coming).
const SENTENCE_BOUNDARY = /[.!?](?:\s|$)/;
// Same, but for text still streaming in (onToken below): deliberately
// excludes the `$` (end-of-string) branch, since "end of the text received
// so far" isn't a sentence boundary, just wherever the stream happens to
// be paused mid-token — confirmed live: with `$` included, this
// intermittently sliced decimal numbers like "29.7" into two separate TTS
// calls ("29." / "7 astronomical...") whenever a streaming chunk happened
// to end right after the decimal point, audibly cutting the number in half.
const SENTENCE_BOUNDARY_LIVE = /[.!?]\s/;

async function handleVoiceFinalTranscript(text) {
  if (!text) return;
  // Captured up front; every UI update and speech call below checks this
  // against the live counter before acting, so if the user interrupts this
  // turn (mic tap → interruptVoiceTurn bumps the counter) to ask something
  // else, this turn's late-arriving result becomes a no-op instead of
  // overwriting the newer question already on screen.
  const myTurnToken = ++voiceTurnToken;
  voiceBusy = true;
  voiceStatus.textContent = "Thinking…";
  voiceTranscript.textContent = `You: ${text}`;
  stopSpeaking();
  // A brand-new question means a brand-new reply to track — old sentences
  // (and whatever "Resume" state they implied) no longer apply, and this
  // fresh reply starts out unpaused regardless of whether the previous one
  // was stopped.
  speechSentences = [];
  voiceSpeechPaused = false;
  updateReplayButtonVisibility();

  const controller = new AbortController();
  voiceTurnAbortController = controller;

  let spokenUpTo = 0;
  const onToken = (_chunk, fullText) => {
    if (myTurnToken !== voiceTurnToken) return;
    voiceTranscript.textContent = `You: ${text}\n\n${fullText}`;
    let boundary;
    while ((boundary = fullText.slice(spokenUpTo).search(SENTENCE_BOUNDARY_LIVE)) !== -1) {
      const cut = spokenUpTo + boundary + 1;
      enqueueSpeech(fullText.slice(spokenUpTo, cut).trim());
      spokenUpTo = cut;
    }
  };

  try {
    const turn = await sendTurn(text, null, onToken, controller.signal);
    if (myTurnToken !== voiceTurnToken) return; // superseded by a newer question
    if (turn && turn.botText) {
      voiceTranscript.textContent = `You: ${text}\n\n${turn.botText}`;
      const remainder = turn.botText.slice(spokenUpTo).trim();
      if (remainder) enqueueSpeech(remainder);
      lastSpokenText = turn.botText;
      updateReplayButtonVisibility();
    } else if (turn && turn.statusText) {
      // No usable reply text (request failed, connection dropped, empty
      // response, etc.) — statusText holds the plain error/status message
      // streamChat wrote, without the sources caption that's appended to
      // the bubble afterward, so surface that instead of silently going
      // back to "Tap the mic to talk" (and instead of reading the sources
      // caption aloud stitched onto the end of it).
      const errorText = turn.statusText;
      voiceTranscript.textContent = `You: ${text}\n\n${errorText}`;
      enqueueSpeech(errorText);
      lastSpokenText = errorText;
      updateReplayButtonVisibility();
    }
  } finally {
    if (voiceTurnAbortController === controller) voiceTurnAbortController = null;
    if (myTurnToken === voiceTurnToken) {
      voiceBusy = false;
      voiceStatus.textContent = "Tap the mic to talk";
    }
  }
}

// Splits fullText into sentences — used both to cut-as-you-go against text
// still streaming in (onToken above) and to rebuild speechSentences from
// scratch when nothing was tracked yet (defensive fallback in the
// voiceReplayBtn handler below).
function splitIntoSentences(fullText) {
  const sentences = [];
  let spokenUpTo = 0;
  let boundary;
  while ((boundary = fullText.slice(spokenUpTo).search(SENTENCE_BOUNDARY)) !== -1) {
    const cut = spokenUpTo + boundary + 1;
    sentences.push(fullText.slice(spokenUpTo, cut).trim());
    spokenUpTo = cut;
  }
  const remainder = fullText.slice(spokenUpTo).trim();
  if (remainder) sentences.push(remainder);
  return sentences;
}

voiceReplayBtn.addEventListener("click", () => {
  if (speechSentences.length === 0 && lastSpokenText) {
    // Defensive fallback — shouldn't normally happen, since enqueueSpeech
    // always records into speechSentences as the reply is spoken.
    speechSentences = splitIntoSentences(lastSpokenText)
      .map((text) => ({ text: sanitizeForSpeech(text), played: false }))
      .filter((s) => s.text);
  }
  if (speechSentences.length === 0) return;
  unlockSpeechSynthesisOnce();
  const hasUnplayed = speechSentences.some((s) => !s.played);
  stopSpeaking(); // safety net: cancel any straggling audio before (re)starting
  voiceSpeechPaused = false; // un-pause so sentences still streaming in resume too
  if (!hasUnplayed) {
    // Everything already finished playing — "Speak again" restarts the
    // whole reply from the top rather than doing nothing.
    for (const s of speechSentences) s.played = false;
  }
  speakSentenceEntries(
    speechSentences.filter((s) => !s.played),
    speechGeneration
  );
});
