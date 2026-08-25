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

async function sendTurn(message, image, onToken) {
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

  await streamChat(body, turn, onToken);
  return turn;
}

// Tracks how many text (non-image) turns the server-side memory currently holds,
// so we only send `truncate_to` when we're actually rewinding history.
let countPriorTextTurnsOnServer = 0;

async function streamChat(body, turn, onToken) {
  let response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    turn.botBubble.textContent = `Network error: ${err.message}`;
    finalizeBotTurn(turn);
    return;
  }

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    turn.botBubble.textContent = `Error: ${data.error || "something went wrong"}`;
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
          turn.botBubble.textContent = `Error: ${data.error}`;
        }
      }
    }
  } catch (err) {
    streamError = err;
  }

  turn.botText = text;
  if (streamError) {
    turn.botBubble.textContent = text
      ? `${text}\n\n[Connection interrupted — tap Retry to continue.]`
      : "Connection interrupted before a response arrived. Tap Retry.";
  } else if (!started && !text) {
    turn.botBubble.textContent = "(no response)";
  }

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

if (!speechSupported) {
  voiceMicBtn.disabled = true;
  voiceMicBtn.title = "Voice input isn't supported in this browser";
  voiceStatus.textContent = "Voice input isn't supported in this browser — try Chrome or Edge.";
}

function getRecognition() {
  if (recognition) return recognition;
  recognition = new SpeechRecognitionCtor();
  recognition.lang = navigator.language || "en-US";
  recognition.continuous = false;
  recognition.interimResults = true;

  recognition.onresult = (event) => {
    let interim = "";
    let final = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const chunk = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        final += chunk;
      } else {
        interim += chunk;
      }
    }
    if (interim) voiceTranscript.textContent = interim;
    if (final) {
      voiceTranscript.textContent = final;
      handleVoiceFinalTranscript(final.trim());
    }
  };

  recognition.onerror = (event) => {
    recognitionActive = false;
    voiceWave.classList.remove("listening");
    voiceMicBtn.classList.remove("active");
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      voiceStatus.textContent = "Microphone access was denied.";
    } else if (event.error === "no-speech") {
      voiceStatus.textContent = "Didn't catch that — tap the mic and try again.";
    } else {
      voiceStatus.textContent = "Voice input error — tap the mic to try again.";
    }
  };

  recognition.onend = () => {
    recognitionActive = false;
    voiceWave.classList.remove("listening");
    voiceMicBtn.classList.remove("active");
    if (!voiceBusy) voiceStatus.textContent = "Tap the mic to talk";
  };

  return recognition;
}

function startListening() {
  if (!speechSupported || voiceBusy) return;
  const rec = getRecognition();
  try {
    rec.start();
  } catch (err) {
    return; // already running
  }
  recognitionActive = true;
  voiceWave.classList.add("listening");
  voiceMicBtn.classList.add("active");
  voiceStatus.textContent = "Listening…";
  voiceTranscript.textContent = "";
}

function stopListening() {
  if (recognition && recognitionActive) {
    recognition.stop();
  }
  recognitionActive = false;
  voiceWave.classList.remove("listening");
  voiceMicBtn.classList.remove("active");
}

voiceMicBtn.addEventListener("click", () => {
  if (recognitionActive) {
    stopListening();
  } else {
    startListening();
  }
});

voiceMuteBtn.addEventListener("click", () => {
  ttsMuted = !ttsMuted;
  voiceMuteBtn.textContent = ttsMuted ? "🔇" : "🔊";
  voiceMuteBtn.title = ttsMuted ? "Spoken replies are off" : "Toggle spoken replies";
  if (ttsMuted && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
});

// Queues one utterance without interrupting what's already speaking — used
// to speak a reply sentence-by-sentence as it streams in, rather than
// waiting for the whole answer (which is what made voice replies feel very
// slow: no sound at all until generation had fully finished).
function enqueueSpeech(text) {
  if (ttsMuted || !window.speechSynthesis || !text) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = navigator.language || "en-US";
  window.speechSynthesis.speak(utterance);
}

// Matches a sentence boundary once trailing whitespace/newlines follow it,
// so we don't speak "Mr." as a full sentence.
const SENTENCE_BOUNDARY = /[.!?](?:\s|$)/;

async function handleVoiceFinalTranscript(text) {
  if (!text) return;
  voiceBusy = true;
  voiceStatus.textContent = "Thinking…";
  voiceMicBtn.disabled = true;
  voiceTranscript.textContent = `You: ${text}`;
  if (window.speechSynthesis) window.speechSynthesis.cancel();

  let spokenUpTo = 0;
  const onToken = (_chunk, fullText) => {
    voiceTranscript.textContent = `You: ${text}\n\n${fullText}`;
    let boundary;
    while ((boundary = fullText.slice(spokenUpTo).search(SENTENCE_BOUNDARY)) !== -1) {
      const cut = spokenUpTo + boundary + 1;
      enqueueSpeech(fullText.slice(spokenUpTo, cut).trim());
      spokenUpTo = cut;
    }
  };

  try {
    const turn = await sendTurn(text, null, onToken);
    if (turn && turn.botText) {
      voiceTranscript.textContent = `You: ${text}\n\n${turn.botText}`;
      const remainder = turn.botText.slice(spokenUpTo).trim();
      if (remainder) enqueueSpeech(remainder);
    }
  } finally {
    voiceMicBtn.disabled = false;
    voiceBusy = false;
    voiceStatus.textContent = "Tap the mic to talk";
  }
}
