const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const ragToggle = document.getElementById("rag-toggle");
const ingestBtn = document.getElementById("ingest-btn");
const resetBtn = document.getElementById("reset-btn");

function addMessage(text, role, sources) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  if (sources && sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    src.textContent = `Sources: ${sources.join(", ")}`;
    bubble.appendChild(src);
  }

  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  chatInput.value = "";
  chatInput.disabled = true;

  const thinkingBubble = addMessage("Thinking…", "bot");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, use_rag: ragToggle.checked }),
    });
    const data = await response.json();

    if (!response.ok) {
      thinkingBubble.textContent = `Error: ${data.error || "something went wrong"}`;
    } else {
      thinkingBubble.textContent = data.reply;
      if (data.sources && data.sources.length) {
        const src = document.createElement("div");
        src.className = "sources";
        src.textContent = `Sources: ${data.sources.join(", ")}`;
        thinkingBubble.appendChild(src);
      }
    }
  } catch (err) {
    thinkingBubble.textContent = `Network error: ${err.message}`;
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
  }
});

ingestBtn.addEventListener("click", async () => {
  ingestBtn.disabled = true;
  ingestBtn.textContent = "Indexing…";
  try {
    const response = await fetch("/api/ingest", { method: "POST" });
    const data = await response.json();
    addMessage(
      data.chunks_indexed
        ? `Indexed ${data.chunks_indexed} chunks from data/docs/.`
        : "No documents found in data/docs/.",
      "bot"
    );
  } finally {
    ingestBtn.disabled = false;
    ingestBtn.textContent = "Reindex docs";
  }
});

resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  chatWindow.innerHTML = "";
  addMessage("Conversation cleared. Ask me anything!", "bot");
});
