from chatbot.memory import ConversationMemory


def test_add_and_as_list():
    memory = ConversationMemory(max_turns=10)
    memory.add("user", "hi")
    memory.add("assistant", "hello")
    assert memory.as_list() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_trims_to_max_turns():
    memory = ConversationMemory(max_turns=2)
    for i in range(10):
        memory.add("user", f"msg-{i}")
        memory.add("assistant", f"reply-{i}")

    assert len(memory) == 4  # 2 turns = 4 messages
    assert memory.messages[0]["content"] == "msg-8"


def test_reset_clears_history():
    memory = ConversationMemory()
    memory.add("user", "hi")
    memory.reset()
    assert len(memory) == 0


def test_truncate_to_turns():
    memory = ConversationMemory()
    for i in range(5):
        memory.add("user", f"msg-{i}")
        memory.add("assistant", f"reply-{i}")

    memory.truncate_to_turns(2)

    assert len(memory) == 4
    assert memory.messages[-1]["content"] == "reply-1"


def test_truncate_to_turns_clamps_negative():
    memory = ConversationMemory()
    memory.add("user", "hi")
    memory.truncate_to_turns(-3)
    assert len(memory) == 0
