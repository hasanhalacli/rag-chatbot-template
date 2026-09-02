from rag_chatbot.generation.memory import ConversationMemory, SlidingWindowMemory, SummaryMemory


def test_memory_keeps_only_the_most_recent_messages():
    mem = ConversationMemory(max_messages=3)
    for i in range(5):
        mem.add_message("c1", "user", f"m{i}")
    assert [m.content for m in mem.get_history("c1")] == ["m2", "m3", "m4"]


def test_conversations_are_isolated_and_clearable():
    mem = ConversationMemory()
    mem.add_message("a", "user", "x")
    mem.add_message("b", "user", "y")
    assert sorted(mem.list_conversations()) == ["a", "b"]
    mem.clear("a")
    assert mem.get_history("a") == [] and len(mem.get_history("b")) == 1
    mem.clear_all()
    assert mem.list_conversations() == []


def test_sliding_window_returns_last_n():
    mem = SlidingWindowMemory(window_size=2, max_messages=10)
    for i in range(4):
        mem.add_message("c", "user", f"m{i}")
    assert [m.content for m in mem.get_history("c")] == ["m2", "m3"]


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, max_tokens=150):
        self.calls += 1

        class R:
            content = "SUMMARY"

        return R()


def test_summary_memory_summarises_old_turns_and_keeps_recent_ones():
    llm = _FakeLLM()
    mem = SummaryMemory(llm, summary_threshold=4)
    for i in range(5):
        mem.add_message("c", "user", f"m{i}")
    history = mem.get_history("c")
    assert llm.calls == 1
    assert history[0].role == "system" and "SUMMARY" in history[0].content
    assert [m.content for m in history[1:]] == ["m1", "m2", "m3", "m4"]
