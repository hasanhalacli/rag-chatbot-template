from rag_chatbot.generation.prompts import CONDENSE_PROMPT, RAG_PROMPT, PromptTemplate


def test_template_substitutes_variables():
    assert PromptTemplate("Hello {name}").format(name="Hasan") == "Hello Hasan"


def test_rag_prompt_carries_context_and_question():
    out = PromptTemplate(RAG_PROMPT).format(context="CTX-42", question="Q-7")
    assert "CTX-42" in out and "Q-7" in out
    assert "{" not in out, "every placeholder must be filled"


def test_condense_prompt_carries_history_and_question():
    out = PromptTemplate(CONDENSE_PROMPT).format(chat_history="HIST-91", question="QST-77")
    assert out.index("HIST-91") < out.index("QST-77")
    assert "{" not in out
