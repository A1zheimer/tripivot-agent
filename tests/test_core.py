from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

import translation_agent.corpus as corpus_module
from translation_agent import cli as cli_module
from translation_agent.agent import LongHorizonTranslationAgent
from translation_agent.backends import MappingBackend
from translation_agent.corpus import CorpusSpec, build_parallel_corpus, collect_opus_corpus
from translation_agent.domain_corpus import extract_term_candidates
from translation_agent.ingestion import (
    IngestionError,
    ensure_ingestable,
    ingest_document,
    write_ingestion_artifacts,
)
from translation_agent.memory import GlossaryMemory, MemorySystem
from translation_agent.models import (
    GlossaryEntry,
    Language,
    StyleGuide,
    TranslationDirection,
    TranslationRequest,
)
from translation_agent.preprocessing import (
    chinese_char_count,
    is_clean_parallel_pair,
    normalize_myanmar,
    normalize_text,
    pack_passages,
)
from translation_agent.training_data import (
    load_glossary,
    prepare_translation_examples,
    prompt_messages,
)


def test_text_normalization_and_quality_filter() -> None:
    assert normalize_text("\ufeffA\u00a0  B\r\n") == "A B"
    accepted, reason = is_clean_parallel_pair(
        "artificial intelligence",
        "人工智能",
        Language.ENGLISH,
        Language.CHINESE,
    )
    assert accepted
    assert reason == "accepted"

    with pytest.raises(ValueError, match="Zawgyi"):
        normalize_myanmar("\u105a")


def test_build_parallel_corpus_deduplicates_and_filters(tmp_path: Path) -> None:
    archive = tmp_path / "my-zh.zip"
    source_lines = [
        "ကွန်ပျူတာသည် အသုံးဝင်သည်။",
        "ကွန်ပျူတာသည် အသုံးဝင်သည်။",
        "this is not Burmese",
        "ဒေတာသည် အရေးကြီးသည်။",
    ]
    target_lines = [
        "计算机很有用。",
        "计算机很有用。",
        "这不是缅甸语。",
        "<b>数据很重要。</b>",
    ]
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("ALT.my-zh.my", "\n".join(source_lines))
        output.writestr("ALT.my-zh.zh", "\n".join(target_lines))

    manifest = build_parallel_corpus(
        archive,
        tmp_path / "processed",
        CorpusSpec(
            corpus="ALT",
            source_language=Language.BURMESE,
            target_language=Language.CHINESE,
            max_records=100,
            license="CC-BY-4.0",
        ),
    )

    assert manifest["stats"]["accepted_pairs"] == 1
    assert manifest["stats"]["duplicate_pairs"] == 1
    assert manifest["stats"]["rejected"]["source_script"] == 1
    assert manifest["stats"]["rejected"]["html"] == 1
    assert sum(manifest["stats"]["splits"].values()) == 1
    records = []
    for output_path in manifest["outputs"].values():
        records.extend(
            json.loads(line)
            for line in Path(output_path).read_text(encoding="utf-8").splitlines()
            if line
        )
    assert len(records) == 1
    assert records[0]["source_language"] == "my"


def test_collect_opus_reuses_local_archive_without_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "raw" / "ALT" / "my-zh.txt.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("ALT.my-zh.my", "ကွန်ပျူတာသည် အသုံးဝင်သည်။")
        output.writestr("ALT.my-zh.zh", "计算机很有用。")

    def fail_if_called(_spec: CorpusSpec) -> dict[str, object]:
        raise AssertionError("OPUS API should not be called when the archive exists")

    monkeypatch.setattr(corpus_module, "resolve_opus_download", fail_if_called)
    manifest = collect_opus_corpus(
        CorpusSpec(
            corpus="ALT",
            source_language=Language.BURMESE,
            target_language=Language.CHINESE,
            max_records=1,
            license="CC-BY-4.0",
        ),
        tmp_path / "raw",
        tmp_path / "processed",
    )

    assert manifest["stats"]["accepted_pairs"] == 1
    assert manifest["archive"]["metadata"]["local_reuse"] is True


def test_long_horizon_agent_plans_remembers_and_enforces_terms(tmp_path: Path) -> None:
    entries = [
        GlossaryEntry(
            concept_id="ai",
            terms={"en": "artificial intelligence", "zh": "人工智能", "my": "ဉာဏ်ရည်တု"},
            domain="technology",
        ),
        GlossaryEntry(
            concept_id="quality",
            terms={"en": "quality", "zh": "质量", "my": "အရည်အသွေး"},
            domain="general",
        ),
    ]
    memory = MemorySystem(tmp_path / "state", GlossaryMemory(entries))
    backend = MappingBackend(
        {
            "Overview": "概述",
            "artificial intelligence improves quality.": "人工智能提升质量。",
        }
    )
    agent = LongHorizonTranslationAgent(backend, tmp_path / "state", memory=memory)
    try:
        artifact = agent.translate_document(
            "# Overview\n\nartificial intelligence improves quality.",
            Language.ENGLISH,
            Language.CHINESE,
            style=StyleGuide(domain="technology"),
        )
        episodes = memory.episodic.document_episodes(artifact.report.document_id)
    finally:
        agent.close()

    assert artifact.text == "# 概述\n\n人工智能提升质量。\n"
    assert artifact.report.completed_chunks == 2
    assert artifact.report.terminology_hits == 2
    assert not artifact.report.issues
    assert len(episodes) == 2
    assert list((tmp_path / "state" / "progress").glob("*.progress.json"))


def test_pack_passages_enforces_hundred_chinese_chars() -> None:
    short = "这是一句不够长的话。"
    long = "机器学习系统需要高质量数据来保持术语一致性。" * 8
    packed = pack_passages(short + "\n\n" + long + "\n\n" + short)
    assert packed
    assert all(chinese_char_count(item) >= 100 for item in packed)


def test_classify_domain_prefers_distinct_keywords() -> None:
    from translation_agent.domain_corpus import classify_domain

    assert classify_domain("机器学习", "机器学习算法与计算机软件系统") == "tech"
    assert classify_domain("外交", "联合国与双边外交条约") == "intl"
    assert classify_domain("央行", "利率、通胀与银行货币政策") == "finance"
    # Near-ties are discarded instead of mislabeled.
    assert classify_domain("混合", "计算机软件与联合国外交") is None
    texts = ["人工智能与机器学习是计算机科学的核心方向。"] * 6
    terms = extract_term_candidates(texts, limit=20)
    assert "人工智能" in terms or "机器学习" in terms
    assert "计算机科学" in terms  # 5-char terms must be mineable whole
    assert all("的" not in term and "与" not in term for term in terms)


def test_code_fence_preserved_verbatim(tmp_path: Path) -> None:
    class Recording(MappingBackend):
        def __init__(self, translations: dict[str, str]) -> None:
            super().__init__(translations)
            self.seen: list[str] = []

        def translate(self, request):  # type: ignore[override]
            self.seen.append(request.text)
            return super().translate(request)

    memory = MemorySystem(tmp_path / "state", GlossaryMemory([]))
    backend = Recording({"Intro": "引言", "Prose here.": "正文内容。"})
    agent = LongHorizonTranslationAgent(backend, tmp_path / "state", memory=memory)
    document = "# Intro\n\nProse here.\n\n```python\n# this is a comment\nx = 1\n```\n"
    try:
        artifact = agent.translate_document(document, Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()

    # The fenced block survives byte-for-byte and is never sent to a model.
    assert "```python\n# this is a comment\nx = 1\n```" in artifact.text
    assert all("# this is a comment" not in text for text in backend.seen)
    assert "this is a comment" not in [section.title for section in artifact.plan.sections]
    assert artifact.text.startswith("# 引言")


def test_resume_skips_completed_chunks(tmp_path: Path) -> None:
    translations = {
        "T": "题",
        "First sentence.": "第一句。",
        "Second sentence.": "第二句。",
        "Third sentence.": "第三句。",
    }

    class Flaky(MappingBackend):
        def __init__(self, fail_on: set[str] | None = None) -> None:
            super().__init__(translations)
            self.fail_on = fail_on or set()
            self.calls: list[str] = []

        def translate(self, request):  # type: ignore[override]
            self.calls.append(request.text)
            if request.text in self.fail_on:
                raise RuntimeError("boom")
            return super().translate(request)

    state = tmp_path / "state"
    document = "# T\n\nFirst sentence.\n\nSecond sentence.\n\nThird sentence."

    memory = MemorySystem(state, GlossaryMemory([]))
    agent = LongHorizonTranslationAgent(Flaky({"Second sentence."}), state, memory=memory)
    try:
        first = agent.translate_document(document, Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()
    assert first.report.failed_chunks == 1
    assert "第二句。" not in first.text  # failed chunk keeps source placeholder
    assert first.text.count("Second sentence.") == 1

    memory = MemorySystem(state, GlossaryMemory([]))
    fixed = Flaky()
    agent = LongHorizonTranslationAgent(fixed, state, memory=memory)
    try:
        second = agent.translate_document(document, Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()
    assert second.report.failed_chunks == 0
    assert "第二句。" in second.text
    assert fixed.calls == ["Second sentence."]  # only the failed chunk reruns


def test_enforcement_matches_injected_glossary(tmp_path: Path) -> None:
    # 25 terms all occur in the text; only 20 fit the prompt budget. The
    # reflector must enforce exactly the injected 20, never the unseen 5.
    entries = [
        GlossaryEntry(concept_id=f"t{index}", terms={"en": f"term{index}", "zh": f"术语{index}"})
        for index in range(25)
    ]
    source = " ".join(f"term{index}" for index in range(25)) + "."
    memory = MemorySystem(tmp_path / "state", GlossaryMemory(entries))
    agent = LongHorizonTranslationAgent(MappingBackend({}), tmp_path / "state", memory=memory)
    try:
        artifact = agent.translate_document(source, Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()
    assert not [issue for issue in artifact.report.issues if "terminology" in issue.kind]


def test_word_boundary_and_casefold_target_matching(tmp_path: Path) -> None:
    ai = GlossaryEntry(concept_id="ai", terms={"en": "AI", "zh": "人工智能"})
    quality = GlossaryEntry(concept_id="q", terms={"zh": "质量", "en": "Quality"})

    # zh->en: the English target appears capitalized; the check must casefold.
    memory = MemorySystem(tmp_path / "a", GlossaryMemory([quality]))
    agent = LongHorizonTranslationAgent(
        MappingBackend({"质量决定成败。": "Quality decides success."}),
        tmp_path / "a",
        memory=memory,
    )
    try:
        artifact = agent.translate_document("质量决定成败。", Language.CHINESE, Language.ENGLISH)
    finally:
        agent.close()
    assert artifact.report.terminology_hits == 1
    assert not artifact.report.issues

    # en->zh: "AI" must not fire inside ordinary words like "said".
    memory = MemorySystem(tmp_path / "b", GlossaryMemory([ai]))
    agent = LongHorizonTranslationAgent(
        MappingBackend({"The robot said it maintained the chain.": "机器人说它维护了链条。"}),
        tmp_path / "b",
        memory=memory,
    )
    try:
        artifact = agent.translate_document(
            "The robot said it maintained the chain.", Language.ENGLISH, Language.CHINESE
        )
    finally:
        agent.close()
    assert artifact.report.terminology_hits == 0
    assert not artifact.report.issues


def test_conflicting_glossary_terms_reported_not_double_enforced(tmp_path: Path) -> None:
    entries = [
        GlossaryEntry(concept_id="q1", terms={"en": "quality", "zh": "质量"}),
        GlossaryEntry(concept_id="q2", terms={"en": "quality", "zh": "品质"}),
        GlossaryEntry(concept_id="q3", terms={"en": "QUALITY", "zh": "质"}),
    ]
    memory = MemorySystem(tmp_path / "state", GlossaryMemory(entries))
    # Both chunks follow the single enforced mapping (质量)…
    backend = MappingBackend({"quality first.": "质量第一。", "quality second.": "质量第二。"})
    agent = LongHorizonTranslationAgent(backend, tmp_path / "state", memory=memory)
    try:
        artifact = agent.translate_document(
            "# H\n\nquality first.\n\nquality second.", Language.ENGLISH, Language.CHINESE
        )
    finally:
        agent.close()
    # Same-source conflicts are deduped at retrieval (one mapping enforced)…
    assert len(memory.semantic.retrieve("quality", Language.ENGLISH, Language.CHINESE)) == 1
    # …and surfaced as a document-level warning, with no terminology errors.
    conflicts = [i for i in artifact.report.issues if i.kind == "glossary_conflict"]
    assert len(conflicts) == 1
    assert "质量" in conflicts[0].message and "品质" in conflicts[0].message
    assert not [i for i in artifact.report.issues if i.kind == "terminology"]


def test_symbol_edge_terms_match(tmp_path: Path) -> None:
    from translation_agent.preprocessing import term_occurs

    assert term_occurs("we use C++ here", "C++", Language.ENGLISH)
    assert term_occurs("built on .NET and C#", ".NET", Language.ENGLISH)
    assert term_occurs("built on .NET and C#", "C#", Language.ENGLISH)
    assert not term_occurs("The robot said it.", "AI", Language.ENGLISH)

    entry = GlossaryEntry(concept_id="cpp", terms={"zh": "C++语言", "en": "C++"})
    memory = MemorySystem(tmp_path / "state", GlossaryMemory([entry]))
    backend = MappingBackend({"我们用C++语言编程。": "We program in C++ language."})
    agent = LongHorizonTranslationAgent(backend, tmp_path / "state", memory=memory)
    try:
        artifact = agent.translate_document(
            "我们用C++语言编程。", Language.CHINESE, Language.ENGLISH
        )
    finally:
        agent.close()
    assert artifact.report.terminology_hits == 1
    assert not artifact.report.issues


def test_mixed_fence_markers_do_not_leak_code(tmp_path: Path) -> None:
    class Recording(MappingBackend):
        def __init__(self, translations: dict[str, str]) -> None:
            super().__init__(translations)
            self.seen: list[str] = []

        def translate(self, request):  # type: ignore[override]
            self.seen.append(request.text)
            return super().translate(request)

    memory = MemorySystem(tmp_path / "state", GlossaryMemory([]))
    backend = Recording({"Intro": "引言", "Tail prose.": "结尾。"})
    agent = LongHorizonTranslationAgent(backend, tmp_path / "state", memory=memory)
    document = "# Intro\n\n```python\nx = 1\n~~~\nsecret_code()\n```\n\nTail prose."
    try:
        artifact = agent.translate_document(document, Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()
    # A ~~~ line inside a ``` fence stays code; nothing leaks to the model.
    assert all("secret_code" not in text for text in backend.seen)
    assert "secret_code()" in artifact.text
    assert artifact.text.endswith("结尾。\n")


def test_indented_code_and_nested_lists_preserved(tmp_path: Path) -> None:
    class Recording(MappingBackend):
        def __init__(self, translations: dict[str, str]) -> None:
            super().__init__(translations)
            self.seen: list[str] = []

        def translate(self, request):  # type: ignore[override]
            self.seen.append(request.text)
            return super().translate(request)

    memory = MemorySystem(tmp_path / "state", GlossaryMemory([]))
    backend = Recording({"Top item.": "顶层项。", "Nested item.": "嵌套项。", "Prose.": "正文。"})
    agent = LongHorizonTranslationAgent(backend, tmp_path / "state", memory=memory)
    # Indented code is pandoc's DOCX output form for code blocks.
    document = (
        "# Title\n"
        "\n"
        "Prose.\n"
        "\n"
        "    x = secret_code()\n"
        "    y = 2\n"
        "\n"
        "- Top item.\n"
        "  - Nested item.\n"
    )
    try:
        artifact = agent.translate_document(document, Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()
    # Indented code never reaches the model and survives verbatim.
    assert all("secret_code" not in text for text in backend.seen)
    assert "    x = secret_code()\n    y = 2" in artifact.text
    # Nested list keeps its marker and indentation.
    assert "- 顶层项。\n  - 嵌套项。" in artifact.text


def test_corrupt_progress_snapshot_is_ignored(tmp_path: Path) -> None:
    from translation_agent.memory import ProgressMemory

    progress = ProgressMemory(tmp_path / "progress")
    for index, junk in enumerate(("null", "123", "[1, 2]", '"str"')):
        snapshot = tmp_path / "progress" / f"junk{index}.progress.json"
        snapshot.write_text(junk, encoding="utf-8")
        assert progress.load(f"junk{index}") is None  # must not raise

    # End-to-end: a corrupt snapshot never crashes the agent.
    memory = MemorySystem(tmp_path / "state", GlossaryMemory([]))
    agent = LongHorizonTranslationAgent(
        MappingBackend({"Hello world.": "你好世界。"}), tmp_path / "state", memory=memory
    )
    try:
        artifact = agent.translate_document("Hello world.", Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()
    assert artifact.report.completed_chunks == 1


def test_empty_backend_content_is_rejected(tmp_path: Path) -> None:
    from translation_agent.backends import BackendResult as _Result

    class Empty(MappingBackend):
        def translate(self, request):  # type: ignore[override]
            return _Result(text="   ", metadata={"backend": "empty"})

    memory = MemorySystem(tmp_path / "state", GlossaryMemory([]))
    agent = LongHorizonTranslationAgent(Empty({}), tmp_path / "state", memory=memory)
    try:
        artifact = agent.translate_document(
            "# T\n\nReal prose here.", Language.ENGLISH, Language.CHINESE
        )
    finally:
        agent.close()
    # Empty output must fail loudly, not render as a green report.
    assert artifact.report.failed_chunks == 2
    assert all(issue.kind == "chunk_error" for issue in artifact.report.issues)
    assert artifact.text.startswith("# T\n")  # failed heading keeps source placeholder
    assert "Real prose here." in artifact.text


def test_openai_backend_retries_then_reports_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import urllib.error
    import urllib.request

    import translation_agent.backends as backends_module
    from translation_agent.backends import OpenAICompatibleBackend, TranslationBackendError

    monkeypatch.setattr(backends_module.time, "sleep", lambda _seconds: None)
    backend = OpenAICompatibleBackend("m", base_url="http://localhost/v1", max_retries=1)
    direction = TranslationDirection(Language.ENGLISH, Language.CHINESE)
    request = TranslationRequest(text="hi", direction=direction, style=StyleGuide())

    calls = {"count": 0}

    def payload_response(payload: dict[str, object]) -> io.TextIOWrapper:
        return io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode()))

    def flaky(_request, timeout=None):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                "url", 503, "unavailable", None, io.BytesIO(b"busy")
            )
        return payload_response(
            {"choices": [{"message": {"content": "好"}, "finish_reason": "stop"}], "usage": {}}
        )

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    result = backend.translate(request)
    assert result.text == "好"
    assert calls["count"] == 2  # one retry on HTTP 503

    def truncated(_request, timeout=None):  # noqa: ANN001
        return payload_response(
            {"choices": [{"message": {"content": "部分"}, "finish_reason": "length"}]}
        )

    monkeypatch.setattr(urllib.request, "urlopen", truncated)
    with pytest.raises(TranslationBackendError, match="truncated"):
        backend.translate(request)


def test_align_on_pivot_rejects_first_side_ambiguity(tmp_path: Path) -> None:
    from translation_agent.corpus import align_on_pivot

    def write(path: Path, rows: list[dict[str, object]]) -> Path:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    first = write(
        tmp_path / "my-zh.jsonl",
        [
            {
                "source_language": "my",
                "target_language": "zh",
                "source_text": "မြန်မာ တစ်",
                "target_text": "共享中文句。",
                "corpus": "ALT",
            },
            {
                "source_language": "my",
                "target_language": "zh",
                "source_text": "မြန်မာ နှစ်",
                "target_text": "共享中文句。",
                "corpus": "ALT",
            },
        ],
    )
    second = write(
        tmp_path / "en-zh.jsonl",
        [
            {
                "source_language": "en",
                "target_language": "zh",
                "source_text": "shared english",
                "target_text": "共享中文句。",
                "corpus": "ALT",
            },
        ],
    )
    stats = align_on_pivot(first, second, tmp_path / "tri.jsonl")
    assert stats["written"] == 0
    assert stats["ambiguous_first_pivots"] == 2


def test_degenerate_and_boundary_detectors() -> None:
    from translation_agent.preprocessing import (
        has_clean_term_boundaries,
        is_degenerate_text,
    )

    loop = "ဒေတာ" + "ဆိုင်ရာ" * 40
    assert is_degenerate_text(loop)
    assert is_degenerate_text("the inaccuracies prove " * 20)
    # Enumerations are legitimate: a 20-char span repeating 4 times stays.
    enumeration = (
        "The college has the Faculty of Electronic Engineering, "
        "the Faculty of Computer Science, the Faculty of Optics "
        "and the Faculty of Communication. "
    )
    assert not is_degenerate_text(enumeration)
    assert not is_degenerate_text("正常的句子没有重复循环的内容。")
    assert not is_degenerate_text("短句")
    assert not has_clean_term_boundaries("的计算机")
    assert has_clean_term_boundaries("计算机科学")


def test_clean_glossary_file_drops_unusable_entries(tmp_path: Path) -> None:

    from translation_agent.domain_corpus import clean_glossary_file

    rows = [
        {
            "concept_id": "keep",
            "terms": {
                "zh": "国际货币基金组织",
                "en": "International Monetary Fund",
                "my": "ကမ္ဘာ့ ငွေကြေးအင်အားစု",
            },
            "domain": "finance",
            "source": "distill-nllb",
            "confidence": 0.6,
        },
        {
            "concept_id": "boundary",
            "terms": {"zh": "的计算机", "en": "It's a computer.", "my": "ကွန်ပျူတာ"},
            "domain": "tech",
            "source": "distill-nllb",
            "confidence": 0.6,
        },
        {
            "concept_id": "punct",
            "terms": {"zh": "操作系统", "en": "Operating system.", "my": "လုပ်ငန်းစနစ်"},
            "domain": "tech",
            "source": "distill-nllb",
            "confidence": 0.6,
        },
        {
            "concept_id": "curated-keep",
            "terms": {"zh": "人工智能", "en": "artificial intelligence", "my": "ဉာဏ်ရည်တု"},
            "domain": "technology",
            "source": "curated",
            "confidence": 1.0,
        },
    ]
    path = tmp_path / "glossary.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    stats = clean_glossary_file(path)
    assert stats["before"] == 4
    assert stats["after"] == 2
    assert stats["removed"] == {"zh_boundary": 1, "en_punctuation": 1}
    kept = [
        json.loads(line)["concept_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert kept == ["keep", "curated-keep"]


def test_agent_preserves_frontmatter_lists_tables_and_code(tmp_path: Path) -> None:
    translations = {
        "Heading": "标题",
        "Alpha prose.": "正文。",
        "First item.": "第一项。",
        "Second item.": "第二项。",
        "Name\nValue": "名称\n数值",
        "Alpha\n1": "甲\n1",
        "Beta\n2": "乙\n2",
        "Tail prose.": "结尾。",
    }
    memory = MemorySystem(tmp_path / "state", GlossaryMemory([]))
    agent = LongHorizonTranslationAgent(
        MappingBackend(translations), tmp_path / "state", memory=memory
    )
    source = (
        "---\n"
        "title: Test\n"
        "---\n\n"
        "# Heading\n\n"
        "Alpha prose.\n\n"
        "- First item.\n"
        "- Second item.\n\n"
        "| Name | Value |\n"
        "|---|---:|\n"
        "| Alpha | 1 |\n"
        "| Beta | 2 |\n\n"
        "```python\n# comment\nx = 1\n```\n\n"
        "Tail prose.\n"
    )
    try:
        artifact = agent.translate_document(source, Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()

    assert artifact.text == (
        "---\n"
        "title: Test\n"
        "---\n\n"
        "# 标题\n\n"
        "正文。\n\n"
        "- 第一项。\n"
        "- 第二项。\n\n"
        "| 名称 | 数值 |\n"
        "|---|---:|\n"
        "| 甲 | 1 |\n"
        "| 乙 | 2 |\n\n"
        "```python\n# comment\nx = 1\n```\n\n"
        "结尾。\n"
    )
    assert artifact.report.completed_chunks == 11


def test_ingestion_normalizes_markdown_and_writes_report(tmp_path: Path) -> None:
    source = (
        "# Title\r\n\r\nArtificial intelligence is useful.\r\n\r\n"
        "```python\r\nx = 1\r\n```\r\n"
    )
    path = tmp_path / "input.md"
    path.write_bytes(source.encode("utf-8"))

    result = ingest_document(path, source_language=Language.ENGLISH)
    assert result.source_format == "markdown"
    assert result.markdown == source.replace("\r\n", "\n")
    assert not result.warnings
    assert result.markdown_metrics["headings"] == 1
    assert result.markdown_metrics["code_blocks"] == 1

    markdown_path = tmp_path / "canonical.md"
    report_path = tmp_path / "canonical.ingestion.json"
    write_ingestion_artifacts(result, markdown_path, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["source"]["format"] == "markdown"
    assert payload["markdown"]["headings"] == 1


def test_ingestion_rejects_unclosed_fence(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text("# Title\n\n```python\nx = 1\n", encoding="utf-8")
    result = ingest_document(path)
    assert result.warnings[0].blocking is True
    with pytest.raises(IngestionError, match="unclosed fenced code block"):
        ensure_ingestable(result)


def test_docx_ingestion_uses_pandoc_and_reports_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from translation_agent import ingestion

    path = tmp_path / "input.docx"
    document_xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/'
        b'wordprocessingml/2006/main"><w:body><w:p/><w:tbl/></w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document_xml)

    def fake_pandoc(_path: Path, source_format: str) -> tuple[str, str]:
        assert source_format == "docx"
        return "# Heading\n\nBody paragraph.\n", "pandoc test"

    monkeypatch.setattr(ingestion, "_run_pandoc", fake_pandoc)
    result = ingest_document(path)
    assert result.source_format == "docx"
    assert result.markdown.startswith("# Heading")
    assert result.source_metrics["paragraphs"] == 1
    assert result.source_metrics["tables"] == 1
    assert not result.warnings


def test_docx_tracked_changes_are_blocking(tmp_path: Path) -> None:
    from translation_agent import ingestion

    path = tmp_path / "tracked.docx"
    document_xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/'
        b'wordprocessingml/2006/main"><w:body><w:p><w:ins/></w:p></w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document_xml)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ingestion,
        "_run_pandoc",
        lambda _path, _source_format: ("Body\n", "pandoc test"),
    )
    try:
        result = ingest_document(path)
        with pytest.raises(IngestionError, match="tracked changes"):
            ensure_ingestable(result)
    finally:
        monkeypatch.undo()


def test_pdf_ingestion_removes_repeated_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self, **_kwargs: object) -> str:
            return self._text

    class Document:
        pages = [
            Page("Company Footer\nFirst page body."),
            Page("Company Footer\nSecond page body."),
            Page("Company Footer\nThird page body."),
        ]

        def __enter__(self) -> Document:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    fake_module = types.SimpleNamespace(open=lambda _path: Document())
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_module)
    path = tmp_path / "input.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    result = ingest_document(path)
    assert result.source_format == "pdf"
    assert result.markdown == "First page body.\n\nSecond page body.\n\nThird page body.\n"
    assert result.source_metrics["pages"] == 3
    assert result.source_metrics["removed_repeated_edge_lines"] == 1
    ensure_ingestable(result)


def test_cli_translate_auto_ingests_before_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("hello world", encoding="utf-8")
    output_path = tmp_path / "output.zh.md"
    monkeypatch.setattr(
        cli_module,
        "OpenAICompatibleBackend",
        lambda _model, **_kwargs: MappingBackend({"hello world": "你好世界"}),
    )
    args = cli_module.build_parser().parse_args(
        [
            "translate",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--source",
            "en",
            "--target",
            "zh",
            "--backend",
            "openai",
            "--model",
            "fake",
        ]
    )

    assert cli_module._translate(args) == 0
    canonical_path = tmp_path / "output.zh.source.md"
    ingestion_path = tmp_path / "output.zh.ingestion.json"
    assert canonical_path.read_text(encoding="utf-8") == "hello world\n"
    assert json.loads(ingestion_path.read_text(encoding="utf-8"))["source"]["format"] == "text"
    assert output_path.read_text(encoding="utf-8") == "你好世界\n"
    report = json.loads((tmp_path / "output.zh.md.report.json").read_text(encoding="utf-8"))
    assert report["ingestion"]["source_format"] == "text"


def test_training_data_mirrors_directions_and_injects_gold_terms(tmp_path: Path) -> None:
    for target in ("en", "my"):
        path = tmp_path / f"zh-{target}.tech.jsonl"
        source = "人工智能 improves finance." if target == "en" else "人工智能很重要。"
        target_text = "AI improves finance." if target == "en" else "ဉာဏ်ရည်တု အရေးကြီးသည်။"
        path.write_text(
            json.dumps(
                {
                    "id": f"tech-{target}",
                    "domain": "tech",
                    "split": "train",
                    "source_text": source,
                    "target_text": target_text,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    glossary_path = tmp_path / "glossary.jsonl"
    glossary_path.write_text(
        json.dumps(
            {
                "concept_id": "ai",
                "terms": {"zh": "人工智能", "en": "AI", "my": "ဉာဏ်ရည်တု"},
                "domain": "tech",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    glossary = load_glossary(glossary_path)
    examples, stats = prepare_translation_examples(
        tmp_path,
        domains=["tech"],
        directions=["zh-en", "en-zh", "zh-my", "my-zh"],
        glossary=glossary,
    )
    assert len(examples) == 4
    assert {item.direction for item in examples} == {"zh-en", "en-zh", "zh-my", "my-zh"}
    assert stats["examples"] == 4

    zh_to_en = next(item for item in examples if item.direction == "zh-en")
    en_to_zh = next(item for item in examples if item.direction == "en-zh")
    assert en_to_zh.source_text == "AI improves finance."
    assert en_to_zh.target_text == "人工智能 improves finance."
    assert "人工智能 => AI" in prompt_messages(zh_to_en, glossary)[1]["content"]
    assert "AI => 人工智能" in prompt_messages(en_to_zh, glossary)[1]["content"]


def test_cjk_leading_utf8_documents_are_accepted(tmp_path: Path) -> None:
    # Regression: a 16-byte prefix used to cut multi-byte characters in half
    # and reject CJK/Burmese-leading documents as "binary".
    path = tmp_path / "zh-first.md"
    path.write_text("这是一份以中文直接开头的文档，没有任何前导 ASCII 字符。\n", encoding="utf-8")
    result = ingest_document(path, source_language=Language.CHINESE)
    assert result.source_format == "markdown"
    assert "这是一份" in result.markdown
    assert not result.warnings

    utf16 = tmp_path / "utf16.md"
    utf16.write_bytes(b"\xff\xfe" + "# Test\n".encode("utf-16-le"))
    result = ingest_document(utf16)
    assert result.source_format == "markdown"
    assert result.markdown == "# Test\n"


def test_long_fence_containing_short_fence_line_stays_code(tmp_path: Path) -> None:
    class Recording(MappingBackend):
        def __init__(self, translations: dict[str, str]) -> None:
            super().__init__(translations)
            self.seen: list[str] = []

        def translate(self, request):  # type: ignore[override]
            self.seen.append(request.text)
            return super().translate(request)

    memory = MemorySystem(tmp_path / "state", GlossaryMemory([]))
    backend = Recording({"Tail.": "尾。"})
    agent = LongHorizonTranslationAgent(backend, tmp_path / "state", memory=memory)
    document = "````\nfence four\n```\nstill code inside four\n````\n\nTail."
    try:
        artifact = agent.translate_document(document, Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()
    # The inner ``` line must not close a ```` fence: no code reaches the model.
    assert all("code inside" not in text for text in backend.seen)
    assert "still code inside four" in artifact.text
    assert artifact.text.endswith("尾。\n")


def test_unclosed_frontmatter_is_body_not_swallower(tmp_path: Path) -> None:
    class Recording(MappingBackend):
        def __init__(self, translations: dict[str, str]) -> None:
            super().__init__(translations)
            self.seen: list[str] = []

        def translate(self, request):  # type: ignore[override]
            self.seen.append(request.text)
            return super().translate(request)

    memory = MemorySystem(tmp_path / "state", GlossaryMemory([]))
    backend = Recording({"Body prose never translated.": "正文永远不翻译。"})
    agent = LongHorizonTranslationAgent(backend, tmp_path / "state", memory=memory)
    try:
        artifact = agent.translate_document(
            "---\n\nBody prose never translated.\n", Language.ENGLISH, Language.CHINESE
        )
    finally:
        agent.close()
    # An unclosed '---' is a thematic break, not frontmatter: the body must
    # actually be translated (regression: whole document used to be swallowed
    # silently with a green report).
    assert backend.seen
    assert "正文永远不翻译。" in artifact.text
    assert artifact.report.completed_chunks >= 1


def test_frontmatter_preserved_verbatim(tmp_path: Path) -> None:
    memory = MemorySystem(tmp_path / "state", GlossaryMemory([]))
    agent = LongHorizonTranslationAgent(
        MappingBackend({"Body.": "正文。"}), tmp_path / "state", memory=memory
    )
    source = (
        "---\n"
        "title: Tom &amp; Jerry\n"
        "book:\n"
        "  title: Hello\n"
        "---\n\n"
        "Body.\n"
    )
    try:
        artifact = agent.translate_document(source, Language.ENGLISH, Language.CHINESE)
    finally:
        agent.close()
    assert artifact.text.startswith("---\ntitle: Tom &amp; Jerry\nbook:\n  title: Hello\n---\n")
    assert artifact.text.endswith("正文。\n")
