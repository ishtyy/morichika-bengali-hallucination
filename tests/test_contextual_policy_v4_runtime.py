import hashlib

import pytest

from pipeline.contextual_policy_v4_runtime import (
    CANONICAL_POLICY_FAMILIES,
    ENGINEERED_EVALUATION_CELLS,
    OPERATION_AXIS,
    VERSION,
    build_aggregation_user,
    build_contextual_policy_packet,
    comparison_view,
    contextual_exact_lexical_policy,
    detected_policy_families,
    full_coverage_windows,
    unicode_receipt,
)
from pipeline.contextual_note_taker_fallback import rank_context_windows


def test_taxonomy_axes_are_separate_and_exactly_sized() -> None:
    assert len(CANONICAL_POLICY_FAMILIES) == len(set(CANONICAL_POLICY_FAMILIES)) == 17
    assert len(ENGINEERED_EVALUATION_CELLS) == len(set(ENGINEERED_EVALUATION_CELLS)) == 26
    assert len(OPERATION_AXIS) == len(set(OPERATION_AXIS)) == 15
    assert {
        "same_passage_different_question", "age_vs_year", "relative_timeline",
        "kinship_composition", "theory_application", "cross_window_conflict",
    } <= set(ENGINEERED_EVALUATION_CELLS)
    assert {"antonym_lookup", "samas_taxonomy", "sandhi_formation"} <= set(OPERATION_AXIS)


def test_short_packet_preserves_complete_literal_context_and_boundary() -> None:
    context = "রহিম ১৯১৯ সালে জন্মান। তিনি ১৯৪০ সালে বিবাহ করেন।"
    user, receipt = build_contextual_policy_packet(
        context, "রহিম কত বছর বয়সে বিবাহ করেন?", "২১ বছর", rank_context_windows
    )
    assert context in user
    assert receipt["context_policy_version"] == VERSION
    assert receipt["full_context_inference_coverage"] is True
    assert receipt["external_retrieval_allowed"] is False
    assert receipt["context_literal"]["literal_sha256"] == hashlib.sha256(context.encode()).hexdigest()
    assert "year_age_duration" in receipt["detected_policy_families"]
    assert "bounded_arithmetic" in receipt["detected_policy_families"]
    assert "বাইরের" not in context  # the evidence itself was not rewritten


def test_long_packet_plans_every_literal_window_and_never_calls_local_miss_refutation() -> None:
    context = "শুরু। " + ("অপ্রাসঙ্গিক বাক্য। " * 500) + "দুই বছর পরে ২০২৪ সালে অনুষ্ঠানটি শেষ হয়।"
    user, receipt = build_contextual_policy_packet(
        context, "অনুষ্ঠানটি কবে শেষ হয়?", "২০২৪ সালে", rank_context_windows,
        full_context_char_limit=200,
    )
    assert receipt["full_context_inference_coverage"] is True
    assert receipt["requires_window_aggregation"] is True
    calls = receipt["window_calls"]
    assert len(calls) > 1
    covered = set()
    for call in calls:
        covered.update(range(call["context_char_start"], call["context_char_end"]))
        assert "local miss is not a contradiction" in call["user"]
    assert covered == set(range(len(context)))
    assert "every_context_character_processed\":true" in user
    assert "outside_world_fact_rescue_allowed\":false" in user
    assert receipt["prompt_sha256"] == hashlib.sha256(user.encode()).hexdigest()


def test_window_ledger_replays_and_aggregation_preserves_exact_frame() -> None:
    context = ("ক ঘটনা ঘটে। " * 400) + "খ ঘটনা দুই বছর পরে ঘটে।"
    windows = full_coverage_windows(context, max_chars=512, overlap_chars=64)
    assert windows[0]["context_char_start"] == 0
    assert windows[-1]["context_char_end"] == len(context)
    assert all(context[w["context_char_start"]:w["context_char_end"]] == w["literal_text"] for w in windows)
    results = [
        {**{k: w[k] for k in ("window_index", "context_char_start", "context_char_end", "literal_span_sha256")},
         "window_verdict": "not_enough_information"}
        for w in windows
    ]
    prompt = build_aggregation_user("খ ঘটনা কখন ঘটে?", "দুই বছর পরে", results)
    assert "QUESTION (literal):\nখ ঘটনা কখন ঘটে?" in prompt
    assert "local span was silent; it is never counter-evidence" in prompt


def test_question_slot_grammar_theory_and_unicode_signals_are_explicit() -> None:
    context = "ব্যাকরণের নিয়মে সমাস শব্দের অর্থ সংযোগ। বিদ্ + অর্থী = বিদ্যার্থী।"
    question = "সমাস শব্দের একটি অর্থ কী?"
    response = "সংযোগ"
    families = detected_policy_families(context, question, response)
    assert "definition_theory_rule_application" in families
    assert "samas_exact_operation" in families
    assert "sandhi_exact_operands" not in families
    user, receipt = build_contextual_policy_packet(
        context, question, response, rank_context_windows
    )
    assert "exact_answer_slot" in receipt["detected_policy_families"]
    assert "same_passage_different_question" in receipt["detected_policy_families"]
    assert "near-looking" in user


def test_unicode_comparison_is_bounded_and_literal_receipt_exposes_delta() -> None:
    decomposed = "কো"
    assert comparison_view(decomposed) == comparison_view("কো")
    report = unicode_receipt(decomposed + "\u200d")
    assert report["nfc_identical"] is False
    assert report["joiner_positions"]
    assert report["literal_sha256"] != report["nfc_sha256"]


def _lexical_record(answer="অধম", *, conflict_status="none"):
    return {
        "source_id": "grammar_book_hashbound", "operation": "antonym_lookup",
        "term_key": "উত্তম", "display_terms": ["উত্তম"],
        "accepted_answers": [answer], "contrast_answers": ["বেয়াদব"],
        "conflict_status": conflict_status,
    }


def test_exact_contextual_lexical_shell_is_allowed_but_model_nonterminal() -> None:
    result = contextual_exact_lexical_policy(
        "'উত্তম' শব্দের শুদ্ধ বিপরীত শব্দ কী?", "অধম", [_lexical_record()]
    )
    assert result["lookup_mode"] == "exact_hash_bound_lexical_policy"
    assert result["operation"] == "antonym_lookup"
    assert result["key"] == "উত্তম"
    assert result["source_ids"] == ["grammar_book_hashbound"]
    assert result["exact_operation"] is True and result["exact_key"] is True
    assert result["conflict_status"] == "none"
    assert result["generic_retrieval_invoked"] is False
    assert result["terminal_label_authority"] is False
    assert result["model_nonterminal"] is True


def test_near_shell_and_ordinary_fact_context_cannot_invoke_lookup() -> None:
    near = contextual_exact_lexical_policy(
        "উত্তম ও অধম কি পরস্পর বিপরীত?", "হ্যাঁ", [_lexical_record()]
    )
    ordinary = contextual_exact_lexical_policy(
        "বাংলাদেশের রাজধানী কী?", "ঢাকা", [_lexical_record()]
    )
    assert near["lookup_mode"] == "forbidden_for_ordinary_context"
    assert ordinary["lookup_mode"] == "forbidden_for_ordinary_context"
    assert near["evidence"] == ordinary["evidence"] == []
    assert near["generic_retrieval_invoked"] is ordinary["generic_retrieval_invoked"] is False


def test_conflicting_exact_pairs_are_nei_nonterminal_and_emit_no_answers() -> None:
    result = contextual_exact_lexical_policy(
        "'উত্তম' শব্দের বিপরীত শব্দ কী?", "অধম",
        [_lexical_record("অধম"), _lexical_record("নিকৃষ্ট")],
    )
    assert result["lookup_mode"] == "exact_lexical_conflict_nei_nonterminal"
    assert result["conflict"] is True
    assert result["conflict_status"] == "conflict"
    assert result["terminal_label_authority"] is False
    assert result["evidence"] == []


def test_samas_never_uses_fixed_lexical_exception() -> None:
    record = {**_lexical_record(), "operation": "samas_taxonomy", "term_key": "রাজপুত্র"}
    result = contextual_exact_lexical_policy(
        "রাজপুত্র কোন সমাস?", "ষষ্ঠী তৎপুরুষ", [record]
    )
    assert result["lookup_mode"] == "forbidden_for_ordinary_context"
    assert result["generic_retrieval_invoked"] is False


def test_router_hash_or_external_retrieval_violation_fails_closed() -> None:
    def bad_router(context: str, question: str, response: str, **_: object) -> dict:
        return {"external_retrieval_allowed": True, "context_sha256": "bad"}

    with pytest.raises(ValueError, match="forbid external retrieval"):
        build_contextual_policy_packet("প্রসঙ্গ", "প্রশ্ন?", "উত্তর", bad_router)


def test_packet_has_no_competition_identity_or_label_replay_fields() -> None:
    user, receipt = build_contextual_policy_packet(
        "ঢাকা বাংলাদেশের রাজধানী।", "বাংলাদেশের রাজধানী কী?", "ঢাকা",
        rank_context_windows,
    )
    combined = (user + repr(receipt)).casefold()
    for forbidden in ("phase1", "gold_label", "route_audit", "example_id", "test.csv"):
        assert forbidden not in combined


def test_generated_runner_persists_context_only_restart_diagnostics() -> None:
    from pipeline.build_morichika_phase2_generalized_hybrid_v3 import transformed_runner

    code = transformed_runner("ishtyy/morichika-retrieval-v3", "a" * 64)
    assert 'output["context_diagnostic"]' in code
    assert '"policy_receipt_sha256": route_receipt.get("receipt_sha256")' in code
    assert '"external_retrieval_allowed": route_receipt.get("external_retrieval_allowed")' in code


def test_generated_runner_maps_every_long_window_then_aggregates_and_conflict_fails_nei(tmp_path) -> None:
    from pipeline.build_morichika_phase2_generalized_hybrid_v3 import transformed_runner
    from pipeline import contextual_policy_v4_runtime as runtime

    namespace = {"__name__": "morichika_v4_behavior"}
    exec(compile(transformed_runner("ishtyy/morichika-retrieval-v3", "a" * 64), "<v4>", "exec"), namespace)
    namespace["build_contextual_policy_packet"] = runtime.build_contextual_policy_packet
    namespace["build_aggregation_user"] = runtime.build_aggregation_user

    calls = []

    class Response:
        def __init__(self, content):
            self.content = content
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4}}

    class Requests:
        @staticmethod
        def post(url, **kwargs):
            calls.append(kwargs["json"]["messages"][-1]["content"])
            return Response(next(contents))

    context = ("প্রথম ঘটনা ২০০০ সালে ঘটে। " * 260) + "দ্বিতীয় ঘটনা পরে ঘটে।"
    planned_windows = full_coverage_windows(context)
    contents = iter(
        ['{"verdict":"supported"}', '{"verdict":"refuted"}']
        + ['{"verdict":"not_enough_information"}'] * (len(planned_windows) - 2)
        + ['{"verdict":"supported"}']  # final result is overridden by conflict veto
    )
    row = {
        "example_id": "context-long", "source_index": 0,
        "model_prompt_bn": "দ্বিতীয় ঘটনা কখন ঘটে?", "model_response_bn": "পরে",
        "model_context_bn": context, "context_available": True,
    }
    namespace["requests"] = Requests
    namespace["RUN_ROOT"] = tmp_path
    output = namespace["run_queue"](
        [row], 8080, 0, {"context-long": {}}, [], rank_context_windows,
        namespace["time"].perf_counter(),
    )[0]
    assert len(calls) == len(planned_windows) + 1
    assert all("WINDOW-LOCAL CONTEXT PASS" in call for call in calls[:-1])
    assert "FINAL FULL-CONTEXT AGGREGATION" in calls[-1]
    assert output["label"] == 0
    assert output["method"] == "cross_window_conflict_fail_closed_nei"
    assert output["context_diagnostic"]["window_count"] == len(planned_windows)
    assert output["context_diagnostic"]["cross_window_conflict_forced_nei"] is True


def test_cross_window_age_derivation_reaches_aggregator_when_every_window_is_nei(tmp_path) -> None:
    from pipeline.build_morichika_phase2_generalized_hybrid_v3 import transformed_runner
    from pipeline import contextual_policy_v4_runtime as runtime

    namespace = {"__name__": "morichika_v4_cross_window_math"}
    exec(compile(transformed_runner("ishtyy/morichika-retrieval-v3", "a" * 64), "<v4>", "exec"), namespace)
    namespace["build_contextual_policy_packet"] = runtime.build_contextual_policy_packet
    namespace["build_aggregation_user"] = runtime.build_aggregation_user
    context = "রহিম ১৯১৯ সালে জন্মান। " + ("অপ্রাসঙ্গিক বিবরণ রয়েছে। " * 310) + "তিনি ১৯৪০ সালে বিবাহ করেন।"
    windows = full_coverage_windows(context)
    contents = iter(
        ['{"verdict":"not_enough_information"}'] * len(windows)
        + ['{"verdict":"supported"}']
    )
    prompts = []

    class Response:
        def __init__(self, content): self.content = content
        def raise_for_status(self): return None
        def json(self):
            return {"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4}}

    class Requests:
        @staticmethod
        def post(url, **kwargs):
            prompts.append(kwargs["json"]["messages"][-1]["content"])
            return Response(next(contents))

    namespace["requests"] = Requests
    namespace["RUN_ROOT"] = tmp_path
    row = {
        "example_id": "age-across-windows", "source_index": 2,
        "model_prompt_bn": "রহিম কত বছর বয়সে বিবাহ করেন?",
        "model_response_bn": "২১ বছর", "model_context_bn": context,
        "context_available": True,
    }
    output = namespace["run_queue"](
        [row], 8080, 0, {"age-across-windows": {}}, [], rank_context_windows,
        namespace["time"].perf_counter(),
    )[0]
    aggregate = prompts[-1]
    assert output["label"] == 1
    assert "FINAL FULL-CONTEXT AGGREGATION" in aggregate
    assert "১৯১৯" in aggregate and "১৯৪০" in aggregate
    assert "1940 - 1919 = 21" in aggregate
    assert '"operand_note_ids"' in aggregate
    assert '"source_linked":true' in aggregate
