from __future__ import annotations

from pipeline.phase2_evidence_priority import annotate_and_rank


def candidate(source: str, record: int, *, query: str, source_question: str, operation: str = "") -> dict:
    return {
        "source_id": source,
        "source_record_index": record,
        "source_record_sha256": f"{record:064x}",
        "rank": 1,
        "exact_normalized": True,
        "query_text": query,
        "question": source_question,
        "supporting_text": "",
        "query_policy_operation": operation,
        "query_context_available": False,
        "answers": ["অধম"],
        "response_answer_relation": "exact",
        "model_facing_eligible": True,
        "model_facing_gate": {"eligible": True},
        "number_set_match": True,
        "negation_set_match": True,
        "source_verdict_candidate": None,
    }


def test_exact_antonym_policy_outranks_semantically_similar_wrong_operation() -> None:
    query = "উত্তম শব্দের শুদ্ধ বিপরীত শব্দ কী?"
    wrong_book = candidate(
        "joykoli_six_part", 1, query=query,
        source_question="উত্তম শব্দের সমার্থক শব্দ কী?",
        operation="antonym_lookup",
    )
    # Explicit metadata catches a source adapter's typed mismatch even when
    # its raw Bengali text is highly similar.
    wrong_book["source_metadata"] = {"operation": "idiom_meaning_lookup"}
    right_dataset = candidate(
        "nctb_qa_87805", 2, query=query,
        source_question="উত্তম শব্দের বিপরীত শব্দ কী?",
        operation="antonym_lookup",
    )
    ranked = annotate_and_rank([wrong_book, right_dataset])
    assert ranked[0]["source_id"] == "nctb_qa_87805"
    assert ranked[1]["model_facing_eligible"] is False
    assert "exclusive_operation_mismatch" in ranked[1]["policy_compatibility_status"]


def test_contextual_external_candidate_is_quarantined() -> None:
    row = candidate(
        "downloads_bcs_10_50", 1, query="বাংলাদেশের রাজধানী কী?",
        source_question="বাংলাদেশের রাজধানী কী?",
    )
    row["query_context_available"] = True
    ranked = annotate_and_rank([row])
    assert ranked[0]["model_facing_eligible"] is False
    assert ranked[0]["source_verdict_candidate"] is None


def test_number_mismatch_beats_book_authority_and_is_nonterminal() -> None:
    bad_book = candidate(
        "downloads_bcs_10_50", 1, query="ঘটনাটি ১৯৭১ সালে কোথায় ঘটে?",
        source_question="ঘটনাটি ১৯৭২ সালে কোথায় ঘটে?",
    )
    bad_book["number_set_match"] = False
    bad_book["query_numbers"] = ["১৯৭১"]
    bad_book["source_numbers"] = ["১৯৭২"]
    good_wiki = candidate(
        "bengali_wikipedia_20210320", 2,
        query="ঘটনাটি ১৯৭১ সালে কোথায় ঘটে?",
        source_question="ঘটনাটি ১৯৭১ সালে কোথায় ঘটে?",
    )
    ranked = annotate_and_rank([bad_book, good_wiki])
    assert ranked[0]["source_id"] == "bengali_wikipedia_20210320"
    assert ranked[1]["model_facing_eligible"] is False


def test_support_and_counter_are_not_ranked_by_agreement_direction() -> None:
    support = candidate(
        "nctb_qa_87805", 2, query="প্রশ্ন", source_question="প্রশ্ন"
    )
    support["evidence_role"] = "support_candidate"
    counter = candidate(
        "downloads_bcs_10_50", 1, query="প্রশ্ন", source_question="প্রশ্ন"
    )
    counter["evidence_role"] = "counter_candidate"
    ranked = annotate_and_rank([support, counter])
    assert ranked[0]["source_id"] == "downloads_bcs_10_50"
