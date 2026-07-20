import pytest

from pipeline.contextual_note_taker_fallback import (
    bounded_fathers_father,
    decision_receipt,
    extract_question_conditioned_notes,
    make_windows,
    rank_context_windows,
    relative_year_offset,
    simple_year_difference,
)


def _notes(context, question, max_chars=96, overlap=24):
    windows = make_windows(context, max_chars=max_chars, overlap_chars=overlap)
    notes = extract_question_conditioned_notes(context, question, windows)
    return windows, notes


def test_date_arithmetic_is_visible_and_source_linked():
    context = "রফিক ১৯১৯ সালে জন্মগ্রহণ করেন। তিনি ১৯৪০ সালে বিবাহ করেন।"
    windows, notes = _notes(context, "রফিক কত বছর বয়সে বিবাহ করেন?")
    dates = [n for n in notes if n.type == "date_or_time"]
    derivation = simple_year_difference(context, windows, dates[0], dates[1])
    assert derivation["visible_expression"] == "1940 - 1919 = 21"
    receipt = decision_receipt(context, "রফিক কত বছর বয়সে বিবাহ করেন?", "২১ বছর", "age_at_marriage", "supported", dates, derivation)
    assert receipt["external_retrieval_allowed"] is False


def test_relative_event_year_is_exposed_as_bounded_derivation():
    context = "প্রথম চুক্তি ২০১৮ সালে স্বাক্ষরিত হয়। দুই বছর পরে প্রকল্পটি চালু হয়।"
    # Numeric form exercises the deterministic operator; Bengali word-number
    # normalization remains a separately gated extension.
    context = context.replace("দুই", "২")
    windows, notes = _notes(context, "প্রকল্পটি কোন সালে চালু হয়?")
    anchor = next(note for note in notes if note.type == "date_or_time")
    relative = next(note for note in notes if note.type == "temporal" and "offset_years" in note.attributes)
    derivation = relative_year_offset(anchor, relative)
    assert derivation["visible_expression"] == "2018 + 2 = 2020"
    routed = rank_context_windows(context, "প্রকল্পটি কোন সালে চালু হয়?", "২০২০", windows, max_windows=3)
    assert routed["bounded_derivation_candidates"][0]["result"]["value"] == 2020
    assert routed["external_retrieval_allowed"] is False


def test_unbound_mid_sentence_duration_is_not_a_derivation():
    context = (
        "১৯২৩ সালের ২৯ অক্টোবর তুরস্ককে প্রজাতন্ত্র ঘোষণা করা হয়। "
        "ফলে প্রায় ৭০০ বছর পর উসমানীয় সাম্রাজ্য আনুষ্ঠানিকভাবে সমাপ্ত হয়।"
    )
    routed = rank_context_windows(context, "উসমানীয় সাম্রাজ্য কবে শেষ হয়?", "১৯২৩", max_windows=3)
    assert routed["bounded_derivation_candidates"] == []


def test_direct_age_evidence_blocks_unrelated_year_subtraction():
    context = "ক্লে ১২ বছর বয়সে প্রশিক্ষণ শুরু করেন। ১৯৬৪ সালে শিরোপা জেতেন। ১৯৭৫ সালে ধর্মান্তরিত হন।"
    routed = rank_context_windows(context, "ক্লে কত বছর বয়সে প্রশিক্ষণ শুরু করেন?", "১২", max_windows=3)
    assert routed["bounded_derivation_candidates"] == []


def test_bounded_kinship_composes_only_cited_edges():
    context = "রহিমের বাবা করিম। করিমের বাবা জলিল।"
    _, notes = _notes(context, "রহিমের দাদা কে?")
    relations = [n for n in notes if n.type == "relation"]
    derivation = bounded_fathers_father(relations[0], relations[1])
    assert derivation["result"]["value"] == "জলিল"
    assert derivation["visible_expression"] == "রহিম→পিতা→করিম→পিতা→জলিল"


def test_indirect_event_selects_requested_phase_not_nearby_date():
    context = "কর্মসূচি ঘোষিত হয় ১২ বৈশাখ। মাঠের কাজ শুরু হয় ১৩ জ্যৈষ্ঠ। সমাপনী প্রতিবেদন জমা হয় ১৬ আষাঢ়।"
    _, notes = _notes(context, "মাঠের কাজ কোন তারিখে শুরু হয়?")
    dates = [n.literal_text for n in notes if n.type == "date_or_time"]
    assert dates == ["১২ বৈশাখ", "১৩ জ্যৈষ্ঠ", "১৬ আষাঢ়"]
    start_date = next(n for n in notes if n.literal_text == "১৩ জ্যৈষ্ঠ")
    receipt = decision_receipt(context, "মাঠের কাজ কোন তারিখে শুরু হয়?", "১৩ জ্যৈষ্ঠ", "field_work_start_date", "supported", [start_date])
    assert receipt["evidence_notes"][0]["literal_text"] == "১৩ জ্যৈষ্ঠ"


def test_negation_is_preserved_as_counterevidence():
    context = "প্রস্তাবটি সভায় তোলা হয়েছিল, কিন্তু প্রশাসনিক অনুমোদন দেওয়া হয়নি।"
    _, notes = _notes(context, "প্রস্তাবটি কি প্রশাসনিক অনুমোদন পেয়েছিল?")
    negation = [n for n in notes if n.type == "negation"]
    assert negation and "হয়নি" in negation[0].literal_text
    receipt = decision_receipt(context, "প্রস্তাবটি কি প্রশাসনিক অনুমোদন পেয়েছিল?", "হ্যাঁ", "administrative_approval", "refuted", negation)
    assert receipt["verdict"] == "refuted"


def test_same_passage_different_question_keeps_relation_slots_separate():
    context = "অরুণের স্রষ্টা বিমল। অরুণের পরিচালক কমল।"
    _, notes = _notes(context, "অরুণের স্রষ্টা কে?")
    relations = [n for n in notes if n.type == "relation"]
    by_relation = {n.attributes["relation"]: n for n in relations}
    assert by_relation["স্রষ্টা"].attributes["object"] == "বিমল"
    assert by_relation["পরিচালক"].attributes["object"] == "কমল"
    receipt = decision_receipt(context, "অরুণের স্রষ্টা কে?", "কমল", "creator", "refuted", [by_relation["স্রষ্টা"]])
    assert receipt["requested_relation_or_property"] == "creator"


def test_grammar_theory_is_context_evidence_and_retrieval_fields_fail_closed():
    context = "ব্যাকরণের নিয়মে সমাস শব্দের অর্থ সংযোগ ও একাধিক পদের একপদীকরণ।"
    _, notes = _notes(context, "সমাস শব্দের একটি অর্থ কী?")
    rules = [n for n in notes if n.type == "rule_or_definition"]
    assert rules and "সংযোগ" in rules[0].literal_text
    receipt = decision_receipt(context, "সমাস শব্দের একটি অর্থ কী?", "সংযোগ", "meaning_of_samas", "supported", rules)
    assert receipt["question_text_is_evidence"] is False
    with pytest.raises(ValueError, match="forbids external/retrieval"):
        decision_receipt(context, "সমাস শব্দের একটি অর্থ কী?", "সংযোগ", "meaning_of_samas", "supported", rules,
                         {"operator_id": "simple_year_difference", "source_linked": True,
                          "operand_note_ids": [rules[0].note_id], "retrieved_evidence": "bad"})


def test_long_windows_preserve_literal_residual_and_full_coverage():
    context = "। ".join([f"ঘটনা {i} সালে শুরু হয়" for i in range(1, 40)]) + "।"
    windows = make_windows(context, max_chars=96, overlap_chars=24)
    assert windows[0].context_char_start == 0
    assert windows[-1].context_char_end == len(context)
    assert all(a.context_char_end - b.context_char_start >= 24 for a, b in zip(windows, windows[1:]))


def test_primary_router_prefers_exact_question_slot_and_keeps_original_offsets():
    context = (
        "অরুণের পরিচালক কমল। কাজটি ২০১৮ সালে প্রকাশিত হয়। "
        "অরুণের স্রষ্টা বিমল। কাজটি ২০২০ সালে পুরস্কার পায়।"
    )
    windows = make_windows(context, max_chars=64, overlap_chars=16)
    result = rank_context_windows(context, "অরুণের স্রষ্টা কে?", "বিমল", windows, max_windows=3)
    assert result["routing_mode"] == "primary_for_long_or_complex_context"
    assert result["external_retrieval_allowed"] is False
    selected = result["selected_windows"]
    assert any("স্রষ্টা বিমল" in row["literal_text"] for row in selected)
    assert all(
        context[row["context_char_start"]:row["context_char_end"]] == row["literal_text"]
        for row in selected
    )


def test_primary_router_reserves_numeric_counterevidence_window():
    context = "জন্ম ১৯১৯ সালে। বিবাহ ১৯৪০ সালে। অন্য নথিতে ভুল করে ১৯৪১ লেখা আছে।"
    windows = make_windows(context, max_chars=64, overlap_chars=16)
    result = rank_context_windows(context, "বিবাহের সাল কত?", "১৯৪১", windows, max_windows=4)
    assert any(row["response_number_support_candidate"] for row in result["ranking"])
    assert any(row["response_number_counter_candidate"] for row in result["ranking"])
