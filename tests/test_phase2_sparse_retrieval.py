from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.phase2_sparse_retrieval import (
    DEFAULT_INDEX_SPECS,
    STRICT_EXACT_CONFLICT_POLICY,
    answer_equivalence_key,
    build,
    composite_retrieval_candidate,
    fuzzy_candidate_gate,
    passage_candidate_gate,
    preferred_cache,
    primary_query_subject_anchor,
    relation,
    rank_and_bound_composite_candidates,
    select_composite_query_rows,
    source_verdict_candidate,
    subject_anchor_match,
    strict_exact_key_conflicts,
    tighten_joykoli_structured_gate,
    write_closed_book_eval_evidence_ledger,
)
from pipeline.phase2_canonicalize import VERSION as CANONICALIZER_VERSION
from pipeline.phase2_mmap_retrieval import (
    NORMALIZER as MMAP_NORMALIZER,
    VERSION as MMAP_CACHE_VERSION,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase2SparseRetrievalTest(unittest.TestCase):
    @staticmethod
    def _authority_candidate(
        source_id: str,
        record_index: int,
        *,
        relation_value: str = "exact",
        exact: bool = True,
        eligible: bool = True,
        number_match: bool = True,
    ) -> dict:
        return {
            "source_id": source_id,
            "source_record_index": record_index,
            "source_record_sha256": f"{record_index:064x}",
            "rank": 1,
            "exact_normalized": exact,
            "answers": ["ঢাকা"],
            "response_answer_relation": relation_value,
            "model_facing_eligible": eligible,
            "model_facing_gate": {"eligible": eligible},
            "number_set_match": number_match,
            "negation_set_match": True,
            "source_verdict_candidate": None,
        }

    def test_composite_final_topk_applies_authority_before_limit(self):
        wikipedia = self._authority_candidate(
            "bengali_wikipedia_20210320", 1
        )
        curated = self._authority_candidate("openai_mmmlu_bn", 2)
        book = self._authority_candidate("downloads_bcs_10_50", 3)
        selected, audit = rank_and_bound_composite_candidates(
            [wikipedia, curated, book], top_k=2
        )
        eligible = [row for row in selected if row["model_facing_eligible"]]
        self.assertEqual(
            [row["source_id"] for row in eligible],
            ["downloads_bcs_10_50", "openai_mmmlu_bn"],
        )
        self.assertEqual(audit["selected_candidates"], 2)
        self.assertFalse(audit["counter_forced_into_top_k"])

    def test_composite_final_topk_retains_exact_counterevidence(self):
        book = self._authority_candidate("downloads_bcs_10_50", 1)
        curated = self._authority_candidate("openai_mmmlu_bn", 2)
        wikipedia_counter = self._authority_candidate(
            "bengali_wikipedia_20210320", 3, relation_value="none"
        )
        selected, audit = rank_and_bound_composite_candidates(
            [wikipedia_counter, curated, book], top_k=2
        )
        eligible = [row for row in selected if row["model_facing_eligible"]]
        self.assertEqual(
            [row["source_id"] for row in eligible],
            ["downloads_bcs_10_50", "bengali_wikipedia_20210320"],
        )
        self.assertTrue(audit["counter_available"])
        self.assertTrue(audit["counter_selected"])
        self.assertTrue(audit["counter_forced_into_top_k"])

    def test_semantic_mismatch_precedes_book_authority(self):
        mismatching_book = self._authority_candidate(
            "downloads_bcs_10_50", 1, number_match=False
        )
        aligned_wikipedia = self._authority_candidate(
            "bengali_wikipedia_20210320", 2
        )
        selected, _ = rank_and_bound_composite_candidates(
            [mismatching_book, aligned_wikipedia], top_k=1
        )
        eligible = [row for row in selected if row["model_facing_eligible"]]
        self.assertEqual(eligible[0]["source_id"], "bengali_wikipedia_20210320")
        self.assertEqual(mismatching_book["number_set_match"], False)

    def test_composite_authority_topk_is_input_order_independent(self):
        values = [
            self._authority_candidate("bengali_wikipedia_20210320", 3),
            self._authority_candidate("openai_mmmlu_bn", 2),
            self._authority_candidate("downloads_bcs_10_50", 1),
        ]
        forward, _ = rank_and_bound_composite_candidates(values, top_k=2)
        reverse, _ = rank_and_bound_composite_candidates(
            list(reversed(values)), top_k=2
        )
        identity = lambda rows: [
            (row["source_id"], row["source_record_index"]) for row in rows
            if row["model_facing_eligible"]
        ]
        self.assertEqual(identity(forward), identity(reverse))

    def test_composite_query_modes_preserve_heavy_counterevidence_and_isolate_residual_ablation(self):
        rows = [
            {"example_id": "closed-with-nonterminal-evidence", "context_available": False},
            {"example_id": "closed-residual", "context_available": False},
            {"example_id": "closed-terminal", "context_available": False},
            {"example_id": "context", "context_available": True},
            {"example_id": "skipped", "context_available": False},
        ]
        candidates = [
            [{"model_facing_eligible": True}],
            [{"model_facing_eligible": False}],
            [{
                "model_facing_eligible": True,
                "source_verdict_candidate": {"verdict": 1},
            }],
            [],
            [],
        ]
        self.assertEqual(
            select_composite_query_rows(
                rows, candidates, skip_example_ids={"skipped"}, mode="all_closed"
            ),
            [0, 1, 2],
        )
        self.assertEqual(
            select_composite_query_rows(
                rows, candidates, skip_example_ids={"skipped"}, mode="unresolved_only"
            ),
            [0, 1],
        )
        self.assertEqual(
            select_composite_query_rows(
                rows, candidates, skip_example_ids={"skipped"}, mode="residual_only"
            ),
            [1],
        )
        with self.assertRaisesRegex(ValueError, "invalid composite query mode"):
            select_composite_query_rows(
                rows, candidates, skip_example_ids=set(), mode="fast"
            )
        with self.assertRaisesRegex(ValueError, "cardinality mismatch"):
            select_composite_query_rows(
                rows, candidates[:-1], skip_example_ids=set(), mode="all_closed"
            )

    def test_composite_evidence_is_closed_book_nonterminal_and_hash_located(self):
        row = {
            "model_prompt_bn": "বাংলাদেশের রাজধানী কী?",
            "model_response_bn": "ঢাকা",
        }
        value = {
            "source_id": "uddipok_v3",
            "source_record_index": 9,
            "source_locator": "RC_Dataset_v2.csv:row:9",
            "record_sha256": "a" * 64,
            "semantic_role": "closed_book_knowledge_evidence_candidate",
            "model_facing": True,
            "terminal_label_authority": False,
            "verdict": "NEI",
            "question": "বাংলাদেশের রাজধানী কি?",
            "answer": "ঢাকা",
            "title": "",
            "evidence_excerpt": "বাংলাদেশের রাজধানী ঢাকা।",
            "choices": [],
            "metadata": {"row": 9},
            "exact_question": True,
            "retrieval_score": -1000.0,
            "retrieval_score_kind": "exact_question_sentinel",
        }
        candidate = composite_retrieval_candidate(row, value, rank=1)
        self.assertTrue(candidate["model_facing_eligible"])
        self.assertIsNone(candidate["source_verdict_candidate"])
        self.assertEqual(candidate["source_record_sha256"], "a" * 64)
        self.assertEqual(candidate["source_locator"], "RC_Dataset_v2.csv:row:9")
        self.assertTrue(candidate["model_facing_gate"]["semantic_verification_required"])
        self.assertEqual(candidate["score_kind"], "exact_question_sentinel")

    def test_composite_component_cannot_enter_factual_evidence(self):
        row = {"model_prompt_bn": "২+২", "model_response_bn": "৪"}
        value = {
            "source_id": "bangla_math",
            "source_record_index": 1,
            "source_locator": "BdMO.csv:row:1",
            "record_sha256": "b" * 64,
            "semantic_role": "query_expansion_or_structured_verifier_resource",
            "model_facing": False,
            "terminal_label_authority": False,
            "verdict": "NEI",
        }
        with self.assertRaises(ValueError):
            composite_retrieval_candidate(row, value, rank=1)

    def test_composite_numeric_mismatch_is_quarantined_before_authority_ranking(self):
        row = {
            "model_prompt_bn": "ঘটনাটি ১৯৭১ সালে কোথায় ঘটেছিল?",
            "model_response_bn": "ঢাকা",
        }
        value = {
            "source_id": "bengali_wikipedia_20210320",
            "source_record_index": 10,
            "source_locator": "page:10",
            "record_sha256": "c" * 64,
            "semantic_role": "closed_book_knowledge_evidence_candidate",
            "model_facing": True,
            "terminal_label_authority": False,
            "verdict": "NEI",
            "question": "ঘটনাটি ১৯৭৪ সালে কোথায় ঘটেছিল?",
            "answer": "ঢাকা",
            "title": "",
            "evidence_excerpt": "ঘটনাটি ১৯৭৪ সালে ঢাকায় ঘটেছিল।",
            "choices": [],
            "metadata": {},
            "exact_question": False,
            "retrieval_score": -5.0,
        }
        candidate = composite_retrieval_candidate(row, value, rank=1)
        self.assertFalse(candidate["model_facing_eligible"])
        self.assertIn("number_set_mismatch", candidate["model_facing_gate"]["reasons"])
        self.assertIsNone(candidate["source_verdict_candidate"])

    def test_closed_book_retrieval_flags_are_saved_for_manual_eval_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [
                {
                    "example_id": "closed-a", "source_index": 7,
                    "context_available": False, "model_prompt_bn": "প্রশ্ন",
                    "model_response_bn": "উত্তর", "prompt_bn": "প্রশ্ন",
                    "response_bn": "উত্তর",
                },
                {
                    "example_id": "context-b", "source_index": 8,
                    "context_available": True, "model_prompt_bn": "প্রশ্ন ২",
                    "model_response_bn": "উত্তর ২",
                },
            ]
            candidate = {
                "source_id": "fixture", "source_record_index": 3,
                "rank": 1, "score": 0.91, "exact_normalized": True,
                "response_answer_relation": "none",
                "model_facing_gate": {"eligible": True, "reasons": []},
                "question": "প্রমাণ প্রশ্ন", "supporting_text": "প্রমাণ পাঠ্য",
                "answers": ["সঠিক"], "choices": ["সঠিক", "ভুল"],
                "source_metadata": {"record_sha256": "a" * 64},
                "source_verdict_candidate": {"verdict": 0, "rule": "fixture"},
            }
            retrievals = [
                {
                    "example_id": "closed-a", "source_index": 7,
                    "query_sha256": "b" * 64, "response_sha256": "c" * 64,
                    "retrieval_candidates": [candidate],
                    "retrieval_audit_quarantined_candidates": [
                        {**candidate, "source_record_index": 4, "score": 0.4}
                    ],
                    "merged_source_candidate": {
                        "status": "source_consensus_candidate", "verdict": 0,
                        "evidence": [],
                    },
                },
                {
                    "example_id": "context-b", "source_index": 8,
                    "query_sha256": "d" * 64, "response_sha256": "e" * 64,
                    "retrieval_candidates": [candidate],
                    "retrieval_audit_quarantined_candidates": [],
                    "merged_source_candidate": {
                        "status": "contextual_external_retrieval_disabled",
                        "verdict": None, "evidence": [],
                    },
                },
            ]
            summary = write_closed_book_eval_evidence_ledger(
                inputs, retrievals, root
            )
            queue = [
                json.loads(line) for line in
                (root / "closed_book_eval_label_queue.private.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(summary["records"], 1)
            self.assertEqual(summary["evidence_items"], 2)
            self.assertEqual(summary["contextual_rows_excluded"], 1)
            self.assertIsNone(queue[0]["eval_label"])
            self.assertEqual(queue[0]["eval_label_status"], "pending_adjudication")
            self.assertEqual(len(queue[0]["possible_counter_evidence_item_ids"]), 2)
            self.assertFalse(summary["model_facing_input"])
            self.assertFalse(summary["terminal_label_assignment"])

    def test_bengali_numeric_counter_spacing_is_exact_but_other_spaces_remain(self):
        self.assertEqual(answer_equivalence_key("১৯ টি"), answer_equivalence_key("১৯টি"))
        self.assertEqual(relation("১৯ টি", ["১৯টি"]), "exact")
        self.assertNotEqual(
            answer_equivalence_key("ঢাকা শহর"), answer_equivalence_key("ঢাকাশহর")
        )

    def test_preferred_cache_prioritizes_compatible_mmap_v2_and_rejects_stale_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "fixture"
            stale = base.with_name(base.name + "_mmap_v1")
            stale.mkdir()
            (stale / "manifest.json").write_text(
                json.dumps({"version": "phase2-mmap-sparse-cache-v1"}),
                encoding="utf-8",
            )
            self.assertEqual(preferred_cache(base), base)
            current = base.with_name(base.name + "_mmap_v2")
            current.mkdir()
            compatible = {
                "version": MMAP_CACHE_VERSION,
                "fingerprint": {
                    "canonicalizer_version": CANONICALIZER_VERSION,
                    "normalizer_sha256": digest(MMAP_NORMALIZER),
                },
            }
            (current / "manifest.json").write_text(
                json.dumps(compatible), encoding="utf-8"
            )
            # Even a compatible historically named shard cannot outrank v2.
            (stale / "manifest.json").write_text(
                json.dumps(compatible), encoding="utf-8"
            )
            self.assertEqual(preferred_cache(base), current)

    def test_opt_in_exact_book_conflict_only_quarantines_same_mcq_signature(self):
        authority = {
            "source_id": "bangla_mmlu", "source_record_index": 4,
            "exact_normalized": True, "answers": ["ঢাকা"],
            "choices": ["ঢাকা", "চট্টগ্রাম"],
            "source_verdict_candidate": {"verdict": 1, "rule": "fixture"},
            "exact_conflict_policy": "none",
        }
        challenger = {
            "source_id": "joykoli_six_part", "source_record_index": 9,
            "exact_normalized": True, "answers": ["চট্টগ্রাম"],
            "choices": ["চট্টগ্রাম", "ঢাকা"],
            "source_verdict_candidate": None,
            "exact_conflict_policy": STRICT_EXACT_CONFLICT_POLICY,
            "exact_conflict_eligible": True,
        }
        conflicts = strict_exact_key_conflicts([authority, challenger])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["effect"], "quarantine_only_no_verdict")
        self.assertNotIn("verdict", conflicts[0])

        challenger["choices"] = ["চট্টগ্রাম", "রাজশাহী"]
        self.assertEqual(strict_exact_key_conflicts([authority, challenger]), [])
        challenger["choices"] = ["চট্টগ্রাম", "ঢাকা"]
        challenger["exact_conflict_policy"] = "none"
        self.assertEqual(strict_exact_key_conflicts([authority, challenger]), [])
        challenger["exact_conflict_policy"] = STRICT_EXACT_CONFLICT_POLICY
        challenger["exact_conflict_eligible"] = False
        self.assertEqual(strict_exact_key_conflicts([authority, challenger]), [])

    def test_bcs_schooltext_and_joykoli_indexes_are_nonterminal(self):
        self.assertTrue(
            {"downloads_bcs_10_50", "nctb_schooltext", "joykoli_six_part"}.issubset(
                {item["source_id"] for item in DEFAULT_INDEX_SPECS}
            )
        )
        for source_id in (
            "downloads_bcs_10_50", "nctb_schooltext", "joykoli_six_part",
        ):
            self.assertIsNone(source_verdict_candidate(
                source_id,
                exact=True,
                response="ঢাকা",
                answers=["ঢাকা"],
                choices=["ঢাকা", "চট্টগ্রাম"],
                response_relation="exact",
                record={},
            ))
        joykoli = next(
            item for item in DEFAULT_INDEX_SPECS
            if item["source_id"] == "joykoli_six_part"
        )
        self.assertEqual(
            joykoli["exact_conflict_policy"], STRICT_EXACT_CONFLICT_POLICY,
        )

    def test_fuzzy_gate_rejects_short_keyword_and_entity_type_conflict(self):
        short = fuzzy_candidate_gate("তিস্তা", "তিস্তা বাগচী কে?", 0.91)
        self.assertFalse(short["eligible"])
        self.assertIn("short_or_keyword_only_query", short["reasons"])
        conflict = fuzzy_candidate_gate(
            "পদ্মা নদীর প্রধান উপনদী কোনটি?",
            "পদ্মা নামের কবি কে?",
            0.81,
        )
        self.assertFalse(conflict["eligible"])
        self.assertIn("answer_type_intent_conflict", conflict["reasons"])

    def test_fuzzy_gate_keeps_exact_full_question_without_partial_logic(self):
        exact = fuzzy_candidate_gate(
            "বাংলাদেশের রাজধানী কী?", "বাংলাদেশের রাজধানী কী?", 1.0, exact=True
        )
        self.assertTrue(exact["eligible"])
        self.assertEqual(exact["policy"], "exact_full_normalized_question")

    def test_passage_gate_requires_grounded_multi_token_overlap(self):
        good = passage_candidate_gate(
            "সিলেট কোন প্রাচীন জনপদের অন্তর্ভুক্ত ছিল?",
            "02 সিলেট কোন প্রাচীন জনপদের অন্তর্ভুক্ত ছিল? হরিকেল গৌড় পুণ্ড্র",
            0.72,
        )
        self.assertTrue(good["eligible"])
        partial_name = passage_candidate_gate(
            "পদ্মা নদীর প্রধান উপনদী কোনটি?",
            "পদ্মা নামের একজন কবির জীবনী ও রচনাবলি",
            0.81,
        )
        self.assertFalse(partial_name["eligible"])
        self.assertIn(
            "passage_requires_two_substantive_overlaps", partial_name["reasons"]
        )

    def test_joykoli_structured_gate_removes_single_long_name_exception(self):
        generic = fuzzy_candidate_gate(
            "মাইকেলমধুসূদন", "মাইকেলমধুসূদন কে ছিলেন?", 0.91,
        )
        # The generic gate may retain a distinctive entity, but noisy Joykoli
        # structured OCR must corroborate it with a second substantive token.
        tightened = tighten_joykoli_structured_gate(generic, exact=False)
        self.assertFalse(tightened["eligible"])
        self.assertIn(
            "joykoli_nonexact_requires_two_substantive_overlaps",
            tightened["reasons"],
        )

    def test_joykoli_anchor_rejects_same_relation_but_wrong_place(self):
        query = "সিলেট কোন প্রাচীন জনপদের অন্তর্ভুক্ত ছিল?"
        wrong = "বগুড়া প্রাচীন কোন জনপদের অন্তর্ভুক্ত ছিল? গৌড় হরিকেল"
        base = passage_candidate_gate(query, wrong, 0.74)
        self.assertTrue(base["eligible"])
        tightened = tighten_joykoli_structured_gate(
            base, exact=False, query=query, evidence=wrong, passage_evidence=True,
        )
        self.assertFalse(tightened["eligible"])
        self.assertIn(
            "query_primary_anchor_missing_from_passage", tightened["reasons"],
        )

    def test_joykoli_anchor_keeps_correct_place_and_bounded_ocr_variant(self):
        sylhet = "সিলেট কোন প্রাচীন জনপদের অন্তর্ভুক্ত ছিল? হরিকেল"
        gate = passage_candidate_gate(
            "সিলেট কোন প্রাচীন জনপদের অন্তর্ভুক্ত ছিল?", sylhet, 0.74,
        )
        tightened = tighten_joykoli_structured_gate(
            gate, exact=False,
            query="সিলেট কোন প্রাচীন জনপদের অন্তর্ভুক্ত ছিল?",
            evidence=sylhet, passage_evidence=True,
        )
        self.assertTrue(tightened["eligible"])
        self.assertEqual(
            tightened["query_primary_subject_anchor"]["method"], "exact_token",
        )
        ocr = subject_anchor_match(
            "বাংলাভাষায় প্রথম সার্চ ইঞ্জিনের নাম কী?",
            "রাংলাভাযষায় প্রথম সার্চ ইঞ্জিনের নান কি? পিপীলিকা",
        )
        self.assertEqual(ocr["method"], "bounded_ocr_similarity")

    def test_primary_anchor_is_subject_or_concept_not_question_relation(self):
        self.assertEqual(
            primary_query_subject_anchor(
                "সিলেট কোন প্রাচীন জনপদের অন্তর্ভুক্ত ছিল?"
            ),
            "সিলেট",
        )
        self.assertEqual(
            primary_query_subject_anchor("কোনটি নবায়নযোগ্য শক্তির উদাহরণ?"),
            "নবায়নযোগ্য",
        )

    def test_joykoli_anchor_rejects_partial_river_and_person_collisions(self):
        river = subject_anchor_match(
            "পদ্মা নদীর প্রধান উপনদী কোনটি?", "পদ্মাবতী নামের কবির পরিচয়",
        )
        self.assertIsNone(river["matched"])
        person = subject_anchor_match(
            "রবীন্দ্রনাথ ঠাকুরের রচিত নাটক কোনটি?",
            "রবীন্দ্র সরোবরের পাশে নাটকের মঞ্চ",
        )
        self.assertIsNone(person["matched"])

    def test_joykoli_exact_gate_is_unchanged_by_anchor(self):
        exact = fuzzy_candidate_gate(
            "বাংলাদেশের রাজধানী কী?", "বাংলাদেশের রাজধানী কী?", 1.0,
            exact=True,
        )
        self.assertIs(
            tighten_joykoli_structured_gate(
                exact, exact=True, query="অন্য বিষয়", evidence="কিছু নেই",
            ),
            exact,
        )

    def test_nctb_aux_exact_consensus_can_reject_a_wrong_mcq_option(self):
        candidate = source_verdict_candidate(
            "nctb_education_aux",
            exact=True,
            response="চট্টগ্রাম",
            answers=["ঢাকা"],
            choices=["ঢাকা", "চট্টগ্রাম"],
            response_relation="none",
            record={
                "terminal_consensus_answer": "ঢাকা",
                "terminal_negative_choices": ["চট্টগ্রাম"],
            },
        )
        self.assertEqual(candidate["verdict"], 0)

    def make_index(self, root: Path) -> dict:
        root.mkdir(parents=True)
        questions = ["বাংলাদেশের রাজধানী কী", "পানির রাসায়নিক সংকেত কী"]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), norm="l2")
        matrix = vectorizer.fit_transform(questions).tocsr()
        joblib.dump(vectorizer, root / "vectorizer.joblib")
        sparse.save_npz(root / "matrix.npz", matrix)
        records = [
            {"index": 0, "normalized_question": questions[0], "question": questions[0], "records": [{"keyed_answer": "ঢাকা", "choices": ["ঢাকা", "চট্টগ্রাম"]}]},
            {"index": 1, "normalized_question": questions[1], "question": questions[1], "records": [{"keyed_answer": "H2O", "choices": ["H2O", "CO2"]}]},
        ]
        with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        files = {name: digest(root / name) for name in ("vectorizer.joblib", "matrix.npz", "records.jsonl")}
        (root / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
        return {
            "source_id": "bangla_mmlu",
            "directory": root,
            "rights_note": "fixture",
            "terminal_policy": "fixture exact only",
        }

    def make_joykoli_index(self, root: Path) -> dict:
        root.mkdir(parents=True)
        question = "বাংলাদেশের রাজধানী কী"
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), norm="l2")
        matrix = vectorizer.fit_transform([question]).tocsr()
        joblib.dump(vectorizer, root / "vectorizer.joblib")
        sparse.save_npz(root / "matrix.npz", matrix)
        record = {
            "index": 0, "normalized_question": question, "question": question,
            "answers": ["চট্টগ্রাম"], "choices": ["ঢাকা", "চট্টগ্রাম"],
            "metadata": {
                "source_part_ids": ["part_1"], "pdf_pages": ["part_1:10"],
                "source_sha256s": ["a" * 64], "raw_block_sha256s": ["b" * 64],
                "exact_conflict_eligible": True,
            },
        }
        (root / "records.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        files = {
            name: digest(root / name)
            for name in ("vectorizer.joblib", "matrix.npz", "records.jsonl")
        }
        (root / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
        return {
            "source_id": "joykoli_six_part", "directory": root,
            "rights_note": "private user-provided fixture",
            "terminal_policy": "corroboration only",
            "exact_conflict_policy": STRICT_EXACT_CONFLICT_POLICY,
        }

    def make_joykoli_passage_index(self, root: Path) -> dict:
        root.mkdir(parents=True)
        passage = (
            "02 সিলেট কোন প্রাচীন জনপদের অন্তর্ভুক্ত ছিল? "
            "হরিকেল বরেন্দ্র গৌড় পুণ্ড্র"
        )
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), norm="l2")
        matrix = vectorizer.fit_transform([passage]).tocsr()
        joblib.dump(vectorizer, root / "vectorizer.joblib")
        sparse.save_npz(root / "matrix.npz", matrix)
        record = {
            "index": 0,
            "record_id": "joykoli_repair:fixture",
            "record_kind": "page_ocr_repair_chunk",
            "normalized_question": "",
            "question": "",
            "answers": [],
            "choices": [],
            "supporting_text": passage,
            "metadata": {
                "fuzzy_eligible": True,
                "exact_search_eligible": False,
                "exact_conflict_eligible": False,
                "terminal_authority": "corroboration_only",
            },
        }
        (root / "records.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        files = {
            name: digest(root / name)
            for name in ("vectorizer.joblib", "matrix.npz", "records.jsonl")
        }
        (root / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
        return {
            "source_id": "joykoli_six_part", "directory": root,
            "rights_note": "private user-provided fixture",
            "terminal_policy": "corroboration only",
            "exact_conflict_policy": STRICT_EXACT_CONFLICT_POLICY,
        }

    def test_opt_in_book_disagreement_reaches_worklist_instead_of_finalizing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = self.make_index(root / "authority")
            challenger = self.make_joykoli_index(root / "challenger")
            inputs = root / "input.jsonl"
            inputs.write_text(json.dumps({
                "example_id": "conflict", "source_index": 0,
                "model_prompt_bn": "বাংলাদেশের রাজধানী কী",
                "model_response_bn": "ঢাকা", "context_available": False,
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            build(
                inputs, root / "out", index_specs=[authority, challenger],
                top_k=1, batch_size=1,
            )
            output = json.loads(
                (root / "out" / "retrieval.jsonl").read_text(encoding="utf-8").strip()
            )
            merged = output["merged_source_candidate"]
            self.assertEqual(merged["status"], "source_conflict_quarantined")
            self.assertIsNone(merged["verdict"])
            self.assertEqual(len(merged["strict_exact_key_conflicts"]), 1)

    def test_repaired_page_chunk_survives_passage_gate_but_stays_nonterminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_joykoli_passage_index(root / "passage")
            inputs = root / "input.jsonl"
            inputs.write_text(json.dumps({
                "example_id": "sylhet", "source_index": 0,
                "model_prompt_bn": "সিলেট কোন প্রাচীন জনপদের অন্তর্ভুক্ত ছিল?",
                "model_response_bn": "হরিকেল", "context_available": False,
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            build(inputs, root / "out", index_specs=[spec], top_k=1, batch_size=1)
            output = json.loads(
                (root / "out" / "retrieval.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(len(output["retrieval_candidates"]), 1)
            candidate = output["retrieval_candidates"][0]
            self.assertTrue(candidate["passage_evidence"])
            self.assertTrue(candidate["model_facing_eligible"])
            self.assertIsNone(candidate["source_verdict_candidate"])
            self.assertEqual(
                output["merged_source_candidate"]["status"],
                "no_terminal_source_candidate",
            )

    def test_exact_candidate_and_fuzzy_nonterminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_index(root / "index")
            inputs = root / "input.jsonl"
            rows = [
                {"example_id": "a", "source_index": 0, "model_prompt_bn": "বাংলাদেশের রাজধানী কী", "model_response_bn": "ঢাকা", "formatting_provenance": {"status": "no_defect"}},
                {"example_id": "b", "source_index": 1, "model_prompt_bn": "বাংলাদেশের প্রধান শহর কোনটি", "model_response_bn": "ঢাকা", "formatting_provenance": {"status": "no_defect"}},
            ]
            inputs.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            manifest = build(inputs, root / "out", index_specs=[spec], top_k=1, batch_size=2)
            outputs = [json.loads(line) for line in (root / "out" / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertFalse(manifest["labels_read"])
            self.assertFalse(manifest["fuzzy_retrieval_terminal_labels"])
            self.assertEqual(outputs[0]["merged_source_candidate"]["verdict"], 1)
            self.assertEqual(outputs[1]["merged_source_candidate"]["status"], "no_terminal_source_candidate")
            self.assertTrue(all(
                candidate["model_facing_eligible"]
                for output in outputs for candidate in output["retrieval_candidates"]
            ))
            self.assertEqual(
                manifest["raw_retrieval_candidate_count"],
                manifest["model_facing_retrieval_candidate_count"]
                + manifest["quarantined_retrieval_candidate_count"],
            )
            self.assertEqual(
                manifest["implementation_sha256"],
                manifest["implementation"]["sparse_retrieval_sha256"],
            )
            self.assertTrue({
                "canonicalizer_sha256",
                "mmap_retrieval_runtime_sha256",
                "composite_fts_runtime_sha256",
                "response_proposition_sha256",
                "source_authority_runtime_sha256",
                "source_authority_policy_sha256",
            }.issubset(manifest["implementation"]))

    def test_preterminal_context_ids_skip_all_source_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_index(root / "index")
            inputs = root / "input.jsonl"
            rows = [
                {"example_id": "a", "source_index": 0, "model_prompt_bn": "বাংলাদেশের রাজধানী কী", "model_response_bn": "ঢাকা"},
                {"example_id": "b", "source_index": 1, "model_prompt_bn": "পানির রাসায়নিক সংকেত কী", "model_response_bn": "H2O"},
            ]
            inputs.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            manifest = build(
                inputs, root / "out", index_specs=[spec], top_k=1,
                batch_size=2, skip_example_ids={"a"},
            )
            outputs = [json.loads(line) for line in (root / "out" / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest["preterminal_skip"]["count"], 1)
            self.assertFalse(manifest["dense_similarity_materialized"])
            self.assertEqual(outputs[0]["merged_source_candidate"]["status"], "skipped_terminal_context")
            self.assertEqual(outputs[1]["merged_source_candidate"]["verdict"], 1)

    def test_nonterminal_context_rows_never_receive_external_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_index(root / "index")
            inputs = root / "input.jsonl"
            rows = [{
                "example_id": "context-a", "source_index": 0,
                "model_prompt_bn": "বাংলাদেশের রাজধানী কী",
                "model_response_bn": "ঢাকা", "context_available": True,
                "model_context_bn": "একটি অসম্পর্কিত সরবরাহকৃত অনুচ্ছেদ।",
            }]
            inputs.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest = build(
                inputs, root / "out", index_specs=[spec], top_k=1, batch_size=1
            )
            output = json.loads(
                (root / "out" / "retrieval.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(manifest["contextual_external_retrieval"]["count"], 1)
            self.assertFalse(manifest["contextual_external_retrieval"]["enabled"])
            self.assertEqual(output["retrieval_candidates"], [])
            self.assertEqual(
                output["merged_source_candidate"]["status"],
                "contextual_external_retrieval_disabled",
            )

    def test_aligned_rule_theory_context_cannot_enable_external_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_index(root / "index")
            inputs = root / "input.jsonl"
            inputs.write_text(json.dumps({
                "example_id": "context-science", "source_index": 0,
                "model_prompt_bn": "পানির রাসায়নিক সংকেত কী",
                "model_response_bn": "H2O", "context_available": True,
                "model_context_bn": "রসায়নে পানির রাসায়নিক সংকেত নির্দিষ্ট নিয়মে লেখা হয়।",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "contextual external lookup IDs are forbidden"
            ):
                build(
                    inputs, root / "out", index_specs=[spec], top_k=1,
                    batch_size=1,
                    context_external_lookup_ids={"context-science"},
                )


if __name__ == "__main__":
    unittest.main()
