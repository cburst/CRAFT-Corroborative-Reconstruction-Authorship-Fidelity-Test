#!/usr/bin/env python3
import csv
import re
import os
import random
import html

from weasyprint import HTML

import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ===============================
# CONFIG
# ===============================

INPUT_TSV = "students.tsv"      # student_id, name, text
PDF_DIR = "pdfs_context_repair"
ANSWER_KEY = "answer_key_context_repair.tsv"

ITEMS_PER_STUDENT = 5           # total items (and also number of longest sentences)
VERB_ITEMS = 3                  # tense / aux / modal change
PREP_ITEMS = 2                  # preposition change

# Preposition and verb-ish sets
PREPOSITIONS = {
    "in","on","at","to","from","for","with","about","over","under","between",
    "through","during","before","after","around","against","into","onto","out",
    "of","by","like","without","within","among","across","behind","beyond",
    "above","below","toward","towards"
}

BE_GROUP = {"am","is","are","was","were","be","been","being"}
HAVE_GROUP = {"have","has","had"}
DO_GROUP = {"do","does","did"}
MODALS = {"can","could","will","would","shall","should","may","might","must"}

VERB_GROUPS = [BE_GROUP, HAVE_GROUP, DO_GROUP, MODALS]


# ===============================
# Utilities
# ===============================

def split_into_sentences(text):
    if not text:
        return []
    sentences = sent_tokenize(str(text))
    return [s.strip() for s in sentences if s.strip()]

def tokenize_words_lower(text):
    """Lowercase word tokens, letters/apostrophes only."""
    return re.findall(r"[A-Za-z']+", str(text).lower())

def sanitize_filename(name):
    forbidden = r'\/:*?"<>|'
    return "".join(c for c in name if c not in forbidden).strip() or "student"


def replace_first_occurrence_case_insensitive(sentence, target_lower, replacement):
    """
    Replace the FIRST occurrence of target_lower (case-insensitive) with replacement.
    Preserve capitalization style on that occurrence.
    """
    pattern = re.compile(r"\b(" + re.escape(target_lower) + r")\b", re.IGNORECASE)
    replaced = False

    def repl(m):
        nonlocal replaced
        if replaced:
            return m.group(0)
        orig = m.group(0)
        replaced = True
        # Match capitalization pattern of original
        if orig.isupper():
            rep = replacement.upper()
        elif orig[0].isupper():
            rep = replacement.capitalize()
        else:
            rep = replacement.lower()
        return rep

    return pattern.sub(repl, sentence)


def choose_verb_replacement(word_lower):
    """
    Given a lowercase verb-ish word, choose a 'wrong' alternative
    from its group (changes tense/person/modal).
    """
    for group in VERB_GROUPS:
        if word_lower in group:
            others = list(group - {word_lower})
            if others:
                return random.choice(others)
    # Fallback: something guaranteed wrong-ish
    return "did" if word_lower != "did" else "do"


def choose_prep_replacement(word_lower):
    """
    Given a lowercase preposition, choose a different preposition.
    """
    others = list(PREPOSITIONS - {word_lower})
    if others:
        return random.choice(others)
    return word_lower  # shouldn't really happen


def get_longest_sentence_indices(sentences, max_n=ITEMS_PER_STUDENT):
    """
    Return indices of the max_n longest sentences in the text,
    measured by number of word tokens (lowercased).
    """
    lengths = []
    for i, s in enumerate(sentences):
        wc = len(tokenize_words_lower(s))
        lengths.append((wc, i))
    # sort by descending length
    lengths.sort(key=lambda x: x[0], reverse=True)
    return [idx for _wc, idx in lengths[:max_n]]


# ===============================
# Candidate selection (within longest sentences)
# ===============================

def find_candidates_in_longest(sentences, longest_indices):
    """
    For each of the longest sentences, find candidate preposition and verb targets.

    Returns:
      prep_candidates: list of dicts:
          {"sent_idx": orig_index, "sentence": s, "word": w}
      verb_candidates: same structure
    """
    prep_candidates = []
    verb_candidates = []

    all_verb_vocab = set().union(*VERB_GROUPS)

    for idx in longest_indices:
        s = sentences[idx]
        tokens = tokenize_words_lower(s)
        if not tokens:
            continue

        # Prepositions
        prep_in_sent = [w for w in tokens if w in PREPOSITIONS]
        if prep_in_sent:
            w = random.choice(prep_in_sent)
            prep_candidates.append({
                "sent_idx": idx,
                "sentence": s,
                "word": w,
            })

        # Verb-ish
        verb_in_sent = [w for w in tokens if w in all_verb_vocab]
        if verb_in_sent:
            w = random.choice(verb_in_sent)
            verb_candidates.append({
                "sent_idx": idx,
                "sentence": s,
                "word": w,
            })

    return prep_candidates, verb_candidates


def select_items_for_student(text, max_items=ITEMS_PER_STUDENT):
    """
    Build items using ONLY the 5 longest sentences in the student's text.

    Target distribution (if possible):
      - PREP_ITEMS sentences with a preposition error
      - VERB_ITEMS sentences with a verb/aux/modal error
      - All from unique sentences among those 5 longest.

    Returns list of items:
      each item is dict:
        {
          "type": "verb" or "prep",
          "sentence": original_sentence,
          "target_word": correct_word_lower,
          "wrong_word": wrong_word_lower,
          "corrupted_sentence": plain_corrupted_sentence
        }
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Find indices of the 5 longest sentences
    longest_indices = get_longest_sentence_indices(sentences, max_n=max_items)
    if not longest_indices:
        return []

    prep_candidates, verb_candidates = find_candidates_in_longest(sentences, longest_indices)
    random.shuffle(prep_candidates)
    random.shuffle(verb_candidates)

    items = []
    used_sentences = set()

    # First: pick up to PREP_ITEMS from distinct longest sentences
    prep_picked = 0
    for cand in prep_candidates:
        if prep_picked >= PREP_ITEMS:
            break
        if cand["sent_idx"] in used_sentences:
            continue

        sent = cand["sentence"]
        target = cand["word"]  # lowercase
        wrong = choose_prep_replacement(target)
        if wrong == target:
            continue

        corrupted = replace_first_occurrence_case_insensitive(sent, target, wrong)

        items.append({
            "type": "prep",
            "sentence": sent,
            "target_word": target,
            "wrong_word": wrong,
            "corrupted_sentence": corrupted,
        })
        used_sentences.add(cand["sent_idx"])
        prep_picked += 1

    # Next: pick up to VERB_ITEMS from remaining longest sentences
    verb_picked = 0
    for cand in verb_candidates:
        if verb_picked >= VERB_ITEMS:
            break
        if cand["sent_idx"] in used_sentences:
            continue

        sent = cand["sentence"]
        target = cand["word"]
        wrong = choose_verb_replacement(target)
        if wrong == target:
            continue

        corrupted = replace_first_occurrence_case_insensitive(sent, target, wrong)

        items.append({
            "type": "verb",
            "sentence": sent,
            "target_word": target,
            "wrong_word": wrong,
            "corrupted_sentence": corrupted,
        })
        used_sentences.add(cand["sent_idx"])
        verb_picked += 1

    # We’re constrained to the 5 longest sentences, so we can't exceed max_items anyway.
    # It’s possible we end up with < 5 items if some longest sentences lack preps/verbs.

    return items


# ===============================
# PDF generation
# ===============================

def generate_pdf_for_student(student_id, name, items, out_dir=PDF_DIR):
    """
    items: list of dicts:
      { "type", "sentence", "target_word", "wrong_word", "corrupted_sentence" }
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(out_dir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_id)

    html_parts = [
        "<html><head><meta charset='utf-8'><style>",
        "@page { margin: 1.5cm; size: A4; }",
        "body { font-family: Arial, sans-serif; font-size: 13pt; line-height: 1.3; }",
        ".header { font-weight: bold; margin-bottom: 0.5em; }",
        ".text { white-space: pre-wrap; margin: 0.3em 0; }",
        "</style></head><body>",
        f"<div class='header'>Name: {esc_name}<br>Student Number: {esc_number}</div>",
        "<div class='header'>Context Repair Quiz</div>",
        "<div class='text'>Each sentence below is one of the longest sentences from your essay.</div>",
        "<div class='text'>In each sentence, exactly <b>one word</b> has been changed.</div>",
        "<div class='text'>Write the <b>one correct word</b> you originally used on the line.</div>",
        "<br>",
    ]

    for q_num, item in enumerate(items, start=1):
        esc_sent = html.escape(item["corrupted_sentence"])
        html_parts.append(
            f"<div class='text'><b>Q{q_num}.</b> {esc_sent}</div>"
        )
        html_parts.append(
            "<div class='text'>Correct word: __________________________</div><br>"
        )

    html_parts.append("</body></html>")
    html_doc = "\n".join(html_parts)

    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📝 Context-repair PDF created: {pdf_path}")


# ===============================
# Main processing
# ===============================

def process_tsv_for_context_repair(input_tsv=INPUT_TSV,
                                   pdf_dir=PDF_DIR,
                                   answer_key_tsv=ANSWER_KEY):
    """
    Input TSV: student_id, name, text

    For each student:
      - Use only the 5 longest sentences from their text.
      - Build up to 5 items (aiming for 2 preposition and 3 verb-ish errors).
      - Generate a PDF.
      - Append answer-key rows.
    """
    with open(answer_key_tsv, "w", newline="", encoding="utf-8") as keyfile:
        keywriter = csv.writer(keyfile, delimiter="\t")
        keywriter.writerow([
            "student_id", "name", "question_number",
            "type", "correct_word", "wrong_word",
            "corrupted_sentence", "original_sentence"
        ])

        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue
                student_id, name, text = row[0], row[1], row[2]

                print(f"\n=== Processing {student_id} / {name} ===")
                items = select_items_for_student(text, max_items=ITEMS_PER_STUDENT)

                if len(items) == 0:
                    print(f"⚠️ No suitable longest sentences for {student_id}; skipping.")
                    continue

                generate_pdf_for_student(student_id, name, items, out_dir=pdf_dir)

                for q_num, item in enumerate(items, start=1):
                    keywriter.writerow([
                        student_id,
                        name,
                        q_num,
                        item["type"],
                        item["target_word"],
                        item["wrong_word"],
                        item["corrupted_sentence"],
                        item["sentence"],
                    ])

                keyfile.flush()
                print(f"✅ Finished {student_id} / {name}")

    print(f"\n🎯 Done. Answer key saved to {answer_key_tsv}")


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    random.seed()  # or set a fixed seed for reproducibility
    process_tsv_for_context_repair()