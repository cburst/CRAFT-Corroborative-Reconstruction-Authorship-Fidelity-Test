#!/usr/bin/env python3
import csv
import os
import re
import random
import html
from collections import Counter

from weasyprint import HTML

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn

import requests  # NEW: for DeepSeek

# Ensure NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_TSV = "students.tsv"       # student_id, name, text
PDF_DIR = "PDFs"
ANSWER_KEY = "answer_key_synonym_replacer.tsv"
FREQ_FILE = "wiki_freq.txt"      # "word count" per line
NUM_WORDS_TO_REPLACE = 5          # how many words you want replaced
NUM_CANDIDATE_OBSCURE = 10        # how many rare words to consider
AVOID_WORDS = {"hufs", "macalister", "minerva"}           # skip HUFS or variants

# Simple English stopwords
STOPWORDS = {
    "the","a","an","and","or","but","if","than","then","therefore","so","because",
    "of","to","in","on","for","at","by","from","with","as","about","into","through",
    "after","over","between","out","against","during","without","before","under","around",
    "among","is","am","are","was","were","be","been","being","have","has","had","do","does",
    "did","can","could","will","would","shall","should","may","might","must","i","you",
    "he","she","it","we","they","me","him","her","us","them","my","your","his","their",
    "our","its","this","that","these","those","there","here","up","down","very","also",
    "just","only","not","no","yes","than","such","many","much","few","several","some",
    "any","all","each","every","both","either","neither","one","two","three","four",
    "five","first","second","third"
}

# 🔑 DeepSeek config (fill in your key)
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_RETRIES = 3


# Ensure POS tagger
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")
    
from nltk import word_tokenize, pos_tag

def guess_pos_for_word(text, word_lower):
    """
    Return (penn_tag, wn_pos) for the first occurrence of word_lower in the text.
    Example: ("NNS", "n"), ("VBD","v"), etc.
    """
    for sent in split_into_sentences(text):
        if re.search(r"\b" + re.escape(word_lower) + r"\b", sent, re.IGNORECASE):
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            for tok, tag in tagged:
                if tok.lower() == word_lower:
                    wn_pos = penn_to_wn_pos(tag)
                    return tag, wn_pos
            break
    return None, None
    
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def penn_to_wn_pos(tag):
    if tag.startswith("N"):
        return "n"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return None
  
def inflect_like(original, base_synonym, penn_tag):
    """
    Heuristic inflection:
    - plural nouns (NNS/NNPS) -> pluralize synonym
    - 3sg present (VBZ)       -> add -s / -es / -ies
    - past tense (VBD/VBN)    -> add -ed / -ied (very rough)
    - gerund (VBG)            -> add -ing
    Otherwise, return base_synonym.
    """
    syn = base_synonym

    # Plural nouns
    if penn_tag in ("NNS", "NNPS"):
        # Very rough pluralization rules
        if syn.endswith(("s","x","z","ch","sh")):
            syn = syn + "es"
        elif re.search(r"[^aeiou]y$", syn):
            syn = syn[:-1] + "ies"
        else:
            syn = syn + "s"
        return syn

    # Verb 3rd person singular present
    if penn_tag == "VBZ":
        if syn.endswith(("s","x","z","ch","sh")):
            syn = syn + "es"
        elif re.search(r"[^aeiou]y$", syn):
            syn = syn[:-1] + "ies"
        else:
            syn = syn + "s"
        return syn

    # Past tense / past participle
    if penn_tag in ("VBD", "VBN"):
        if re.search(r"[^aeiou]y$", syn):
            syn = syn[:-1] + "ied"
        elif syn.endswith("e"):
            syn = syn + "d"
        else:
            syn = syn + "ed"
        return syn

    # Gerund / present participle
    if penn_tag == "VBG":
        if syn.endswith("e") and not syn.endswith("ee"):
            syn = syn[:-1] + "ing"
        else:
            syn = syn + "ing"
        return syn

    return syn  
    
def find_sentence_and_surface_word(text, word_lower):
    """
    Find the first sentence containing word_lower (case-insensitive),
    and return (sentence, surface_form_as_it_appears).
    """
    for sent in split_into_sentences(text):
        m = re.search(r"\b" + re.escape(word_lower) + r"\b", sent, re.IGNORECASE)
        if m:
            return sent, m.group(0)  # surface form in that sentence
    return None, None    
    
# -----------------------------
# Utilities
# -----------------------------
def load_frequency_ranks(freq_file):
    freq_ranks = {}
    try:
        with open(freq_file, encoding="utf-8") as f:
            rank = 1
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0].lower()
                if word not in freq_ranks:
                    freq_ranks[word] = rank
                    rank += 1
    except FileNotFoundError:
        print(f"⚠️ Frequency file {freq_file} not found — all words treated equally.")
    return freq_ranks

def split_into_sentences(text):
    return [s.strip() for s in sent_tokenize(str(text)) if s.strip()]

def tokenize_words_lower(text):
    return re.findall(r"[A-Za-z']+", str(text).lower())

def sanitize_filename(name):
    forbidden = r'\/:*?"<>|'
    return "".join(c for c in name if c not in forbidden).strip() or "student"

# -----------------------------
# Core logic
# -----------------------------
def find_obscure_words(text, freq_ranks, num_candidates=NUM_CANDIDATE_OBSCURE):
    """
    Return up to num_candidates obscure words (rarest first).
    We'll later attempt to find synonyms for these and replace
    up to NUM_WORDS_TO_REPLACE of them.
    """
    tokens = tokenize_words_lower(text)
    counts = Counter(tokens)
    candidates = []

    for w, c in counts.items():
        if len(w) < 4:
            continue
        if w in STOPWORDS or w in AVOID_WORDS:
            continue
        if "'" in w:
            # skip possessives / contracted forms like "teacher's"
            continue
        rank = freq_ranks.get(w, 10**9)  # unseen = very rare
        candidates.append((rank, w))

    # sort by rarity: highest rank first
    candidates.sort(key=lambda x: x[0], reverse=True)

    result = []
    for _, w in candidates:
        if w not in result:
            result.append(w)
        if len(result) >= num_candidates:
            break

    return result
    
# ---------- DeepSeek helper ----------

def get_synonym_from_deepseek(surface_word, sentence):
    """
    Ask DeepSeek for a single-word synonym for surface_word that can
    replace it in the given sentence without breaking grammar.
    The synonym should match part of speech, inflection, and casing.

    Returns a string or None.
    """
    if not DEEPSEEK_API_KEY:
        return None

    system_prompt = (
        "You are a precise thesaurus assistant. Given an English word as it appears "
        "inside a sentence, you must produce exactly one synonym that could replace "
        "that word in the sentence without breaking grammar.\n\n"
        "Requirements:\n"
        "1) The synonym must have the SAME part of speech as the original word.\n"
        "2) It must have the SAME inflection (singular/plural, verb tense, etc.).\n"
        "3) It must have the SAME capitalization pattern as the original word.\n"
        "4) Respond with ONLY the replacement word, no explanation.\n"
        "5) If there is no good synonym, respond with the original word."
    )

    user_prompt = (
        f"Sentence:\n{sentence}\n\n"
        f"Original word in this sentence: {surface_word}\n\n"
        "Return exactly one synonym that can replace this word in the sentence."
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 20,
        "temperature": 0.5,
        "stream": False,
    }

    attempt = 0
    while attempt < DEEPSEEK_MAX_RETRIES:
        try:
            print("\n----------------------------")
            print(f"DeepSeek synonym lookup for: {surface_word!r} (attempt {attempt+1})")
            print("Sentence context:")
            print(sentence)
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=20)

            if resp.status_code >= 400:
                print(f"⚠️ DeepSeek HTTP {resp.status_code}")
                print("Response body:")
                print(resp.text)
                attempt += 1
                continue

            job = resp.json()
            text = None
            if "choices" in job and job["choices"]:
                choice = job["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    text = choice["message"]["content"]
                elif "text" in choice:
                    text = choice["text"]
            if text is None:
                text = str(job)

            print("DeepSeek raw content:")
            print(text)
            print("----------------------------")

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                attempt += 1
                continue

            candidate = lines[0]
            if (candidate.startswith('"') and candidate.endswith('"')) or \
               (candidate.startswith("'") and candidate.endswith("'")):
                candidate = candidate[1:-1].strip()

            tokens = re.findall(r"[A-Za-z]+", candidate)
            if not tokens:
                attempt += 1
                continue

            synonym = tokens[0]
            # if DeepSeek just echoed the original, reject and retry
            if synonym.lower() == surface_word.lower():
                attempt += 1
                continue

            print(f"Accepted DeepSeek synonym for {surface_word!r}: {synonym}")
            return synonym

        except Exception as e:
            print(f"⚠️ DeepSeek error while getting synonym for {surface_word!r}: {e}")
            attempt += 1

    print(f"⚠️ DeepSeek failed to find synonym for {surface_word!r}.")
    return None


def get_synonym(surface_word, sentence):
    return get_synonym_from_deepseek(surface_word, sentence)


def replace_word_case_preserving(text, original_lower, synonym):
    def repl(m):
        orig = m.group(0)
        if orig.isupper():
            rep = synonym.upper()
        elif orig[0].isupper():
            rep = synonym.capitalize()
        else:
            rep = synonym.lower()
        return rep
    pattern = re.compile(r"\b(" + re.escape(original_lower) + r")\b", re.IGNORECASE)
    return pattern.sub(repl, text)

def transform_text_with_synonyms(text, freq_ranks):
    """
    1) Find up to NUM_CANDIDATE_OBSCURE rare words.
    2) For each, find a sentence and the surface form of the word.
    3) Ask DeepSeek for a synonym that matches POS + inflection.
    4) Replace up to NUM_WORDS_TO_REPLACE words in the text.

    Returns:
      modified_text, replacements_list (list of (original_surface, synonym)).
    """
    candidate_words = find_obscure_words(text, freq_ranks, num_candidates=NUM_CANDIDATE_OBSCURE)

    modified_text = text
    replacements = []

    for w_lower in candidate_words:
        if len(replacements) >= NUM_WORDS_TO_REPLACE:
            break

        # Find a sentence and surface form (e.g. "Fulfilling" vs "fulfilling")
        sentence, surface_word = find_sentence_and_surface_word(modified_text, w_lower)
        if not sentence or not surface_word:
            continue

        synonym = get_synonym(surface_word, sentence)
        if not synonym:
            continue

        # Replace all occurrences of w_lower with synonym, preserving capitalization per occurrence
        pattern = re.compile(r"\b" + re.escape(w_lower) + r"\b", re.IGNORECASE)

        if not pattern.search(modified_text):
            continue

        def repl(m):
            orig = m.group(0)
            # if DeepSeek already matched the casing, just use it;
            # but we can still adjust for safety:
            if orig.isupper():
                return synonym.upper()
            elif orig[0].isupper():
                return synonym[0].upper() + synonym[1:]
            else:
                return synonym.lower()

        modified_text = pattern.sub(repl, modified_text)
        replacements.append((surface_word, synonym))

    return modified_text, replacements
    
# -----------------------------
# PDF generation
# -----------------------------
def generate_pdf(student_id, name, modified_text, num_blanks, out_dir=PDF_DIR):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(out_dir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_id)
    esc_text = html.escape(modified_text)

    # Text for "N words"
    plural = "word" if num_blanks == 1 else "words"

    # Build blanks dynamically
    blanks_html = []
    for i in range(1, num_blanks + 1):
        blanks_html.append(
            f"<div class='text'>{i}. ____________________ → ____________________</div>"
        )
    blanks_block = "\n".join(blanks_html)

    html_doc = f"""
    <html>
    <head>
    <meta charset='utf-8'>
<style>
	@page {{
		margin: 1.5cm;
		size: A4;
	}}
	body {{
		font-family: Arial, sans-serif;
		font-size: 13pt;
		line-height: 1.4;
		margin: 0;
		padding: 0;
	}}
	.header {{
		font-weight: bold;
		margin-bottom: 0.5em;
	}}
	.instructions {{
		white-space: normal;
		margin: 0;
		text-indent: 0;
		margin-left: 0;
		padding-left: 0;
		text-align: left;
	}}
	.text {{
		white-space: pre-wrap;
		margin: 0.5em 0;
		text-indent: 2em;        /* ← normal paragraph indent */
	}}
</style>
    </head>
    <body>
      <div class='header'>
        Name: {esc_name}<br>
        Student Number: {esc_number}
      </div>
      <div class='header'>Synonym Replacer</div>
      <div class='instructions' style="text-indent: 0;">
        In the text below, <b>{num_blanks} {plural}</b> have been replaced with synonyms.<br>
        Identify the replaced words and write the original words below.
      </div><br>
      <div class='text'>{esc_text}</div>
      <br>
      {blanks_block}
    </body>
    </html>
    """
    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📝 PDF created: {pdf_path}")

# -----------------------------
# Main
# -----------------------------
def process_tsv(input_tsv, output_tsv, freq_file=FREQ_FILE):
    freq_ranks = load_frequency_ranks(freq_file)
    with open(output_tsv, "w", newline="", encoding="utf-8") as keyfile:
        writer = csv.writer(keyfile, delimiter="\t")
        writer.writerow(["student_id", "name", "original", "synonym"])
        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                student_id, name, text = row[0], row[1], row[2]

                print(f"\n=== Processing {student_id} / {name} ===")
                modified, replacements = transform_text_with_synonyms(text, freq_ranks)
                if not replacements:
                    print(f"⚠️ No suitable replacements for {name}. Skipping.")
                    continue

                num_blanks = len(replacements)
                generate_pdf(student_id, name, modified, num_blanks)

                for orig, syn in replacements:
                    writer.writerow([student_id, name, orig, syn])

    print(f"✅ Done. Output TSV: {output_tsv}")

# -----------------------------
if __name__ == "__main__":
    INPUT_TSV = "students.tsv"
    OUTPUT_TSV = "answer_key_synonym_replacer.tsv"
    process_tsv(INPUT_TSV, OUTPUT_TSV)