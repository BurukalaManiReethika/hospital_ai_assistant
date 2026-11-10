"""Small RAG pipeline for the Hospital AI Assistant.

Flow:  question -> safety checks -> retrieve top chunks (TF-IDF + cosine)
       -> answer grounded only in the retrieved text -> fallback if nothing relevant.

Retrieval is TF-IDF so it runs with no model download or network. If the
ANTHROPIC_API_KEY environment variable is set, the retrieved chunks are also
passed to an LLM to phrase the reply; any failure falls back to the plain
grounded answer. The LLM path needs network access and an API key.
"""
import json
import os
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
MIN_SCORE = 0.12      # below this, we do not trust the match
DOCTOR_RATIO = 0.6    # list other doctors scoring at least this fraction of the best

FALLBACK = (
    "Sorry, I don't have that information. I can help with OPD timings, appointments, "
    "doctor availability, visiting hours, insurance, reports and billing. For anything "
    "else, please call the hospital helpline or ask at the front desk."
)
EMERGENCY = (
    "This may be an emergency. Please call 112 right now or go straight to the "
    "emergency department, which is open 24 hours. Do not wait for an appointment."
)
MEDICAL_ADVICE = (
    "I can't give medical advice, diagnosis or medicine suggestions. A doctor needs to "
    "examine you. I can tell you which doctor to see and help you plan an appointment. "
    "If symptoms are severe, call 112 or visit the emergency department."
)

EMERGENCY_PATTERNS = [
    r"chest pain", r"heart attack", r"can('|no)?t breathe", r"cannot breathe",
    r"difficulty breathing", r"unconscious", r"not breathing", r"severe bleeding",
    r"bleeding heavily", r"stroke", r"seizure now", r"overdose", r"poison",
]
MEDICAL_ADVICE_PATTERNS = [
    r"which (medicine|tablet|drug)", r"what (medicine|tablet|drug)",
    r"should i take", r"what dose", r"dosage", r"how many tablets",
    r"do i have", r"diagnos", r"is it serious", r"treatment for",
]


def load_chunks():
    """Turn faqs.json and doctors.json into searchable chunks."""
    faqs = json.loads((BASE / "faqs.json").read_text(encoding="utf-8"))
    doctors = json.loads((BASE / "doctors.json").read_text(encoding="utf-8"))
    chunks = []
    for f in faqs:
        chunks.append({
            "id": f["id"],
            "type": "faq",
            "title": f["question"],
            "search_text": f"{f['question']} {f['answer']} {' '.join(f.get('keywords', []))}",
            "answer": f["answer"],
        })
    for d in doctors:
        answer = (
            f"{d['name']} ({d['department']}) sees patients {d['days']}, {d['timings']}, "
            f"{d['room']}. Consultation fee: Rs {d['fee']}."
        )
        chunks.append({
            "id": d["name"].lower().replace(" ", "-").replace(".", ""),
            "type": "doctor",
            "title": f"{d['name']} - {d['department']}",
            "search_text": (
                f"doctor {d['name']} {d['department']} {d['specialization']} "
                f"{' '.join(d['keywords'])} {d['days']} "
                "timing schedule available availability fee room"
            ),
            "answer": answer,
        })
    return chunks


def tokenize(text):
    """Lowercase words, drop stop words, and fold simple plurals (bones -> bone)."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in ENGLISH_STOP_WORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(c["search_text"] for c in self.chunks)

    def search(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in order if scores[i] > 0]


_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _matches(patterns, text):
    return any(re.search(p, text) for p in patterns)


def _llm_answer(question, hits):
    """Optional: let an LLM phrase the answer using only the retrieved context."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import requests

        context = "\n".join(f"- {h['answer']}" for h in hits)
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-5-5",
                "max_tokens": 300,
                "system": (
                    "You are a hospital front-desk assistant. Answer ONLY from the "
                    "context. If the context does not contain the answer, say you "
                    "don't know. Never give medical advice. Keep it short."
                ),
                "messages": [{
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }],
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    except Exception:
        return None


def answer(question):
    q = (question or "").strip()
    if not q:
        return {"answer": "Please type a question.", "mode": "empty", "sources": [], "score": 0.0}

    low = q.lower()
    if _matches(EMERGENCY_PATTERNS, low):
        return {"answer": EMERGENCY, "mode": "emergency", "sources": [], "score": 1.0}
    if _matches(MEDICAL_ADVICE_PATTERNS, low):
        return {"answer": MEDICAL_ADVICE, "mode": "safety", "sources": [], "score": 1.0}

    hits = get_retriever().search(q, k=3)
    if not hits or hits[0]["score"] < MIN_SCORE:
        return {"answer": FALLBACK, "mode": "fallback", "sources": [], "score": hits[0]["score"] if hits else 0.0}

    top = hits[0]
    if top["type"] == "doctor":
        used = [h for h in hits if h["type"] == "doctor" and h["score"] >= DOCTOR_RATIO * top["score"]]
    else:
        used = [top]

    text = _llm_answer(q, used) or "\n".join(h["answer"] for h in used)
    return {
        "answer": text,
        "mode": top["type"],
        "sources": [h["title"] for h in used],
        "score": round(top["score"], 3),
    }


if __name__ == "__main__":
    print("Hospital AI Assistant (type 'quit' to exit)")
    while True:
        try:
            user = input("You: ")
        except EOFError:
            break
        if user.strip().lower() in {"quit", "exit"}:
            break
        print("Bot:", answer(user)["answer"], "\n")
