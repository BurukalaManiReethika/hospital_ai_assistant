# Hospital AI Assistant

A small RAG chatbot that answers hospital FAQs and doctor-availability questions
from a knowledge base (`faqs.json`, `doctors.json`). The hospital and doctor data
is sample data for demonstration.

## How it works

1. **Chunking**: each FAQ and each doctor profile becomes one chunk.
2. **Retrieval**: TF-IDF vectors (unigrams + bigrams) with cosine similarity pick the top 3 chunks.
3. **Grounded answer**: the reply is built only from the retrieved chunks, and the sources are shown.
4. **Fallback**: if the best score is below a threshold, the bot says it doesn't know instead of guessing.
5. **Safety rules**: emergency phrases (e.g. chest pain) show emergency guidance; requests for diagnosis or medicine advice are declined.
6. **Optional LLM step**: if `ANTHROPIC_API_KEY` is set, the retrieved chunks are sent to an LLM to phrase the answer (needs network access; falls back to the plain grounded answer on any error).

## Run

```
pip install -r requirements.txt
python app.py            # open http://localhost:5000
python rag.py            # command-line chat
python -m unittest -v    # run tests
```

## API

`POST /api/chat` with `{"message": "..."}` returns `{answer, mode, sources, score}`.

## Possible improvements

- Replace TF-IDF with sentence embeddings and a vector index (e.g. sentence-transformers + FAISS) to handle paraphrases and typos better.
- Load doctors and schedules from a database instead of JSON.
- Add appointment booking, conversation history and Telugu/Hindi support.
