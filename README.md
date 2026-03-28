# Climate Change Knowledge Graph

End-to-end Knowledge Graph pipeline on Climate Change — Web Mining & Semantics, ESILV 2026.

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

For RAG, install [Ollama](https://ollama.com/download) then:
```bash
ollama pull gemma:2b
```

## How to Run

| Module | Command |
|---|---|
| Crawling + NER | Run `notebooks/lab1.ipynb` |
| KB Construction + Alignment | Run `notebooks/lab2.ipynb` |
| SWRL + KGE | Run `notebooks/lab3.ipynb` |
| RAG demo | `python src/rag/rag.py` |

## RAG Demo Example
```
Question (or 'quit'): What organizations are in the graph?
```

## Hardware

CPU only, 8 GB RAM minimum.