import re
import requests
from typing import List, Tuple
from rdflib import Graph, URIRef
from pathlib import Path

# configuration
nt_file = Path(__file__).parent.parent.parent / "kg_artifacts" / "expanded_kb.nt"
ollama_url = "http://localhost:11434/api/generate"
model = "gemma:2b"

cc_ns = "http://climate-change.org/kg/"
max_predicates = 30
max_classes = 15
sample_triples = 20


# call local LLM via Ollama
def ask_local_llm(prompt):
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(ollama_url, json=payload, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")
    return response.json().get("response", "")


# load RDF graph
def load_graph(path):
    g = Graph()
    g.parse(str(path), format="nt")
    print(f"Graph loaded: {len(g)} triples from {path.name}")
    return g


# build schema summary from CC namespace only
def build_schema_summary(g):
    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    cc_triples = [(s, p, o) for s, p, o in g if str(s).startswith(cc_ns)]

    preds = sorted(set(str(p) for s, p, o in cc_triples))[:max_predicates]
    clss = sorted(set(str(o) for s, p, o in cc_triples if str(p) == rdf_type))[:max_classes]
    samples = [
        (str(s), str(p), str(o))
        for s, p, o in cc_triples
        if isinstance(o, URIRef) and str(p) != rdf_type
    ][:sample_triples]
    entities = sorted(set(str(s) for s, p, o in cc_triples))[:15]

    pred_lines = "\n".join(f"  <{p}>" for p in preds)
    cls_lines = "\n".join(f"  <{c}>" for c in clss)
    sample_lines = "\n".join(f"  <{s}> <{p}> <{o}> ." for s, p, o in samples)
    entity_lines = "\n".join(f"  <{e}>" for e in entities)

    return f"""# Base namespace: {cc_ns}
PREFIX cc: <{cc_ns}>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

# Classes
{cls_lines}

# Predicates
{pred_lines}

# Sample entity URIs
{entity_lines}

# Sample triples
{sample_lines}
""".strip()


# few-shot examples for NL to SPARQL
FEW_SHOT = """
EXAMPLE 1:
Question: What organizations are in the graph?
```sparql
SELECT DISTINCT ?org ?label WHERE {
  ?org rdf:type <http://climate-change.org/kg/Organization> .
  OPTIONAL { ?org rdfs:label ?label . }
}
LIMIT 20
```

EXAMPLE 2:
Question: Which persons are mentioned?
```sparql
SELECT DISTINCT ?person ?label WHERE {
  ?person rdf:type <http://climate-change.org/kg/Person> .
  OPTIONAL { ?person rdfs:label ?label . }
}
LIMIT 20
```

EXAMPLE 3:
Question: What geopolitical entities are in the graph?
```sparql
SELECT DISTINCT ?place ?label WHERE {
  ?place rdf:type <http://climate-change.org/kg/GeopoliticalEntity> .
  OPTIONAL { ?place rdfs:label ?label . }
}
LIMIT 20
```
"""

SPARQL_INSTRUCTIONS = """You are a SPARQL expert. Convert the QUESTION into a valid SPARQL 1.1 SELECT query.
Rules:
- Use ONLY the IRIs and prefixes from the SCHEMA SUMMARY.
- The main namespace for entities is: http://climate-change.org/kg/
- Use rdf:type to filter by class.
- Always add OPTIONAL { ?x rdfs:label ?label } to get labels.
- Add LIMIT 20 to all queries.
- Return ONLY the SPARQL query in one ```sparql code block.
- No text outside the code block."""


def make_sparql_prompt(schema_summary, question):
    return f"""{SPARQL_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

{FEW_SHOT}

QUESTION:
{question}

Return only the SPARQL query in a ```sparql code block."""


code_block_re = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sparql(text):
    m = code_block_re.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def generate_sparql(question, schema_summary):
    raw = ask_local_llm(make_sparql_prompt(schema_summary, question))
    return extract_sparql(raw)


# execute SPARQL query against the graph
def run_sparql(g, query):
    res = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows


REPAIR_INSTRUCTIONS = """The SPARQL query below failed. Fix it using the SCHEMA SUMMARY and ERROR MESSAGE.
Rules:
- Use ONLY IRIs from the schema. Main namespace: http://climate-change.org/kg/
- Keep the query simple.
- Return ONLY the corrected SPARQL in one ```sparql code block."""


def repair_sparql(schema_summary, question, bad_query, error_msg):
    prompt = f"""{REPAIR_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

QUESTION: {question}

BAD SPARQL:
{bad_query}

ERROR: {error_msg}

Return only the corrected SPARQL in a ```sparql code block."""
    raw = ask_local_llm(prompt)
    return extract_sparql(raw)


def answer_with_rag(g, schema_summary, question, try_repair=True):
    sparql = generate_sparql(question, schema_summary)
    try:
        vars_, rows = run_sparql(g, sparql)
        return {"query": sparql, "vars": vars_, "rows": rows, "repaired": False, "error": None}
    except Exception as e:
        err = str(e)
        if try_repair:
            repaired = repair_sparql(schema_summary, question, sparql, err)
            try:
                vars_, rows = run_sparql(g, repaired)
                return {"query": repaired, "vars": vars_, "rows": rows, "repaired": True, "error": None}
            except Exception as e2:
                return {"query": repaired, "vars": [], "rows": [], "repaired": True, "error": str(e2)}
        return {"query": sparql, "vars": [], "rows": [], "repaired": False, "error": err}


# direct LLM answer without the knowledge graph
def answer_no_rag(question):
    prompt = f"Answer the following question briefly and factually:\n\n{question}"
    return ask_local_llm(prompt)


# display RAG result
def pretty_print(result):
    print("\n[SPARQL Query Used]")
    print(result["query"])
    print(f"\n[Repaired] {result['repaired']}")
    if result.get("error"):
        print(f"[Error] {result['error']}")
    vars_ = result.get("vars", [])
    rows = result.get("rows", [])
    if not rows:
        print("[No results returned]")
        return
    print("\n[Results]")
    print(" | ".join(vars_))
    print("-" * 60)
    for r in rows[:20]:
        print(" | ".join(r))
    if len(rows) > 20:
        print(f"... (showing 20 of {len(rows)})")


# CLI demo
if __name__ == "__main__":
    print("Loading graph...")
    g = load_graph(nt_file)
    print("Building schema summary...")
    schema = build_schema_summary(g)
    print("Schema summary built.")
    print("\nClimate Change RAG demo — type 'quit' to exit.\n")

    while True:
        q = input("Question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        print("\n--- Baseline (No RAG) ---")
        print(answer_no_rag(q))
        print("\n--- RAG (NL->SPARQL + self-repair) ---")
        result = answer_with_rag(g, schema, q, try_repair=True)
        pretty_print(result)
        print()