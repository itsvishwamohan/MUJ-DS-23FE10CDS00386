"""JSON schemas. They are sent to Ollama (structured outputs) and re-checked locally."""

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
NOT_ENOUGH = "not_enough_evidence"
LABELS = (SUPPORTED, CONTRADICTED, NOT_ENOUGH)

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "minLength": 5},
                    "original_text": {"type": "string"},
                    "search_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "original_text", "search_queries"],
            },
        }
    },
    "required": ["claims"],
}

# Property order matters: the model writes its reasoning first, then commits to a label.
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "reasoning": {"type": "string"},
                    "label": {"type": "string", "enum": list(LABELS)},
                    "evidence_quote": {"type": "string"},
                    "passage_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["id", "reasoning", "label", "evidence_quote", "passage_id", "confidence"],
            },
        }
    },
    "required": ["verdicts"],
}
