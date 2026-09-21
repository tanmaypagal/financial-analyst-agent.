"""Company-name normalisation shared by the MCP server (lookup) and the agent (grounding). Pure functions, no DB code."""
import re

# Corporate suffixes that never distinguish one company from another in this universe.
SUFFIXES = ("group holdings", "corporation", "corp", "incorporated", "inc", "ltd", "plc", "asa", "ag", "se", "ab")


def normalize_company(name: str) -> str:
    """'Oracle Corp.' -> 'oracle'; 'Kongsberg Gruppen ASA' -> 'kongsberg gruppen'. Lower-case, no punctuation, suffixes removed."""
    s = re.sub(r"[.,]", " ", (name or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    changed = True
    while changed and s:
        changed = False
        for suf in SUFFIXES:
            if s == suf:
                return s
            if s.endswith(" " + suf):
                s, changed = s[: -len(suf) - 1].rstrip(), True
    return s
