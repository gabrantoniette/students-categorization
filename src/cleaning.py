import re 
import unicodedata as unicode

# pre defined null_tokens to validate null values in data
NULL_TOKENS = {
    "", 
    "-", 
    "--", 
    "?", 
    "n/a", 
    "na", 
    "nan", 
    "null", 
    "none", 
    "nao informado", 
    "sem informacao", 
    "desconhecido"
}


def strip_accent(raw):
    text = unicode.normalize("NFKD", raw)
    return "".join(c for c in text if not unicode.combining(c))   


def normalize_token(raw):
    return  " ".join(strip_accent(str(raw)).lower().split())


def normalize_headers(token):
    return re.sub(r"[^a-z0-9]+", "_", normalize_token(token)).strip("_")


def is_null(raw, extra_tokens=()): # add extra_tokens paramenters to defined more null values beyond the existing ones. 
    token = normalize_token(raw)
    return token in NULL_TOKENS or token in extra_tokens 


def parse_int(raw):
    match = re.fullmatch(r"(\d+)(?:[.,]0+)?(?:\s*anos?)?", normalize_token(raw))
    return int(match.group(1)) if match else None


def _text(raw, spec):
    text = " ".join(unicode.normalize("NFC", raw).split())
    return text.title() if spec.get("case") == "title" else text


def _category(raw, spec):
    token = normalize_token(raw)
    return spec.get("synonyms", {}).get(token, token)


PARSERS = {
    "int": lambda raw, spec: parse_int(raw),
    "text": _text,
    "category": _category,
}


def clean_values(raw, spec):
    if is_null(raw, spec.get("nulls", ())):
        return None, ("missing" if spec.get("required") else None)

    value = PARSERS[spec["type"]](raw, spec)
    if value is None:
        return None, (f"cannot parse {raw!r} as {spec['type']}")
    if "min" in spec and value < spec["min"] or "max" in spec and value > spec["max"]:
        return  None, (f"{value!r} out of range, required a value between {spec['min']}, {spec['max']}")
    if "allowed" in spec and value not in spec["allowed"]:
        return None, (f"category {value!r} is not allowed.")
    return value, None