"""Robust parsing of model replies into {address, confidence, abstain, parse_ok}."""
import json, re

ADDR_RE = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")
ANY_HEX_RE = re.compile(r"0x[0-9a-zA-Z]+")


def _strip_fences(t):
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t


THINK_RE = re.compile(r"<(think|thought|thinking|reasoning)>.*?(</\1>|$)", re.S | re.I)
OBJ_RE = re.compile(r"\{[^{}]*\}", re.S)


def parse_reply(text):
    out = {"address": None, "raw_address": None, "confidence": None, "abstain": False, "parse_ok": False,
           "valid_format": False, "placeholder": False}
    if text is None:
        return out
    t = _strip_fences(THINK_RE.sub("", text))  # drop visible reasoning blocks (incl. unterminated ones)
    obj = None
    for cand in reversed(OBJ_RE.findall(t)):  # last well-formed flat JSON object wins
        for c in (cand, cand.replace("'", '"')):
            try:
                o = json.loads(c)
            except Exception:
                continue
            if isinstance(o, dict) and ({"address", "abstain", "confidence_0_100"} & set(o)):
                obj = o; break
        if obj is not None:
            break
    if isinstance(obj, dict):
        if obj.get("abstain") is True or str(obj.get("abstain")).lower() == "true":
            out.update(abstain=True, parse_ok=True)
            return out
        a = obj.get("address")
        c = obj.get("confidence_0_100", obj.get("confidence"))
        if isinstance(a, str):
            out["raw_address"] = a.strip()
        try:
            if c is not None:
                c = float(str(c).strip().rstrip("%"))
                out["confidence"] = max(0.0, min(100.0, c))
        except Exception:
            pass
    else:  # fallback: regex on free text
        if re.search(r'"?abstain"?\s*:\s*true', t, re.I):
            out.update(abstain=True, parse_ok=True)
            return out
        h = ANY_HEX_RE.search(t)
        if h:
            out["raw_address"] = h.group(0)
        cm = re.search(r"confidence[^0-9]{0,15}(\d{1,3}(?:\.\d+)?)", t, re.I)
        if cm:
            out["confidence"] = max(0.0, min(100.0, float(cm.group(1))))
    ra = out["raw_address"]
    if ra:
        out["parse_ok"] = True
        if "..." in ra or "…" in ra or len(ra) < 10:   # echoed template like "0x..." = no real answer
            out["placeholder"] = True
            return out
        if ADDR_RE.fullmatch(ra):
            out["valid_format"] = True
            out["address"] = ra
    return out
