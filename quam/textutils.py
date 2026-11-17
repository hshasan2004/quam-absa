"""Text helpers shared by EDA and preprocessing."""
import re
import unicodedata

BN = re.compile(r"[ঀ-৿]")
LAT = re.compile(r"[A-Za-z]")
# Frequent romanised-Bangla tokens. Heuristic only: used to separate Banglish from English.
BANGLISH = set("""valo bhalo valoi kharap kharapi onek anek ase ache achhe nai nei na hoy hoyeche hoise
hoyse dise diyeche dilam pelam paisi paichi peyechi ekdom amar ami apni apnar tader taka dam bhai vai vaia
kintu tobe ar r jonno moto mone kore korse korche korlam chilo silo khub khubi sob shob kono kichu ekta
eta oita ota ta ti gulo gula thik shundor sundor sondor dorkar lagbe lage lagse laglo deoya deya
oneek valoo""".split())


def script_of(text) -> str:
    """Return 'bangla', 'english', 'banglish', 'mixed' or 'other' (heuristic)."""
    t = str(text)
    bn, lat = len(BN.findall(t)), len(LAT.findall(t))
    if bn + lat == 0:
        return "other"
    if bn / (bn + lat) >= 0.8:
        return "bangla"
    if lat / (bn + lat) >= 0.8:
        toks = re.findall(r"[a-z]+", t.lower())
        hits = sum(tok in BANGLISH for tok in toks)
        return "banglish" if toks and hits / len(toks) >= 0.15 else "english"
    return "mixed"


def clean_text(text) -> str:
    """Unicode-normalise and collapse whitespace; keeps casing and punctuation."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", t).strip()


def dedup_key(text) -> str:
    """Key used to keep duplicate texts in the same split."""
    return re.sub(r"[^\w]+", " ", clean_text(text).lower()).strip()


def aspect_query(name: str) -> str:
    """'Click_Buttons' -> 'click buttons', 'LowLight' -> 'low light', 'ANC_ENC' -> 'anc enc'."""
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name.strip())
    return re.sub(r"[_\s]+", " ", s).strip().lower()
