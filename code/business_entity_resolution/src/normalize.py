"""Text normalisation for business names and addresses.

Everything here is offline and hand-written or learned from the training pairs:
  * a native-script -> English token dictionary learned from train matches
    (falls back to `anyascii` romanisation for unseen tokens),
  * abbreviation / legal-suffix / state-name canonicalisation tables,
  * splitting of website names / hashtags into words with a unigram model built
    from the name tokens of the data itself.
"""
import math
import re
from collections import Counter, defaultdict

from anyascii import anyascii

from config import NORM_V7

# --------------------------------------------------------------------------
# canonicalisation tables
# --------------------------------------------------------------------------
LEGAL = {
    "private": "private", "pvt": "private", "pvte": "private", "priv": "private",
    "limited": "limited", "ltd": "limited", "ltda": "limited", "limitee": "limited",
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cos": "co",
    "llc": "llc", "llp": "llp", "pllc": "pllc", "pc": "pc", "plc": "plc",
    "lp": "lp", "llllp": "llp", "pa": "pa", "ltee": "limited",
    "public": "public", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sci": "sci", "eurl": "eurl",
    "sa": "sa", "snc": "snc", "scop": "scop", "selarl": "selarl", "scm": "scm",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
}
# multi-token legal forms collapsed before tokenising
LEGAL_PHRASES = [
    (r"\bl\s*\.?\s*l\s*\.?\s*c\b\.?", " llc "),
    (r"\bl\s*\.?\s*l\s*\.?\s*p\b\.?", " llp "),
    (r"\bp\s*\.\s*l\s*\.\s*l\s*\.\s*c\b\.?", " pllc "),
    (r"\bp\s*\.\s*c\s*\.", " pc "),
    (r"\bs\s*\.\s*a\s*\.\s*r\s*\.\s*l\b\.?", " sarl "),
    (r"\bs\s*\.\s*a\s*\.\s*s\s*\.\s*u\b\.?", " sasu "),
    (r"\bs\s*\.\s*a\s*\.\s*s\b\.?", " sas "),
    (r"\bs\s*\.\s*a\b\.", " sa "),
    (r"\bpvt\.?\s*ltd\b\.?", " private limited "),
]
NAME_STOP = {"the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "a", "an", "for", "en"}
DBA_SPLIT = re.compile(r"\b(?:dba|d\s*/\s*b\s*/\s*a|t\s*/\s*a|aka|a\s*/\s*k\s*/\s*a|trading as|doing business as|also known as)\b")
TLD = re.compile(r"(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|fr|biz|info|us|io|org\.in|net\.in|shop|store|online|site|eu)\b")
DIGIT2CHAR = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g"})
ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th|er|e|eme|ere)$")
# v7: house numbers written together with a letter suffix / prefix ("60bis", "17b", "b177", "w7615")
_NUM_SUFFIX = re.compile(r"^(\d+)([a-z]{1,6})$")
_NUM_PREFIX = re.compile(r"^([a-z]{1,3})(\d+)$")
# v7: honorific prefixes injected in front of S2/S3 names ("M/s", "Shri", "Smt" never start an S1 name)
_HONORIFIC = re.compile(r"^\s*(?:m\s*/\s*s|mrs|mr|ms|dr|smt|shri|sri)\b\.?\s*(?=\S)")

ADDR_ABBR = {
    # english street types
    "st": "street", "str": "street", "street": "street", "rd": "road", "road": "road",
    "ave": "avenue", "av": "avenue", "avn": "avenue", "avenue": "avenue", "aven": "avenue",
    "dr": "drive", "drv": "drive", "ln": "lane", "ct": "court", "crt": "court",
    "cir": "circle", "circ": "circle", "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard",
    "bvd": "boulevard", "hwy": "highway", "hiway": "highway", "pkwy": "parkway", "pky": "parkway",
    "pl": "place", "ter": "terrace", "terr": "terrace", "trl": "trail", "tr": "trail",
    "sq": "square", "pt": "point", "mt": "mount", "ft": "fort", "cres": "crescent",
    "xing": "crossing", "expy": "expressway", "fwy": "freeway", "tpke": "turnpike",
    "cty": "county", "hts": "heights", "jct": "junction", "mdws": "meadows", "rdg": "ridge",
    "vly": "valley", "vw": "view", "vlg": "village", "sta": "station", "spg": "spring",
    "spgs": "springs", "cv": "cove", "pnt": "point", "rte": "route", "rt": "route",
    "cswy": "causeway", "aly": "alley", "br": "branch", "brg": "bridge", "bnd": "bend",
    "clf": "cliff", "crk": "creek", "cyn": "canyon", "est": "estate", "ests": "estates",
    "grv": "grove", "hbr": "harbor", "hl": "hill", "hls": "hills", "holw": "hollow",
    "is": "island", "lk": "lake", "lks": "lakes", "ldg": "lodge", "mnr": "manor",
    "orch": "orchard", "pass": "pass", "psge": "passage", "pne": "pine", "pnes": "pines",
    "plz": "plaza", "prt": "port", "rnch": "ranch", "run": "run", "shr": "shore",
    "smt": "summit", "trce": "trace", "trak": "track", "tunl": "tunnel", "un": "union",
    "vis": "vista", "wl": "well", "wls": "wells", "wy": "way",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    # indian
    "nr": "near", "opp": "opposite", "bldg": "building", "apts": "apartment",
    "apartments": "apartment", "apt": "apartment", "appt": "apartment", "soc": "society",
    "ngr": "nagar", "mkt": "market", "chwk": "chowk", "clny": "colony", "col": "colony",
    "extn": "extension", "ext": "extension", "sec": "sector", "sect": "sector",
    "flr": "floor", "fl": "floor", "gf": "ground floor", "ff": "first floor",
    "mg": "mahatma gandhi", "ph": "phase", "stn": "station", "hosp": "hospital",
    "indl": "industrial", "ind": "industrial", "estt": "estate", "comp": "complex",
    "cplx": "complex", "twr": "tower", "po": "post office",
    # french
    "r": "rue", "imp": "impasse", "all": "allee", "che": "chemin", "chem": "chemin",
    "fbg": "faubourg", "rpt": "rond point", "sq": "square", "qu": "quai", "crs": "cours",
    "pas": "passage", "res": "residence", "lot": "lotissement", "za": "zone",
    "zi": "zone", "zac": "zone", "bis": "bis", "ter": "terrace",
}
# tokens that carry no identity information in an address
ADDR_STOP = {
    "null", "none", "nan", "na", "n", "a", "unit", "suite", "ste", "apt", "no", "nos",
    "number", "num", "h", "hno", "hn", "house", "door", "dno", "plot", "flat", "shop",
    "office", "room", "box", "pobox", "c", "o", "s", "w", "d", "the", "of", "and",
    "township", "twp", "cdp", "city", "town", "village", "region", "district", "dist",
    "taluk", "tehsil", "tal", "teh", "dt", "via", "post", "de", "du", "des", "la", "le",
    "les", "en", "sur", "et", "l", "cedex", "apartment", "floor", "fl", "bldg",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "goa", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mnp", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "lad", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "andaman and nicobar islands": "an",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
}
STATE_CODES = set(US_STATES.values()) | {c for v in IN_STATES.values() for c in v.split()} | {"or", "od", "tg", "ct", "ua"}
_STATE_MAP = {**US_STATES, **IN_STATES}

_NONALNUM = re.compile(r"[^a-z0-9]+")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_WS = re.compile(r"\s+")
_LONGNUM = re.compile(r"\d{7,}")
_HASHNUM = re.compile(r"#\s*\d+")
_BRACKET_NOISE = re.compile(r"[\[\]\(\)\{\}<>|]")
_NUM = re.compile(r"\d+")
_DEGREE = re.compile(r"\b[nN]\s*[°º]")


_NON_LATIN = re.compile(r"[^\x00-ɏḀ-ỿ]")


def is_native(s):
    """True when the text contains non-ASCII characters (accents or other scripts)."""
    return bool(_NON_ASCII.search(s))


def is_non_latin(s):
    """True when the text contains a non-Latin script (Devanagari, Tamil, ...)."""
    return bool(_NON_LATIN.search(s))


# --------------------------------------------------------------------------
# learned transliteration dictionary (native-script token -> english token)
# --------------------------------------------------------------------------
_ASCII_ALNUM = re.compile(r"[a-z0-9]+")


def _plain_tokens(s):
    """Lower-case ASCII alphanumeric tokens of a string (after transliteration with anyascii)."""
    return _ASCII_ALNUM.findall(anyascii(s).lower())


def learn_translit(pairs, min_count=2, min_share=0.5):
    """pairs: iterable of (english_name, other_name, english_addr, other_addr).

    Name tokens are aligned by position when both names have the same number of
    whitespace tokens; comma-separated address components that are fully in a
    native script are aligned to the S1 address component they co-occur with most.
    """
    tok_counts = defaultdict(Counter)
    comp_counts = defaultdict(Counter)
    comp_total = Counter()
    for n1, n2, a1, a2 in pairs:
        if n2 and is_native(n2) and not is_native(n1):
            t2 = n2.split()
            t1 = n1.lower().replace("&", " and ").split()
            t1 = [re.sub(r"[^a-z0-9]", "", t) for t in t1]
            t1 = [t for t in t1 if t]
            if len(t1) == len(t2):
                for x, y in zip(t2, t1):
                    if is_native(x):
                        tok_counts[x][y] += 1
        if a2 and is_native(a2):
            comps1 = {c.strip().lower() for c in a1.split(",") if c.strip()}
            for c in a2.split(","):
                c = c.strip()
                if c and is_native(c):
                    comp_total[c] += 1
                    for d in comps1:
                        comp_counts[c][d] += 1
    tok_map = {}
    for x, cnt in tok_counts.items():
        y, c = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if c >= min_count and c / tot >= min_share:
            tok_map[x] = y
    comp_map = {}
    for x, cnt in comp_counts.items():
        if _ASCII_ALNUM.search(x.lower()) or len(x) < 2:
            continue
        y, c = cnt.most_common(1)[0]
        if c >= min_count and c / comp_total[x] >= min_share:
            comp_map[x] = y
    return {"tok": tok_map, "comp": comp_map}


def translit_name(s, tmap):
    """Native-script name -> Latin: learned word map first, anyascii as the fallback."""
    if not is_native(s):
        return s
    tok = tmap["tok"]
    return " ".join(tok.get(t) or anyascii(t) for t in s.split())


def translit_addr(s, tmap):
    """Native-script address -> Latin, per comma component: learned component map (e.g. state
    names), then the learned word map, then anyascii."""
    if not is_native(s):
        return s
    comp, tok = tmap["comp"], tmap["tok"]
    out = []
    for c in s.split(","):
        cs = c.strip()
        if cs in comp:
            out.append(comp[cs])
        elif is_native(cs):
            out.append(" ".join(tok.get(t) or anyascii(t) for t in cs.split()))
        else:
            out.append(c)
    return ",".join(out)


# --------------------------------------------------------------------------
# word segmentation for concatenated names (domains / hashtags)
# --------------------------------------------------------------------------
class Segmenter:
    """Split concatenated words (website / hashtag names) with a unigram language model:
    the most probable split under word frequencies counted in the data (Viterbi)."""
    def __init__(self, counts, max_len=20):
        """counts: word -> frequency; max_len: longest word considered."""
        total = sum(counts.values()) or 1
        self.logp = {w: math.log(c / total) for w, c in counts.items() if len(w) >= 2 or w in ("a",)}
        self.max_len = max_len
        self.unk = math.log(1.0 / total) - 5

    def split(self, text):
        """Most probable sequence of words for `text`; unknown pieces are penalised per character."""
        n = len(text)
        if n < 6:
            return [text]
        best = [0.0] + [-1e18] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                w = text[j:i]
                lp = self.logp.get(w)
                if lp is None:
                    lp = self.unk * (i - j)
                sc = best[j] + lp
                if sc > best[i]:
                    best[i], back[i] = sc, j
        out, i = [], n
        while i > 0:
            out.append(text[back[i]:i])
            i = back[i]
        return out[::-1]


# --------------------------------------------------------------------------
# name normalisation
# --------------------------------------------------------------------------
def _fix_digits(tok):
    """Fix OCR-style digit typos inside words ("6reen" -> "green"); numbers and ordinals are kept."""
    if tok.isdigit() or ORDINAL.match(tok):
        return tok
    if any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok):
        return tok.translate(DIGIT2CHAR)
    return tok


def _name_tokens(s):
    """Tokenise a name: digit typos fixed, legal suffixes canonicalised, immediate repeats removed."""
    toks = []
    prev = None
    for t in _NONALNUM.sub(" ", s).split():
        t = _fix_digits(t)
        t = LEGAL.get(t, t)
        if t == prev:  # "Global Global"
            continue
        toks.append(t)
        prev = t
    return toks


def name_features(raw, tmap, country=""):
    """Return dict with normalised name representations.

    v7 (config.NORM_V7): a leading honorific is dropped, a website / hashtag name is kept as a
    word when the other words are weak ("M/s alluredigitalindia.com"), and the words of the
    record's own country label ("(India)", "(France)") are dropped from the name."""
    s = translit_name(_DEGREE.sub(" no ", raw), tmap)
    s = anyascii(s).lower()
    if NORM_V7:
        s = _HONORIFIC.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " plus ").replace("'", "")
    domain = None
    m = TLD.search(s)
    if m:
        domain = m.group(1).replace("-", "")
        s = TLD.sub(" ", s)
    hashtag = None
    if "#" in s or "@" in s:
        s = _HASHNUM.sub(" ", s)
        mh = re.search(r"[#@]([a-z0-9]+)", s)
        if mh:
            hashtag = mh.group(1)
            s = s.replace(mh.group(0), " ")
    s = _LONGNUM.sub(" ", s)
    for pat, rep in LEGAL_PHRASES:
        s = re.sub(pat, rep, s)
    s = _BRACKET_NOISE.sub(" ", s)
    parts = DBA_SPLIT.split(s)
    main = parts[0]
    alts = parts[1:]
    toks = _name_tokens(s if not alts else " ".join(parts))
    main_toks = _name_tokens(main)
    alt_toks = [t for a in alts for t in _name_tokens(a)]
    concat = domain or hashtag
    if NORM_V7 and country:
        cw = set(anyascii(country).lower().split())
        if [t for t in toks if t not in cw]:
            toks = [t for t in toks if t not in cw]
            main_toks = [t for t in main_toks if t not in cw] or main_toks
            alt_toks = [t for t in alt_toks if t not in cw]
    weak = not [t for t in toks if len(t) > 2 and t not in _LEGAL_CANON and t not in NAME_STOP]
    if concat and (not toks or (NORM_V7 and weak and concat not in toks)):
        toks = [t for t in toks if len(t) > 2] + [concat]
        main_toks = [t for t in main_toks if len(t) > 2] + [concat]
    legal = sorted({t for t in toks if t in _LEGAL_CANON})
    core = [t for t in toks if t not in _LEGAL_CANON and t not in NAME_STOP]
    main_core = [t for t in main_toks if t not in _LEGAL_CANON and t not in NAME_STOP]
    alt_core = [t for t in alt_toks if t not in _LEGAL_CANON and t not in NAME_STOP]
    return {
        "name_full": " ".join(toks),
        "name_core": " ".join(core),
        "name_main": " ".join(main_core),
        "name_alt": " ".join(alt_core),
        "name_legal": " ".join(legal),
        "name_concat": concat or "",
        "name_native": is_non_latin(raw),
    }


_LEGAL_CANON = set(LEGAL.values())


def expand_concat(nf, seg):
    """Split a website / hashtag style name ("mishawangreliable") into words
    using the corpus unigram model. Only applied to such concatenated names."""
    concat = nf["name_concat"]
    if not concat:
        return nf
    core = nf["name_core"].split()
    out = []
    for t in core:
        if t == concat:
            parts = seg.split(t)
            out.extend(parts)
        else:
            out.append(t)
    nf = dict(nf)
    nf["name_core"] = " ".join(t for t in out if t not in _LEGAL_CANON and t not in NAME_STOP)
    if not nf["name_alt"]:
        nf["name_main"] = nf["name_core"]
    nf["name_full"] = " ".join(out + nf["name_legal"].split())
    return nf


# --------------------------------------------------------------------------
# address normalisation
# --------------------------------------------------------------------------
def addr_components(raw, tmap):
    """Normalised comma components of an address (transliterated, ASCII, lower case, punctuation
    replaced by spaces, fillers removed); empty components are dropped."""
    s = translit_addr(_DEGREE.sub(" no ", raw), tmap)
    s = anyascii(s).lower().replace("<null>", " ").replace("n/a", " ")
    return [c for c in (_NONALNUM.sub(" ", comp).strip() for comp in s.split(",")) if c]


def addr_features(raw, tmap):
    """Return dict with normalised address representations: tokens (abbreviations expanded,
    fillers removed), words only, house numbers, state (from names or codes), empty flag.

    tmap may carry maps learned for countries without training labels (address_align.py):
    "addr_alias_learned" (component -> canonical spelling) and "addr_state_learned"
    (component -> state code, for region-like components)."""
    learned_state = tmap.get("addr_state_learned", {})
    learned_alias = tmap.get("addr_alias_learned", {})
    states, toks, nums = [], [], []
    for c in addr_components(raw, tmap):
        c = learned_alias.get(c, c)
        st = _STATE_MAP.get(c)
        if st is not None:
            states.extend(st.split())
            continue
        if c in learned_state:
            states.append(learned_state[c])
            continue
        if c in STATE_CODES:
            states.append(c)
            continue
        words = c.split()
        if NORM_V7:
            split = []
            for t in words:
                ms, mp = (None, None) if ORDINAL.match(t) else (_NUM_SUFFIX.match(t), _NUM_PREFIX.match(t))
                split.extend(ms.groups() if ms else mp.groups() if mp else (t,))
            words = split
        for t in words:
            m = ORDINAL.match(t)
            if m:
                t = m.group(1)
            if t.isdigit():
                t = t.lstrip("0") or "0"
                if len(t) > 6:
                    continue
                toks.append(t)
                nums.append(t)
                continue
            t = ADDR_ABBR.get(t, t)
            for u in t.split():
                if u not in ADDR_STOP:
                    toks.append(u)
    words = [t for t in toks if not t.isdigit()]
    return {
        "addr_tokens": " ".join(toks),
        "addr_words": " ".join(words),
        "addr_nums": " ".join(dict.fromkeys(nums)),
        "addr_state": " ".join(sorted(set(states))),
        "addr_empty": len(toks) == 0,
    }
