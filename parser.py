import re

COLOR_WORDS = ['red','orange','yellow','green','blue','purple','pink','brown','black','white','gray','grey']
NUM_WORDS = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12}
UNCERTAINTY = ['cannot confirm', 'not sure', "can't tell", 'unable to determine', 'difficult to determine', 'hard to tell', 'unclear']

MEDICAL_ATTR_WORDS = [
    'almost the same',
    'irregular', 'oval', 'round', 'elongated',
    'hypodense', 'hyperdense', 'isodense',
    'lung', 'liver', 'kidney', 'spleen', 'heart',
]


def try_forced_choice(question, response):
    m = re.search(r'(\w[\w\s]*?)\s+or\s+(\w[\w\s]*?)\??$', question.strip())
    if not m:
        return None
    opt_a, opt_b = m.group(1).strip().lower(), m.group(2).strip().lower()
    r = response.lower()
    a_in, b_in = opt_a in r, opt_b in r
    if a_in and not b_in:
        return opt_a
    if b_in and not a_in:
        return opt_b
    return None


def parse_response(text, category, question=None):
    t = text.lower().strip()

    if any(u in t for u in UNCERTAINTY):
        return 'unparseable'

    if question and category in ('spatial', 'attribute'):
        forced = try_forced_choice(question, text)
        if forced:
            return forced

    if category in ('existence', 'spatial'):
        neg_words = ['no', 'not', "don't", "doesn't", "isn't", "aren't", 'cannot', "can't", 'none', 'absent']
        pos_words = ['yes']
        if any(re.search(r'\b' + w + r'\b', t) for w in pos_words):
            return 'yes'
        if any(re.search(r'\b' + w + r'\b', t) for w in neg_words):
            return 'no'
        return 'unparseable'

    if category == 'count':
        neg_words = ['no ', 'none', 'zero', 'not visible', 'not present']
        if any(w in t for w in neg_words) and not re.search(r'\b\d+\b', t):
            return '0'
        m = re.search(r'\b(\d+)\b', t)
        if m:
            return m.group(1)
        for word, num in NUM_WORDS.items():
            if re.search(r'\b' + word + r'\b', t):
                return str(num)
        return 'unparseable'

    if category == 'ocr_numeric':
        # Exclude any 4-digit year that appears in the question itself --
        # e.g. "What is the value of 2016 for X?" -> "2016" in the response
        # is the question echoing back, not the answer. Prefer numbers
        # with %, $, or that come right after "is"/"was"/":" in the response.
        years_in_question = set(re.findall(r'\b(?:19|20)\d{2}\b', question or ''))

        # try percent/currency first -- most specific, least ambiguous
        m = re.search(r'[\d,]+\.?\d*\s*%', t)
        if m:
            return m.group(0).strip().replace(' ', '')
        m = re.search(r'\$\s*[\d,]+\.?\d*', t)
        if m:
            return m.group(0).strip().replace(' ', '')

        # fall back to any number, but skip ones matching a year already
        # present in the question
        candidates = re.findall(r'[\d,]+\.?\d*', t)
        for c in candidates:
            cleaned = c.replace(',', '')
            if cleaned in years_in_question:
                continue
            if cleaned:
                return c
        return 'unparseable'

    if category == 'attribute':
        for c in COLOR_WORDS:
            if re.search(r'\b' + c + r'\b', t):
                return 'gray' if c == 'grey' else c
        for w in MEDICAL_ATTR_WORDS:
            if w in t:
                return w
        return 'unparseable'

    if category == 'confabulation':
        return text

    return 'unparseable'
