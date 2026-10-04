"""Input guard: detect personal data before the message goes anywhere.

Runs first, on the raw user message. A message with PII is refused without
being sent to the LLM, stored in chat history, or logged. The brief: "Do not
accept/store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers."

Design goal: catch real identifiers without flagging ordinary finance text such
as "₹1,000", "2024", "80C", "1.25 lakh", "0.65%" or a NAV. (The old notebook's
`\\b\\d{4,6}\\b` "OTP" pattern blocked all of those.) Bare digit runs are
therefore flagged only when long enough to be an identifier, or when a nearby
word says what they are ("account", "folio", "OTP"...).
"""

import re
from dataclasses import dataclass

# Card numbers are validated with the Luhn checksum, so random digit runs don't count
def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


_PATTERNS = {
    # ABCDE1234F
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I),
    # 12 digits, first 2-9, optionally grouped 4-4-4
    "Aadhaar": re.compile(r"(?<![\d₹.,])[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?![\d.,%])"),
    # Indian mobile: optional +91/0, then 10 digits starting 6-9 (optionally 5+5)
    "phone number": re.compile(r"(?<![\d₹.,])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?![\d.,%])"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # UPI handle: name@bank (no dot after @, unlike email)
    "UPI ID": re.compile(r"\b[\w.-]{2,}@(?:ok)?[a-z]{2,}\b(?!\.)", re.I),
    # A secret value right after OTP/PIN/password/CVV; the value must contain a digit,
    # so "I haven't received the OTP" or "the password is my PAN" don't count
    "OTP / PIN / password": re.compile(
        r"\b(?:otp|m?pin|cvv|password|passcode)\b\s*(?:is|was|:|=|-)?\s*[A-Za-z0-9@#$!]*\d[A-Za-z0-9@#$!]*", re.I),
    # Account / folio / client numbers: digits after a word naming them
    "account or folio number": re.compile(
        r"\b(?:a/?c|acc(?:ount)?|acct|folio|client\s*id|customer\s*id|demat)\b(?:\s*(?:no\.?|number|#|:))?\s*[:#-]?\s*\d[\d/\s-]{5,}\d", re.I),
    # Very long bare digit runs (11-18 digits) are identifiers, not amounts
    "long number (account/ID)": re.compile(r"(?<![\d₹.,])\d{11,18}(?![\d.,%])"),
}

_CARD = re.compile(r"(?<!\d)(?:\d[\s-]?){13,19}(?!\d)")


@dataclass
class PIIResult:
    found: bool
    types: list[str]  # which kinds were found; the values themselves are never kept


def detect_pii(text: str) -> PIIResult:
    types = [name for name, pat in _PATTERNS.items() if pat.search(text)]
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            types.append("card number")
            break
    return PIIResult(bool(types), types)
