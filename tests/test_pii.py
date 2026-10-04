import pytest

from mf_assistant.guardrails.pii import detect_pii

# Messages that contain personal data and must be refused
PII_MESSAGES = [
    ("My PAN is ABCDE1234F, show my holdings", "PAN"),
    ("my pan abcde1234f", "PAN"),
    ("Aadhaar 2345 6789 0123 linked?", "Aadhaar"),
    ("aadhaar number 234567890123", "Aadhaar"),
    ("call me on 9876543210", "phone number"),
    ("my number is +91 98765 43210", "phone number"),
    ("mail me at ravi.k@gmail.com", "email"),
    ("pay to ravi@okhdfcbank", "UPI ID"),
    ("my OTP is 482913", "OTP / PIN / password"),
    ("otp 4829", "OTP / PIN / password"),
    ("password: Abc@1234", "OTP / PIN / password"),
    ("my account number 50100234567", "account or folio number"),
    ("folio no 1234567/89 balance?", "account or folio number"),
    ("check a/c 123456789", "account or folio number"),
    ("card 4111 1111 1111 1111 declined", "card number"),
    ("my client id is 123456789012345", "long number (account/ID)"),
]

# Ordinary finance questions that must NOT be flagged
CLEAN_MESSAGES = [
    "What is the minimum SIP of ₹1,000 for ELSS?",
    "Can I invest 5000 per month?",
    "returns in 2024 and 2025",
    "Is there a 3 year lock-in?",
    "Tax on gains above 1.25 lakh?",
    "Does ELSS qualify under Section 80C?",
    "NAV is 1576.03, why?",
    "expense ratio 0.65% vs 0.69%",
    "I haven't received the OTP even after trying multiple times",
    "What password can I use to open my CAS?",
    "Is the CAS password my PAN in capital letters?",
    "What is a folio number?",
    "SIP on the 5th of every month",
    "Groww has 10 Cr+ users",
    "AUM is ₹1,47,404.50 Cr",
    "Is there a Rs 100000 limit on autopay?",
    "What is the exit load within 365 days?",
]


@pytest.mark.parametrize("message,expected_type", PII_MESSAGES)
def test_detects_pii(message, expected_type):
    result = detect_pii(message)
    assert result.found, f"missed PII in: {message!r}"
    assert expected_type in result.types, f"{message!r}: got {result.types}"


@pytest.mark.parametrize("message", CLEAN_MESSAGES)
def test_no_false_positive(message):
    result = detect_pii(message)
    assert not result.found, f"false positive {result.types} in: {message!r}"
