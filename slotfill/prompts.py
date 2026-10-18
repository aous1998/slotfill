"""One prompt shared by training, the prompted baselines and evaluation.

Using the identical instruction everywhere means the comparison measures what
fine-tuning adds, not differences in prompt wording.
"""

SYSTEM_PROMPT = """You extract fields from the OCR text of a scanned receipt or invoice.

Return ONLY a JSON object with exactly these keys, no markdown, no commentary:
{"company": string, "date": string, "address": string or null, "total": string}

Rules:
- company: the seller's business name as printed (usually near the top, often ending in SDN BHD, BHD, ENTERPRISE, etc.). Not the cashier or customer.
- date: the transaction date in ISO format YYYY-MM-DD. Printed dates are day-first (DD/MM/YYYY).
- address: the seller's full address joined into one line with single spaces, or null if none is printed.
- total: the final amount paid (grand total, rounded total, including tax), digits with exactly two decimals and no currency symbol, e.g. "60.30".
- Copy text exactly as printed; fix only whitespace."""


def user_message(ocr_text: str) -> str:
    return f"OCR text:\n{ocr_text}"


def build_messages(ocr_text: str, answer: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(ocr_text)},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs
