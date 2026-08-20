"""
Government Document Text Extraction Validator for Practical 9.

Includes regex patterns for Indian Government IDs (Aadhaar, PAN, Passport, Dates, Phone),
structured key-value entity parsing, dictionary verification, and WER/CER metric calculation.
"""

import re
from typing import Dict, Any, List, Tuple, Optional


class DocumentValidator:
    """
    Validation engine for government documents and extracted OCR text.
    """

    PATTERNS = {
        "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
        "passport": r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b",
        "date": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b",
        "pincode": r"\b[1-9][0-9]{5}\b"
    }

    KEY_ALIASES = {
        "name": ["name", "full name", "applicant name", "holder name"],
        "dob": ["dob", "date of birth", "birth date"],
        "gender": ["gender", "sex"],
        "aadhaar_no": ["aadhaar", "aadhaar no", "uid", "aadhaar number"],
        "pan_no": ["pan", "pan no", "permanent account number"],
        "issue_date": ["issue date", "date of issue", "dated"],
        "authority": ["authority", "issuing authority", "govt of india", "government of india"],
        "address": ["address", "residence"]
    }

    def __init__(self):
        pass

    def validate_patterns(self, text: str) -> Dict[str, List[str]]:
        """Run regex pattern extractors against text."""
        results = {}
        for key, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            results[key] = matches
        return results

    def extract_key_values(self, text: str) -> Dict[str, str]:
        """Parse structured key-value pairs from document text (e.g. 'Name: Rahul Sharma')."""
        extracted = {}
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        for line in lines:
            if ":" in line or "-" in line:
                parts = re.split(r"[:\-]", line, maxsplit=1)
                if len(parts) == 2:
                    raw_key = parts[0].strip().lower()
                    val = parts[1].strip()

                    # Match with known aliases
                    matched_field = None
                    for std_field, aliases in self.KEY_ALIASES.items():
                        if any(alias in raw_key for alias in aliases):
                            matched_field = std_field
                            break

                    field_name = matched_field if matched_field else raw_key
                    extracted[field_name] = val
        return extracted

    def calculate_levenshtein(self, seq1: str, seq2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    def compute_cer(self, reference: str, hypothesis: str) -> float:
        """Compute Character Error Rate (CER)."""
        ref = reference.strip()
        hyp = hypothesis.strip()
        if not ref:
            return 0.0 if not hyp else 1.0
        distance = self.calculate_levenshtein(ref, hyp)
        return min(1.0, float(distance / len(ref)))

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate (WER)."""
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        # Calculate Levenshtein distance on word level
        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return min(1.0, float(dp[m][n] / len(ref_words)))

    def validate_document(self, ocr_text: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete validation summary of OCR extracted text.
        """
        pattern_matches = self.validate_patterns(ocr_text)
        key_values = self.extract_key_values(ocr_text)

        # Check overall validity of government entities
        validations = {
            "has_valid_aadhaar": len(pattern_matches["aadhaar"]) > 0,
            "has_valid_pan": len(pattern_matches["pan"]) > 0,
            "has_valid_date": len(pattern_matches["date"]) > 0,
            "has_valid_email": len(pattern_matches["email"]) > 0,
            "has_valid_phone": len(pattern_matches["phone"]) > 0
        }

        # Calculate evaluation metrics if ground truth is supplied
        cer = None
        wer = None
        if ground_truth:
            cer = round(self.compute_cer(ground_truth, ocr_text), 4)
            wer = round(self.compute_wer(ground_truth, ocr_text), 4)

        return {
            "entities": pattern_matches,
            "key_values": key_values,
            "validations": validations,
            "cer": cer,
            "wer": wer,
            "total_lines": len(ocr_text.splitlines())
        }
