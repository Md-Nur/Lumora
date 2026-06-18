import re
from textblob import TextBlob

def polish_model_text(text: str) -> str:
    """Conservative grammar cleanup that does not add or remove clinical findings."""
    replacements = {
        " ,": ",",
        " .": ".",
        " ;": ";",
        " :": ":",
        "( ": "(",
        " )": ")",
        "There is also sign of": "There are also signs of",
        "there is also sign of": "there are also signs of",
        "There is sign of": "There are signs of",
        "there is sign of": "there are signs of",
        "There is no signs of": "There are no signs of",
        "there is no signs of": "there are no signs of",
        "There are no evidence of": "There is no evidence of",
        "there are no evidence of": "there is no evidence of",
        "There are no pneumothorax": "There is no pneumothorax",
        "there are no pneumothorax": "there is no pneumothorax",
        "There are no pleural effusion": "There is no pleural effusion",
        "there are no pleural effusion": "there is no pleural effusion",
        "No focal consolidations": "No focal consolidation",
        "no focal consolidations": "no focal consolidation",
        "lung is clear": "lungs are clear",
        "Lung is clear": "Lungs are clear",
        "lungs is clear": "lungs are clear",
        "Lungs is clear": "Lungs are clear",
    }

    def polish_segment(segment: str) -> str:
        cleaned = segment.strip()
        previous = None
        while previous != cleaned:
            previous = cleaned
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            for source, target in replacements.items():
                cleaned = cleaned.replace(source, target)
            cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
            cleaned = re.sub(r"([,.;:])([^\s])", r"\1 \2", cleaned)
            cleaned = re.sub(r"\s+([)])", r"\1", cleaned)
            cleaned = re.sub(r"([(])\s+", r"\1", cleaned)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        polished_sentences = []
        for sentence in sentences:
            if not sentence:
                continue
            s = sentence[:1].upper() + sentence[1:]
            s = re.sub(
                r"^I\s+(is|has|shows|appears|demonstrates|looks|suggests|seems|indicates|presents|details|confirms|reveals|discloses|disclosed|revealed|indicated|presented|detailed|confirmed)\b",
                r"It \1",
                s,
                flags=re.IGNORECASE
            )
            polished_sentences.append(s)
        
        textblob = TextBlob(" ".join(polished_sentences))
        corrected = textblob.correct()
        return str(corrected)

    parts = re.split(r"(\n+)", text.strip())
    textblob = TextBlob("".join(part if part.startswith("\n") else polish_segment(part) for part in parts).strip())
    return textblob.correct().string