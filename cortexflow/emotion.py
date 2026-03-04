"""
Emotion tracking for CortexFlow.

Detects and tracks emotional states in conversation messages, enabling
emotionally-aware response generation for companion AI applications.
"""
from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cortexflow")

# Plutchik's primary emotions
EMOTIONS = (
    "joy", "sadness", "anger", "fear",
    "surprise", "disgust", "trust", "anticipation",
)

# Valence / arousal mapping for each primary emotion
_EMOTION_VA: dict[str, tuple[float, float]] = {
    "joy":          ( 0.8,  0.6),
    "sadness":      (-0.7,  -0.3),
    "anger":        (-0.6,   0.8),
    "fear":         (-0.7,   0.7),
    "surprise":     ( 0.1,   0.8),
    "disgust":      (-0.6,  -0.1),
    "trust":        ( 0.6,   0.2),
    "anticipation": ( 0.4,   0.5),
    "neutral":      ( 0.0,   0.0),
}


@dataclass
class EmotionalState:
    """A snapshot of detected emotion at a point in time."""

    primary_emotion: str = "neutral"
    intensity: float = 0.0  # 0–1
    valence: float = 0.0    # -1 (negative) to +1 (positive)
    arousal: float = 0.0    # -1 (calm) to +1 (excited)
    secondary_emotions: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_emotion": self.primary_emotion,
            "intensity": round(self.intensity, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "secondary_emotions": {k: round(v, 3) for k, v in self.secondary_emotions.items()},
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionalState:
        return cls(
            primary_emotion=data.get("primary_emotion", "neutral"),
            intensity=data.get("intensity", 0.0),
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            secondary_emotions=data.get("secondary_emotions", {}),
            timestamp=data.get("timestamp", time.time()),
            confidence=data.get("confidence", 0.0),
        )


# ------------------------------------------------------------------
# Detector ABC (Strategy pattern)
# ------------------------------------------------------------------

class EmotionDetector(ABC):
    """Strategy interface for emotion detection."""

    @abstractmethod
    def detect(self, text: str) -> EmotionalState:
        """Analyse *text* and return the detected emotional state."""


# ------------------------------------------------------------------
# Rule-based detector (zero external deps)
# ------------------------------------------------------------------

# Lexicon: word → (emotion, weight)
_LEXICON: dict[str, tuple[str, float]] = {}

def _build_lexicon() -> dict[str, tuple[str, float]]:
    if _LEXICON:
        return _LEXICON

    raw: dict[str, list[tuple[str, float]]] = {
        "joy": [
            ("happy", 0.8), ("glad", 0.7), ("excited", 0.8), ("love", 0.9),
            ("wonderful", 0.8), ("great", 0.7), ("amazing", 0.8), ("fantastic", 0.8),
            ("delighted", 0.8), ("cheerful", 0.7), ("thrilled", 0.9),
            ("blessed", 0.7), ("grateful", 0.7), ("thankful", 0.7),
            ("awesome", 0.7), ("perfect", 0.7), ("beautiful", 0.6),
            ("laugh", 0.7), ("smile", 0.6), ("fun", 0.6), ("enjoy", 0.7),
            ("pleased", 0.6), ("content", 0.5), ("elated", 0.9),
        ],
        "sadness": [
            ("sad", 0.8), ("unhappy", 0.7), ("depressed", 0.9), ("lonely", 0.8),
            ("miserable", 0.9), ("heartbroken", 0.9), ("grief", 0.9),
            ("crying", 0.8), ("cry", 0.7), ("tears", 0.7), ("sorrow", 0.8),
            ("hopeless", 0.8), ("melancholy", 0.7), ("gloomy", 0.7),
            ("devastated", 0.9), ("hurt", 0.7), ("pain", 0.6),
            ("miss", 0.5), ("lost", 0.5), ("empty", 0.7), ("broken", 0.7),
        ],
        "anger": [
            ("angry", 0.8), ("furious", 0.9), ("mad", 0.7), ("hate", 0.9),
            ("annoyed", 0.6), ("irritated", 0.6), ("frustrated", 0.7),
            ("rage", 0.9), ("pissed", 0.8), ("outraged", 0.9),
            ("livid", 0.9), ("hostile", 0.7), ("resent", 0.7),
            ("bitter", 0.6), ("aggravated", 0.7), ("infuriated", 0.9),
        ],
        "fear": [
            ("afraid", 0.8), ("scared", 0.8), ("terrified", 0.9), ("anxious", 0.7),
            ("worried", 0.6), ("nervous", 0.6), ("panic", 0.9), ("dread", 0.8),
            ("frightened", 0.8), ("uneasy", 0.5), ("paranoid", 0.7),
            ("phobia", 0.8), ("horror", 0.8), ("alarmed", 0.7),
            ("insecure", 0.5), ("overwhelmed", 0.6), ("stressed", 0.6),
        ],
        "surprise": [
            ("surprised", 0.7), ("shocked", 0.8), ("astonished", 0.8),
            ("amazed", 0.7), ("stunned", 0.8), ("unexpected", 0.6),
            ("wow", 0.6), ("whoa", 0.6), ("omg", 0.7), ("unbelievable", 0.7),
        ],
        "disgust": [
            ("disgusted", 0.8), ("gross", 0.7), ("revolting", 0.8),
            ("sick", 0.5), ("nasty", 0.7), ("repulsive", 0.8),
            ("vile", 0.8), ("horrible", 0.6), ("awful", 0.6),
            ("yuck", 0.6), ("ew", 0.5), ("cringe", 0.5),
        ],
        "trust": [
            ("trust", 0.7), ("believe", 0.5), ("faith", 0.7), ("reliable", 0.6),
            ("honest", 0.6), ("loyal", 0.7), ("confident", 0.6),
            ("safe", 0.6), ("secure", 0.5), ("comfortable", 0.5),
            ("depend", 0.5), ("support", 0.5),
        ],
        "anticipation": [
            ("excited", 0.6), ("eager", 0.7), ("looking forward", 0.7),
            ("hope", 0.6), ("expect", 0.5), ("waiting", 0.4),
            ("curious", 0.6), ("wonder", 0.5), ("can't wait", 0.8),
            ("impatient", 0.5), ("ready", 0.4),
        ],
    }

    for emotion, words in raw.items():
        for word, weight in words:
            if word not in _LEXICON or weight > _LEXICON[word][1]:
                _LEXICON[word] = (emotion, weight)
    return _LEXICON


# Emoji → (emotion, weight)
_EMOJI_MAP: dict[str, tuple[str, float]] = {
    "😊": ("joy", 0.7), "😃": ("joy", 0.7), "😄": ("joy", 0.8),
    "😁": ("joy", 0.7), "🥰": ("joy", 0.8), "❤️": ("joy", 0.7),
    "💕": ("joy", 0.7), "😍": ("joy", 0.8), "🎉": ("joy", 0.7),
    "😢": ("sadness", 0.7), "😭": ("sadness", 0.9), "💔": ("sadness", 0.8),
    "😞": ("sadness", 0.6), "😔": ("sadness", 0.6),
    "😠": ("anger", 0.7), "😡": ("anger", 0.8), "🤬": ("anger", 0.9),
    "😱": ("fear", 0.8), "😰": ("fear", 0.7), "😨": ("fear", 0.7),
    "😲": ("surprise", 0.7), "😮": ("surprise", 0.6), "🤯": ("surprise", 0.8),
    "🤢": ("disgust", 0.7), "🤮": ("disgust", 0.8),
    "🤗": ("trust", 0.6), "🙏": ("trust", 0.5),
}


class RuleBasedEmotionDetector(EmotionDetector):
    """Lexicon + regex + emoji + punctuation heuristics. Zero dependencies."""

    _INTENSIFIERS = re.compile(
        r"\b(very|really|so|extremely|incredibly|absolutely|totally|utterly)\b",
        re.IGNORECASE,
    )
    _NEGATORS = re.compile(
        r"\b(not|no|never|don't|doesn't|didn't|won't|can't|isn't|aren't|wasn't|weren't|neither|nor)\b",
        re.IGNORECASE,
    )

    def detect(self, text: str) -> EmotionalState:
        lexicon = _build_lexicon()
        text_lower = text.lower()
        words = re.findall(r"\b[\w']+\b", text_lower)

        # 1. Lexicon scoring
        scores: dict[str, float] = {e: 0.0 for e in EMOTIONS}
        match_count = 0

        has_negation = bool(self._NEGATORS.search(text_lower))
        intensifier_count = len(self._INTENSIFIERS.findall(text_lower))
        intensifier_boost = 1.0 + 0.15 * intensifier_count

        for word in words:
            hit = lexicon.get(word)
            if hit is None:
                continue
            emotion, weight = hit
            adjusted = weight * intensifier_boost
            if has_negation:
                # Negation flips valence: positive → sadness, negative → neutral
                if emotion in ("joy", "trust", "anticipation"):
                    scores["sadness"] += adjusted * 0.5
                else:
                    adjusted *= 0.3  # dampen
                    scores[emotion] += adjusted
            else:
                scores[emotion] += adjusted
            match_count += 1

        # Multi-word phrases
        for phrase, (emotion, weight) in lexicon.items():
            if " " in phrase and phrase in text_lower:
                scores[emotion] += weight * intensifier_boost
                match_count += 1

        # 2. Emoji scoring
        for emoji_char, (emotion, weight) in _EMOJI_MAP.items():
            count = text.count(emoji_char)
            if count:
                scores[emotion] += weight * min(count, 3)
                match_count += count

        # 3. Punctuation heuristics
        excl = text.count("!")
        if excl >= 3:
            scores["surprise"] += 0.3
            match_count += 1
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.5 and len(text) > 5:
            # ALL-CAPS suggests strong emotion — boost top score
            top_emo = max(scores, key=scores.get)  # type: ignore[arg-type]
            if scores[top_emo] > 0:
                scores[top_emo] *= 1.3
            else:
                scores["anger"] += 0.3

        # 4. Aggregate
        if match_count == 0:
            return EmotionalState(primary_emotion="neutral", confidence=0.5)

        primary = max(scores, key=scores.get)  # type: ignore[arg-type]
        primary_score = scores[primary]

        if primary_score <= 0:
            return EmotionalState(primary_emotion="neutral", confidence=0.5)

        # Normalise intensity to 0-1
        intensity = min(primary_score / 3.0, 1.0)

        # Secondary emotions (anything > 30 % of primary)
        threshold = primary_score * 0.3
        secondaries = {
            e: min(s / 3.0, 1.0)
            for e, s in scores.items()
            if e != primary and s > threshold
        }

        va = _EMOTION_VA.get(primary, (0.0, 0.0))
        valence = va[0] * intensity
        arousal = va[1] * intensity

        confidence = min(0.5 + match_count * 0.1, 0.9)

        return EmotionalState(
            primary_emotion=primary,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            secondary_emotions=secondaries,
            confidence=confidence,
        )


class LLMEmotionDetector(EmotionDetector):
    """Uses the configured LLM for nuanced emotion detection.

    Falls back to ``RuleBasedEmotionDetector`` if the LLM call fails.
    Results are cached (keyed by text hash) to reduce API cost.
    """

    def __init__(self, llm_client, cache_size: int = 256):
        self._llm = llm_client
        self._fallback = RuleBasedEmotionDetector()
        self._cache: dict[int, EmotionalState] = {}
        self._cache_size = cache_size

    def detect(self, text: str) -> EmotionalState:
        import json as _json

        key = hash(text)
        if key in self._cache:
            return self._cache[key]

        prompt = (
            "Analyze the emotional content of the following message. "
            "Return ONLY a JSON object with these fields:\n"
            '- "primary_emotion": one of [joy, sadness, anger, fear, surprise, disgust, trust, anticipation, neutral]\n'
            '- "intensity": float 0-1\n'
            '- "valence": float -1 to 1 (negative to positive)\n'
            '- "arousal": float -1 to 1 (calm to excited)\n'
            '- "secondary_emotions": dict of emotion→intensity\n'
            '- "confidence": float 0-1\n\n'
            f'Message: "{text}"'
        )

        try:
            raw = self._llm.generate_from_prompt(prompt)
            # Try to parse JSON from the response
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            data = _json.loads(cleaned)
            state = EmotionalState.from_dict(data)
        except Exception as exc:
            logger.debug("LLM emotion detection failed (%s), using rule-based fallback", exc)
            state = self._fallback.detect(text)

        # LRU eviction
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = state
        return state


# ------------------------------------------------------------------
# Emotion Tracker (per-session state over time)
# ------------------------------------------------------------------

class EmotionTracker:
    """Tracks emotional state over time within a session.

    Maintains a rolling window of ``EmotionalState`` snapshots and
    exposes trend / shift detection helpers.
    """

    def __init__(self, detector: EmotionDetector, window_size: int = 20):
        self._detector = detector
        self._window_size = window_size
        self._history: deque[EmotionalState] = deque(maxlen=window_size)
        self._current: EmotionalState = EmotionalState()

    # -- Public API --

    def process_message(self, text: str) -> EmotionalState:
        """Detect emotion in *text*, update history, return the new state."""
        state = self._detector.detect(text)
        self._history.append(state)
        self._current = state
        return state

    def get_current_state(self) -> EmotionalState:
        return self._current

    def get_emotional_trend(self, last_n: int | None = None) -> dict[str, Any]:
        """Return a summary of the emotional trend over recent messages.

        Keys: ``avg_valence``, ``avg_arousal``, ``dominant_emotion``,
        ``valence_direction`` (rising / falling / stable).
        """
        history = list(self._history)
        if last_n is not None:
            history = history[-last_n:]
        if not history:
            return {
                "avg_valence": 0.0,
                "avg_arousal": 0.0,
                "dominant_emotion": "neutral",
                "valence_direction": "stable",
                "sample_count": 0,
            }

        avg_val = sum(s.valence for s in history) / len(history)
        avg_aro = sum(s.arousal for s in history) / len(history)

        # Dominant emotion by frequency
        counts: dict[str, int] = {}
        for s in history:
            counts[s.primary_emotion] = counts.get(s.primary_emotion, 0) + 1
        dominant = max(counts, key=counts.get)  # type: ignore[arg-type]

        # Valence direction (compare first half vs second half)
        if len(history) >= 4:
            mid = len(history) // 2
            first_half = sum(s.valence for s in history[:mid]) / mid
            second_half = sum(s.valence for s in history[mid:]) / (len(history) - mid)
            diff = second_half - first_half
            if diff > 0.15:
                direction = "rising"
            elif diff < -0.15:
                direction = "falling"
            else:
                direction = "stable"
        else:
            direction = "stable"

        return {
            "avg_valence": round(avg_val, 3),
            "avg_arousal": round(avg_aro, 3),
            "dominant_emotion": dominant,
            "valence_direction": direction,
            "sample_count": len(history),
        }

    def detect_emotional_shift(self) -> dict[str, Any] | None:
        """If the most recent emotion differs significantly from the prior
        trend, return a dict describing the shift. Otherwise ``None``."""
        if len(self._history) < 3:
            return None

        recent = list(self._history)
        current = recent[-1]
        prior = recent[:-1]

        avg_val = sum(s.valence for s in prior) / len(prior)
        avg_aro = sum(s.arousal for s in prior) / len(prior)

        val_delta = current.valence - avg_val
        aro_delta = current.arousal - avg_aro

        if abs(val_delta) > 0.3 or abs(aro_delta) > 0.3:
            return {
                "detected": True,
                "from_emotion": prior[-1].primary_emotion,
                "to_emotion": current.primary_emotion,
                "valence_delta": round(val_delta, 3),
                "arousal_delta": round(aro_delta, 3),
                "description": self._describe_shift(val_delta, aro_delta),
            }
        return None

    def get_history(self) -> list[EmotionalState]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self._current = EmotionalState()

    # -- Internal --

    @staticmethod
    def _describe_shift(val_delta: float, aro_delta: float) -> str:
        parts = []
        if val_delta > 0.3:
            parts.append("becoming more positive")
        elif val_delta < -0.3:
            parts.append("becoming more negative")
        if aro_delta > 0.3:
            parts.append("becoming more agitated")
        elif aro_delta < -0.3:
            parts.append("becoming calmer")
        return "; ".join(parts) if parts else "subtle shift"
