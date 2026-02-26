# AriVa — Code assistant engine. Session-backed suggestions, completions, validation.
# All config pre-set; addresses and hex are unique to this module (not reused elsewhere).

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Unique config (not reused from Robotank, BacklineLedger, Spella, FrostVow, etc.)
# -----------------------------------------------------------------------------
ARIVA_COORDINATOR = "0x9f8E7d6C5b4A39281706f5e4d3c2B1a098765432"
ARIVA_VAULT = "0x8e7D6c5B4a39281706F5e4d3C2b1A0987654321"
ARIVA_RELAY = "0x7d6C5b4A39281706f5E4d3c2B1a09876543210"
ARIVA_ORACLE = "0x6c5B4a39281706f5e4D3c2b1A09876543210fe"
ARIVA_SENTINEL = "0x5b4A39281706f5e4d3C2b1a09876543210fedc"

ARIVA_DOMAIN_SALT = "0xe9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0"
ARIVA_SESSION_SALT = "0xd8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9"
ARIVA_COMPLETION_SEED = "0xc7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8"

MAX_SUGGESTIONS_PER_REQUEST = 24
MAX_COMPLETIONS_PER_LINE = 12
MAX_SESSION_DURATION_SEC = 7200
MAX_SESSIONS_PER_USER = 16
MIN_QUERY_LEN = 1
MAX_QUERY_LEN = 8192
SUGGESTION_CACHE_TTL = 300
VALIDATION_RULESET_VERSION = 3
CODE_CONTEXT_WINDOW = 2048
ASSISTANT_RESPONSE_MAX_LEN = 4096


class SuggestionKind(IntEnum):
    COMPLETION = 0
    FIX = 1
    HINT = 2
    REFACTOR = 3


class SessionStatus(IntEnum):
    ACTIVE = 0
    IDLE = 1
    CLOSED = 2


# -----------------------------------------------------------------------------
# AriVa-specific exceptions (not FV_*, SPEL_*, Tank*, Ledger_*, etc.)
# -----------------------------------------------------------------------------
class AriVaNotCoordinator(Exception):
    """Caller is not the coordinator address."""


class AriVaSessionNotFound(Exception):
    """Session id does not exist."""


class AriVaSessionLimitReached(Exception):
    """Max sessions per user reached."""


class AriVaQueryTooShort(Exception):
    """Query length below minimum."""


class AriVaQueryTooLong(Exception):
    """Query length above maximum."""


class AriVaSuggestionLimitReached(Exception):
    """Too many suggestions in one request."""


class AriVaZeroDisallowed(Exception):
    """Zero address or empty required value."""


class AriVaValidationFailed(Exception):
    """Code validation rule failed."""


class AriVaContextOverflow(Exception):
    """Code context exceeds window size."""


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
@dataclass
class SuggestionRecord:
    kind: int
    text: str
    offset_start: int
    offset_end: int
    confidence: float
    rule_id: Optional[str]
    created_at: float


@dataclass
class ValidationResult:
    passed: bool
    rule_id: str
    message: str
    line: Optional[int]
    column: Optional[int]


@dataclass
class AssistantSession:
    session_id: str
    user_ref: str
    status: int
    created_at: float
