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
    last_activity_at: float
    query_count: int
    context_buffer: str


@dataclass
class CompletionRequest:
    session_id: str
    prefix: str
    line_context: str
    language: str
    max_completions: int


# -----------------------------------------------------------------------------
# In-memory state
# -----------------------------------------------------------------------------
class AriVaState:
    def __init__(self) -> None:
        self.sessions: Dict[str, AssistantSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}
        self.suggestion_log: List[Dict[str, Any]] = []
        self.request_count: int = 0
        self.total_suggestions_served: int = 0


# -----------------------------------------------------------------------------
# Validation rules (code assistant style)
# -----------------------------------------------------------------------------
def _rule_no_trailing_whitespace(code: str) -> List[ValidationResult]:
    results = []
    for i, line in enumerate(code.split("\n"), start=1):
        if line != line.rstrip() and line.strip():
            results.append(
                ValidationResult(
                    passed=False,
                    rule_id="ARIVA_NO_TRAILING_WS",
                    message="Trailing whitespace",
                    line=i,
                    column=len(line) - len(line.rstrip()) + 1,
                )
            )
    return results


def _rule_max_line_length(code: str, max_len: int = 120) -> List[ValidationResult]:
    results = []
    for i, line in enumerate(code.split("\n"), start=1):
        if len(line) > max_len:
            results.append(
                ValidationResult(
                    passed=False,
                    rule_id="ARIVA_MAX_LINE_LEN",
                    message=f"Line exceeds {max_len} characters",
                    line=i,
                    column=max_len + 1,
                )
            )
    return results


def _rule_balanced_braces(code: str) -> List[ValidationResult]:
    stack = []
    open_b = {"(": ")", "[": "]", "{": "}"}
    close_b = set(open_b.values())
    for i, c in enumerate(code):
        if c in open_b:
            stack.append((open_b[c], i))
        elif c in close_b:
            if not stack or stack[-1][0] != c:
                line = code[:i].count("\n") + 1
                col = i - code.rfind("\n", 0, i) if "\n" in code[:i] else i + 1
                results = [
                    ValidationResult(
                        passed=False,
                        rule_id="ARIVA_BALANCED_BRACES",
                        message="Unbalanced bracket",
                        line=line,
                        column=col,
                    )
                ]
                return results
            stack.pop()
    if stack:
        _, pos = stack[-1]
        line = code[:pos].count("\n") + 1
        return [
            ValidationResult(
                passed=False,
                rule_id="ARIVA_BALANCED_BRACES",
                message="Unclosed bracket",
                line=line,
                column=0,
            )
        ]
    return []


def _run_all_validation_rules(code: str) -> List[ValidationResult]:
    out = []
    out.extend(_rule_no_trailing_whitespace(code))
    out.extend(_rule_max_line_length(code))
    out.extend(_rule_balanced_braces(code))
    return out


# -----------------------------------------------------------------------------
# Core engine
# -----------------------------------------------------------------------------
class AriVaEngine:
    """Code assistant engine: sessions, suggestions, completions, validation."""

    def __init__(self) -> None:
        self.state = AriVaState()
        self._coordinator = ARIVA_COORDINATOR
        self._vault = ARIVA_VAULT
        self._relay = ARIVA_RELAY
        self._oracle = ARIVA_ORACLE
        self._sentinel = ARIVA_SENTINEL

    def _require_coordinator(self, caller: str) -> None:
        if caller != self._coordinator:
            raise AriVaNotCoordinator()

    def _new_session_id(self) -> str:
        raw = f"{ARIVA_SESSION_SALT}{time.time()}{uuid.uuid4().hex}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def create_session(self, user_ref: str, caller: str) -> str:
        self._require_coordinator(caller)
        if not user_ref:
            raise AriVaZeroDisallowed()
        if user_ref not in self.state.user_sessions:
            self.state.user_sessions[user_ref] = []
        if len(self.state.user_sessions[user_ref]) >= MAX_SESSIONS_PER_USER:
            raise AriVaSessionLimitReached()
        session_id = self._new_session_id()
        now = time.time()
        self.state.sessions[session_id] = AssistantSession(
            session_id=session_id,
            user_ref=user_ref,
            status=int(SessionStatus.ACTIVE),
            created_at=now,
            last_activity_at=now,
            query_count=0,
            context_buffer="",
        )
        self.state.user_sessions[user_ref].append(session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id not in self.state.sessions:
            return None
        s = self.state.sessions[session_id]
        return {
            "session_id": s.session_id,
            "user_ref": s.user_ref,
            "status": s.status,
            "created_at": s.created_at,
            "last_activity_at": s.last_activity_at,
            "query_count": s.query_count,
        }

    def close_session(self, session_id: str, caller: str) -> None:
        self._require_coordinator(caller)
        if session_id not in self.state.sessions:
            raise AriVaSessionNotFound()
        s = self.state.sessions[session_id]
        s.status = int(SessionStatus.CLOSED)
        if s.user_ref in self.state.user_sessions:
            self.state.user_sessions[s.user_ref] = [
                x for x in self.state.user_sessions[s.user_ref] if x != session_id
            ]

    def update_context(self, session_id: str, context: str, caller: str) -> None:
        self._require_coordinator(caller)
        if session_id not in self.state.sessions:
            raise AriVaSessionNotFound()
        if len(context) > CODE_CONTEXT_WINDOW:
            raise AriVaContextOverflow()
        s = self.state.sessions[session_id]
        s.context_buffer = context
        s.last_activity_at = time.time()

    def validate_code(self, code: str) -> List[Dict[str, Any]]:
        results = _run_all_validation_rules(code)
        return [
            {
                "passed": r.passed,
                "rule_id": r.rule_id,
                "message": r.message,
                "line": r.line,
                "column": r.column,
            }
            for r in results
        ]

    def get_completions(
        self,
        session_id: str,
        prefix: str,
        line_context: str,
        language: str,
        max_n: int = MAX_COMPLETIONS_PER_LINE,
    ) -> List[Dict[str, Any]]:
        if session_id not in self.state.sessions:
            raise AriVaSessionNotFound()
        if len(prefix) > MAX_QUERY_LEN or len(line_context) > MAX_QUERY_LEN:
            raise AriVaQueryTooLong()
        if max_n > MAX_COMPLETIONS_PER_LINE:
            max_n = MAX_COMPLETIONS_PER_LINE
        s = self.state.sessions[session_id]
        s.last_activity_at = time.time()
        s.query_count += 1
        self.state.request_count += 1
        completions = _fake_completions(prefix, line_context, language, max_n)
        self.state.total_suggestions_served += len(completions)
        return completions

    def get_suggestions(
        self,
        session_id: str,
        query: str,
        kind: int,
        max_n: int = MAX_SUGGESTIONS_PER_REQUEST,
    ) -> List[Dict[str, Any]]:
        if session_id not in self.state.sessions:
            raise AriVaSessionNotFound()
        if len(query) < MIN_QUERY_LEN:
            raise AriVaQueryTooShort()
        if len(query) > MAX_QUERY_LEN:
            raise AriVaQueryTooLong()
        if max_n > MAX_SUGGESTIONS_PER_REQUEST:
            raise AriVaSuggestionLimitReached()
        s = self.state.sessions[session_id]
        s.last_activity_at = time.time()
        s.query_count += 1
        self.state.request_count += 1
        suggestions = _fake_suggestions(query, kind, max_n)
        self.state.total_suggestions_served += len(suggestions)
