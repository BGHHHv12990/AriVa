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
        return suggestions

    def config_snapshot(self) -> Dict[str, Any]:
        return {
            "coordinator": self._coordinator,
            "vault": self._vault,
            "relay": self._relay,
            "oracle": self._oracle,
            "sentinel": self._sentinel,
            "domain_salt": ARIVA_DOMAIN_SALT,
            "session_salt": ARIVA_SESSION_SALT,
            "completion_seed": ARIVA_COMPLETION_SEED,
        }


def _fake_completions(
    prefix: str, line_context: str, language: str, max_n: int
) -> List[Dict[str, Any]]:
    out = []
    for i in range(min(max_n, 5)):
        token = f"token_{language}_{hashlib.sha256((prefix + str(i)).encode()).hexdigest()[:8]}"
        out.append({
            "text": token,
            "kind": int(SuggestionKind.COMPLETION),
            "confidence": 0.7 + i * 0.05,
        })
    return out


def _fake_suggestions(query: str, kind: int, max_n: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(min(max_n, 4)):
        out.append({
            "suggestion_id": hashlib.sha256((query + str(i) + ARIVA_COMPLETION_SEED).encode()).hexdigest()[:16],
            "kind": kind,
            "text": f"suggestion_{i}",
            "confidence": 0.8 - i * 0.1,
        })
    return out


# -----------------------------------------------------------------------------
# Unified API (single entry)
# -----------------------------------------------------------------------------
class AriVaPlatform:
    def __init__(self) -> None:
        self._engine = AriVaEngine()

    @property
    def engine(self) -> AriVaEngine:
        return self._engine

    def api_create_session(self, user_ref: str, caller: str) -> Dict[str, Any]:
        sid = self._engine.create_session(user_ref, caller)
        return {"session_id": sid, "user_ref": user_ref}

    def api_get_session(self, session_id: str) -> Dict[str, Any]:
        out = self._engine.get_session(session_id)
        if out is None:
            return {"error": "AriVaSessionNotFound"}
        return out

    def api_close_session(self, session_id: str, caller: str) -> Dict[str, Any]:
        self._engine.close_session(session_id, caller)
        return {"session_id": session_id}

    def api_update_context(self, session_id: str, context: str, caller: str) -> Dict[str, Any]:
        self._engine.update_context(session_id, context, caller)
        return {"session_id": session_id}

    def api_validate_code(self, code: str) -> Dict[str, Any]:
        results = self._engine.validate_code(code)
        return {"results": results, "passed": all(r["passed"] for r in results) is False or len(results) == 0}

    def api_get_completions(
        self,
        session_id: str,
        prefix: str,
        line_context: str,
        language: str,
        max_n: int = MAX_COMPLETIONS_PER_LINE,
    ) -> Dict[str, Any]:
        comps = self._engine.get_completions(session_id, prefix, line_context, language, max_n)
        return {"completions": comps}

    def api_get_suggestions(
        self,
        session_id: str,
        query: str,
        kind: int = 0,
        max_n: int = MAX_SUGGESTIONS_PER_REQUEST,
    ) -> Dict[str, Any]:
        sugs = self._engine.get_suggestions(session_id, query, kind, max_n)
        return {"suggestions": sugs}

    def api_config(self) -> Dict[str, Any]:
        return self._engine.config_snapshot()

    def api_stats(self) -> Dict[str, Any]:
        return {
            "request_count": self._engine.state.request_count,
            "total_suggestions_served": self._engine.state.total_suggestions_served,
            "active_sessions": sum(1 for s in self._engine.state.sessions.values() if s.status == int(SessionStatus.ACTIVE)),
        }


def create_ariva() -> AriVaPlatform:
    return AriVaPlatform()


# -----------------------------------------------------------------------------
# Address/hex uniqueness confirmation
# -----------------------------------------------------------------------------
def _is_eth_address(addr: str) -> bool:
    if not addr or len(addr) != 42 or not addr.startswith("0x"):
        return False
    try:
        int(addr[2:], 16)
        return True
    except ValueError:
        return False


def confirm_ariva_addresses_unique() -> bool:
    addrs = [ARIVA_COORDINATOR, ARIVA_VAULT, ARIVA_RELAY, ARIVA_ORACLE, ARIVA_SENTINEL]
    return len(addrs) == len(set(addrs)) and all(_is_eth_address(a) for a in addrs)


def confirm_ariva_hex_unique() -> bool:
    salts = [ARIVA_DOMAIN_SALT, ARIVA_SESSION_SALT, ARIVA_COMPLETION_SEED]
    return len(salts) == len(set(salts))


# -----------------------------------------------------------------------------
# Event log (code assistant events)
# -----------------------------------------------------------------------------
@dataclass
class AriVaEvent:
    event_type: str
    payload: Dict[str, Any]
    timestamp: float
    event_id: str


class AriVaEventLog:
    def __init__(self, max_events: int = 5000) -> None:
        self._events: List[AriVaEvent] = []
        self._max = max_events

    def emit(self, event_type: str, payload: Dict[str, Any]) -> str:
        eid = str(uuid.uuid4())
        self._events.append(
            AriVaEvent(
                event_type=event_type,
                payload={**payload, "event_id": eid},
                timestamp=time.time(),
                event_id=eid,
            )
        )
        while len(self._events) > self._max:
            self._events.pop(0)
        return eid

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        out = self._events[-limit:]
        return [
            {"event_type": e.event_type, "payload": e.payload, "timestamp": e.timestamp}
            for e in reversed(out)
        ]


# -----------------------------------------------------------------------------
# Session cleanup (stale sessions)
# -----------------------------------------------------------------------------
def cleanup_stale_sessions(engine: AriVaEngine) -> int:
    now = time.time()
    to_close = []
    for sid, s in engine.state.sessions.items():
        if s.status != int(SessionStatus.ACTIVE):
            continue
        if now - s.last_activity_at > MAX_SESSION_DURATION_SEC:
            to_close.append(sid)
    for sid in to_close:
        s = engine.state.sessions[sid]
        s.status = int(SessionStatus.CLOSED)
        if s.user_ref in engine.state.user_sessions:
            engine.state.user_sessions[s.user_ref] = [
                x for x in engine.state.user_sessions[s.user_ref] if x != sid
            ]
    return len(to_close)


# -----------------------------------------------------------------------------
# Additional validation rules
# -----------------------------------------------------------------------------
def _rule_no_tabs(code: str) -> List[ValidationResult]:
    results = []
    for i, line in enumerate(code.split("\n"), start=1):
        if "\t" in line:
            results.append(
                ValidationResult(
                    passed=False,
                    rule_id="ARIVA_NO_TABS",
                    message="Use spaces instead of tabs",
                    line=i,
                    column=line.index("\t") + 1,
                )
            )
    return results


def _rule_indent_consistent(code: str, spaces: int = 4) -> List[ValidationResult]:
    results = []
    for i, line in enumerate(code.split("\n"), start=1):
        if not line.strip():
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent % spaces != 0:
            results.append(
                ValidationResult(
                    passed=False,
                    rule_id="ARIVA_INDENT",
                    message=f"Indent must be multiple of {spaces}",
                    line=i,
                    column=1,
                )
            )
    return results


def _run_full_validation(code: str) -> List[ValidationResult]:
    out = []
    out.extend(_run_all_validation_rules(code))
    out.extend(_rule_no_tabs(code))
    out.extend(_rule_indent_consistent(code))
    return out


# -----------------------------------------------------------------------------
# Request handler (JSON API style)
# -----------------------------------------------------------------------------
def handle_ariva_request(
    platform: AriVaPlatform,
    method: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    coord = ARIVA_COORDINATOR
    try:
        if method == "config":
            return platform.api_config()
        if method == "create_session":
            return platform.api_create_session(
                params.get("user_ref", ""),
                params.get("caller", coord),
            )
        if method == "get_session":
            return platform.api_get_session(params.get("session_id", ""))
        if method == "close_session":
            return platform.api_close_session(
                params.get("session_id", ""),
                params.get("caller", coord),
            )
        if method == "update_context":
            return platform.api_update_context(
                params.get("session_id", ""),
                params.get("context", ""),
                params.get("caller", coord),
            )
        if method == "validate_code":
            return platform.api_validate_code(params.get("code", ""))
        if method == "get_completions":
            return platform.api_get_completions(
                params.get("session_id", ""),
                params.get("prefix", ""),
                params.get("line_context", ""),
                params.get("language", "py"),
                params.get("max_n", MAX_COMPLETIONS_PER_LINE),
            )
        if method == "get_suggestions":
            return platform.api_get_suggestions(
                params.get("session_id", ""),
                params.get("query", ""),
                params.get("kind", 0),
                params.get("max_n", MAX_SUGGESTIONS_PER_REQUEST),
            )
        if method == "stats":
            return platform.api_stats()
        if method == "cleanup_stale":
            n = cleanup_stale_sessions(platform._engine)
            return {"removed": n}
        return {"error": f"Unknown method: {method}"}
    except AriVaNotCoordinator as e:
        return {"error": "AriVaNotCoordinator", "message": str(e)}
    except AriVaSessionNotFound as e:
        return {"error": "AriVaSessionNotFound", "message": str(e)}
    except AriVaSessionLimitReached as e:
        return {"error": "AriVaSessionLimitReached", "message": str(e)}
    except AriVaQueryTooShort as e:
        return {"error": "AriVaQueryTooShort", "message": str(e)}
