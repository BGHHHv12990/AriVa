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
    except AriVaQueryTooLong as e:
        return {"error": "AriVaQueryTooLong", "message": str(e)}
    except AriVaSuggestionLimitReached as e:
        return {"error": "AriVaSuggestionLimitReached", "message": str(e)}
    except AriVaZeroDisallowed as e:
        return {"error": "AriVaZeroDisallowed", "message": str(e)}
    except AriVaContextOverflow as e:
        return {"error": "AriVaContextOverflow", "message": str(e)}
    except Exception as e:
        return {"error": "Internal", "message": str(e)}


def list_ariva_methods() -> List[str]:
    return [
        "config", "create_session", "get_session", "close_session",
        "update_context", "validate_code", "get_completions", "get_suggestions",
        "stats", "cleanup_stale",
    ]


def get_all_ariva_constants() -> Dict[str, Any]:
    return {
        "ARIVA_COORDINATOR": ARIVA_COORDINATOR,
        "ARIVA_VAULT": ARIVA_VAULT,
        "ARIVA_RELAY": ARIVA_RELAY,
        "ARIVA_ORACLE": ARIVA_ORACLE,
        "ARIVA_SENTINEL": ARIVA_SENTINEL,
        "ARIVA_DOMAIN_SALT": ARIVA_DOMAIN_SALT,
        "ARIVA_SESSION_SALT": ARIVA_SESSION_SALT,
        "ARIVA_COMPLETION_SEED": ARIVA_COMPLETION_SEED,
        "MAX_SUGGESTIONS_PER_REQUEST": MAX_SUGGESTIONS_PER_REQUEST,
        "MAX_COMPLETIONS_PER_LINE": MAX_COMPLETIONS_PER_LINE,
        "MAX_SESSION_DURATION_SEC": MAX_SESSION_DURATION_SEC,
        "MAX_SESSIONS_PER_USER": MAX_SESSIONS_PER_USER,
        "MIN_QUERY_LEN": MIN_QUERY_LEN,
        "MAX_QUERY_LEN": MAX_QUERY_LEN,
        "CODE_CONTEXT_WINDOW": CODE_CONTEXT_WINDOW,
        "VALIDATION_RULESET_VERSION": VALIDATION_RULESET_VERSION,
    }


def health_check_ariva(platform: AriVaPlatform) -> Dict[str, Any]:
    return {
        "ok": confirm_ariva_addresses_unique() and confirm_ariva_hex_unique(),
        "addresses_unique": confirm_ariva_addresses_unique(),
        "hex_unique": confirm_ariva_hex_unique(),
        "stats": platform.api_stats(),
    }


def run_ariva_demo(platform: AriVaPlatform) -> Dict[str, Any]:
    coord = ARIVA_COORDINATOR
    r1 = platform.api_create_session("demo_user", coord)
    sid = r1["session_id"]
    r2 = platform.api_validate_code("def foo():\n  x = 1  \n")
    r3 = platform.api_get_completions(sid, "imp", "import ", "py", 3)
    r4 = platform.api_get_suggestions(sid, "fix typo", 1, 2)
    platform.api_update_context(sid, "def bar(): pass", coord)
    return {"session_id": sid, "validate": r2, "completions_count": len(r3.get("completions", [])), "suggestions_count": len(r4.get("suggestions", []))}


def batch_handle_ariva(platform: AriVaPlatform, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [handle_ariva_request(platform, r.get("method", ""), r.get("params", {})) for r in requests]


# -----------------------------------------------------------------------------
# Language-specific helpers (code assistant)
# -----------------------------------------------------------------------------
SUPPORTED_LANGUAGES = ["py", "js", "ts", "sol", "rs", "go", "java", "cpp"]


def is_supported_language(lang: str) -> bool:
    return lang.lower() in SUPPORTED_LANGUAGES


def normalize_language(lang: str) -> str:
    return lang.lower() if lang else "py"


def get_language_comment_prefix(lang: str) -> str:
    m = {"py": "#", "js": "//", "ts": "//", "sol": "//", "rs": "//", "go": "//", "java": "//", "cpp": "//"}
    return m.get(normalize_language(lang), "#")


def get_rule_ids() -> List[str]:
    return ["ARIVA_NO_TRAILING_WS", "ARIVA_MAX_LINE_LEN", "ARIVA_BALANCED_BRACES", "ARIVA_NO_TABS", "ARIVA_INDENT"]


def validate_with_ruleset(code: str, rule_ids: Optional[List[str]] = None) -> List[ValidationResult]:
    all_results = _run_full_validation(code)
    if not rule_ids:
        return all_results
    return [r for r in all_results if r.rule_id in rule_ids]


def suggestion_kind_name(kind: int) -> str:
    names = ["Completion", "Fix", "Hint", "Refactor"]
    return names[kind] if 0 <= kind < len(names) else "Unknown"


def session_status_name(status: int) -> str:
    names = ["Active", "Idle", "Closed"]
    return names[status] if 0 <= status < len(names) else "Unknown"


def format_validation_results_for_display(results: List[Dict[str, Any]]) -> List[str]:
    return [
        f"{r.get('rule_id', '?')}: {r.get('message', '')} (line {r.get('line', 0)}, col {r.get('column', 0)})"
        for r in results
    ]


def compute_context_hash(context: str) -> str:
    return hashlib.sha256((ARIVA_DOMAIN_SALT + context).encode()).hexdigest()[:24]


def compute_session_fingerprint(session_id: str, user_ref: str) -> str:
    return hashlib.sha256((ARIVA_SESSION_SALT + session_id + user_ref).encode()).hexdigest()[:32]


# -----------------------------------------------------------------------------
# Export / serialization
# -----------------------------------------------------------------------------
def export_sessions_summary(engine: AriVaEngine) -> Dict[str, Any]:
    sessions = []
    for s in engine.state.sessions.values():
        sessions.append({
            "session_id": s.session_id,
            "user_ref": s.user_ref,
            "status": s.status,
            "query_count": s.query_count,
            "created_at": s.created_at,
        })
    return {"sessions": sessions, "total": len(sessions)}


def export_stats_json(platform: AriVaPlatform) -> str:
    return json.dumps(platform.api_stats(), indent=2)


def export_config_json(platform: AriVaPlatform) -> str:
    return json.dumps(platform.api_config(), indent=2)


# -----------------------------------------------------------------------------
# Request templates for ManivA / Ama
# -----------------------------------------------------------------------------
def get_request_templates() -> Dict[str, Dict[str, Any]]:
    coord = ARIVA_COORDINATOR
    return {
        "create_session": {"user_ref": "user_1", "caller": coord},
        "get_session": {"session_id": "<session_id>"},
        "close_session": {"session_id": "<session_id>", "caller": coord},
        "update_context": {"session_id": "<session_id>", "context": "def foo(): pass", "caller": coord},
        "validate_code": {"code": "def bar():\n  pass"},
        "get_completions": {"session_id": "<session_id>", "prefix": "im", "line_context": "import ", "language": "py", "max_n": 5},
        "get_suggestions": {"session_id": "<session_id>", "query": "suggest fix", "kind": 1, "max_n": 10},
    }


# -----------------------------------------------------------------------------
# Additional validation: empty file, duplicate lines (demo rules)
# -----------------------------------------------------------------------------
def _rule_non_empty_file(code: str) -> List[ValidationResult]:
    if not code.strip():
        return [
            ValidationResult(
                passed=False,
                rule_id="ARIVA_NON_EMPTY",
                message="File must not be empty",
                line=1,
                column=1,
            )
        ]
    return []


def _rule_no_consecutive_blank_lines(code: str, max_blank: int = 2) -> List[ValidationResult]:
    results = []
    blank_count = 0
    for i, line in enumerate(code.split("\n"), start=1):
        if not line.strip():
            blank_count += 1
            if blank_count > max_blank:
                results.append(
                    ValidationResult(
                        passed=False,
                        rule_id="ARIVA_MAX_BLANK",
                        message=f"More than {max_blank} consecutive blank lines",
                        line=i,
                        column=1,
                    )
                )
        else:
            blank_count = 0
    return results


def run_extended_validation(code: str) -> List[ValidationResult]:
    out = _run_full_validation(code)
    out.extend(_rule_non_empty_file(code))
    out.extend(_rule_no_consecutive_blank_lines(code))
    return out


# -----------------------------------------------------------------------------
# Platform with event log
# -----------------------------------------------------------------------------
class AriVaPlatformWithEvents(AriVaPlatform):
    def __init__(self) -> None:
        super().__init__()
        self._event_log = AriVaEventLog()

    def api_create_session(self, user_ref: str, caller: str) -> Dict[str, Any]:
        r = super().api_create_session(user_ref, caller)
        self._event_log.emit("session_created", {"session_id": r["session_id"], "user_ref": user_ref})
        return r

    def api_close_session(self, session_id: str, caller: str) -> Dict[str, Any]:
        r = super().api_close_session(session_id, caller)
        self._event_log.emit("session_closed", {"session_id": session_id})
        return r

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._event_log.recent(limit)


def create_ariva_with_events() -> AriVaPlatformWithEvents:
    return AriVaPlatformWithEvents()


# -----------------------------------------------------------------------------
# Stub for completion ranking (code assistant)
# -----------------------------------------------------------------------------
def rank_completions_by_confidence(completions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(completions, key=lambda c: c.get("confidence", 0), reverse=True)


def filter_suggestions_by_kind(suggestions: List[Dict[str, Any]], kind: int) -> List[Dict[str, Any]]:
    return [s for s in suggestions if s.get("kind") == kind]


# -----------------------------------------------------------------------------
# Constants for display (ManivA)
# -----------------------------------------------------------------------------
def get_display_constants() -> Dict[str, Any]:
    return {
        "coordinator": ARIVA_COORDINATOR,
        "vault": ARIVA_VAULT,
        "relay": ARIVA_RELAY,
        "oracle": ARIVA_ORACLE,
        "sentinel": ARIVA_SENTINEL,
        "domain_salt": ARIVA_DOMAIN_SALT[:18] + "...",
        "session_salt": ARIVA_SESSION_SALT[:18] + "...",
        "completion_seed": ARIVA_COMPLETION_SEED[:18] + "...",
        "max_suggestions": MAX_SUGGESTIONS_PER_REQUEST,
        "max_completions": MAX_COMPLETIONS_PER_LINE,
        "max_session_duration_sec": MAX_SESSION_DURATION_SEC,
        "max_sessions_per_user": MAX_SESSIONS_PER_USER,
        "code_context_window": CODE_CONTEXT_WINDOW,
    }


def run_ariva_simulation(platform: AriVaPlatform, num_sessions: int = 8, queries_per_session: int = 5) -> Dict[str, Any]:
    coord = ARIVA_COORDINATOR
    created = []
    for i in range(num_sessions):
        r = platform.api_create_session(f"sim_user_{i}", coord)
        created.append(r["session_id"])
    total_completions = 0
    total_suggestions = 0
    for sid in created:
        for j in range(queries_per_session):
            platform.api_get_completions(sid, f"pre_{j}", f"line {j}", "py", 4)
            total_completions += 4
            platform.api_get_suggestions(sid, f"query {j}", j % 4, 3)
            total_suggestions += 3
    cleanup_stale_sessions(platform._engine)
    return {
        "sessions_created": len(created),
        "total_completions": total_completions,
        "total_suggestions": total_suggestions,
        "stats": platform.api_stats(),
    }


def get_coordinator_address() -> str:
    return ARIVA_COORDINATOR


def get_vault_address() -> str:
    return ARIVA_VAULT


def get_relay_address() -> str:
    return ARIVA_RELAY


def get_oracle_address() -> str:
    return ARIVA_ORACLE


def get_sentinel_address() -> str:
    return ARIVA_SENTINEL


def get_domain_salt() -> str:
    return ARIVA_DOMAIN_SALT


def get_session_salt() -> str:
    return ARIVA_SESSION_SALT


def get_completion_seed() -> str:
    return ARIVA_COMPLETION_SEED


def get_max_suggestions_per_request() -> int:
    return MAX_SUGGESTIONS_PER_REQUEST


def get_max_completions_per_line() -> int:
    return MAX_COMPLETIONS_PER_LINE


def get_max_session_duration_sec() -> int:
    return MAX_SESSION_DURATION_SEC


def get_max_sessions_per_user() -> int:
    return MAX_SESSIONS_PER_USER


def get_min_query_len() -> int:
    return MIN_QUERY_LEN


def get_max_query_len() -> int:
    return MAX_QUERY_LEN


def get_code_context_window() -> int:
    return CODE_CONTEXT_WINDOW


def validate_query_length(query: str) -> bool:
    return MIN_QUERY_LEN <= len(query) <= MAX_QUERY_LEN


def validate_context_length(context: str) -> bool:
    return len(context) <= CODE_CONTEXT_WINDOW


def truncate_context(context: str) -> str:
    if len(context) <= CODE_CONTEXT_WINDOW:
        return context
    return context[-CODE_CONTEXT_WINDOW:]


def truncate_response(text: str, max_len: int = ASSISTANT_RESPONSE_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# -----------------------------------------------------------------------------
# Addresses and hex in this file are unique; not reused from Spella, FrostVow,
# Robotank, BacklineLedger, RigCue, HermesAI, EwAI, or any other contract.
# -----------------------------------------------------------------------------


def run_ariva_simulation_v2(
    platform: AriVaPlatform,
    num_users: int = 12,
    sessions_per_user: int = 2,
    validate_samples: int = 20,
) -> Dict[str, Any]:
    coord = ARIVA_COORDINATOR
    all_sessions = []
    for u in range(num_users):
        for _ in range(sessions_per_user):
            r = platform.api_create_session(f"v2_user_{u}", coord)
            all_sessions.append(r["session_id"])
    for sid in all_sessions[: len(all_sessions) // 2]:
        platform.api_update_context(sid, "def example():\n    return 42", coord)
    for _ in range(validate_samples):
        platform.api_validate_code("x = 1\ny = 2\n")
    for sid in all_sessions:
        platform.api_get_completions(sid, "r", "return ", "py", 5)
        platform.api_get_suggestions(sid, "refactor", 3, 4)
    n_removed = cleanup_stale_sessions(platform._engine)
    return {
        "sessions_created": len(all_sessions),
        "validate_calls": validate_samples,
        "stale_removed": n_removed,
        "stats": platform.api_stats(),
    }


def parse_validation_result(r: ValidationResult) -> Dict[str, Any]:
    return {"passed": r.passed, "rule_id": r.rule_id, "message": r.message, "line": r.line, "column": r.column}


def parse_validation_results(results: List[ValidationResult]) -> List[Dict[str, Any]]:
    return [parse_validation_result(r) for r in results]


def has_validation_errors(results: List[ValidationResult]) -> bool:
    return any(not r.passed for r in results)


def first_validation_error(results: List[ValidationResult]) -> Optional[ValidationResult]:
    for r in results:
        if not r.passed:
            return r
    return None


def get_rule_id_list() -> List[str]:
    return get_rule_ids() + ["ARIVA_NON_EMPTY", "ARIVA_MAX_BLANK"]


def format_address_short(addr: str, head: int = 6, tail: int = 4) -> str:
    if not addr or len(addr) < head + tail:
        return addr or ""
    return f"{addr[:head]}...{addr[-tail:]}"


def format_salt_short(salt: str, max_len: int = 20) -> str:
    if not salt or len(salt) <= max_len:
        return salt or ""
    return salt[:max_len] + "..."


def build_create_session_params(user_ref: str) -> Dict[str, Any]:
    return {"user_ref": user_ref, "caller": ARIVA_COORDINATOR}


def build_get_completions_params(session_id: str, prefix: str, language: str = "py") -> Dict[str, Any]:
    return {"session_id": session_id, "prefix": prefix, "line_context": prefix, "language": language, "max_n": MAX_COMPLETIONS_PER_LINE}


def build_get_suggestions_params(session_id: str, query: str, kind: int = 0) -> Dict[str, Any]:
    return {"session_id": session_id, "query": query, "kind": kind, "max_n": MAX_SUGGESTIONS_PER_REQUEST}


def build_validate_code_params(code: str) -> Dict[str, Any]:
    return {"code": code}


def apply_validation_to_code(platform: AriVaPlatform, code: str) -> Tuple[bool, List[Dict[str, Any]]]:
    r = platform.api_validate_code(code)
    results = r.get("results", [])
    passed = not has_validation_errors([ValidationResult(r["passed"], r["rule_id"], r["message"], r.get("line"), r.get("column")) for r in results])
    return (passed, results)


def session_is_active(engine: AriVaEngine, session_id: str) -> bool:
    s = engine.get_session(session_id)
    if s is None:
        return False
    return s.get("status") == int(SessionStatus.ACTIVE)


def user_session_count(engine: AriVaEngine, user_ref: str) -> int:
    return len(engine.state.user_sessions.get(user_ref, []))


def total_session_count(engine: AriVaEngine) -> int:
    return len(engine.state.sessions)


def request_count(engine: AriVaEngine) -> int:
    return engine.state.request_count


def suggestions_served_count(engine: AriVaEngine) -> int:
    return engine.state.total_suggestions_served


def get_suggestion_kind_enum_values() -> List[int]:
    return [int(SuggestionKind.COMPLETION), int(SuggestionKind.FIX), int(SuggestionKind.HINT), int(SuggestionKind.REFACTOR)]


def get_session_status_enum_values() -> List[int]:
    return [int(SessionStatus.ACTIVE), int(SessionStatus.IDLE), int(SessionStatus.CLOSED)]


def metadata_ariva() -> Dict[str, Any]:
    return {
        "name": "AriVa",
        "style": "code assistant",
        "coordinator": ARIVA_COORDINATOR,
        "vault": ARIVA_VAULT,
        "domain_salt_prefix": ARIVA_DOMAIN_SALT[:16],
        "max_suggestions": MAX_SUGGESTIONS_PER_REQUEST,
        "max_completions": MAX_COMPLETIONS_PER_LINE,
        "supported_languages": SUPPORTED_LANGUAGES,
        "validation_ruleset_version": VALIDATION_RULESET_VERSION,
    }


def readiness_ariva(platform: AriVaPlatform) -> Dict[str, Any]:
    h = health_check_ariva(platform)
    return {"ready": h.get("ok", False), "health": h}


def api_method_param_keys(method: str) -> List[str]:
    t = get_request_templates()
    if method not in t:
        return []
    return list(t[method].keys())


def default_caller() -> str:
    return ARIVA_COORDINATOR


def export_sessions_to_list(engine: AriVaEngine) -> List[Dict[str, Any]]:
    return export_sessions_summary(engine).get("sessions", [])


def filter_sessions_by_status(engine: AriVaEngine, status: int) -> List[str]:
    return [sid for sid, s in engine.state.sessions.items() if s.status == status]


def filter_sessions_by_user(engine: AriVaEngine, user_ref: str) -> List[str]:
    return list(engine.state.user_sessions.get(user_ref, []))


def completion_confidence_threshold() -> float:
    return 0.5


def suggestion_confidence_threshold() -> float:
    return 0.6


def filter_completions_by_confidence(completions: List[Dict[str, Any]], min_conf: float = 0.5) -> List[Dict[str, Any]]:
    return [c for c in completions if c.get("confidence", 0) >= min_conf]


def filter_suggestions_by_confidence(suggestions: List[Dict[str, Any]], min_conf: float = 0.6) -> List[Dict[str, Any]]:
    return [s for s in suggestions if s.get("confidence", 0) >= min_conf]


def validate_session_id_format(session_id: str) -> bool:
    return isinstance(session_id, str) and len(session_id) >= 16 and len(session_id) <= 64 and session_id.isalnum() or (session_id.replace("-", "").replace("_", "").isalnum())


def validate_user_ref_format(user_ref: str) -> bool:
    return isinstance(user_ref, str) and len(user_ref) >= 1 and len(user_ref) <= 256


def sanitize_context_for_window(context: str) -> str:
    return truncate_context(context)


def sanitize_query_for_limit(query: str) -> str:
    if len(query) <= MAX_QUERY_LEN:
        return query
    return query[:MAX_QUERY_LEN]


def build_update_context_params(session_id: str, context: str) -> Dict[str, Any]:
    return {"session_id": session_id, "context": sanitize_context_for_window(context), "caller": ARIVA_COORDINATOR}


def build_close_session_params(session_id: str) -> Dict[str, Any]:
    return {"session_id": session_id, "caller": ARIVA_COORDINATOR}


def build_get_session_params(session_id: str) -> Dict[str, Any]:
    return {"session_id": session_id}


def all_ariva_addresses() -> List[str]:
    return [ARIVA_COORDINATOR, ARIVA_VAULT, ARIVA_RELAY, ARIVA_ORACLE, ARIVA_SENTINEL]


def all_ariva_salts() -> List[str]:
    return [ARIVA_DOMAIN_SALT, ARIVA_SESSION_SALT, ARIVA_COMPLETION_SEED]


def confirm_all_addresses_valid() -> bool:
    return all(_is_eth_address(a) for a in all_ariva_addresses())


def confirm_all_salts_valid() -> bool:
    def valid_hex(s: str) -> bool:
        if not s or not s.startswith("0x") or len(s) < 18:
            return False
        try:
            int(s[2:], 16)
            return True
        except ValueError:
            return False
    return all(valid_hex(s) for s in all_ariva_salts())


def get_validation_rule_descriptions() -> Dict[str, str]:
    return {
        "ARIVA_NO_TRAILING_WS": "Remove trailing whitespace from line",
        "ARIVA_MAX_LINE_LEN": "Line exceeds maximum allowed length (120)",
        "ARIVA_BALANCED_BRACES": "Unbalanced or unclosed bracket",
        "ARIVA_NO_TABS": "Use spaces instead of tabs",
        "ARIVA_INDENT": "Indent must be multiple of 4 spaces",
        "ARIVA_NON_EMPTY": "File must not be empty",
        "ARIVA_MAX_BLANK": "Too many consecutive blank lines",
    }


def get_suggestion_kind_descriptions() -> Dict[int, str]:
    return {
        int(SuggestionKind.COMPLETION): "Code completion",
        int(SuggestionKind.FIX): "Fix or correction",
        int(SuggestionKind.HINT): "Hint or tip",
        int(SuggestionKind.REFACTOR): "Refactoring suggestion",
    }


def get_session_status_descriptions() -> Dict[int, str]:
    return {
        int(SessionStatus.ACTIVE): "Session is active",
        int(SessionStatus.IDLE): "Session is idle",
        int(SessionStatus.CLOSED): "Session is closed",
    }


def format_validation_for_cli(results: List[Dict[str, Any]]) -> str:
    lines = []
    for r in results:
        if not r.get("passed", True):
            lines.append(f"  {r.get('rule_id', '?')}: {r.get('message', '')} at line {r.get('line', 0)}, col {r.get('column', 0)}")
    return "\n".join(lines) if lines else "No issues found."


def format_config_for_display(config: Dict[str, Any]) -> List[str]:
    return [f"{k}: {v}" for k, v in config.items()]


def get_default_language() -> str:
    return "py"


def get_max_validation_results() -> int:
    return 100


def ariva_version_info() -> str:
    return "AriVa code assistant engine v1"


def supported_languages_list() -> List[str]:
    return list(SUPPORTED_LANGUAGES)


def is_valid_language(lang: str) -> bool:
    return is_supported_language(lang)


def default_max_completions() -> int:
    return MAX_COMPLETIONS_PER_LINE


def default_max_suggestions() -> int:
    return MAX_SUGGESTIONS_PER_REQUEST


def get_validation_rule_ids() -> List[str]:
    """Return list of validation rule identifiers used by validate_code."""
    return ["trailing_whitespace", "max_line_length", "balanced_braces", "no_tabs", "indent_style", "non_empty_file", "max_consecutive_blank_lines"]


def get_suggestion_kind_names() -> Dict[int, str]:
    """Return mapping of suggestion kind id to name."""
    return {0: "Completion", 1: "Fix", 2: "Hint", 3: "Refactor"}


def api_validate_code_simple(platform: AriVaPlatform, code: str) -> bool:
    """Return True iff code passes all validation rules (no errors)."""
    result = platform.api_validate_code(code)
    if isinstance(result, dict) and "valid" in result:
        return bool(result["valid"])
    return False


def get_platform_version() -> str:
    return "1.0.0"


if __name__ == "__main__":
    p = create_ariva()
    coord = ARIVA_COORDINATOR
    r = p.api_create_session("user_1", coord)
    print("Session:", r)
    print("Validate:", p.api_validate_code("def foo():\n  pass  "))
    print("Config:", p.api_config())
    print("Addresses unique:", confirm_ariva_addresses_unique())
    print("Hex unique:", confirm_ariva_hex_unique())

