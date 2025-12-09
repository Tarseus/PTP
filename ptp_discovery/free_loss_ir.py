from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


@dataclass
class FreeLossImplementationHint:
    expects: Sequence[str]
    returns: str


@dataclass
class FreeLossIR:
    """Intermediate representation for a free-form preference loss.

    This mirrors the JSON contract expected from the LLM.
    """

    name: str
    intuition: str
    pseudocode: str
    hyperparams: Dict[str, Any]
    operators_used: List[str]
    implementation_hint: FreeLossImplementationHint


def ir_from_json(obj: Mapping[str, Any]) -> FreeLossIR:
    """Convert a JSON-like mapping into a FreeLossIR instance."""

    name = str(obj.get("name", "")).strip()
    intuition = str(obj.get("intuition", "")).strip()
    pseudocode = str(obj.get("pseudocode", "")).strip()
    hyperparams_raw = obj.get("hyperparams", {}) or {}
    operators_raw = obj.get("operators_used", []) or {}
    impl_raw = obj.get("implementation_hint", {}) or {}

    debug_prefix = "[FreeLossIR debug]"

    # hyperparams: prefer an object, but fall back to empty dict on mismatch.
    if not isinstance(hyperparams_raw, dict):
        print(f"{debug_prefix} hyperparams not object; raw={hyperparams_raw!r}")
        hyperparams_raw = {}

    # operators_used: prefer an array; if not, log and coerce.
    if isinstance(operators_raw, (list, tuple)):
        operators_list = [str(op) for op in operators_raw]
    elif operators_raw is None:
        operators_list = []
    else:
        print(f"{debug_prefix} operators_used not array; raw={operators_raw!r}")
        operators_list = [str(operators_raw)]

    # implementation_hint: prefer an object; if not, log and replace.
    if not isinstance(impl_raw, Mapping):
        print(f"{debug_prefix} implementation_hint not object; raw={impl_raw!r}")
        impl_raw = {}

    expects_raw = impl_raw.get("expects", []) or []
    if isinstance(expects_raw, (list, tuple)):
        expects = [str(x) for x in expects_raw]
    elif expects_raw is None:
        expects = []
    else:
        # Be tolerant to models that emit a single string or other scalar.
        print(f"{debug_prefix} implementation_hint.expects not array; raw={expects_raw!r}")
        expects = [str(expects_raw)]

    returns = str(impl_raw.get("returns", "")).strip()

    impl = FreeLossImplementationHint(
        expects=expects,
        returns=returns or "scalar",
    )

    return FreeLossIR(
        name=name or "unnamed_free_loss",
        intuition=intuition,
        pseudocode=pseudocode,
        hyperparams=dict(hyperparams_raw),
        operators_used=operators_list,
        implementation_hint=impl,
    )
