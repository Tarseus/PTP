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
    operators_raw = obj.get("operators_used", []) or []
    impl_raw = obj.get("implementation_hint", {}) or {}

    if not isinstance(hyperparams_raw, dict):
        raise ValueError("hyperparams must be a JSON object.")
    if not isinstance(operators_raw, (list, tuple)):
        raise ValueError("operators_used must be a JSON array.")
    if not isinstance(impl_raw, Mapping):
        raise ValueError("implementation_hint must be a JSON object.")

    expects = impl_raw.get("expects", []) or []
    if not isinstance(expects, (list, tuple)):
        raise ValueError("implementation_hint.expects must be a JSON array.")
    returns = str(impl_raw.get("returns", "")).strip()

    impl = FreeLossImplementationHint(
        expects=[str(x) for x in expects],
        returns=returns or "scalar",
    )

    return FreeLossIR(
        name=name or "unnamed_free_loss",
        intuition=intuition,
        pseudocode=pseudocode,
        hyperparams=dict(hyperparams_raw),
        operators_used=[str(op) for op in operators_raw],
        implementation_hint=impl,
    )

