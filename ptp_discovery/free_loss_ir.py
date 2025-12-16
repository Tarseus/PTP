from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence
import logging


LOGGER = logging.getLogger("ptp_discovery.free_loss_ir")


@dataclass
class FreeLossImplementationHint:
    expects: Sequence[str]
    returns: str


@dataclass
class FreeLossIR:
    """Intermediate representation for a free-form preference loss.

    This mirrors the JSON contract expected from the LLM. In the new
    code-first design, the JSON is expected to include a `code` field
    containing a concrete Python implementation of the loss function.
    """

    name: str
    intuition: str
    pseudocode: str
    hyperparams: Dict[str, Any]
    operators_used: List[str]
    implementation_hint: FreeLossImplementationHint
    code: str = ""
    theoretical_basis: str = ""


def ir_from_json(obj: Mapping[str, Any]) -> FreeLossIR:
    """Convert a JSON-like mapping into a FreeLossIR instance."""

    name = str(obj.get("name", "")).strip()
    intuition = str(obj.get("intuition", "")).strip()
    pseudocode = str(obj.get("pseudocode", "")).strip()
    code = str(obj.get("code", "")).strip()
    theoretical_basis = str(obj.get("theoretical_basis", "")).strip()
    hyperparams_raw = obj.get("hyperparams", {}) or {}
    operators_raw = obj.get("operators_used", []) or {}
    impl_raw = obj.get("implementation_hint", {}) or {}

    # hyperparams: prefer an object, but fall back to empty dict on mismatch.
    if not isinstance(hyperparams_raw, dict):
        LOGGER.debug("hyperparams not object; raw=%r", hyperparams_raw)
        hyperparams_raw = {}

    # operators_used: prefer an array; if not, log and coerce.
    if isinstance(operators_raw, (list, tuple)):
        operators_list = [str(op) for op in operators_raw]
    elif operators_raw is None:
        operators_list = []
    elif isinstance(operators_raw, Mapping):
        operators_list = [str(k) for k in operators_raw.keys()]
    else:
        LOGGER.debug(
            "operators_used not array or Mapping; type=%s, raw=%r",
            type(operators_raw),
            operators_raw,
        )
        operators_list = [str(operators_raw)]

    # implementation_hint: prefer an object; if not, log and replace.
    if not isinstance(impl_raw, Mapping):
        LOGGER.debug("implementation_hint not object; raw=%r", impl_raw)
        impl_raw = {}

    expects_raw = impl_raw.get("expects", []) or []
    if isinstance(expects_raw, (list, tuple)):
        expects = [str(x) for x in expects_raw]
    elif expects_raw is None:
        expects = []
    elif isinstance(expects_raw, Mapping):
        expects = [str(k) for k in expects_raw.keys()]
    else:
        # Be tolerant to models that emit a single string or other scalar.
        LOGGER.debug(
            "implementation_hint.expects not array or Mapping; type=%s, raw=%r",
            type(expects_raw),
            expects_raw,
        )
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
        code=code,
        theoretical_basis=theoretical_basis,
    )
