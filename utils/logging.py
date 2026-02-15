from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# -------------------------
# Bitstring parsing
# -------------------------
@dataclass(frozen=True)
class ParsedShot:
    # Index-order bits (NOT Qiskit's printed order):
    # - d_bits: d0 d1 d2 (data qubits q0,q1,q2)
    # - s_bits: s0..s(2*n_cycles-1) where per cycle k:
    #     s[2k]   = Z0Z1 parity (0=even, 1=odd)
    #     s[2k+1] = Z1Z2 parity (0=even, 1=odd)
    d_bits: str  # length 3
    s_bits: str  # length 2*n_cycles


def parse_counts_key_default(key: str, n_cycles: int) -> ParsedShot:
    """
    Parser for repetition-code circuits built by core.circuit.build_core_circuit_with_syndrome().

    Qiskit prints each classical register MSB->LSB and separates registers with spaces.
    For our circuit (d:3 bits, s:2*n_cycles bits), typical key looks like:
        "d2d1d0 s(2n-1)...s1s0"

    This function returns index-order strings:
        d_bits = d0d1d2
        s_bits = s0...s(2n-1)
    """
    n_cycles = int(n_cycles)
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")

    parts = key.strip().split()
    d_group: str
    s_group: str
    s_len = 2 * n_cycles

    if len(parts) == 2:
        a, b = parts[0].replace(" ", ""), parts[1].replace(" ", "")
        if len(a) == 3 and len(b) == s_len:
            d_group, s_group = a, b
        elif len(b) == 3 and len(a) == s_len:
            d_group, s_group = b, a
        else:
            bits = (a + b).replace(" ", "")
            if len(bits) < 3 + s_len:
                raise ValueError(f"Invalid counts key: {key!r} (expected 3+{s_len} bits)")
            d_group = bits[:3]
            s_group = bits[3 : 3 + s_len]
    else:
        bits = key.replace(" ", "")
        if len(bits) < 3 + s_len:
            raise ValueError(f"Invalid counts key: {key!r} (expected 3+{s_len} bits)")
        d_group = bits[:3]
        s_group = bits[3 : 3 + s_len]

    # Convert printed order (MSB->LSB) to index order (0->...).
    return ParsedShot(d_bits=d_group[::-1], s_bits=s_group[::-1])


# -------------------------
# Basic ops
# -------------------------
def majority(bits: str) -> int:
    ones = bits.count("1")
    zeros = len(bits) - ones
    return 1 if ones > zeros else 0  # tie -> 0


def decode_repetition_syndrome_majority(s_bits: str, n_cycles: int) -> Optional[int]:
    """
    Decode a 3-qubit repetition code from repeated syndrome measurements.

    We take a per-stabilizer majority vote over cycles to suppress measurement noise,
    then map the (s01, s12) pair to a single-qubit X correction target:
      s01 s12 -> target
       0   0  -> None
       1   0  -> 0
       1   1  -> 1
       0   1  -> 2
    """
    n_cycles = int(n_cycles)
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
    if len(s_bits) != 2 * n_cycles:
        raise ValueError(f"s_bits must be length {2*n_cycles}, got {len(s_bits)}: {s_bits!r}")

    s01_bits = "".join(s_bits[2 * k] for k in range(n_cycles))
    s12_bits = "".join(s_bits[2 * k + 1] for k in range(n_cycles))
    s01 = majority(s01_bits)
    s12 = majority(s12_bits)

    if s01 == 0 and s12 == 0:
        return None
    if s01 == 1 and s12 == 0:
        return 0
    if s01 == 1 and s12 == 1:
        return 1
    return 2  # s01 == 0 and s12 == 1


def _block_consensus(bits: str, W: int) -> int:
    """
    Split a bitstring into blocks of length W and do majority vote per block,
    then majority vote over block votes.
    """
    if W <= 0:
        return majority(bits)
    blocks = [bits[i : i + W] for i in range(0, len(bits), W) if len(bits[i : i + W]) > 0]
    votes = [majority(b) for b in blocks]
    return 1 if sum(votes) > (len(votes) - sum(votes)) else 0  # tie -> 0


def decode_repetition_syndrome_block_consensus(s_bits: str, n_cycles: int, W: int) -> Optional[int]:
    """
    Block-consensus variant of decode_repetition_syndrome_majority.
    We run block consensus on each stabilizer stream (s01, s12) separately.
    """
    n_cycles = int(n_cycles)
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
    if len(s_bits) != 2 * n_cycles:
        raise ValueError(f"s_bits must be length {2*n_cycles}, got {len(s_bits)}: {s_bits!r}")

    s01_bits = "".join(s_bits[2 * k] for k in range(n_cycles))
    s12_bits = "".join(s_bits[2 * k + 1] for k in range(n_cycles))
    s01 = _block_consensus(s01_bits, W=W)
    s12 = _block_consensus(s12_bits, W=W)

    if s01 == 0 and s12 == 0:
        return None
    if s01 == 1 and s12 == 0:
        return 0
    if s01 == 1 and s12 == 1:
        return 1
    return 2


def apply_x_correction(d_bits: str, target: Optional[int]) -> str:
    """
    Apply an X correction (bit flip) to the chosen data index, in post-processing.
    d_bits is index order: d0 d1 d2.
    """
    if len(d_bits) != 3:
        raise ValueError(f"d_bits must be length 3, got {d_bits!r}")
    if target is None:
        return d_bits
    if target not in (0, 1, 2):
        raise ValueError(f"target must be one of 0,1,2 or None, got {target!r}")

    out = list(d_bits)
    out[target] = "1" if out[target] == "0" else "0"
    return "".join(out)


def decode_logical_bit_majority(d_bits: str) -> str:
    """
    Decode logical Z-basis bit from a 3-qubit repetition code by majority vote.
    """
    return "1" if majority(d_bits) == 1 else "0"


# -------------------------
# Scoring
# -------------------------
def score_logical_success(
    counts: Dict[str, int],
    n_cycles: int,
    ideal_logical: str = "0",
) -> float:
    """
    Success probability of the *decoded logical bit* for Z-basis logical states ("0" or "1").

    Decoding pipeline per shot:
    - parse (d_bits, s_bits)
    - syndrome majority -> correction target (optional)
    - apply correction to data bits (post-processing)
    - majority vote on corrected data bits -> logical bit
    """
    ideal_logical = str(ideal_logical).strip()
    if ideal_logical not in ("0", "1"):
        raise ValueError(f"ideal_logical must be '0' or '1', got {ideal_logical!r}")

    success = 0
    total = 0
    for key, c in counts.items():
        parsed = parse_counts_key_default(key, n_cycles)
        target = decode_repetition_syndrome_majority(parsed.s_bits, n_cycles=n_cycles)
        corrected = apply_x_correction(parsed.d_bits, target)
        logical = decode_logical_bit_majority(corrected)

        if logical == ideal_logical:
            success += c
        total += c

    return success / total if total else 0.0


def score_raw_majority_success(
    counts: Dict[str, int],
    n_cycles: int,
    ideal_logical: str = "0",
) -> float:
    """
    Baseline: ignore syndrome bits and decode logical bit by majority vote on raw data measurement.
    """
    ideal_logical = str(ideal_logical).strip()
    if ideal_logical not in ("0", "1"):
        raise ValueError(f"ideal_logical must be '0' or '1', got {ideal_logical!r}")

    success = 0
    total = 0
    for key, c in counts.items():
        parsed = parse_counts_key_default(key, n_cycles)
        logical = decode_logical_bit_majority(parsed.d_bits)
        if logical == ideal_logical:
            success += c
        total += c
    return success / total if total else 0.0


def parse_syndromes_from_counts(counts: dict) -> list[str]:
    """
    Extract only the syndrome register substring (index order) from get_counts() output.
    Useful for quick observability stats.
    """
    syndromes: list[str] = []
    for key, c in counts.items():
        parts = key.strip().split()
        if len(parts) == 2:
            a, b = parts[0], parts[1]
            s_group = b if len(b) != 3 else a
            syndromes.extend([s_group[::-1]] * c)
        elif len(parts) == 1:
            syndromes.extend([parts[0][::-1]] * c)
        else:
            bits = key.replace(" ", "")
            if len(bits) > 3:
                syndromes.extend([bits[3:][::-1]] * c)
    return syndromes
