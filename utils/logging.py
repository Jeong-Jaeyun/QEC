from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional


# -------------------------
# Bitstring parsing
# -------------------------
@dataclass(frozen=True)
class ParsedShot:
    d_bits: str  # length 2 (as stored in counts key slice)
    s_bits: str  # length n_cycles


def parse_counts_key_default(key: str, n_cycles: int) -> ParsedShot:
    """
    Default parser aligned with your current experiments:
    bits = d(2) + s(n_cycles)
    where the key is returned by Qiskit get_counts().
    """
    bits = key.replace(" ", "")
    d_bits = bits[:2]
    s_bits = bits[2:2 + n_cycles]
    return ParsedShot(d_bits=d_bits, s_bits=s_bits)


# -------------------------
# Basic ops
# -------------------------
def majority(bits: str) -> int:
    ones = bits.count("1")
    zeros = len(bits) - ones
    return 1 if ones > zeros else 0  # tie -> 0


def apply_correction_to_data(d_bits: str, corr: int, flip_target: int = 0) -> str:
    """
    Minimal correction model:
        - if corr==1, flip chosen data bit (0 or 1).
    We keep consistent ordering with previous code:
        d_bits = d1 d0  (2 chars)
    Return decoded string with same convention used in scoring:
        return d1 + d0
    """
    if len(d_bits) != 2:
        raise ValueError(f"d_bits must be length 2, got {d_bits}")

    d1, d0 = d_bits[0], d_bits[1]
    if corr == 1:
        if flip_target == 0:
            d0 = "1" if d0 == "0" else "0"
        else:
            d1 = "1" if d1 == "0" else "0"
    return d1 + d0


# -------------------------
# Decoder policies
# -------------------------
def decode_flat_majority(s_bits: str) -> int:
    """S2(A) baseline: majority over the whole syndrome sequence."""
    return majority(s_bits)


def decode_block_consensus(s_bits: str, W: int) -> int:
    """
    S3(MVP): split syndrome into blocks of length W
        block_vote = majority(block)
        consensus  = majority(block_vote list)
    """
    if W <= 0:
        return decode_flat_majority(s_bits)

    blocks = [s_bits[i:i+W] for i in range(0, len(s_bits), W) if len(s_bits[i:i+W]) > 0]
    block_votes = [majority(b) for b in blocks]
    ones = sum(block_votes)
    zeros = len(block_votes) - ones
    return 1 if ones > zeros else 0  # tie -> 0


# ---- 확장: weighted consensus ----
def block_vote_and_margin(block: str) -> Tuple[int, float]:
    """
    vote: majority(block)
    margin: |#1-#0| / len(block)  in [0,1]
    """
    n = len(block)
    if n == 0:
        return 0, 0.0
    ones = block.count("1")
    zeros = n - ones
    vote = 1 if ones > zeros else 0
    margin = abs(ones - zeros) / n
    return vote, margin


def decode_weighted_consensus(s_bits: str, W: int) -> int:
    """
    S3-2: weighted consensus using block margin as weight.
        score = sum(weight_i * (+1 if vote=1 else -1))
        return 1 if score>0 else 0
    """
    if W <= 0:
        return decode_flat_majority(s_bits)

    blocks = [s_bits[i:i+W] for i in range(0, len(s_bits), W) if len(s_bits[i:i+W]) > 0]
    score = 0.0
    for b in blocks:
        vote, margin = block_vote_and_margin(b)
        score += margin * (1.0 if vote == 1 else -1.0)
    return 1 if score > 0 else 0  # tie -> 0


# ---- 확장: slashing consensus ----
def decode_slashing_consensus(s_bits: str, W: int, tau: float = 0.2) -> int:
    """
    S3-3: slashing/pruning low-confidence blocks.
        - compute (vote, margin) per block
        - discard if margin < tau
        - consensus by majority on remaining votes
        - if all discarded -> fallback to flat majority
    """
    if W <= 0:
        return decode_flat_majority(s_bits)

    blocks = [s_bits[i:i+W] for i in range(0, len(s_bits), W) if len(s_bits[i:i+W]) > 0]
    votes = []
    for b in blocks:
        vote, margin = block_vote_and_margin(b)
        if margin >= tau:
            votes.append(vote)

    if len(votes) == 0:
        return decode_flat_majority(s_bits)

    ones = sum(votes)
    zeros = len(votes) - ones
    return 1 if ones > zeros else 0


# -------------------------
# Scoring
# -------------------------
DecoderFn = Callable[[str], int]


def score_decoded_success(
    counts: Dict[str, int],
    n_cycles: int,
    decoder: DecoderFn,
    ideal_data: str = "00",
    flip_target: int = 0,
    parser: Callable[[str, int], ParsedShot] = parse_counts_key_default,
) -> float:
    """
    Compute decoded success probability from raw get_counts() output.
    """
    success = 0
    total = 0

    for key, c in counts.items():
        parsed = parser(key, n_cycles)
        corr = decoder(parsed.s_bits)
        decoded = apply_correction_to_data(parsed.d_bits, corr, flip_target=flip_target)

        if decoded == ideal_data:
            success += c
        total += c

    return success / total if total else 0.0

def parse_syndromes_from_counts(counts: dict) -> list[str]:
    syndromes = []
    for key, c in counts.items():
        parts = key.split(maxsplit=1)
        if len(parts) == 1:
            # fallback: if only syndrome exists
            s = parts[0].replace(" ", "")
        else:
            s = parts[1].replace(" ", "")
        syndromes.extend([s] * c)
    return syndromes
