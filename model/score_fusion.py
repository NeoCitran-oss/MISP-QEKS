"""Score fusion utilities for XEQ-Matcher / TVA_KWS_PLCL_AVmask."""

from __future__ import annotations

from typing import Dict, Optional

PAIR_NAMES = ('t_v', 't_a', 'v_v', 'v_a', 'a_v', 'a_a')

# Default: equal weight on all six pairwise experts (score fusion baseline).
DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {name: 1.0 for name in PAIR_NAMES}


def parse_fusion_weights(spec: Optional[str]) -> Dict[str, float]:
    """
    Parse fusion weights from a CLI string.

    Examples:
        "1,1,1,1,1,1"              -> equal weights in PAIR_NAMES order
        "t_v=2,t_a=1,v_v=0.5,..."    -> named weights
    """
    if not spec:
        return dict(DEFAULT_FUSION_WEIGHTS)

    spec = spec.strip()
    if '=' not in spec and ',' in spec:
        values = [float(x) for x in spec.split(',')]
        if len(values) != len(PAIR_NAMES):
            raise ValueError(
                f'Expected {len(PAIR_NAMES)} comma-separated weights, got {len(values)}.'
            )
        return dict(zip(PAIR_NAMES, values))

    weights = dict(DEFAULT_FUSION_WEIGHTS)
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, value = item.split('=')
        name = name.strip()
        if name not in PAIR_NAMES:
            raise ValueError(f'Unknown pair name "{name}". Valid: {PAIR_NAMES}')
        weights[name] = float(value)
    return weights
