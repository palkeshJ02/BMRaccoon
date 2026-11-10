# BMRaccoon Parameters check

This folder contains a standalone script for the BMRaccoon concrete
parameters. It reproduces the security checks.
## Dependencies

- Python 3.9 or newer (tested with 3.10.12)
- SageMath, importable from that Python (`from sage.all import ...`; tested with 9.5)
- A copy of `lattice-estimator` is bundled in this folder and is used by
  default.

## Run

From this folder:

```bash
python3 BlindMultiRaccoon.py
```

The script is not a parameter search. It contains the four concrete rows
`(q_s, n) ∈ {2^32, 2^64} × {32, 64}` and calls the lattice estimator directly
on the fixed instances used for:

- signature MSIS,
- key MLWE,
- binary MLWE-BGV IND-CPA security,
- hiding and simulated-setup security of the extractable commitment.

It also recomputes the statistical terms (bad hash, commitment guessing, nonce
and prehash collisions, correctness, blindness), the BGV and
extraction no-wrap inequalities, signature/public-key/ciphertext/commitment
sizes, the SHA3-384/SHAKE256 nonlinear constraint count, and the communication
table.

The approximately 115-125 KB size of each ZK-LaBRADOR proof is an analytical input
from the cited implementation paper; it is not derived by this script.
