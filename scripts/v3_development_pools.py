"""Recreate only v3 development/selection pools; never draw assessment cases."""
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dreamer.provenance import record


def main():
    for split,count in {'development':24,'selection':64,'fit':64,'fit-validation':24}.items():
        seeds=[int.from_bytes(hashlib.sha256(f'namazu-unknown-v3-wave1:{split}:{i}'.encode()).digest()[:8],'big')%(2**63)
               for i in range(count)]
        record(ROOT/'results/private'/f'v3-{split}-seeds.json',seeds)
        print(split,count,'layout cases; three regimes each')


if __name__=='__main__':main()
