# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""A10b paired threefold-compute exploratory test, frozen after A10 main."""
from __future__ import annotations
import argparse
from pathlib import Path
from train import run_training

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--length',choices=('short','long'),required=True)
    a=p.parse_args()
    steps=(1000,300) if a.length=='short' else (3000,900)
    config=dict(worlds=list(range(906,910)),seeds=[0,1],methods=['mirror','shift','film'],caps=[1024],
                meta_steps=steps[0],adapt_steps=steps[1],batch=128,experiment='A10B_'+a.length.upper())
    run_training(a.out,config,workers=4)
if __name__=='__main__':main()
