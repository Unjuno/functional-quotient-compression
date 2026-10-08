#!/usr/bin/env python3
"""MAT03R fixed-code fresh-only replication. Original MAT03 source unchanged."""
import argparse
from pathlib import Path
import run_mat03
run_mat03.SEEDS['fresh']=(401,402,403,404,405)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results_replica')
    opt=p.parse_args()
    run_mat03.run('fresh',Path(opt.out))
if __name__=='__main__':main()
