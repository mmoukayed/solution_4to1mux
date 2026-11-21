#!/usr/bin/env python3
"""Read-only check that the loaded install matches this source release."""
import argparse
import hashlib
import importlib
from pathlib import Path
import sys

EXPECTED_VERSION = '0.8.1'


def compare_sources(source, installed, files):
    problems = []
    for name in files:
        expected, actual = source/name, installed/name
        if not expected.is_file() or not actual.is_file():
            problems.append(f'missing source/install file: {name}')
        elif hashlib.sha256(expected.read_bytes()).digest() != hashlib.sha256(actual.read_bytes()).digest():
            problems.append(f'different source/install contents: {name}')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Path to the extracted solution package root')
    args = parser.parse_args(argv)
    source = args.source.resolve()/'solution_4to1mux'
    try:
        module = importlib.import_module('solution_4to1mux.build_info')
    except ImportError:
        print('NOT READY: loaded package is missing the Build 8.1 identity. Rebuild and source the correct install/setup.bash.')
        return 1
    installed = Path(module.__file__).resolve().parent
    problems = compare_sources(source, installed, module.CORE_FILES)
    if module.VERSION != EXPECTED_VERSION:
        problems.append(f'loaded version {module.VERSION}; expected {EXPECTED_VERSION}')
    print('Requested source:', source)
    print('Loaded source:   ', installed)
    print(module.build_banner())
    for problem in problems:
        print('FAIL:', problem)
    if problems:
        print('NOT READY: replace the full package, rebuild, source the correct install and rerun this check.')
        return 1
    print('READY: all loaded core modules match this release. This checks installation, not live robot safety or controller activation.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
