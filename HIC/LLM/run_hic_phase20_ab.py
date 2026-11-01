"""Thin command-line entry point for the approved phase-20 HIC A/B run."""

import sys

from LLM.run_hic_full_corpus_development import main as _phase20_main


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--phase20-ab" not in arguments:
        arguments.insert(0, "--phase20-ab")
    return _phase20_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
