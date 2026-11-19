# Contributing to Computational RCS Analysis

Thanks for helping. Small, focused changes are easiest to review.

## Getting set up

```bash
git clone https://github.com/shanthoshkv/Computational-electromagnetic-scattering-for-aircraft-radar-cross-section
cd Computational-electromagnetic-scattering-for-aircraft-radar-cross-section
pip install numpy scipy matplotlib
```

## Check your change

```bash
python -m py_compile *.py
python -c "import solvers, geometries, plot_style"   # no test suite exists yet
python main.py                                       # optional: regenerates every figure under output_ppt/ (slow)
```

## Pull requests

1. Open an issue first for anything bigger than a bug fix or doc tweak.
2. Branch from `main`, keep the diff small, one concern per PR.
3. Make sure the check above passes and CI is green.
4. Describe what changed and why. Add a line to `CHANGELOG.md` under **Unreleased** for user-visible changes.

## Good first issues

Look for the [`good first issue`](https://github.com/shanthoshkv/Computational-electromagnetic-scattering-for-aircraft-radar-cross-section/labels/good%20first%20issue) label. Comment on one to claim it.

## Questions

Use [Discussions](https://github.com/shanthoshkv/Computational-electromagnetic-scattering-for-aircraft-radar-cross-section/discussions) for questions and ideas, Issues for bugs and concrete requests.

By contributing you agree your work is released under this repository's license.
