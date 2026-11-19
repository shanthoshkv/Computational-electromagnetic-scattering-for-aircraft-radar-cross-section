# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-11-19
### Added
- `solvers.py`: Physical Optics, Physical Theory of Diffraction, double- and triple-bounce solvers, SIBC / Salisbury / Dallenbach radar-absorbing material models, and an analytic Mie sphere reference.
- `geometries.py`: procedural meshes for a sphere, flat plate, conventional aircraft, faceted stealth aircraft, flying wing and chined-nose aircraft.
- Analysis scripts: ISAR imaging, wideband impulse response, SIBC comparison, stealth shape optimizer with sensitivity study, and mesh-convergence studies.
- `main.py` batch-generates the figure set under `output_ppt/`; report PDFs and poster under `docs/`.
- Pixel-art demo GIF (`docs/demo.gif`), MIT `LICENSE`, `CONTRIBUTING.md`, issue templates and a CI workflow.
