# GlueBind

A collaborative project for binding and gluing things together, developed on Bohrium platform with GPU support.

## Overview

GlueBind is designed to leverage GPU acceleration for efficient computation and data processing tasks.

## Development Environment

- **Platform**: [Bohrium](https://bohrium.dp.tech) - AI4S Research Platform
- **GPU**: NVIDIA T4 (15GB VRAM)
- **CUDA**: 12.2
- **Driver**: 535.274.02
- **Python**: 3.10+
- **OS**: Ubuntu 22.04

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/yiyanliao/GlueBind.git
cd GlueBind
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Check GPU Availability

```bash
python -c "from src.gluebind import check_gpu; print(f'GPU available: {check_gpu()}')"
```

## Project Structure

```
GlueBind/
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
├── src/               # Source code
│   ├── __init__.py
│   └── gluebind.py    # Main module with GPU utilities
├── tests/             # Test files
│   ├── __init__.py
│   └── test_gluebind.py
├── docs/              # Documentation
└── examples/          # Example scripts
```

## Features

- GPU availability detection
- CUDA environment checking
- Modular architecture
- Comprehensive test suite

## Contributors

- yiyanliao - Project Owner
- BohrClaw Assistant - Collaborator

## License

MIT License

## Acknowledgments

Developed with support from Bohrium platform.
