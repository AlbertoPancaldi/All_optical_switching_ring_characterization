# All-Optical Switching Ring Characterization

This repository contains Python scripts and data for the characterization of all-optical switching in ring resonator and MZI structures.

The code is intended for research and internal use.

---

## Repository structure

```text
Internship/
├─ ring_characterization.py        # Main analysis / characterization script
├─ ring_geometry_emode.py          # Geometry and effective index simulation using EMode
├─ data/                           # Experimental or simulated data (ignored by Git)
├─ requirements.txt                # Python dependencies
├─ README.md
└─ venv_PIC_Internship/            # Local virtual environment (ignored by Git)
```

### Notes
- `venv_PIC_Internship/` is **not tracked** by Git and must be created locally.
- The data folders are **ignored** by Git. Required CSV or measurement files must be placed there manually.

---

## Requirements

- Python **3.10+** (tested with Python 3.12)
- macOS or Linux (Windows should also work)

---

## Setup

1. **Clone the repository**
```bash
git clone https://github.com/AlbertoPancaldi/All_optical_switching_ring_characterization.git
cd All_optical_switching_ring_characterization
```
2. **Create the virtual environment**
```bash
python -m venv venv_ring
```
3. **Activate the virtual environment**
```bash
source venv_ring/bin/activate
```
4. **Install dependancies**
```bash
pip install -r requirements.txt
```

---

