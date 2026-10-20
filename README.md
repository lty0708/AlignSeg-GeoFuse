# AlignSeg–GeoFuse

Core code and selected aggregate data for *Aligned Multi-View Segmentation and Image–Geometry Fusion for Multi-Task PitNET MRI Assessment*.

`src/nets.py`: AlignSeg. `src/mva_multitask_s2r1.py`: GeoFuse. `src/knosp_r_experiments.py`: geometry and frozen normalization. `prepare_public.py`: PitNET136 preprocessing.

Python 3.10; tested with PyTorch 2.8.0. From this directory in PowerShell:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe test_contracts.py --output outputs/contracts.json
& .\.venv\Scripts\python.exe smoke_training.py --output outputs/training.json
& .\.venv\Scripts\python.exe reproduce_aggregate.py --output-dir outputs/aggregate
```

For CUDA, install the corresponding PyTorch build and pass `--device cuda` to `smoke_training.py`. Tests use generated tensors; study data are unchanged. AlignSeg accepts `[B,5,H,W]` and returns `[B,3,H,W]`; GeoFuse accepts `[B,3,1,H,W]` and `[B,32]` and returns seven task heads.

Obtain PitNET136 from [the original dataset](https://doi.org/10.6084/m9.figshare.27894084.v1) ([source article](https://doi.org/10.1038/s41597-024-04218-8)):

```powershell
& .\.venv\Scripts\python.exe prepare_public.py --data-root data/PitNET136/Pituitary_MRI_tumor_carotids --metadata data/PitNET136/SciData_pitdataset_v2.xlsx --output-dir outputs/public_smoke --max-cases 3
```

Preprocessing uses reference masks for offline view selection and preserves the original affine handling, recorded in each output `meta.json`. The public workbook does not supply all seven task labels.

`data/aggregate` contains four unchanged manuscript CSVs: segmentation comparison (15 rows), task performance (21), calibration summaries (4), and Knosp error counts (3). This partial release excludes patient-level records, labels, partitions, trained weights and full experiment orchestration. It is not sufficient to reconstruct every paper result.

Code: [MIT](LICENSE). Four aggregate CSVs: [CC BY 4.0](data/LICENSE). Original public datasets retain their own licenses and are not redistributed here.
