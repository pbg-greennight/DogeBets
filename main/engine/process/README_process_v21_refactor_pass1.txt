process_v21_refactor_pass2

process_v21_refactor_pass1
==========================

This package reorganizes the audited process code into the planned folder layout:
  - core/
  - features/
  - export/
  - printing/
  - utils/

Notes:
  - Existing code was preserved and reorganized; this is a structural refactor snapshot.
  - v21 scaffold/patch files are included in export/.
  - __pycache__ files were excluded.
  - Some imports in the code may still need updating to match the new folder layout.
  - Feature builder files are still a mix of live code and scaffold/TODO placeholders depending on the module.

Unmapped audited root files kept out of the package because they were not placed in the planned layout:


Pass2 changes:
- HYST compute moved into features/DB_process_hysteresis.py
- printing/DB_process_printing_hyst.py now imports HYST compute from features
- export/DB_process_dataset_export.py now imports feature modules via package-relative imports and uses live HYST signature
