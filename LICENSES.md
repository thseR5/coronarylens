# Licences

CoronaryLens combines our own code with third-party data and anatomy. Each part keeps its own licence.

| Part | Licence | Where |
|---|---|---|
| Source code, configuration, scripts and documentation written for this project | MIT | [`LICENSE`](LICENSE) |
| Patient data: Extension of Z-Alizadeh Sani dataset | CC BY 4.0 | `data/raw/` |
| 3D anatomy: BodyParts3D 4.0 (derived meshes) | CC BY 4.0 | `frontend/public/models/anatomy.glb` |

The MIT licence covers only our own work. It does not relicense the dataset or the anatomy, which remain under CC BY 4.0.

## Dataset

- Alizadehsani R, Roshanzamir M, Sani Z. *Extension of Z-Alizadeh Sani dataset.* UCI Machine Learning Repository, id 411, doi:10.24432/C5461K.
- Licence: Creative Commons Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/).
- Changes: none to the file, which is included unchanged in `data/raw/`. The pipeline encodes columns at load time (`ml/ingest.py`), e.g. Y/N to 1/0, and drops the constant column `Exertional CP`.

## BodyParts3D

- BodyParts3D, © The Database Center for Life Science (DBCLS), licensed under CC Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/).
- Source: https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html
- Changes: meshes merged per structure, decimated, re-oriented and rescaled, left-ventricle surface split into AHA segments, compressed to glTF. Full list of files, FMA concepts and modifications: [`frontend/src/three/LICENSES.md`](frontend/src/three/LICENSES.md).

## Where attribution is shown

README ("Data, anatomy and licences"), the app footer, the 3D scene caption and this file.
