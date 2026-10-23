# Data and Weight Licence Register

Every dataset and pretrained weight file the model program may use, with the
terms read from its primary source, the date they were read, and what is still
unresolved. Nothing listed here has been downloaded. A row is updated, never
silently rewritten: a changed term gets a new dated entry beside the old one.

This register records what the sources say. It is not legal advice, and a
permissive *code* licence on a repository says nothing about the data or
weights it points to.

**Last checked:** 2026-10-24.

---

## 1. Summary

| Asset | Intended use | Terms (short) | Access | Status |
|---|---|---|---|---|
| FaceForensics++ (c23) | Train, validation, calibration, in-domain test | Non-commercial research and educational use only | Request form; approval required | Not requested |
| Celeb-DF v2 | External benchmark only | Non-commercial research only; no publishing of derived data | Request form; academic email requested | Not requested |
| DINOv2 ViT-B/14 (`facebook/dinov2-base`) | Frozen backbone for the MVP probe | Apache-2.0 (code and weights) | Public download | Not downloaded |
| CLIP ViT-B/16 | Comparison backbone | Not checked | — | Not checked |
| torch, transformers, scikit-learn | Training extra | Not checked | PyPI | Check from package metadata when the `train` extra is locked |
| DFDC, DeeperForensics-1.0, WildDeepfake | Deferred | Not checked | — | Not checked |
| SCRFD weights, WIDER FACE | Face detector (optional) and its R13 measurement corpus | See `docs/SCRFD_ASSET_AUDIT.md` | — | Audited earlier; unchanged |

---

## 2. FaceForensics++

| Field | Value |
|---|---|
| Sources | Terms of use: `https://kaldir.vc.in.tum.de/faceforensics_tos.pdf` (the README links the `http://` URL, which redirects). Repository: `https://github.com/ondyari/FaceForensics` (`README.md`, `dataset/README.md`, `dataset/splits/`, `LICENSE`). Request form: `https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform` |
| Date checked | 2026-10-24 |
| Data terms | Use "only for non-commercial research and educational purposes" (term 1). Sharing is allowed only with research associates and colleagues who first agree to the same terms (term 4). The researcher indemnifies TUM and contributors, including for copies of copyrighted images and videos made from the data (term 3). Access can be terminated at any time (term 5). If the researcher works for a for-profit entity, the employer is also bound (term 6). German law (term 7). Citation required. |
| Code licence | MIT (repository `LICENSE`). Covers the scripts, not the data. |
| Access | Google form; a download script is sent after approval. Required fields: email, name, lab/department/affiliation, principal investigator or advisor name and email, research purpose, agreement to the terms. |
| Size | FF++ h264 videos at CRF 23: ~10 GB; raw ~500 GB; original YouTube downloads 38.5 GB (`dataset/README.md`). The DeepFakeDetection subset is listed separately (c23 ~3 GB originals + ~22 GB manipulated) and is not needed. |
| Official splits | `dataset/splits/{train,val,test}.json`: 360 / 70 / 70 pairs = 720 / 140 / 140 source videos (counted from the files). |
| Not addressed by the terms | Publishing models trained on the data; serving such a model publicly; publishing aggregate metrics (citation guidance implies publications are expected). |

**Open questions**

1. Is an applicant without an academic lab or supervisor eligible? The form
   makes both required fields.
2. Does a self-directed portfolio project count as "educational purposes"?
3. May weights trained on FF++ be published, or served in a public demo?
4. May a public README show illustrative frames? (Default: no.)

---

## 3. Celeb-DF v2

| Field | Value |
|---|---|
| Sources | Repository: `https://github.com/yuezunli/celeb-deepfakeforensics` (`README.md`; no licence file in the repository). Terms and request form: `https://forms.gle/2jYBby6y1FBU3u6q9` (resolves to a Google form titled "Terms to use Celeb-DF"). |
| Date checked | 2026-10-24 |
| Data terms | (1) "non-commercial research purposes only". (2) Source videos are from the Internet and not the authors' property. (3) Must not "reproduce, duplicate, copy, sell, trade, resell or exploit" any portion of the videos **or any derived data**, for any purpose. (4) Must not further "copy, publish or distribute" any portion of the dataset **or any derived data**, for any purpose. (5) User indemnifies the authors. (6) Publications using the dataset must cite the paper. (7) Access can be terminated at any time. |
| Access | Google form (or a Tencent form). Asks for an **academic** email address, name, affiliation, city, country, purpose description, version, and a signature. |
| Composition | 590 celebrity real videos, 300 additional YouTube real videos, 5,639 synthesized videos, `List_of_testing_videos.txt` with 518 videos (repository README). |
| Size | Not stated in the README. |
| Privacy | The README offers removal on request for people who appear in the dataset. |

**Open questions**

1. Do published aggregate metrics (AUC, ECE) count as "derived data"? Term 6
   implies results may be published with a citation; terms 3–4 read literally
   are broader. Ask the authors before publishing any Celeb-DF number.
2. Are local crops, cached features and per-video scores, kept private,
   covered by permitted research use? (Assumed yes for local processing only;
   confirm.)
3. Is an applicant without an academic email eligible?

**Handling rule until answered:** Celeb-DF crops, features, per-video scores
and error-analysis material never leave local storage, and no Celeb-DF number
is published.

---

## 4. DINOv2 ViT-B/14

| Field | Value |
|---|---|
| Sources | `https://github.com/facebookresearch/dinov2` (`README.md` "License" section, `LICENSE`, `MODEL_CARD.md`, commit history of `LICENSE`). `https://huggingface.co/facebook/dinov2-base` (model card metadata via the Hugging Face API). |
| Date checked | 2026-10-24 |
| Licence | README: "DINOv2 code and model weights are released under the Apache License 2.0." Repository `LICENSE` is Apache-2.0. `MODEL_CARD.md`: Apache License 2.0. Hugging Face `facebook/dinov2-base` card: `apache-2.0`. |
| History | Commit `81b2b641` (2023-08-31), "Update license everywhere", changed code and models from CC-BY-NC to Apache 2.0. |
| Different licences in the same repository | XRay-DINO weights: FAIR Noncommercial Research License (`LICENSE_XRAY_DINO_MODEL`). Cell-DINO code and models have their own licence files (`LICENSE_CELL_DINO_CODE`, `LICENSE_CELL_DINO_MODELS`; not read). **Neither is used.** |
| Pinned checkpoint | Hugging Face `facebook/dinov2-base`, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, `model.safetensors`, 346,345,912 bytes, SHA-256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`. |
| Official alternative | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`, 346,378,731 bytes, last modified 2023-04-13 (HTTP `HEAD` only; not downloaded). |
| Training data | LVD-142M (`MODEL_CARD.md`). |
| Obligations | Apache-2.0: keep the licence and any NOTICE when redistributing; state changes. Not redistributed by this project; weights stay in local `artifacts/`. |

**Open questions**

1. Is the Hugging Face `model.safetensors` numerically identical to the
   official `.pth`? Checking it needs both files downloaded; it can be done in
   smoke test S0 only if that extra download is approved.
2. The official `.pth` predates the relicence. The current `LICENSE` and README
   state Apache-2.0 for the weights; no separate statement for older downloads
   was found.

---

## 5. Not yet checked

| Asset | Needed by | Check before |
|---|---|---|
| CLIP ViT-B/16 weights | M2 comparison baseline | M2 |
| torch, transformers, scikit-learn | `train` extra | M0d (smoke test S0) |
| EfficientNet (timm) and ImageNet-derived weights | M6 | M6 |
| ConvNeXt V2 weights | Option C only | Before any use |
| DFDC, DeeperForensics-1.0, WildDeepfake | Deferred benchmarks | Before any request |
