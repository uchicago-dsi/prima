# Omoleye 2023 Radiology: AI reference

Citation:

Omoleye OJ, Woodard AE, Howard FM, Zhao F, Yoshimatsu TF, Zheng Y, Pearson AT, Levental M, Aribisala BS, Kulkarni K, Karczmar GS, Olopade OI, Abe H, Huo D. **External Evaluation of a Mammography-based Deep Learning Model for Predicting Breast Cancer in an Ethnically Diverse Population.** *Radiology: Artificial Intelligence.* 2023;5(6):e220299. doi: `10.1148/ryai.220299`

Stable identifiers:

- DOI: `https://doi.org/10.1148/ryai.220299`
- RSNA full text: `https://pubs.rsna.org/doi/full/10.1148/ryai.220299`
- RSNA issue TOC entry: `https://pubs.rsna.org/toc/ai/5/6`

Notes:

- The article is labeled **Free access** on the RSNA full-text page.
- The direct RSNA PDF endpoint (`/doi/pdf/...` or `/doi/epdf/...`) is cookie/Cloudflare-protected from CLI access in this environment, so a browser session may still be required to download the PDF file itself.
- The full-text HTML page was successfully accessible via browser tooling during debugging and was used to extract cohort definitions and headline results below.

Key paper details used in Mirai debugging:

- Study title: **External Evaluation of a Mammography-based Deep Learning Model for Predicting Breast Cancer in an Ethnically Diverse Population**
- Dataset: `6435` screening mammograms in `2096` female patients
- Main filtered analysis after excluding `time to cancer < 6 months`: `6266` exams in `2043` patients
- Included examinations were:
  - standard four-view mammograms
  - **For Presentation** mode
  - excluded if implants / foreign devices / burned-in annotations / insufficient follow-up
- Reported Mirai AUCs:
  - unfiltered set: `1y 0.71`, `5y 0.65`
  - after excluding `TTC < 6 months`: `1y 0.64`, `5y 0.63`

Important analysis details from the paper:

- Positive exam at year `x`: cancer developed from mammography until year `x`
- Negative exam at year `x`: confirmed cancer free from mammography until year `x`
- Control exams without sufficient follow-up at year `x` were right-censored for the `x`-year AUC
- DeLong used for most AUC comparisons
- Bootstrap used when subgroup AUCs shared the same control set

Why this matters for the current repo:

- This paper is the main reference point for the CHiMEC Mirai validation/debugging work in:
  - `docs/lab_meeting_mirai_debug.md`
  - `docs/lab_meeting_mirai_debug.html`
- The exact paper-era input/output subsets we compared against live in Omoleye's read-only workspace, but this note records the citation and URLs in-repo so they are easy to recover later.
