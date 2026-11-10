# IJIST (Wiley, ISSN 0899-9457 / 1098-1098): Submission Requirements for an Original Research Article, 2025–2026, as a Conversion Checklist from the Springer Nature LaTeX Template

Your ConvNeXt-Tiny + Swin-Tiny skin-disease ensemble is squarely in scope for IJIST, and the journal's rules for a Research Article are easy to meet. The limits are a 5,000-word main text, a 300-word abstract, up to seven keywords, and no more than 15 figures and tables combined.\[1\]\[2\] You also need a running title under 40 characters, a Data Availability Statement, an IRB/ethics statement and a PDF cover letter.\[1\] Because IJIST offers "Free Format" submission, you can submit a compiled PDF from LaTeX at first submission without reformatting references.\[1\]\[3\] The real risks are editorial, not typographic: about half of all IJIST rejections are desk rejections, and the Editor screens for formatting compliance, scope, writing quality and methodological rigor in ML reporting.\[4\]\[5\]

**How to read the tags.** Each point is marked:
- **[Stated]**: written in the IJIST Author Guidelines, the Overview page or the Wiley Journal Finder.
- **[Wiley-general]**: from Wiley-wide policy pages.
- **[Inferred]**: deduced from IJIST articles published in 2024–2026.
- **[Unverified]**: could not be confirmed.

The IJIST guidelines page (https://onlinelibrary.wiley.com/page/journal/10981098/homepage/forauthors.html) blocks automated fetching. Points tagged [Stated] that come from that page were checked against indexed extracts of the live page, not a full read.

## TL;DR
- **Hard limits for a Research Article [Stated]:**
  - main text 5,000 words; above 6,000 words needs prior Editor-in-Chief approval requested in the cover letter
  - abstract of at most 300 words
  - at most 7 keywords
  - at most 15 figures and tables combined
  - running title under 40 characters; no abbreviations in the title
  - a Data Availability Statement, an IRB statement and a PDF cover letter
  - ORCID required; peer review is single-anonymized
- **Conversion path:** drop the Springer `sn-jnl` class. Rebuild in Wiley's NJD LaTeX template (optional but recommended), or keep a clean `article`-class file.
  - At first submission, upload the compiled PDF as "Main Document – LaTeX PDF" and the sources as "LaTeX Supplementary File".
  - References may be in any consistent style at first submission. IJIST's nominal style is APA, but published IJIST papers use numbered references, so numbered citations are safe.
- **Biggest acceptance levers:** IJIST is a subscription journal with an optional OnlineOpen APC of US$4,190, a 3.0 Impact Factor and a 12% acceptance rate. What decides the outcome is methodological reporting:
  - patient/image-level split integrity across the five aggregated datasets
  - confidence intervals and statistical tests
  - fair comparison against the state of the art on identical data
  - calibration and per-skin-tone subgroup analysis
  - public code
  - a CLAIM-style checklist

---

## 1. Article types and length limits
- **Research Article [Stated]:** "reports of new research findings or conceptual analyses that make a significant contribution to knowledge."\[2\]
  - Word limit 5,000; abstract 300 words.\[2\]
  - Other requirements: "Data Availability Statement, IRB Statement, Cover Letter in PDF."\[1\]
- **Over-length papers [Stated]:** "Authors should reach out to our Editor-in-Chief to request approval in their cover letter for the articles whose main text exceeds 6000 words."\[1\]
  - **Interpretation:** treat 5,000 words as the target. 5,000–6,000 words is tolerated. Above 6,000 words needs explicit permission.
  - The word count is of the main text. The guidelines do not spell out exclusions, but abstract, references, legends and tables are conventionally excluded [Inferred].
- **Figures and tables [Stated]:** "Manuscripts should include no more than 15 Illustrations in total (Figures plus Tables)."\[1\]
  - Count carefully. A typical ensemble paper has many tables: dataset composition, ablation, calibration, SOTA comparison, per-class metrics, Fitzpatrick subgroup results. Move overflow to Supporting Information.
- **Other article types [Stated, partial]:** Editorials are "short, invited opinion pieces." The journal also considers "advanced algorithmics, software, databases, replication studies as well as negative results."\[2\]\[3\]
- **References:** no reference-count limit was found [Unverified]. The Editor lists "missing references" as a formatting problem, not excess references.\[5\]

## 2. Abstract
- **Limit [Stated]:** 300 words maximum.\[2\]
- **Format:** no requirement for a structured abstract was found [Unverified/likely unstructured].
  - IJIST research articles published in 2024–2026 use single-paragraph unstructured abstracts under an "ABSTRACT" heading [Inferred].
  - **Recommendation:** one paragraph that covers the following, with no citations and with every abbreviation defined:
    - the problem (under-representation of melanin-rich/Indian skin)
    - the data (the five sources, the total images, the five classes)
    - the method (ConvNeXt-Tiny + Swin-Tiny soft voting)
    - the key numbers (accuracy, macro-F1 or balanced accuracy with 95% CI, ECE before and after calibration)
    - a clinical-relevance sentence
- **Optional "Key points" [Stated]:** "Authors will need to provide no more than 3 'key points', that summarize the key messages of their paper to be published with their article." The guidelines mark this optional.\[2\] Supply three anyway; it costs nothing.

## 3. Keywords
- **[Stated]:** "Up to seven keywords."\[2\]
- If you used research resources with RRIDs, the guidelines say "you must include the RRIDs in the list of keywords."\[1\] This rarely applies to DL papers.
- **Suggested set (7):** skin disease classification; deep learning ensemble; ConvNeXt; Swin Transformer; model calibration; skin of color; Fitzpatrick skin type.
- Separate keywords with semicolons and use lowercase except for proper nouns [Inferred from published style].

## 4. Manuscript structure and section order
**Main document contents [Stated]:**
- A short informative title containing the major keywords, with no abbreviations.\[2\]
- Full author names with institutional affiliations where the work was conducted, and a footnote for any present address that differs.\[2\]
- Acknowledgments.\[1\]
- Abstract (300 words maximum).\[2\]
- Up to seven keywords.
- Main body: "formatted as introduction, materials & methods, results, discussion, conclusion."\[1\]
- References.
- Tables, each complete with title and footnotes.\[1\]
- Figure legends: "Legends should be supplied as a complete list in the text."\[1\]

**Discussion vs Results:** the stated main-body format lists Results and Discussion as separate sections [Stated].\[1\] Keep them separate. Do not use a combined "Results and Discussion".

**Recommended final order** (statement placement is inferred from IJIST articles published in 2025, e.g., e70224, October 2025):
1. Title page: title, running title, authors, affiliations, corresponding author, ORCIDs.
2. Abstract.
3. Keywords.
4. Key points (optional).
5. 1 Introduction.
6. 2 Materials and Methods: datasets, preprocessing, models, training, calibration, evaluation, statistics.
7. 3 Results: main results, ablation, calibration, subgroup analysis.
8. 4 Discussion, including limitations.
9. 5 Conclusion.
10. Author Contributions (CRediT).\[6\]
11. Acknowledgments, with funding sources listed here per the guidelines.\[1\]
12. Funding (published papers show a separate "Funding" line) [Inferred].\[6\]
13. Ethics Statement.
14. Conflicts of Interest.
15. Data Availability Statement.
16. References.
17. Tables, if not embedded.
18. Figure legends list.
19. Supporting Information legends.

**Numbering:** published IJIST articles use numbered headings ("1 | Introduction") [Inferred]. Wiley's typesetters apply these, so plain numbered sections are fine.

**What to remove from the Springer template:** Springer's "Declarations" block (Funding, Competing interests, Ethics approval, Consent, Data availability, Materials availability, Code availability, Author contribution).\[7\] Replace it with the separate Wiley-style headed statements listed above.

## 5. Title page
**[Stated]:**
- "A brief informative title containing the major key words. The title should not contain abbreviations."\[1\]
  - This matters for you: spell out the architectures, or describe them generically ("a convolutional–transformer soft-voting ensemble"). Avoid bare "CNN" or "ViT".
- "A short running title of less than 40 characters."\[1\] Example: "Calibrated Ensemble for Skin of Color", which is 37 characters.
- Co-author details are required: "including affiliation and email address."\[1\]
- Affiliations: "institutional affiliations where the work was conducted, with a footnote for the author's present address if different."\[2\]
- **ORCID:** "This journal requires ORCID."\[1\] At minimum the submitting/corresponding author must link an ORCID iD in Research Exchange. Collecting iDs for all co-authors is best practice [Wiley-general].
- **Corresponding author:** give full postal address and email. IJIST articles print a "Correspondence:" line with the email [Inferred].\[8\]
- **Title length:** no character or word limit is stated [Unverified]. Aim for 20 words or fewer.

## 6. References and citations
- **[Stated]:** "This journal uses APA reference style; as the journal offers Free Format submission, however, this is for information only and you do not need to format the references in your article."\[1\]
- **[Stated]:** "References may be submitted in any style or format, as long as it is consistent throughout the manuscript."\[1\]
- **What IJIST actually prints [Inferred]:** recent IJIST articles (2024–2025) print numbered references in a Wiley house style. An example from an IJIST 2024 editorial: "M. L. Seghier, 'Ten Simple Rules for Reporting Machine Learning Methods' Implementation and Evaluation on Biomedical Data,' International Journal of Imaging Systems and Technology 32 (2022): 5–11."\[8\]\[9\]
  - **Conflict to note:** the guideline says APA, but printed papers are numbered. Wiley's copyeditors restyle references at production, so this does not affect you at submission.
  - **Recommendation:** keep numeric citations in square brackets with your existing BibTeX. In the NJD template, pick the numbered option (e.g., `vancouver`/`WileyNJD-AMA`) or keep `unsrtnat`. Consistency is all that is required.
- **DOIs:** include them wherever available. Wiley production links references through Crossref, and printed IJIST references carry DOIs (e.g., "https://doi.org/10.1002/ima.22973") [Inferred/Wiley-general].\[10\]
- **Preprints and arXiv:** acceptable. IJIST explicitly accepts preprints, and printed IJIST papers cite arXiv.\[1\] Cite the peer-reviewed version where one exists (e.g., the ConvNeXt CVPR 2022 paper and the Swin ICCV 2021 paper, not the arXiv versions).
- **Datasets:** Wiley's "Expects Data" policy calls for data citations in the reference list [Wiley-general].\[11\]
  - Cite each dataset by its descriptor paper. Where possible, also cite the dataset itself: creator, year, title, repository, version, URL/DOI, access date. This covers Fitzpatrick17k, PAD-UFES-20 (Data in Brief paper plus Mendeley Data DOI), DermaCon-IN, the Kaggle Melanoma Cancer Image Dataset and the Skin Disease Image Dataset.
  - Kaggle datasets usually have no DOI. Cite them as "Creator. (Year). *Title* [Data set]. Kaggle. URL (accessed date)."

## 7. Figures and tables
- **[Stated]:** "Figures should be uploaded in the highest resolution possible. If the figures are not of sufficiently high quality your manuscript may be delayed."\[1\]
- **[Stated]:** "Figures should be uploaded as separate files." Also: "Figures, supporting information, and appendices should be supplied as separate files."\[1\]
  - In Free Format, the guidelines also allow "an editable file including text, figures, and tables, or separate files—whichever you prefer."\[1\]
  - **Practical rule:** embed figures in the review PDF, and also upload each figure as its own file at revision.
- **Legends [Stated]:** "Figures and tables should have legends."\[1\] Legends must also appear as a complete list in the main text.\[1\]
- **Formats and resolution [Wiley-general, from Wiley's "Guidelines for the Preparation of Figures"]:**
  - Line art (plots, flowcharts, reliability diagrams): EPS or PDF preferred.
  - Images (dermatology photographs, Grad-CAM overlays): TIFF, PNG or EPS.\[12\]
  - Name files "Figure_1.tiff" and so on.\[12\]
  - For peer review, line art must be "legible when viewed as an 80 mm or 1800 pixel width".\[12\]
  - Post-acceptance, Wiley's usual targets are 300 dpi for photographs/halftones, 600 dpi for combination figures, and 600–1200 dpi for line art at final size.
- **Color:** colour is free online [Wiley-general]. Use colour-blind-safe palettes for confusion matrices and reliability diagrams.
- **Patient images:** use only images whose dataset licence permits reproduction. Note this in the legend (e.g., "Image from PAD-UFES-20, CC BY 4.0") [Inferred best practice].
- **Tables:** submit as editable text (LaTeX `tabular`), never as images. Each table needs a title and footnotes [Stated: "Tables (each table complete with title and footnotes)"].\[1\] Avoid vertical rules (Wiley house style) [Wiley-general].

## 8. Equations, units, abbreviations, spelling
- **Equations:** typeset in LaTeX math, numbered consecutively. Define every symbol, e.g., the soft-voting rule p̂ = w·p_ConvNeXt + (1−w)·p_Swin, temperature scaling, and the ECE formula. IJIST editorials ask authors to give "the exact equation used" for metrics [Stated in Seghier's "Image Segmentation Evaluation With the Dice Index: Methodological Issues," IJIST, October 2024, DOI 10.1002/ima.23203].
- **Units:** use SI units throughout (image sizes in pixels; times in s or ms; memory in GB) [Wiley-general].
- **Abbreviations:**
  - none in the title [Stated]
  - define at first use in both the abstract and the main text
  - an abbreviation list is optional [Inferred]
- **English:** IJIST does not mandate British or American spelling [Unverified]. Be consistent. Published IJIST articles mostly use American spelling [Inferred].
- **Writing quality:** the Editor screens for it personally before review [Stated in the 2022 desk-rejection editorial].\[5\]

## 9. LaTeX submission
- **Accepted [Stated]:** "The main manuscript file can be submitted in Microsoft Word (.doc or .docx) or LaTex (.tex) format."
  - Upload the compiled PDF as "Main Document – LaTeX PDF".
  - Upload every file referenced by the main document (.tex, .bib/.bbl, figures, .cls/.sty) as "LaTeX Supplementary File".\[1\]
  - "When submitting a Latex Main Document, you must also provide a PDF version of the manuscript for Peer Review."\[1\]\[13\]
- **Free Format submission: Yes [Stated in the Journal Finder].**\[1\]\[3\] At first submission you can upload the PDF compiled from your current file. You do not need to reformat references.
  - Even so, strip all Springer-specific branding and the Declarations block before submitting.
- **Template:** IJIST does not mandate a journal-specific template [Unverified]. Wiley's recommended template is the **New Journal Design (NJD) LaTeX authoring template** (`WileyNJDv5.cls`), available from Wiley Author Services and Overleaf.\[14\]
  - Wiley's NJD page links a spreadsheet listing each journal's reference style, font and column options.\[14\]
  - Wiley notes that "many journals at Wiley accept LaTeX files at submission, while others will require you to upload a PDF at submission and supply the native LaTeX files at a later date."\[14\]\[15\]
  - Wiley's LaTeX helpdesk: latexsupport@wiley.com.\[14\]
- **At revision or acceptance [Wiley-general/Inferred]:**
  - full editable source (.tex plus .bib or the .bbl)
  - separate high-resolution figure files
  - Supporting Information as separate files
  - a clean manuscript and a marked-up (changes-highlighted) manuscript
  - a point-by-point response letter
  - CRediT roles entered in the system\[13\]

## 10. Mandatory statements
| Statement | Status | What to write for your study |
|---|---|---|
| **Data Availability Statement** | **Required** for Research Articles [Stated]. IJIST policy tier: **"Expects data sharing"** [Stated].\[1\] A DAS must be published; actually sharing data is optional but encouraged [Wiley-general]\[1\]\[11\]\[16\] | Use Wiley's template for public-domain resources: "The data that support the findings of this study are openly available in [repository name] at [URL], reference number [reference number]." or "These data were derived from the following resources available in the public domain: [list resources and URLs]."\[17\]\[18\] List all five datasets with URLs/DOIs. Add your split files (image IDs per train/val/test) and code via a GitHub release archived on Zenodo with a DOI |
| **Ethics / IRB statement** | **Required** for Research Articles [Stated].\[1\] "The journal requires that you include in the manuscript details IRB approvals, ethical treatment of human and animal research participants, and gathering of informed consent, as appropriate" [Stated]\[1\]\[2\] | Example: "This study used only publicly available, de-identified image datasets (Fitzpatrick17k, PAD-UFES-20, DermaCon-IN, …) released under their respective licences; no new data were collected from human participants. Ethical approval and informed consent were therefore not required [or: the [Institution] IRB granted an exemption, ref. no. …]." An institutional exemption letter is the safest choice for an Indian-skin study |
| **Patient consent statement** | Listed among Free Format ethics statements [Stated]\[1\] | "Not applicable: secondary analysis of publicly released, de-identified datasets; consent was obtained by the original dataset creators where required." |
| **Permission to reproduce material** | Listed [Stated]\[1\] | Confirm the licence for every dermatology image shown in figures |
| **Conflict of Interest** | "You will be expected to declare all conflicts of interest, or none, on submission" [Stated]\[1\] | IJIST print form [Inferred, e70224]: "The authors declare no conflicts of interest."\[6\] |
| **Funding** | "You should list all funding sources in the Acknowledgments section" [Stated];\[1\] published papers also show a "Funding:" line [Inferred]\[6\] | Give agency names and grant numbers, or "This research received no specific grant from any funding agency." |
| **Author Contributions (CRediT)** | IJIST-specific wording [Unverified]; Wiley journals collect CRediT roles in the submission system [Wiley-general].\[13\]\[19\]\[20\] IJIST 2025 papers print role-based contribution lists [Inferred, e.g., "Javed Rashid: project administration; validation; conceptualization…"]\[6\] | List the 14 CRediT roles per author\[21\] |
| **Acknowledgments** | Required section [Stated]\[1\] | Non-author help, compute resources, dataset creators |
| **AI / LLM use disclosure** | IJIST-specific clause [Unverified]. Wiley policy: AI tools cannot be authors; substantive use must be described "transparently and in detail, in the Methods or Acknowledgements section" [Wiley-general].\[22\] The IJIST Editor has published on ChatGPT-assisted language editing (2023) and on harnessing AI for reporting (2025)\[23\]\[24\] | Example: "During manuscript preparation the authors used [tool, version] for language editing only; all content was reviewed and the authors take full responsibility." Do not disclose the ensemble itself here; that belongs in Methods |
| **Preprint** | Accepted [Stated]: "IMA will consider for review articles previously available as preprints"\[1\] | Disclose any arXiv/medRxiv preprint in the cover letter and link the final article after publication |

## 11. Cover letter, reviewers, highlights and Supporting Information
- **Cover letter [Stated]:** "A cover letter in PDF is required for Research Articles."\[1\] Include:
  - the novelty claim in two or three sentences
  - scope fit (AI classification, skin of color)
  - the word count, plus an approval request if over 6,000 words
  - a statement that the work is not under consideration elsewhere
  - any preprint DOI
  - a conflict summary
- **Suggested or opposed reviewers:** no IJIST-specific requirement was found [Unverified]. Research Exchange typically offers optional fields. Suggest 3–4 independent dermatology-AI experts, with no co-authors in the past three years and none from your institution.
- **Highlights or graphical abstract:** none required [Inferred]. The optional "key points" (at most 3) serve that purpose [Stated]. A cover-image option exists [Stated: front-cover images get "35% higher readership"].\[1\]
- **Supporting Information [Stated]:** supplied as separate files and hosted free on Wiley Online Library. Put the following here to stay within the 15-illustration cap:
  - per-dataset class distributions
  - hyperparameter tables
  - extra Grad-CAMs
  - full per-class confusion matrices
  - a CLAIM checklist
- **Submission portal [Stated]:** https://authors.wiley.com/journal/IMA (Wiley Research Exchange).\[1\] The Overview page still shows the legacy ScholarOne address (mc.manuscriptcentral.com/ima).\[25\] The Author Guidelines name Research Exchange as current.

## 12. Peer review, open access, metrics and scope fit
- **Peer review [Stated]:** single-anonymized (single-blind).\[1\] Reviewers see author names, so no anonymization is needed. Manuscripts are "screened by the Editor for methodological innovation and impact" and those deemed suitable are "reviewed by at least two referees."\[3\]
- **Editor-in-Chief:** Mohamed L. Seghier, Khalifa University [Stated].\[4\]\[26\]\[27\]
- **Business model [Stated]:** "This journal is a Subscription journal that offers an Open Access option."\[1\] The OnlineOpen APC is **US$4,190**.\[3\]
  - Check whether your institution is covered by a Wiley transformative agreement. For Indian authors, coverage is institution-specific: a Wiley press release (Noida, 15 July 2024) announced an IIT Kharagpur agreement "effective through the entire year 2024" that lets affiliated researchers "publish open access in Wiley's portfolio of over 1,300 hybrid open access journals," and Wiley reports "more than 100 agreements, covering more than 2,800 institutions"; whether such agreements were renewed beyond 2024 is unconfirmed.
  - Subscription publication costs nothing.
- **Speed and selectivity [Stated, Wiley Journal Finder]:**
  - "Submission to first decision": **5.0 days**.\[3\] This reflects fast desk decisions, not full review.
  - Acceptance rate: **12%**.\[3\]
  - Acceptance to online publication: **12** (days).\[3\]
  - Full-text views: 121,240 (2024).\[3\]
- **Impact Factor:** **3.0** on the current Wiley product page and Journal Finder [Stated].\[3\]\[26\] An earlier Wiley page snapshot showed 2.5, the previous year's JIF.\[28\]
  - SCImago lists SJR 2025 at 0.601 (best quartile Q2; 0.561 in 2024). The VU Amsterdam Journal Publishing Guide gives a 2024 Scopus CiteScore of 6.7 (SNIP 0.902; ranked 42/346, Q1, in Radiology, Nuclear Medicine & Imaging); one third-party aggregator shows 6.3.
- **Scope fit [Stated]:** the revised aims say the journal's aim is "to provide a leading venue for research on AI/ML in medical imaging."\[3\]\[26\] It explicitly lists:\[3\]
  - "classification"
  - "computer-aided diagnosis"\[26\]
  - "optical imaging"\[3\]
  - "benchmarking, reproducibility, open datasets, and performance evaluation of imaging algorithms"\[3\]
  - It stresses "clear clinical value."\[3\]
  - **Verdict:** a calibrated, fairness-oriented ensemble for under-represented skin tones fits well. Frame the contribution around clinical reliability (calibration, subgroup equity), not raw accuracy.

## 13. Reporting expectations for DL/medical imaging, and desk-rejection triggers
**Desk rejection at IJIST.** Mohamed L. Seghier's editorial "Demystifying desk rejection: A call to action for our authors" (IJIST 32(3):701–703, first published 4 April 2022, DOI 10.1002/ima.22733) gives the context, noting that "given the shortage of available reviewers… my priorities were to ensure a reasonable number of review requests and to speed up the review process":
- Submission volume "almost quadrupled, from 217 papers received in 2018 to 845 papers in 2021."\[4\]\[5\]
- Desk rejection accounts for "nearly 50% of all rejections at IMA."\[4\]\[5\]

The named recurring triggers are:
1. **Inadequate formatting:** "missing abstracts, long papers that far exceed the word count limit, papers with too many illustrations, missing references, figures without legends, or wrong file types."\[5\]
2. **Out-of-scope topics.**\[5\]
3. **Poor writing quality:** "not only about grammatical soundness but also about flow and cohesion"; the Editor checks this personally.\[5\]

Further reasons appear on pages 702–703, which could not be retrieved [Unverified]. The current guidelines tell authors to read this editorial "before submitting your manuscript to avoid desk-rejection" [Stated].\[1\]

**Editor-endorsed ML reporting standards [Stated in IJIST editorials]:**
- **"Ten simple rules for reporting machine learning methods implementation and evaluation on biomedical data"** (Seghier, IJIST 32(1):5–11, 2022). It targets these failures:
  - vague aims and scope
  - missing details about the data
  - ambiguous preprocessing procedures
  - unclear implementation
  - poor validation and testing
  - invalid comparisons between methods
  - no rationale for performance metric choices\[5\]
  - Its abstract states that "All stages of data processing and method's performance evaluation should be clearly described, and parameters and metric choices must be justified"; the wording of the individual rules was not retrieved.
- **"Harnessing AI for Comprehensive Reporting of Medical AI Research"** (IJIST 2025;35(2):e70047).\[23\]\[29\] It names MAIC-10, **CLAIM**, STARD-AI, MI-CLAIM, MINIMAR, RQS, QAMAI and TRIPOD+AI as the relevant checklists.\[24\]
- **Seghier's "Image Segmentation Evaluation With the Dice Index: Methodological Issues" (IJIST, October 2024, DOI 10.1002/ima.23203):** "Only compare between methods from prior work using Dice indices calculated on exactly the same data."\[9\] The same principle applies to accuracy and F1.

**What this means for your paper [Inferred, decision-ready]:**
1. **Data provenance and leakage.**
   - Give a table of images per source × class × Fitzpatrick type (or ITA/Monk proxy).
   - Report deduplication across sources. Kaggle skin sets often contain ISIC/HAM10000 duplicates, so use perceptual hashing.
   - Use patient-level splitting where IDs exist (PAD-UFES-20 has patient IDs).
   - Release the split files.
2. **Validation.**
   - Use a fixed held-out test set and report seeds.
   - Prefer k-fold CV with mean ± SD.
   - Ideally run leave-one-dataset-out or external testing (e.g., train without DermaCon-IN and test on it) to show generalization to Indian skin.
3. **Statistics.**
   - Give 95% bootstrap CIs for accuracy, macro-F1, balanced accuracy and AUROC.
   - Use McNemar's or DeLong tests for ensemble vs single models.
   - Report ECE/MCE and Brier score with reliability diagrams, before and after temperature scaling.
4. **Ablation.** Include ConvNeXt alone, Swin alone, equal versus tuned soft-voting weights, and the effect of augmentation and class-balancing choices.
5. **SOTA comparison.** Retrain baselines on your exact splits. Do not copy numbers from papers that used different data; the Editor flags this explicitly.
6. **Fairness.** Report per-skin-tone subgroup metrics, which is the core of your claim, with CIs. Acknowledge the small-n subgroups.
7. **Reproducibility.**
   - Public code (GitHub + Zenodo DOI) with environment file and weights.
   - Hardware, training time and inference latency.
   - A CLAIM 2024 checklist uploaded as Supporting Information.
8. **Clinical value.** Discuss the intended use (triage or teledermatology), the limits of web-scraped images, and the absence of prospective validation.

---

## Conversion checklist: Springer `sn-jnl` to IJIST
- [ ] Replace `\documentclass{sn-jnl}` with `WileyNJDv5` (or plain `article`). Remove `\bmhead`, `\backmatter` and the "Declarations" section.
- [ ] Title has no abbreviations; running title is under 40 characters.
- [ ] Authors, affiliations, emails and corresponding author are complete; ORCID is linked in Research Exchange.
- [ ] Abstract is 300 words or fewer, one paragraph, no citations. Up to 7 keywords. Optional 3 key points.
- [ ] Sections are Introduction / Materials and Methods / Results / Discussion / Conclusion, with Results and Discussion separate.
- [ ] Main text is 5,000 words or fewer (6,000 or more only with EIC approval).
- [ ] Figures + tables total 15 or fewer; the rest go to Supporting Information.
- [ ] Each figure is a separate high-resolution file (TIFF/PNG for images, EPS/PDF for plots). All legends are listed in the text.
- [ ] Tables are editable, with titles and footnotes.
- [ ] References are consistent, numbered, with DOIs. Datasets are cited in the reference list. Peer-reviewed versions are preferred over arXiv.
- [ ] Statements are present: Author Contributions (CRediT), Acknowledgments (with funding), Funding, Ethics/IRB, Patient consent, Conflicts of Interest, Data Availability (Wiley template wording), AI-use disclosure.
- [ ] Cover letter is a PDF: novelty, scope fit, word count, no dual submission, preprint disclosure.
- [ ] Files uploaded: "Main Document – LaTeX PDF" plus "LaTeX Supplementary File(s)".
- [ ] The CLAIM checklist, split files and code DOI are ready.

## Caveats
- The IJIST Author Guidelines page blocked direct retrieval. Points tagged [Stated] were checked against indexed extracts of the live page, so recheck it on the day you submit.
- The 5,000-word limit (article-type table) and the 6,000-word EIC-approval note both appear on the page. Treat 5,000 as the target.
- The guidelines name APA style while published papers are numbered. Under Free Format either style is accepted at submission.
- IJIST-specific wording on CRediT timing, AI disclosure, suggested reviewers and abstract structure could not be verified. The entries above use Wiley-wide policy and published-paper conventions.
- The "5.0 days to first decision" figure is dominated by desk decisions. Expect full-review rounds to take considerably longer.
- Third-party template sites (e.g., SciSpace) list IJIST templates dated 2020.\[30\] Do not rely on them; use Wiley's NJD template.

## Sources

1. [International Journal of Imaging Systems and Technology](https://onlinelibrary.wiley.com/page/journal/10981098/homepage/forauthors.html)
2. [International Journal of Imaging Systems and Technology](https://onlinelibrary.wiley.com/page/journal/10981098/homepage/ForAuthors.html)
3. [International Journal of Imaging Systems and Technology](https://www.wiley.com/en-us/journal-finder/journal/ima)
4. [International Journal of Imaging Systems and Technology](https://onlinelibrary.wiley.com/doi/full/10.1002/ima.22733)
5. [Demystifying desk rejection: A call to action for our authors | Request PDF](https://www.researchgate.net/publication/359712021_Demystifying_desk_rejection_A_call_to_action_for_our_authors)
6. [International Journal of Imaging Systems and Technology](https://onlinelibrary.wiley.com/doi/10.1002/ima.70224)
7. [Physics-informed Partitioned Coupled Neural Operator for Complex Networks](https://arxiv.org/pdf/2410.21025)
8. [International Journal of Imaging Systems and Technology](https://onlinelibrary.wiley.com/doi/10.1002/ima.23203)
9. [International Journal of Imaging Systems and Technology](https://onlinelibrary.wiley.com/doi/full/10.1002/ima.23203)
10. [Diagnostic Performance of Artificial Intelligence in Detecting Periodontal Diseases Using Cone‐Beam Computed Tomography: A Systematic Review - Dong - Oral Diseases - Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1111/odi.70528?af=R)
11. [Paving the Way to Open Data](https://direct.mit.edu/dint/article/1/4/368/9984/Paving-the-Way-to-Open-Data)
12. [Updated 1 September 2016 Guidelines for the Preparation of Figures Overview](https://onlinelibrary.wiley.com/pb-assets/25673173/Guidelines%20for%20the%20Preparation%20of%20Figures-1573182704117.pdf)
13. [Journal of Medical Imaging and Radiation Oncology](https://onlinelibrary.wiley.com/page/journal/17549485/homepage/forauthors.html)
14. [Wiley's New Journal Design (NJD) LaTex authoring template](https://authorservices.wiley.com/author-resources/Journal-Authors/Prepare/new-journal-design.html)
15. [Wiley's PDF design LaTex authoring template](https://authors.aws.wiley.com/author-resources/Journal-Authors/Prepare/latex-template.html)
16. [The state of play of reproducibility in Statistics: an empirical analysis](https://arxiv.org/pdf/2209.15602)
17. [Data Sharing Policy](https://authors.wiley.com/author-resources/Journal-Authors/open-access/data-sharing-citation/data-sharing-policy.html)
18. [Author Guidelines - Current Protocols - Wiley](https://currentprotocols.onlinelibrary.wiley.com/hub/authorguidelines)
19. [WIREs Author Guidelines](https://wires.onlinelibrary.wiley.com/hub/for_authors)
20. [European Journal of Education](https://onlinelibrary.wiley.com/page/journal/14653435/homepage/forauthors.html)
21. [Submission Guidelines - SciOpen](https://www.sciopen.com/journal/join_journal/submission_guidelines?id=1452804496562978818&issn=2097-0684)
22. [Author Guidelines - Learned Publishing](https://onlinelibrary.wiley.com/page/journal/17414857/homepage/forauthors.html)
23. [dblp: Mohamed L. Seghier](https://dblp.org/pid/28/7494.html)
24. [International Journal of Imaging Systems and Technology](https://onlinelibrary.wiley.com/doi/full/10.1002/ima.70047)
25. [Overview - International Journal of Imaging Systems and Technology - Wiley Online Library](https://onlinelibrary.wiley.com/page/journal/10981098/homepage/productinformation.html)
26. [International Journal of Imaging Systems and Technology](https://www.wiley.com/en-us/International+Journal+of+Imaging+Systems+and+Technology-p-10981098)
27. [International Journal of Imaging Systems and Technology](https://onlinelibrary.wiley.com/doi/10.1002/ima.70047)
28. [International Journal of Imaging Systems and Technology](https://www.wiley.com/en-us/journals/International+Journal+of+Imaging+Systems+and+Technology-p-10981098)
29. [Harnessing AI for Comprehensive Reporting of Medical AI Research - Khalifa University](https://khazna.ku.ac.ae/en/publications/harnessing-ai-for-comprehensive-reporting-of-medical-ai-research/)
30. [International Journal of Imaging Systems and Technology Template - Wiley](https://scispace.com/formats/wiley/international-journal-of-imaging-systems-and-technology/d702bd327c03d496c0ae0ea353659363)
