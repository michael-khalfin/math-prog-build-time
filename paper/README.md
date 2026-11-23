# Paper

*The Oracle–Sandbox Method: An LLM Agent Builds a Validated Pyomo-to-Gurobi Transpiler*
(submitted to CPAIOR 2027, Springer LNCS format).

| Path | Contents |
|---|---|
| `writeup/paper.tex` | Main document |
| `writeup/sec_epidemic.tex` | Section 4.2, the epidemic case study (`\input` by `paper.tex`) |
| `writeup/references.bib` | Bibliography |
| `writeup/llncs.cls`, `writeup/splncs04.bst` | Springer LNCS class and bibliography style |
| `writeup/paper.pdf` | Compiled paper |
| `figstyle.tex` | Shared TikZ style for Figures 1–3 and 5 |
| `diagram1/`–`diagram3/` | Figures 1–3 (TikZ, `\input` by `paper.tex`) |
| `arxiv_notes.md` | The only differences between this version and the arXiv version |

## Building

From `writeup/`:

```sh
latexmk -pdf paper.tex
```

or by hand: `pdflatex paper`, `bibtex paper`, then `pdflatex paper` twice.
The template's fonts (`newtxtext`, `newtxmath`) come with a full TeX Live install.
