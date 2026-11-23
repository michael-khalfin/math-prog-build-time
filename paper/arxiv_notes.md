# Differences between the CPAIOR submission and the arXiv version

The arXiv version is the CPAIOR paper (`writeup/paper.tex`) with the two tables
below restored to full length. Everything else is identical.

## 1. Table 4 (constraint shapes): add two rows before "Grouped sum"

```latex
Direct bound & $x_i \le b_i$ & One nonzero per row (identity-like) \\
Linear combination & $\sum_i (\alpha x_i + \beta y_i) \le b$ & One block per variable term, each with its own coefficient \\
```

## 2. Table 5 (build times): restore the second and third size of each family

Supply–Demand, after the 2300^2 row:
```latex
\(3300^2\) & 10.89 & 1.955 & 0.410 & 0.766 & 0.107 \\
\(4700^2\) & 22.09 & 6.328 & 0.900 & 1.605 & 0.227 \\
```
Multi-Commodity Network Flow, after the n=5000 row:
```latex
\(n=7500,k=225\) & 10.13 & 2.340 & 0.602 & 0.900 & 0.166 \\
\(n=11000,k=285\) & 18.81 & 8.707 & 1.272 & 1.741 & 0.322 \\
```
Bill-of-Materials, after the p=10k row:
```latex
\(p=15k,c=30k,t=100\) & 4.50 & 1.479 & 0.337 & 0.328 & 0.098 \\
\(p=22k,c=44k,t=120\) & 7.92 & 2.948 & 0.635 & 0.594 & 0.177 \\
```
