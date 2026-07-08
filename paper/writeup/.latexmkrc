# minted needs two things that a bare editor build (e.g. VSCode LaTeX Workshop)
# does not provide by default:
#   1. pdflatex invoked with -shell-escape, and
#   2. pygmentize (Pygments) on PATH.
# pygmentize is installed only in the conda 'base' env, so we prepend it here.
# Because latexmk reads this file, any latexmk-driven build -- terminal or
# VSCode -- picks up both, with no per-editor configuration.

$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S';
$ENV{'PATH'} = '/opt/homebrew/Caskroom/miniforge/base/bin:' . $ENV{'PATH'};
