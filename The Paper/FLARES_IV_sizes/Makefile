flaresIV: *.tex *.bib Figures/*.png Figures/*.p??
	pdflatex flaresIV
	bibtex flaresIV
	pdflatex flaresIV
	pdflatex flaresIV

mnras: 	*.tex *.bib Figures/*.png Figures/*.pdf
	rm -rf mnras.tar.gz
	tar cvzf mnras.tar.gz Makefile *.tex *.bib Figures/*.png Figures/*.pdf

arxiv: 	*.tex *.bbl Figures/*.png Figures/*.pdf
	rm -rf arxiv.tgz
	tar cvzf arxiv.tar.gz *.tex *.bbl Figures/*.png Figures/*.pdf

overleaf:
	rm -rf overleaf.z
	zip overleaf.z Makefile *.tex *.bbl Figures/*.p??

clean:
	/bin/rm -rf *~ *.bbl *.blg *.dvi *.log *aux main.pdf
