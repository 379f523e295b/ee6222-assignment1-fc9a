latexmk = latexmk

default: *.pdf clean

%.pdf: %.tex
	$(latexmk) -pdf -use-make -halt-on-error $<

clean:
	$(latexmk) -c
