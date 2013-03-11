TEMPLATE = template.tex
STYLESHEET = style.css
MD = $(wildcard *.md)
TEX = $(patsubst %.md,%.tex,$(MD))
HTML = $(patsubst %.md,%.html,$(MD))
PDF = $(patsubst %.md,%.pdf,$(MD))

.PHONY: clean pdf html

all: pdf html

html: $(HTML)

pdf: $(PDF)

%.html: %.md $(STYLESHEET)
	pandoc $< -s -S -c $(STYLESHEET) -o $@

%.pdf: %.tex
	pdflatex $<

%.tex: %.md $(TEMPLATE)
	pandoc $< -s --template=$(TEMPLATE) -o $@

clean:
	rm -f $(TEX) $(HTML) $(PDF) *.log *.aux *.blg *.bbl *.out
