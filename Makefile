.PHONY: setup test demo clean
setup:
	bash scripts/setup_kali.sh
test:
	python -m pytest -q
demo:
	secleak demo --out results/demo
clean:
	rm -rf results/demo data/cache
