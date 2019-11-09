init: 
	pip3 install -r requirements.txt
	pip3 install -e .

update: 
	pip3 install -r requirements.txt --upgrade

clean:
	rm -rf *.egg *.egg-info

	find . -type f -name '*.png' -delete
	find . -type f -name '*.xlsx' -delete
	find . -type f -name '*.log' -delete
	find . | grep -E "(__pycache__)" | xargs rm -rf

.PHONY: init