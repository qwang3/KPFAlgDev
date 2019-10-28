init: 
	pip3 install -r requirements.txt

update: 
	pip3 install -r requirements.txt --upgrade

clean:
	rm -rf *.egg *.egg-info *.png 
	find . | grep -E "(__pycache__)" | xargs rm -rf

.PHONY: init