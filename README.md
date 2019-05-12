## DeepTriage
Implementation of 'DeepTriage: Exploring the Effectiveness of Deep Learning for Bug Triaging'

###  How to use
 1. Install required packages.
	`pip install -U nltk gensim tensorflow keras scikit-learn`
 2. Clone the repository.
	`git clone https://github.com/hacetin/deep-triage.git`
 3. Download datasets into the repository as following:
	 - Download `deep_data.json` , `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json` and `classifier_data_20.json` files in [http://bugtriage.mybluemix.net/#chrome](http://bugtriage.mybluemix.net/#chrome) into **data/google_chromium** folder.
	 - Download `deep_data.json` , `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json` and `classifier_data_20.json` files in [http://bugtriage.mybluemix.net/#core](http://bugtriage.mybluemix.net/#core) into **data/mozilla_core** folder.
	 - Download `deep_data.json` , `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json` and `classifier_data_20.json` files in [http://bugtriage.mybluemix.net/#firefox](http://bugtriage.mybluemix.net/#firefox) into **data/mozilla_firefox** folder.
 4. Run `main.py`.
	```python
	cd deep-triage
	python main.y
	```

###  Contribution
Any contribution (pull request etc.) is welcome.
