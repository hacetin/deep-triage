## DeepTriage
Implementation of 'DeepTriage: Exploring the Effectiveness of Deep Learning for Bug Triaging'

###  How to use
 1. Install required packages.
	`pip install -U nltk gensim tensorflow keras scikit-learn`
 2. Clone the repository.
	`git clone https://github.com/hacetin/deep-triage.git`
 3. Download datasets into the repository as following:
	 - Download `deep_data.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tZlIzRjVXRTA1YjA/view) and `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json`, `classifier_data_20.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tc2t0aTFyTGhBOTA/view), then put them into **data/google_chromium** folder.
	 - Download `deep_data.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tZkZRblM2cGRXc3c/view) and `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json`, `classifier_data_20.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tc1pkREhFQVNYczA/view), then put them into **data/mozilla_core** folder.
	 - Download `deep_data.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tVngxY0o5cnQ3MTg/view) and `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json`, `classifier_data_20.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tOTB0eXBrVHRfWDQ/view), then put them into **data/mozilla_firefox** folder.
 4. Run `main.py`.
	```python
	cd deep-triage
	python main.y
	```

###  Contribution
Any contribution (pull request etc.) is welcome.
