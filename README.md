## DeepTriage
Implementation of 'DeepTriage: Exploring the Effectiveness of Deep Learning for Bug Triaging'

###  How to use
 1. You need a Python version of 3.6.x or later.
 2. Install required packages.
	`pip install -U nltk gensim tensorflow keras scikit-learn`
 3. Clone the repository.
	`git clone https://github.com/hacetin/deep-triage.git`
 4. Download datasets into the repository as following:
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



## Datasets
Here are the links for the datasets:

- [Google Chromium](https://drive.google.com/file/d/0Bz07ySZGa87tdENrZjAxelBPdFE/view) - 383,104 bug reports
- [Mozilla Core](https://drive.google.com/file/d/0Bz07ySZGa87tSkVDcWoybWtuNHc/view) - 314,388 bug reports
- [Mozilla Firefox](https://drive.google.com/file/d/0Bz07ySZGa87tXzB3cDlHWm9OQWc/view) - 162,307 bug reports


A sample bug report from datasets is given below:


#### Google Chromium:
```json
{
		"id" : 1,
		"issue_id" : 2,
		"issue_title" : "Testing if chromium id works",
		"reported_time" : "2008-08-30 16:00:21",
		"owner" : "",
		"description" : "\nWhat steps will reproduce the problem?\n1.\n2.\n3.\n\r\nWhat is the expected output? What do you see instead?\n\r\n\r\nPlease use labels and text to provide additional information.\n \n ",
		"status" : "Invalid",
		"type" : "Bug"
}
```

#### Mozilla Core and Firefox:
```json
{
		"id" : 1,
		"issue_id" : 91,
		"issue_title" : "document properties cannot be listed",
		"reported_time" : "1998-04-07 23:05:23",
		"owner" : "rickg@formerly-netscape.com.tld",
		"description" : "Created by Till Krech (till@berlin.snafu.de) on Tuesday, April 7, 1998 9:05:23 AM PDT\nAdditional Details :\nthe JavaScript \"for in\" statement does not work on the\ndocument object. At least not in the Linux version.",
		"status" : "VERIFIED",
		"resolution" : "FIXED"
}
```
