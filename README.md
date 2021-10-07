## DeepTriage
Implementation of 'DeepTriage: Exploring the Effectiveness of Deep Learning for Bug Triaging'

### File Contents
- `preprocess.py` includes text cleaning and tokenization parts.  
- `dataset.py` includes dataset  reading and slicing methods for chronological cross validation.
- `dbrnna.py` is the model implementation in Keras.
- `main.py` includes example method calls.

###  How to use
 1. You need a Python version of 3.6.x or later.
 2. Install required packages (Using a virtual environment is recommended).
	`pip install -r requirements.txt`
 3. Clone the repository.
	`git clone https://github.com/hacetin/deep-triage.git`
 4. Download datasets into the repository as following:
	 - Download `deep_data.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tZlIzRjVXRTA1YjA/view?usp=sharing&resourcekey=0-nGbGv3dUSNwR2SphE_X6Ig) and `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json`, `classifier_data_20.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tc2t0aTFyTGhBOTA/view?usp=sharing&resourcekey=0-gz3rhBj22o03rk3_Xnpm5A), then put them into **data/google_chromium** folder.
	 - Download `deep_data.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tZkZRblM2cGRXc3c/view?usp=sharing&resourcekey=0-5_rsgTX54eUcnojnN_0MNg) and `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json`, `classifier_data_20.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tc1pkREhFQVNYczA/view?usp=sharing&resourcekey=0-q6zYsLPkZPSmF1DL2hJ5lQ), then put them into **data/mozilla_core** folder.
	 - Download `deep_data.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tVngxY0o5cnQ3MTg/view?usp=sharing&resourcekey=0-f5cbZhUOx2LSKlcknjXdnw) and `classifier_data_0.json`, `classifier_data_5.json`, `classifier_data_10.json`, `classifier_data_20.json` from [here](https://drive.google.com/file/d/0Bz07ySZGa87tOTB0eXBrVHRfWDQ/view?usp=sharing&resourcekey=0-HJGIOGz2BgWe9H3DdY1G3A), then put them into **data/mozilla_firefox** folder.
 4. Run `main.py`.
	```python
	cd deep-triage
	python main.py
	```

###  Contribution
Any contribution (pull request etc.) is welcome.



## Datasets
Here are the links for the datasets:

- [Google Chromium](https://drive.google.com/file/d/0Bz07ySZGa87tdENrZjAxelBPdFE/view?usp=sharing&resourcekey=0-wtiL-j5GT5XYS2LTjjSjFw) - 383,104 bug reports
- [Mozilla Core](https://drive.google.com/file/d/0Bz07ySZGa87tSkVDcWoybWtuNHc/view?usp=sharing&resourcekey=0-SvdReUneUjZgZjgDxY7EeA) - 314,388 bug reports
- [Mozilla Firefox](https://drive.google.com/file/d/0Bz07ySZGa87tXzB3cDlHWm9OQWc/view?usp=sharing&resourcekey=0-kzn3rDnULxdsdC68QcY0LQ) - 162,307 bug reports


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
