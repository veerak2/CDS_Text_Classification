# CDS Text Classification

The Citizen Data Science program is an initiative by the Muma College of Business made in association with Tableau. The aim of the program is to teach the undergraduate students about deriving insights from data using Tableau. More information about the program can be found in the below link.
https://www.usf.edu/business/centers/analytics-creativity/tableau-citizen-data-science-certificate.aspx

What is the problem?
The Tableau based assignments are embedded in different courses. The assignment contains a dataset and questions pertaining to it. The students find the answers to the questions by creating visualizations using Tableau. The students then make a video explaining their findings. The videos are uploaded on YouTube as unlisted videos.

These videos are then validated by the student assistants in the program. These student assistants look for certain features when they are validating. For instance, one of the requirements is to have a dashboard that is interactive. The goal is to automate this process.

To achieve this, we obtained the transcripts from YouTube from historical data ie assignments that have already been validated. The transcripts will be our main source of signal.

Things completed:
- Obtained 1000 transcripts from YouTube. 
- Preprocessed the corpus using nltk, spacy.
- Designed a simple heuristic for classification by looking for key words such as 'interactive' and 'dashboard' in the transcript. Observed that some of the existing labels were wrong.
- Relabelled the dataset. Implemented a quicker way to label by using IPython widgets.
- Modularized the code. In the process of moving code from notebook to scripts.

To do:
- Experiment with more models. Implement MLflow for tracking experiments. 
- Use FastAPI for backend and streamlit for frontend.
- Experiment with DVC(Data Version Control)
- Containerize the application using Docker.

Given more time:
- Unit testing for code using Pytest
- Explore Great Expectations for testing data

