# content-moderation-cookbook
This repo contains entries for exploring content moderation using LLMs to tackle the problem statement from TikTok's Tech Jam: https://bytedance.sg.larkoffice.com/docx/PLdZdxbXOoTs1Wx3s5klHVySg2w.

# Summary

- The solution uses the Google Review Data: https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews
- It performs some EDA to understand the shape of the data as well as patterns.
- Based on this, we can surmise that the dataset does not contain much in terms of toxic comments or advertisements. 
- The model defines relevance as spam or not spam where spam is defined as reviews that do not contribute enough detail to the review or is not about the restaurant or the food at all. Note that the details need not necessarily be about the restaurant or the food itself - personal anecdotes or experiences can form important signals for the kinds of occasions that people typically
visit the location for and therefore, provide important data points for anyone reading the review. Visitors are able to use the data to determine if the location would help them host similar events / be amenable for similar occasions. Owners can use the data to classify their user groups based on the context behind their visit and consequently, determine how best to improve service offerings for said groups or occasions. 
- The blind spots or areas of improvement / further exploration include:
    - Using larger datasets for location based inference as well as greater samples on which to run the tests.
    - Fine tuning the model on more positive expressions that might still be spam - at the moment, a large proportion of comments
        classified as "spam" are negative reviews, which might skew the overall rating of locations that truly deserve the negative. From a UX perspective, a simple workaround might be to encourage the user to leave more descriptive reviews
        OR encourage them to remove the short reviews and simply use the rating.
    - Accounting for informal variants of English including contractions and colloquials, as well as emoji / unicode.
    - Multi class classification instead of binary - the model is suited for overlapping class classifications (spam + toxicity + irrelevant) - for classifications at higher granularities. 
    - Different ways of modelling the problem - multiple choice classification or prompt based classification to generate labels for unlabelled data and test against a separate corpus of labelled data. 
    - Hyperparameter tuning - the training process takes a significant amount of time, so only one run was performed optimised for reduced speed, however, tuning the parameters might provide different sets of results and possibly greater accuracy on the test data. At the moment validation and training are both performed on the labelled data whereas the efficacy of the performance on unlabelled data can be verified less precisely (aside from manual review).
        - To address the above, human-in-the-loop methods can be employed to rerun the same train-test loop after annotating any false positives or false negatives in the tested data.


# Design

- This uses a DistilBERT model for sequence classification since the model is fine tuned for faster training and inference and the dataset is only in English.
- It performs some experiments on tokenizers to determine whether different methods of tokenization expose any security loopholes (based on truncation / length of review) or in performance. Tokenization also poses some impact on accuracy based on some preliminary tests where contracted words are not precisely tokenized by some models while others split the tokens more meaningfully.
- Only the text of the review is used as the feature, with binary labels.
- The model is fine tuned on two spam / content moderation datasets from HuggingFace: ontocord/OIG-moderation and Deysi/spam-detection-dataset with label encoding. 
- It has reasonably high accuracy and F1 scores on the training / validation datasets but on the test dataset, the correctness is up to human judgement and definition of relevance. Since the sample data is relatively clean, some obervations of the general types of reviews picked out as irrelevant / spam include:
    - Generic qualifiers - good, bad, terrible, without associated details. They don't provide enough signals for informed decision making and therefore, make sense to disqualify.
    - Short reviews - same as above.
    - Reviews that specifically target people or quality of service - since there is no dedicated category for this, all the reviews here are likely to be false positives as they do provide decent signals regarding the type of service provided (especially those specifying length of wait etc.). An alternative might be to add this on as a granular category of review. 
    - Rant-like negative reviews. Although they are relevant in that the subject of the review, those with less emotional and more factual reactions are valuable to reannotate as a false positive. There are some that explicitly insult employees which are true positives and can be categorized as abusive.

# Setup

- Get a kaggle.json file containing the API Key. 
- Run the notebooks in sequence. 
- There is one prompt to input the WANDB API Key however this does not need to be set up beforehand.

# Experiments

- One hot encoding of labelled data - since this is binary classified and BERT models require string labels, the encoding was not necessary. 
- Tokenizer experiments - truncation, accuracy etc. It's interesting to note that the spam datasets used as reference do include contracted and colloquial language.
- Fine tuned training - single set of results so limited comparative potential.

( To be expanded as more experiments are performed )