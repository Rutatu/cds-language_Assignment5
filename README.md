# lang_analytics_-assignment5


## Assignment 5: LDA model on American television sitcom´s Friends full script


For this assignment I have chosen to train an LDA model on American television sitcom´s Friends full script (10 seasons and 236 episodes). It is one of the  most popular TV series in history and I am a huge fan of it (I have seen the full series at least 3 times). Therefore, it is very interesting to test whether the conversations of the characters can be used to find meaningful topics. If so, then it can be used to define what Friends were talking most about, which does not have much of scientific value, but could be used for other sociological purposes. My investigation is purely exploratory. I will investigate the following questions:

QUESTION 1: Can LDA model find coherent and semantically meaningful topics Friends are talking about?
QUESTION 2: Can the dialogues/monologues of different characters be easily compared and does it reflect relevant topics according to a character? 


A to Q1: The visualization of the full script shows that it is possible to distinguish between relatively separate topics (the bubbles do not overlap too much and there is some distance between them). Most of the topics are rather specific (the majority of bubbles are quite small), probably reflecting  specific episodes or seasons. Nevertheless, as expected, most frequent terms are friend, baby, wedding, coffee, date, sex, which to some extent reflect the most crucial topics in the show.



A to Q2: The overall structure (size, position of the bubbles) is similar to the full script visualization, thus we can say that it does distinguish meaningful topics to some extent and they tend to be pretty specific. Interestingly, most frequent terms for each character represent them pretty well (you must have seen the episodes many times to say that).
Frequent terms for each character:
	Chandler: wedding, game, sleep, work, relationship
	Ross: date, baby, dad, love, wedding
	Rachel: ross, baby, coffee
	Phoebe: mom, grandmother, fun
	Monica: baby, problem, cookie, potato, wedding
	Joye: girl, butt, sex, night, pheebs, woman


Models Metrics:

Full script
Perplexity: - 10.55
Coherence: 0.28

Chandler
Perplexity: - 7.87
Coherence: 0.30

Monica
Perplexity: - 7.88
Coherence: 0.27

Ross
Perplexity: - 7.51
Coherence: 0.28

Rachel
Perplexity: - 7.82
Coherence: 0.31

Phoebe
Perplexity: - 8.09
Coherence: 0.33

Joey
Perplexity: - 8.01
Coherence: 0.33

The best model based on perplexity is Full script model. If we compare only characters´ models, then it is Phoebe´s model.
The best model based on coherence is Joey´s and Phoebe´s models.
However, the differences between scores are very small and, thus, it does not tell us much.

Limitations:
- only some of the semantically meaningless words were filtered out from the script. Filtering out more of them could lead to better performance.
- in this script the number of chunks in preprocessing stage: 30, min_count of bigrams: 10 and number of topics in lda model: 15 were set to be identical for each character. Each character would need further investigation and it would make more sense to set individualized metrics.
- the comparison of characters is very basic and shallow. It could have been more informative if chosen different, more advanced methods


Overall, the models perform not too badly. If invested more time in data cleaning and more thoughtful metrics choice, it could perform pretty well.
