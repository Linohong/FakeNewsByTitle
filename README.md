# FakeNewsByTitle

1. **Model** 
   * Hierarchical encoder 
     * training set consists of article with multiple sentences, abstracts with multiple sentences.
     * 'Article'
       * (1) encode each sentences of article using CNN (outputs hidden states of 300 dimension)
       * (2) then encode all sentences using bi-LSTM 
     * 'Abstract' : same as the article above but fewer number of sentences.
       
   * Calculate the Score
     * For each hidden states of the (2) above, inner product with abstract's last hidden state 
     * 
2. **Data**
   * take first half of the randomly shuffled NN/DailyMail Dataset.
     * swap their abstracts so that one's article and abstract doesn't fit each other 
     * 10% of the entire data are extracted when these limits are set
       * sent_num = 72
       * max_sent = 40
       * abs_num = 3
         
3. **Present (08.16)**
   * 99% Accuracy over closed sets 
   * 57% Accuracy when trained over 10,000 training examples 
   * ScoreAdam02 model : trained over about 100,000 training sets
     
4. **Issues** 
   * Score comparison between abstract and even 'document hidden state' necessary? 
     * proper way of adding both scores in question
   * 

