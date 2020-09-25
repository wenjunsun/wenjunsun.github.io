In this notebook we will look at what each candidate talks about in their speeches. My expectation is that Biden talks more about environment and race issues and Trump talks more about economics. But we will see if that is really the case.

There are obviously a lot of ways to do topic modeling. In this notebook I will explore two techniques that fall into the realm of matrix decomposition: SVD and NMF.

I won't provide details about how NMF or SVD works, interested readers can read more about these methods online.

Motivation for using matrix decomposition to do topic modeling by Rachel Thomas from Fast AI: 

> "Consider the most extreme case - reconstructing the matrix using an outer product of two vectors. Clearly, in most cases we won't be able to reconstruct the matrix exactly. But if we had one vector with the relative frequency of each vocabulary word out of the total word count, and one with the average number of words per document, then that outer product would be as close as we can get.
Now consider increasing that matrices to two columns and two rows. The optimal decomposition would now be to cluster the documents into two groups, each of which has as different a distribution of words as possible to each other, but as similar as possible amongst the documents in the cluster. We will call those two groups "topics". And we would cluster the words into two groups, based on those which most frequently appear in each of the topics."


```
import numpy as np # basical linear algbera numerical computation package.
import pandas as pd # for reading data.
```

# 1. Read Data

Data is collected by Elijah Greisz, from [this website](https://www.rev.com/blog/transcript-category/2020-election-transcripts), we used speeches of both Biden and Trump from Aug 1 to Sep 10 as our data. 


```
ls
```

    [0m[01;34mdrive[0m/  [01;34msample_data[0m/
    


```
# before running this line need to connect to Google Drive.
data = pd.read_csv('drive/My Drive/political_sentiment_analysis/text_data.csv')
```


```
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>speech</th>
      <th>candidate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good afternoon folks. Sorry I’m a little late....</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
    <tr>
      <th>1</th>
      <td>When my son volunteered and joined the United ...</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
    <tr>
      <th>2</th>
      <td>He stood by failing, failing to take action or...</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I’ve talked to a lot of real working people, a...</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
    <tr>
      <th>4</th>
      <td>This is a special place for the Biden family. ...</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
  </tbody>
</table>
</div>




```
# select speech texts from biden
biden_data = data[data['candidate'] == 'biden']
biden_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>speech</th>
      <th>candidate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good afternoon folks. Sorry I’m a little late....</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
    <tr>
      <th>1</th>
      <td>When my son volunteered and joined the United ...</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
    <tr>
      <th>2</th>
      <td>He stood by failing, failing to take action or...</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I’ve talked to a lot of real working people, a...</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
    <tr>
      <th>4</th>
      <td>This is a special place for the Biden family. ...</td>
      <td>Joe Biden Press Conference Transcript September 4</td>
      <td>biden</td>
    </tr>
  </tbody>
</table>
</div>




```
# select speech texts from trump
trump_data = data[data['candidate'] == 'trump']
trump_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>speech</th>
      <th>candidate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>362</th>
      <td>We brought you a lot of car plants, Michigan. ...</td>
      <td>Donald Trump Freeland MI Campaign Rally Speech...</td>
      <td>trump</td>
    </tr>
    <tr>
      <th>363</th>
      <td>Long time, it’s been a long time since you had...</td>
      <td>Donald Trump Freeland MI Campaign Rally Speech...</td>
      <td>trump</td>
    </tr>
    <tr>
      <th>364</th>
      <td>This is the most important election in the his...</td>
      <td>Donald Trump Freeland MI Campaign Rally Speech...</td>
      <td>trump</td>
    </tr>
    <tr>
      <th>365</th>
      <td>We shouldn’t be smiling about it, but we’ve ch...</td>
      <td>Donald Trump Freeland MI Campaign Rally Speech...</td>
      <td>trump</td>
    </tr>
    <tr>
      <th>366</th>
      <td>If Biden wins, China wins. If Biden wins, the ...</td>
      <td>Donald Trump Freeland MI Campaign Rally Speech...</td>
      <td>trump</td>
    </tr>
  </tbody>
</table>
</div>




```
print(f'there are {biden_data.shape[0]} records of Biden speech documents')
print(f'there are {trump_data.shape[0]} records of Trump speech documents')
```

    there are 362 records of Biden speech documents
    there are 886 records of Trump speech documents
    

Now we want to print out some sample texts from both Trump and Biden, just to see what their speeches are like.


```
from random import randint
```


```
# randomly print 5 speeches from Biden.
for i in range(5):
  random_index = randint(0, biden_data.shape[0] - 1)
  print(biden_data.iloc[random_index]['text'])
  print()
```

    And speaking of President Obama, a man I was honored to serve alongside for eight years as vice president. Let me take this moment to say something we don’t say nearly enough. Thank you, Mr. President, you were a great president, a president that our children could and did look up to. No one’s going to say that about the current occupant of the White House. What you know about this president is if he’s given four more years, he’ll be what he’s been for the last four years. The president takes no responsibility, refuses to lead, blames others, cozies up to dictators, and fans the flames of hate and division. He’ll wake up every day believing that job is all about him, never about you. Is that the America you want for you, your family, your children? I see a different America, one that’s generous and strong, selfless, and humble. It’s an America we could rebuild together.
    
    He invited companies to the White House to make what he called the pledge to American workers. He couldn’t even keep those firms from outsourcing. Many were given lucrative federal contracts, but then some of them turned around and shipped 7,000 jobs overseas. Under President Trump, US trade deficit has grown. It’s hit an all time high. Let me say that again. US trade deficit is at an all time high under Trump in the last three years.
    
    And instead of telling Vladimir Putin that there’d be no putting up with this, that there’d be a heavy price to pay if they dare touch an American soldier, this president doesn’t even bring up the subject in his multiple phone calls with Putin. It’s been reported that Russian forces just attacked American troops in Syria, injuring our service members. Did you hear the president say a single word? Did he lift one finger? Never before has an American president played such a subservient role to a Russian leader. It’s not only dangerous, it’s humiliating and embarrassing for the rest of the world to see, it weakens us. Not even American troops can feel safer under Trump. Donald Trump’s role as a bystander in his own presidency extends to the economic plan and pain. The plan he doesn’t have and the pain being felt by millions of Americans. He said this week, and I quote, “You better vote for me, or you’re going to have the greatest depression you’ve ever seen.” Does he not understand and see the tens of millions of people who’ve had to file for unemployment this year so far?
    
    One of the most important conversations I’ve had this entire campaign. It was so someone who was much too young to vote. I met with six year old Gianna Floyd the day before her daddy, George Lloyd was laid to rest. She’s an incredibly brave little girl and I’ll never forget it. When I leaned down to speak to her, she looked in my eyes and she said, and I quote, “Daddy changed the world.” Daddy changed the world. Her words burrowed deep into my heart. Maybe George Floyd’s murder was a breaking point. Maybe John Lewis’ passing is the inspiration. But however it’s come to be, however it’s happened, America’s ready in John’s words, to lay down, quote, “The heavy burden of hate at last.” And in the hard work of rooting out our systemic racism. American history tells us that it’s been in our darkest moments that we’ve made our greatest progress, that we found the light. In this dark moment, I believe we’re poised to make great progress again, that we can find the light once more.
    
    It’s not what Dr. King or John Lewis taught and it must end. Fires are burning and we have a president who fans the flames rather than fighting the flames. But we must not burn, we have to build. This president, long ago, forfeited any moral leadership in this country. He can’t stop the violence because for years he’s fomented it. He may believe mouthing the words law and order makes him strong. But his failure to call on his own supporters to stop acting as an armed militia in this country shows how weak he is. Does anyone believe there’ll be less violence in America if Donald Trump is reelected?
    
    


```
# randomly print 5 speeches from Trump.
for i in range(5):
  random_index = randint(0, trump_data.shape[0] - 1)
  print(trump_data.iloc[random_index]['text'])
  print()
```

    And I saw where these phonies, you know they want to end everything we’ve done. They want to end it. They want to go to wind. They don’t even know if they want to go to wind. I think they want to just basically close up our country, because they’ve taken away our strength, but they want to do something. But, there is no such thing. Solar can’t do it. I love solar. It’s all fine. Very, very heavily expensive. Very expensive. But they want to go to other forms of alternative energy. And I think that’s okay, except we don’t have them. And it’s not going to power these massive factories.
    
    We can’t loose.
    
    That’s what they want to do. They want to take away your Second Amendment. If I weren’t president, you would either have an obliterated Second Amendment or it would be gone entirely. I am standing between them and your Second Amendment. And that’s it. That’s it. They know.
    
    They said manufacturing jobs will never come back. Remember, you need a magic wand. Where’s the magic wand? Well, we have the magic wand. My first week in office, I withdrew from the Trans-Pacific partnership. It would have been totally destructive to your jobs. It would have been a horror show. I withdrew from the one-sided Paris climate accord, which would have cost us so many billions of dollars. And all it would have done is made the competition even tougher all over the world. I believe it was designed to hurt the United States and to get jobs away from us and companies. I stood up to China’s rampant cheating, plunder, and theft. I repealed that horrible tax. So many taxes. How many taxes did I repeal? I’m getting a list of them now, I’ll have it for you for the next meeting. Because I’ll be back to Pennsylvania, that I promise. And I’m standing up to the special interests and to big pharma. Weeks ago, I signed four historic directives to dramatically reduce the cost of prescription drugs.
    
    It was the biggest China ordered two days. The biggest order in the history of corn. Then they did, if you look at soybeans, it was the largest soybean order in history, in history, and also beef cattle, et cetera. So it just shows you how smart China is. We sign a deal and the plague comes in. It might’ve been a mistake. It might’ve been on purpose. Who knows what happened? We’ll figure it out. But whatever it was, it was no good. They could have stopped it. But they know my attitude, I don’t like it. I don’t like it. So a normal country that’s not so smart in that position, Scott would have said, “Oh, we don’t like the way he’s talking about us. Let’s immediately shut down.” He didn’t do that. They said, “Let’s order more from the farmers than we’ve ever ordered. Biggest corn, biggest soybean, biggest cattle.”
    
    

We see that Trump talks about China, manufacture jobs, second Amendment.

Biden talks about moral leadership, George Lloyd, American soldiers, President Obama, etc.

# 2. Applying SVD, NMF to extract what Trump and Biden each talks about in their speeches.


```
biden_speeches = biden_data['text'].tolist()
```


```
biden_speeches[0]
```




    'Good afternoon folks. Sorry I’m a little late. I was a mesmerized. I was walking out of the office, that listening to an interview of former General Barry McCaffrey and Bill Cohen, former Secretary of Defense. Before I begin, I want to speak a little bit to what they talked about and the revelations about President Trump’s disregard for our military and our veterans. Quite frankly, if what is written in the Atlantic is true, it’s disgusting. At affirms what most of us believe to be true, that Donald Trump is not fit to do the job of President, to be the Commander in Chief. The President reportedly said, and I emphasize reportedly, said that those who sign up to serve, instead of doing something more lucrative are suckers. Let me be real clear. When my son was an Assistant US Attorney and he volunteered to go to Kosovo, when the war is going on as a civilian, he wasn’t a sucker.'




```
trump_speeches = trump_data['text'].tolist()
```


```
trump_speeches[0]
```




    'We brought you a lot of car plants, Michigan. We brought you a lot of car plants. You know that, right?'



## 2.1 Preprocessing/tokenizing:


```
from sklearn.feature_extraction.text import TfidfVectorizer
```


```
# in python there is a stemming library called snowballStemmer.
from nltk.stem.snowball import SnowballStemmer
```


```
import nltk
```


```
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    




    True




```
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```
stemmer = SnowballStemmer(language='english')
```


```
stopwords = nltk.corpus.stopwords.words('english')

def stemming_tokenizer(text):
  tokens = []
  sentences = nltk.sent_tokenize(text)
  for sentence in sentences:
    for word in nltk.word_tokenize(sentence):
      if word not in stopwords:
        tokens.append(word.lower())

  # filter out any tokens that are not words - filter out punctuations and numbers
  onlyWordTokens = [word for word in filter(lambda x: x.isalpha(), tokens)]

  # replace every word by its stem:
  onlyWordTokens = [stemmer.stem(word) for word in onlyWordTokens]

  return onlyWordTokens
```


```
# see how the first biden speech will get tokenized.
stemming_tokenizer(biden_speeches[0])
```




    ['good',
     'afternoon',
     'folk',
     'sorri',
     'i',
     'littl',
     'late',
     'i',
     'mesmer',
     'i',
     'walk',
     'offic',
     'listen',
     'interview',
     'former',
     'general',
     'barri',
     'mccaffrey',
     'bill',
     'cohen',
     'former',
     'secretari',
     'defens',
     'befor',
     'i',
     'begin',
     'i',
     'want',
     'speak',
     'littl',
     'bit',
     'talk',
     'revel',
     'presid',
     'trump',
     'disregard',
     'militari',
     'veteran',
     'quit',
     'frank',
     'written',
     'atlant',
     'true',
     'disgust',
     'at',
     'affirm',
     'us',
     'believ',
     'true',
     'donald',
     'trump',
     'fit',
     'job',
     'presid',
     'command',
     'chief',
     'the',
     'presid',
     'report',
     'said',
     'i',
     'emphas',
     'report',
     'said',
     'sign',
     'serv',
     'instead',
     'someth',
     'lucrat',
     'sucker',
     'let',
     'real',
     'clear',
     'when',
     'son',
     'assist',
     'us',
     'attorney',
     'volunt',
     'go',
     'kosovo',
     'war',
     'go',
     'civilian',
     'sucker']




```
tfidf_trump = TfidfVectorizer(max_df=0.99, max_features= 300,
                        min_df=0.01, stop_words='english',
                        use_idf=True, tokenizer = stemming_tokenizer, # our custom tokenizer that ignores the tense of words.
                        ngram_range=(1,1))
```


```
tfidf_matrix_trump = tfidf_trump.fit_transform(trump_speeches)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py:385: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['afterward', 'alon', 'alreadi', 'alway', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becom', 'besid', 'cri', 'describ', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti', 'forti', 'henc', 'hereaft', 'herebi', 'howev', 'hundr', 'inde', 'mani', 'meanwhil', 'moreov', 'nobodi', 'noon', 'noth', 'nowher', 'otherwis', 'perhap', 'pleas', 'sever', 'sinc', 'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 'thenc', 'thereaft', 'therebi', 'therefor', 'togeth', 'twelv', 'twenti', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev'] not in stop_words.
      'stop_words.' % sorted(inconsistent))
    


```
# convert a sparse matrix to dense matrix, for later SVD.
tfidf_matrix_trump = tfidf_matrix_trump.todense()
```


```
# each of trump's speech is represented
# by a 300 dimensional vector
tfidf_matrix_trump.shape
```




    (886, 300)




```
tfidf_biden = TfidfVectorizer(max_df=0.99, max_features= 250,
                        min_df=0.01, stop_words='english',
                        use_idf=True, tokenizer = stemming_tokenizer, # our custom tokenizer that ignores the tense of words.
                        ngram_range=(1,1))
```


```
tfidf_matrix_biden = tfidf_biden.fit_transform(biden_speeches)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py:385: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['afterward', 'alon', 'alreadi', 'alway', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becom', 'besid', 'cri', 'describ', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti', 'forti', 'henc', 'hereaft', 'herebi', 'howev', 'hundr', 'inde', 'mani', 'meanwhil', 'moreov', 'nobodi', 'noon', 'noth', 'nowher', 'otherwis', 'perhap', 'pleas', 'sever', 'sinc', 'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 'thenc', 'thereaft', 'therebi', 'therefor', 'togeth', 'twelv', 'twenti', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev'] not in stop_words.
      'stop_words.' % sorted(inconsistent))
    


```
tfidf_matrix_biden = tfidf_matrix_biden.todense()
```


```
tfidf_matrix_biden.shape
```




    (362, 250)




```
# 20 words biden used
tfidf_biden.get_feature_names()[:20]
```




    ['abl',
     'act',
     'administr',
     'afford',
     'ago',
     'alreadi',
     'alway',
     'america',
     'american',
     'anyth',
     'ask',
     'away',
     'bad',
     'begin',
     'believ',
     'best',
     'better',
     'biden',
     'big',
     'billion']




```
# 20 words trump used
tfidf_trump.get_feature_names()[:20]
```




    ['achiev',
     'actual',
     'administr',
     'agenda',
     'ago',
     'allow',
     'alreadi',
     'alway',
     'amaz',
     'amend',
     'america',
     'american',
     'anoth',
     'anybodi',
     'anyth',
     'ask',
     'away',
     'bad',
     'ballot',
     'ban']




```
trump_vocab = tfidf_trump.get_feature_names()
```


```
biden_vocab = tfidf_biden.get_feature_names()
```


```
# write a function that given a vector,
# show the words that correspond to the most importance.
# (largest in magnitude)
# assume the element that has more magnitute are more important.
def getMostImportantWords(vector, numOfWords, isTrump):
  indices_of_most_important_words = np.argsort(vector)[::-1][:numOfWords]
  if isTrump:
    return ' '.join([trump_vocab[index] for index in indices_of_most_important_words])
  else:
    return ' '.join([biden_vocab[index] for index in indices_of_most_important_words])
```

## 2.2 SVD.


```
from scipy import linalg
```


```
%time U, s, Vh = linalg.svd(tfidf_matrix_trump, full_matrices=False)
```

    CPU times: user 99.5 ms, sys: 36.2 ms, total: 136 ms
    Wall time: 76.1 ms
    


```
# print out 5 topics Trump talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = Vh[i], numOfWords= 10, isTrump= True))
```

    topic 0: russia favor trillion plan post term judg free illeg play
    topic 1: biden year china joe happen countri like left citi deal
    topic 2: peopl said know say like think good thing look guy
    topic 3: want peopl citi polic left democrat america win biden law
    topic 4: great peopl love state job nation american good america unit
    


```
# print out 5 topics Trump talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = Vh[i], numOfWords= 15, isTrump= True))
```

    topic 0: russia favor trillion plan post term judg free illeg play stock achiev yeah polici oil
    topic 1: biden year china joe happen countri like left citi deal mani look thing said talk
    topic 2: peopl said know say like think good thing look guy lot big happen right poll
    topic 3: want peopl citi polic left democrat america win biden law run everi york american crime
    topic 4: great peopl love state job nation american good america unit right open million work make
    


```
%time U, s, Vh = linalg.svd(tfidf_matrix_biden, full_matrices=False)
```

    CPU times: user 44.9 ms, sys: 16 ms, total: 60.9 ms
    Wall time: 37.3 ms
    


```
# print out 5 topics biden talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = Vh[i], numOfWords= 10, isTrump= False))
```

    topic 0: program shot local crosstalk justic afford buy general citi death
    topic 1: thank want god think elect realli crosstalk hope got better
    topic 2: crosstalk know right got think said talk someth vote thing
    topic 3: crosstalk right worker tax thank job trump pay american union
    topic 4: school abl educ need home health safe sure children make
    


```
# print out 5 topics biden talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = Vh[i], numOfWords= 15, isTrump= False))
```

    topic 0: program shot local crosstalk justic afford buy general citi death learn test cut street idea
    topic 1: thank want god think elect realli crosstalk hope got better day number import state lot
    topic 2: crosstalk know right got think said talk someth vote thing lot anyth want say happen
    topic 3: crosstalk right worker tax thank job trump pay american union compani build treat feder make
    topic 4: school abl educ need home health safe sure children make kid open billion parent think
    

## 2.3 NMF


```
from sklearn import decomposition
```


```
clf = decomposition.NMF(n_components=5, random_state=1)

W1 = clf.fit_transform(tfidf_matrix_trump)
H1 = clf.components_
```


```
# print out 5 topics Trump talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = H1[i], numOfWords= 10, isTrump= True))
```

    topic 0: said happen like come know thing say think got right
    topic 1: thank great job want love let everybodi man friend michigan
    topic 2: china year job american world america countri nation biden histori
    topic 3: want biden left citi polic joe law look berni radic
    topic 4: peopl great love good win state right know job realli
    


```
# print out 5 topics Trump talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = H1[i], numOfWords= 15, isTrump= True))
```

    topic 0: said happen like come know thing say think got right time let look big year
    topic 1: thank great job want love let everybodi man friend michigan respect children god carolina realli
    topic 2: china year job american world america countri nation biden histori billion deal unit million state
    topic 3: want biden left citi polic joe law look berni radic suburb democrat crime everi run
    topic 4: peopl great love good win state right know job realli like vote lot say governor
    


```
W1 = clf.fit_transform(tfidf_matrix_biden)
H1 = clf.components_
```


```
# print out 5 topics Biden talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = H1[i], numOfWords= 10, isTrump= False))
```

    topic 0: american trump job presid worker tax america donald pay union
    topic 1: thank god want better day covid labor elect hope number
    topic 2: peopl think know said say talk got thing want look
    topic 3: crosstalk right treat know got someth life love want health
    topic 4: school need educ safe abl make latino sure children year
    


```
# print out 5 topics Biden talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = H1[i], numOfWords= 15, isTrump= False))
```

    topic 0: american trump job presid worker tax america donald pay union million build work make compani
    topic 1: thank god want better day covid labor elect hope number state promis realli protect union
    topic 2: peopl think know said say talk got thing want look countri lot time come presid
    topic 3: crosstalk right treat know got someth life love want health deal problem okay happen number
    topic 4: school need educ safe abl make latino sure children year home open everi mask health
    

## 2.4 K-means


```
from sklearn.cluster import KMeans
```


```
km = KMeans(n_clusters = 5)
km.fit(tfidf_matrix_trump)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)




```
# Trump:
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(km.cluster_centers_[i], numOfWords = 10, isTrump=True))
```

    topic 0: china year job countri world american biden billion america deal
    topic 1: thank great job love want peopl right realli good man
    topic 2: peopl great want know say right good state like win
    topic 3: said think like thing say right come know happen got
    topic 4: biden want left citi know polic look joe american yeah
    


```
# Trump:
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(km.cluster_centers_[i], numOfWords = 15, isTrump=True))
```

    topic 0: china year job countri world american biden billion america deal nation histori right number million
    topic 1: thank great job love want peopl right realli good man friend make america know said
    topic 2: peopl great want know say right good state like win lot love vote new got
    topic 3: said think like thing say right come know happen got want time year mani let
    topic 4: biden want left citi know polic look joe american yeah countri berni america law radic
    


```
km = KMeans(n_clusters = 5)
km.fit(tfidf_matrix_biden)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)




```
# biden
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(km.cluster_centers_[i], numOfWords = 10, isTrump=False))
```

    topic 0: crosstalk inaud got tell togeth chanc know campaign want right
    topic 1: trump job presid american worker make tax donald pay america
    topic 2: peopl think say presid thing talk know said like countri
    topic 3: thank god want elect state better day number covid think
    topic 4: latino abl make peopl school union know sure okay everi
    


```
# biden
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(km.cluster_centers_[i], numOfWords = 15, isTrump=False))
```

    topic 0: crosstalk inaud got tell togeth chanc know campaign want right way time generat lot think
    topic 1: trump job presid american worker make tax donald pay america work million school countri peopl
    topic 2: peopl think say presid thing talk know said like countri look make vote want understand
    topic 3: thank god want elect state better day number covid think hope labor good union honor
    topic 4: latino abl make peopl school union know sure okay everi program billion way state educ
    

# 3. repeat analysis for slightly different parameters


```
tfidf_trump = TfidfVectorizer(max_df=0.95, max_features= 500, # use more words.
                        min_df=0.02, stop_words='english',
                        use_idf=True, tokenizer = stemming_tokenizer, # our custom tokenizer that ignores the tense of words.
                        ngram_range=(1,1))
```


```
tfidf_matrix_trump = tfidf_trump.fit_transform(trump_speeches)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py:385: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['afterward', 'alon', 'alreadi', 'alway', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becom', 'besid', 'cri', 'describ', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti', 'forti', 'henc', 'hereaft', 'herebi', 'howev', 'hundr', 'inde', 'mani', 'meanwhil', 'moreov', 'nobodi', 'noon', 'noth', 'nowher', 'otherwis', 'perhap', 'pleas', 'sever', 'sinc', 'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 'thenc', 'thereaft', 'therebi', 'therefor', 'togeth', 'twelv', 'twenti', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev'] not in stop_words.
      'stop_words.' % sorted(inconsistent))
    


```
# convert a sparse matrix to dense matrix, for later SVD.
tfidf_matrix_trump = tfidf_matrix_trump.todense()
```


```
# each of trump's speech is represented
# by a 300 dimensional vector
tfidf_matrix_trump.shape
```




    (886, 392)




```
tfidf_biden = TfidfVectorizer(max_df=0.99, max_features= 250,
                        min_df=0.01, stop_words='english',
                        use_idf=True, tokenizer = stemming_tokenizer, # our custom tokenizer that ignores the tense of words.
                        ngram_range=(1,1))
```


```
tfidf_matrix_biden = tfidf_biden.fit_transform(biden_speeches)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py:385: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['afterward', 'alon', 'alreadi', 'alway', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becom', 'besid', 'cri', 'describ', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti', 'forti', 'henc', 'hereaft', 'herebi', 'howev', 'hundr', 'inde', 'mani', 'meanwhil', 'moreov', 'nobodi', 'noon', 'noth', 'nowher', 'otherwis', 'perhap', 'pleas', 'sever', 'sinc', 'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 'thenc', 'thereaft', 'therebi', 'therefor', 'togeth', 'twelv', 'twenti', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev'] not in stop_words.
      'stop_words.' % sorted(inconsistent))
    


```
tfidf_matrix_biden = tfidf_matrix_biden.todense()
```


```
tfidf_matrix_biden.shape
```




    (362, 250)




```
# 20 words biden used
tfidf_biden.get_feature_names()[:20]
```




    ['abl',
     'act',
     'administr',
     'afford',
     'ago',
     'alreadi',
     'alway',
     'america',
     'american',
     'anyth',
     'ask',
     'away',
     'bad',
     'begin',
     'believ',
     'best',
     'better',
     'biden',
     'big',
     'billion']




```
# 20 words trump used
tfidf_trump.get_feature_names()[:20]
```




    ['abl',
     'achiev',
     'actual',
     'administr',
     'advantag',
     'agenda',
     'ago',
     'agre',
     'alien',
     'allow',
     'alreadi',
     'alway',
     'amaz',
     'amend',
     'america',
     'american',
     'announc',
     'anoth',
     'anybodi',
     'anymor']




```
trump_vocab = tfidf_trump.get_feature_names()
```


```
biden_vocab = tfidf_biden.get_feature_names()
```


```
# write a function that given a vector,
# show the words that correspond to the most importance.
# (largest in magnitude)
# assume the element that has more magnitute are more important.
def getMostImportantWords(vector, numOfWords, isTrump):
  indices_of_most_important_words = np.argsort(vector)[::-1][:numOfWords]
  if isTrump:
    return ' '.join([trump_vocab[index] for index in indices_of_most_important_words])
  else:
    return ' '.join([biden_vocab[index] for index in indices_of_most_important_words])
```

## 3.2 SVD.


```
from scipy import linalg
```


```
%time U, s, Vh = linalg.svd(tfidf_matrix_trump, full_matrices=False)
```

    CPU times: user 172 ms, sys: 52.4 ms, total: 224 ms
    Wall time: 146 ms
    


```
# print out 5 topics Trump talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = Vh[i], numOfWords= 10, isTrump= True))
```

    topic 0: terrorist alien sudden manifesto short kind fair speak histor immedi
    topic 1: thank great love job friend good repres john realli fantast
    topic 2: china america biden year american world thank job joe nation
    topic 3: said year china billion deal farmer good job tariff thing
    topic 4: great job love peopl state good open american nation million
    


```
# print out 5 topics Trump talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = Vh[i], numOfWords= 15, isTrump= True))
```

    topic 0: terrorist alien sudden manifesto short kind fair speak histor immedi honest anywher coupl sort futur
    topic 1: thank great love job friend good repres john realli fantast man everybodi honor carolina god
    topic 2: china america biden year american world thank job joe nation countri histori billion unit deal
    topic 3: said year china billion deal farmer good job tariff thing took sir come right guy
    topic 4: great job love peopl state good open american nation million right win work john governor
    


```
%time U, s, Vh = linalg.svd(tfidf_matrix_biden, full_matrices=False)
```

    CPU times: user 39 ms, sys: 19.1 ms, total: 58.1 ms
    Wall time: 35.1 ms
    


```
# print out 5 topics biden talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = Vh[i], numOfWords= 10, isTrump= False))
```

    topic 0: program shot local crosstalk justic afford buy general citi death
    topic 1: thank want god think elect realli crosstalk hope got better
    topic 2: crosstalk know right got think said talk someth vote thing
    topic 3: crosstalk right worker tax thank job trump pay american union
    topic 4: school abl educ need home health safe sure children make
    


```
# print out 5 topics biden talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = Vh[i], numOfWords= 15, isTrump= False))
```

    topic 0: program shot local crosstalk justic afford buy general citi death learn test cut street idea
    topic 1: thank want god think elect realli crosstalk hope got better day number import state lot
    topic 2: crosstalk know right got think said talk someth vote thing lot anyth want say happen
    topic 3: crosstalk right worker tax thank job trump pay american union compani build treat feder make
    topic 4: school abl educ need home health safe sure children make kid open billion parent think
    

## 2.3 NMF


```
from sklearn import decomposition
```


```
clf = decomposition.NMF(n_components=5, random_state=1)

W1 = clf.fit_transform(tfidf_matrix_trump)
H1 = clf.components_
```


```
# print out 5 topics Trump talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = H1[i], numOfWords= 10, isTrump= True))
```

    topic 0: said peopl like say know thing happen come think right
    topic 1: thank let want repres great everybodi god make america respect
    topic 2: year china billion world job countri biden american deal america
    topic 3: want biden left citi polic law joe everi america radic
    topic 4: great love job good state win right john peopl guy
    


```
# print out 5 topics Trump talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = H1[i], numOfWords= 15, isTrump= True))
```

    topic 0: said peopl like say know thing happen come think right look lot got let big
    topic 1: thank let want repres great everybodi god make america respect children proud minnesota job honor
    topic 2: year china billion world job countri biden american deal america histori nation took joe economi
    topic 3: want biden left citi polic law joe everi america radic look countri democrat new berni
    topic 4: great love job good state win right john peopl guy governor realli friend open vote
    


```
W1 = clf.fit_transform(tfidf_matrix_biden)
H1 = clf.components_
```


```
# print out 5 topics Biden talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = H1[i], numOfWords= 10, isTrump= False))
```

    topic 0: american trump job presid worker tax america donald pay union
    topic 1: thank god want better day covid labor elect hope number
    topic 2: peopl think know said say talk got thing want look
    topic 3: crosstalk right treat know got someth life love want health
    topic 4: school need educ safe abl make latino sure children year
    


```
# print out 5 topics Biden talks about
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(vector = H1[i], numOfWords= 15, isTrump= False))
```

    topic 0: american trump job presid worker tax america donald pay union million build work make compani
    topic 1: thank god want better day covid labor elect hope number state promis realli protect union
    topic 2: peopl think know said say talk got thing want look countri lot time come presid
    topic 3: crosstalk right treat know got someth life love want health deal problem okay happen number
    topic 4: school need educ safe abl make latino sure children year home open everi mask health
    

## 2.4 K-means


```
from sklearn.cluster import KMeans
```


```
km = KMeans(n_clusters = 5)
km.fit(tfidf_matrix_trump)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)




```
# Trump:
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(km.cluster_centers_[i], numOfWords = 10, isTrump=True))
```

    topic 0: said come like happen say know yeah thing got want
    topic 1: citi polic left want look york biden know law new
    topic 2: thank great job love want peopl realli good right know
    topic 3: year china job biden american countri america world billion histori
    topic 4: peopl know want right great think say like lot good
    


```
# Trump:
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(km.cluster_centers_[i], numOfWords = 15, isTrump=True))
```

    topic 0: said come like happen say know yeah thing got want right let time countri think
    topic 1: citi polic left want look york biden know law new peopl radic run portland crime
    topic 2: thank great job love want peopl realli good right know repres said america friend make
    topic 3: year china job biden american countri america world billion histori deal nation joe state number
    topic 4: peopl know want right great think say like lot good state thing way win time
    


```
km = KMeans(n_clusters = 5)
km.fit(tfidf_matrix_biden)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)




```
# biden
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(km.cluster_centers_[i], numOfWords = 10, isTrump=False))
```

    topic 0: talk togeth know got say said anyth thing come lot
    topic 1: presid trump american job america worker countri make donald work
    topic 2: thank inaud tell want state god way better elect day
    topic 3: peopl make think school abl vote need look right sure
    topic 4: crosstalk right treat love someth life got know watch matter
    


```
# biden
for i in range(5):
  print(f'topic {i}: ' + getMostImportantWords(km.cluster_centers_[i], numOfWords = 15, isTrump=False))
```

    topic 0: talk togeth know got say said anyth thing come lot someth want stand time son
    topic 1: presid trump american job america worker countri make donald work peopl nation union famili build
    topic 2: thank inaud tell want state god way better elect day come number good covid friend
    topic 3: peopl make think school abl vote need look right sure realli know way thing year
    topic 4: crosstalk right treat love someth life got know watch matter fact want say thank think
    

# Conclusion: 

As we can see, it is pretty clear that Trump talks a lot about China, job, America crime, radical, left. Biden talks more about school, education, health, safe, worker, vote. There isn't much surprise here. 

- We can try with different topic modeling models and different hyperparameters in the future.
