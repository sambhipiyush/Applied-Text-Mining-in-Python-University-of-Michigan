Applied Text Mining in Python
==============================

by University of Michigan

# Module 2
#
## Title: Basic Natural Language Processing

### Overview

* What is Natural Language?
	* Well, any language that is used in everyday communication by humans is natural language
		* Languages such as English, or Chinese, or Hindi, or Russian, or Spanish are all natural languages
		* But you know, also the language we use in short text messages or on tweets is also, by this definition natural language
* What is Natural Language Processing? 
	* Any computation or manipulation of natural language to get some insights about how words mean and how sentences are constructed is natural language processing

###### NLP Tasks

* Counting words, counting frequency of words
* Finding sentence boundaries
* Part of speech tagging
* Parsing the sentence structure
* Identifying semantic roles
	* For example, if you have a sentence like Mary loves John. Then you know that Mary is the subject. John is the object, and love is the verb that connects them
* Identifying entities in a sentence --> this is called name entity recognition
* Finding which pronoun refer to which entity --> This is called co-ref resolution, or co-reference resolution

### Basic NLP tasks with NLTK

##### Introduction NLTK

* **NLTK** --> **N**atural **L**anguage **T**ool**K**it
* Open Source Library in Python
* Has support to most NLP Tasks
* Also provide access to numerous text corpora
	* to download text corpora, cmd used is
		```python
		nltk.download()
		# OR
		nltk.download('all') # this is for jupyter notebook
		```

##### Simple NLTK Tasks

1. Counting vocabulary of words
	```python
	>>> text7
		<Text: Wall Street Journal>
	>>> sent7
		['Pierre', 'Vinken', ',', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'Nov.', '29', '.']
	>>> len(sent7)
		18
	>>> len(text7)
		100676
	>>> len(set(text7))
		12408
	```
1. To get first 10 unique worlds
	```python
	>>> list(set(text7))[:10]
		['PAP', 'lap-shoulder', 'INS', '73', 'price', 'delisted', 'Calif', 'elaborate', 'pre-emptive', 'Shrum']
	```
1. Frequency of words
	```python
	>>> dist = FreqDist(text7) # Creating Frequency Distribution from Text 7
	>>> len(dist)
		12408
	>>> vocab1 = dist.keys()
	>>> list(vocab1)[:10]
		['Pierre', 'Vinken', ',', '61', 'years', 'old', 'will', 'join', 'the', 'board']
	>>> dist['four'] # How many times a particular word occured --> in this case word 'four'
		20
	>>> freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100] # Frequency of a word which has length of at-least 5 and occured at-least 100 times
	>>> freqwords
		['billion', 'company', 'president', 'because', 'market', 'million', 'shares', 'trading', 'program']
	>>>
	```
1. Normalization and stemming
	* Normalization - is when you have to transform a word to make it appear the same way or the count even though they look very different
		* Example - there might be different forms in which the same word occurs
			* "List listed lists listing listings"
	```python
	>>> input1 = "List listed lists listing listings"
	>>> words1 = input1.lower().split(' ') # To bring all words to same case, so that we can compare then easily
	>>> words1 # This is an example of Normalization
		['list', 'listed', 'lists', 'listing', 'listings']
	```
	* Stemming - is to find the root word or the root form of any given word
		* Algorithm that is quite popular and used widely is **Porter Stemmer**
	```python
	# This is an example of Stemming
	>>> porter = nltk.PorterStemmer()
	>>> [porter.stem(t) for t in words1]
		['list', 'list', 'list', 'list', 'list']
	```
1. Lemmatization
	* Lemitization is where you want to have the words that come out to be actually meaningful
	* Lemmatization would do stemming, but really keep the resulting tense to be valid words
	* It is sometimes useful because you want to somehow normalize it, but normalize it to something that is also meaningful
	* We could use something like a **Wordnet Lemmatizer**
	```python
	>>> udhr = nltk.corpus.udhr.words('English-Latin1')
	>>> udhr[:20]
		['Universal', 'Declaration', 'of', 'Human', 'Rights', 'Preamble', 'Whereas', 'recognition', 'of', 'the', 'inherent', 'dignity', 'and', 'of', 'the', 'equal', 'and', 'inalienable', 'rights', 'of']
	>>> WNlemma = nltk.WordNetLemmatizer()
	>>> #  you will also notice that the fifth word here, universal declaration of human rights is not lemmatized because that is with a capital R, it's a different word that was not lemmatized to right
	>>> # Second last word "RIGHTS" is lemmatized to "RIGHT"
	>>> [WNlemma.lemmatize(t) for t in udhr[:20]] 
		['Universal', 'Declaration', 'of', 'Human', 'Rights', 'Preamble', 'Whereas', 'recognition', 'of', 'the', 'inherent', 'dignity', 'and', 'of', 'the', 'equal', 'and', 'inalienable', 'right', 'of']
	```
1. Tokenization
	* Syntax
		```python
		>>> nltk.word_tokenize(<PASS_STRING_HERE>)
		```
	* Sentence Splitting
		* Syntax
		```python
		>>> nltk.sent_tokenize(<PASS_STRING_HERE>)
		>>> text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
		>>> sentences = nltk.sent_tokenize(text12)
		>>> len(sentences)
			4
		>>> sentences
			['This is the first sentence.',
			 'A gallon of milk in the U.S. costs $2.99.',
			 'Is this the third sentence?',
			 'Yes, it is!']
		```

##### Review

1. NLTK is a widely used toolkit for text and natural language processing
1. It has quite a few tools and very handy tools to tokenize and split a sentence and then go from there, lemmatize and stem
1. It gives access to many text corpora as well
1. These tasks of sentence splitting and tokenization and lemmatization are quite important preprocessing tasks and they are non-trivial
	* So you cannot really write a regular expression in a trivial fashion and expect it to work well.


### Advanced NLP tasks with NLTK

```ruby
Read the following sentence:
Visiting aunts can be a nuisance.

Quest: What can be a nuisance?
Ans:
1. Traveling to see aunts
2. Welcoming aunts into your home

Reason:	
	Besides the tongue-in-cheek responses, this sentence is another example of a syntactic ambiguity. Depending on how the sentence is parsed, both the first and second interpretations are possible. In this case, the ambiguity arises because the word “Visiting” could either be an adjective or a gerund, leading to two different ways to parse the sentence to derive two different meanings.
```

```ruby
Read the following sentence and select the most appropriate response.
Mary saw the man with the telescope.
Quest: Who has the telescope?
Ans: Could be either
Reason:
	It could be either. This sentence is an example of an inherent ambiguity in the prepositional phrase attachment. The statement can be parsed in two different ways to derive two different meanings.
```

1. **POS** ( **P**art **O**f **S**peech ) tagging
	* Some tags
		|TAG|Word Class|
		|:---:|:---:|
		|CC|Conjunction|
		|CD|Cardinal|
		|DT|Determiner|
		|IN|Preposition|
		|JJ|Adjective|
		|MD|Modal|
		|NN|Noun|
		|PRP|Pronoun|
		|RB|Adverb|
		|SYM|Symbol|
		|VB|Verb|
	* Tagsets for English examples
		|Tag|Description|Example|
		|:---:|:---:|:---:|
		|CC|Coordin. Conjunction|and, but, or|
		|CD|Cardinal number|one, two, three|
		|DT|Determiner|a, the|
		|EX|Existential|‘there’ there|
		|FW|Foreign word|mea culpa|
		|IN|Preposition/sub-conj|of, in, by|
		|JJ|Adjective|yellow|
		|JJR|Adj., comparative|bigger|
		|JJS|Adj., superlative|wildest|
		|LS|List item marker|1, 2, One|
		|MD|Modal|can, should|
		|NN|Noun, sing. or mass|llama|
		|NNS|Noun, plural|llamas|
		|NNP|Proper noun, singular|IBM|
		|NNPS|Proper noun, plural|Carolinas|
		|PDT|Predeterminer|all, both|
		|POS|Possessive ending|’s|
		|PRP|Personal pronoun|I, you, he|
		|PRP$|Possessive pronoun|your, one’s|
		|RB|Adverb|quickly, never , Comma ,|
		|RBR|Adverb, comparative|faster|
		|RBS|Adverb, superlative|fastest|
		|RP|Particle|up, off|
		|SYM|Symbol|+,%, &|
		|TO|“to”|to|
		|UH|Interjection|ah, oops|
		|VB|Verb, base form|eat|
		|VBD|Verb, past tense|ate|
		|VBG|Verb, gerund|eating|
		|VBN|Verb, past participle|eaten|
		|VBP|Verb, non-3sg pres|eat|
		|VBZ|Verb, 3sg pres|eats|
		|WDT|Wh-determiner which, that|
		|WP|Wh-pronoun|what, who|
		|WP$|Possessive wh-|whose|
		|WRB|Wh-adverb|how, where|
		|$|Dollar sign|$|
		|#|Pound sign|#|
		|“|Left quote|‘ or “|
		|”|Right quote|’ or ”|
		|(|Left parenthesis|[, (, {, <|
		|)|Right parenthesis|], ), }, >|
		|.|Sentence-final punc|. ! ?|
		|:|Mid-sentence punc|: ; ... – -|
	* POS tags and examples
		|Tag|Description|Example|
		|:---:|:---:|:---:|
		|(|opening parenthesis|(, [|
		|)|closing parenthesis|),]|
		|*|negator|not n’t|
		|,|comma|,|
		|–|dash|–|
		|.|sentence terminator|. ; ? !|
		|:|colon|:|
		|ABL|pre-qualifier|quite, rather, such|
		|ABN|pre-quantifier|half, all,|
		|ABX|pre-quantifier, double conjunction|both|
		|AP|post-determiner|many, next, several, last|
		|AT|article|a the an no a every|
		|BE/BED/BEDZ/BEG/BEM/BEN/BER/BEZ|be/were/was/being/am/been/are/is|
		|CC|coordinating conjunction|and or but either neither|
		|CD|cardinal numeral|two, 2, 1962, million|
		|CS|subordinating conjunction|that as after whether before|
		|DO/DOD/DOZ|do, did, does|
		|DT|singular determiner,|this, that|
		|DTI|singular or plural determiner|some, any|
		|DTS|plural determiner|these those them|
		|DTX|determiner, double conjunction|either, neither|
		|EX|existential|there there|
		|HV/HVD/HVG/HVN/HVZ|have, had, having, had, has|
		|IN|preposition|of in for by to on at|
		|JJ|adjective|
		|JJR|comparative adjective|better, greater, higher, larger, lower|
		|JJS|semantically superlative adj.|main, top, principal, chief, key, foremost|
		|JJT|morphologically superlative adj.|best, greatest, highest, largest, latest, worst|
		|MD|modal auxiliary|would, will, can, could, may, must, should|
		|NN|(common) singular or mass noun|time, world, work, school, family, door|
		|NN$|possessive singular common noun|father’s, year’s, city’s, earth’s|
		|NNS|plural common noun|years, people, things, children, problems|
		|NNS$|possessive plural noun|children’s, artist’s parent’s years’|
		|NP|singular proper noun|Kennedy, England, Rachel, Congress|
		|NP$|possessive singular proper noun|Plato’s Faulkner’s Viola’s|
		|NPS|plural proper noun|Americans Democrats Belgians Chinese Sox|
		|NPS$|possessive plural proper noun|Yankees’, Gershwins’ Earthmen’s|
		|NR|adverbial noun|home, west, tomorrow, Friday, North,|
		|NR$|possessive adverbial noun|today’s, yesterday’s, Sunday’s, South’s|
		|NRS|plural adverbial noun|Sundays Fridays|
		|OD|ordinal numeral|second, 2nd, twenty-first, mid-twentieth|
		|PN|nominal pronoun|one, something, nothing, anyone, none,|
		|PN$|possessive nominal pronoun|one’s someone’s anyone’s|
		|PP$|possessive personal pronoun|his their her its my our your|
		|PP$$|second possessive personal pronoun|mine, his, ours, yours, theirs|
		|PPL|singular reflexive personal pronoun|myself, herself|
		|PPLS|plural reflexive pronoun|ourselves, themselves|
		|PPO|objective personal pronoun|me, us, him|
		|PPS|3rd. sg. nominative pronoun|he, she, it|
		|PPSS|other nominative pronoun|I, we, they|
		|QL|qualifier|very, too, most, quite, almost, extremely|
		|QLP|post-qualifier|enough, indeed|
		|RB|adverb|
		|RBR|comparative adverb|later, more, better, longer, further|
		|RBT|superlative adverb|best, most, highest, nearest|
		|RN|nominal adverb|here, then|
		|RP|adverb or particle|across, off, up|
		|TO|infinitive|marker to|
		|UH|interjection, exclamation|well, oh, say, please, okay, uh, goodbye|
		|VB|verb, base form|make, understand, try, determine, drop|
		|VBD|verb, past tense|said, went, looked, brought, reached kept|
		|VBG|verb, present participle, gerund|getting, writing, increasing|
		|VBN|verb, past participle|made, given, found, called, required|
		|VBZ|verb, 3rd singular present|says, follows, requires, transcends|
		|WDT|wh- determiner|what, which|
		|WP$|possessive wh- pronoun|whose|
		|WPO|objective wh- pronoun|whom, which, that|
		|WPS|nominative wh- pronoun|who, which, that|
		|WQL|how|
		|WRB|wh- adverb|how, when|
		|AJ0|adjective (unmarked)|good, old|
		|AJC|comparative adjective|better, older|
		|AJS|superlative adjective|best, oldest|
		|AT0|article|the, a, an|
		|AV0|adverb (unmarked)|often, well, longer, furthest|
		|AVP|adverb particle|up, off, out|
		|AVQ|wh-adverb|when, how, why|
		|CJC|coordinating conjunction|and, or|
		|CJS|subordinating conjunction|although, when|
		|CJT|the conjunction that|
		|CRD|cardinal numeral (except one)|3, twenty-five, 734|
		|DPS|possessive determiner|your, their|
		|DT0|general determiner|these, some|
		|DTQ|wh-determiner|whose, which|
		|EX0|existential there|
		|ITJ|interjection or other isolate|oh, yes, mhm|
		|NN0|noun (neutral for number)|aircraft, data|
		|NN1|singular noun|pencil, goose|
		|NN2|plural noun|pencils, geese|
		|NP0|proper noun|London, Michael, Mars|
		|ORD|ordinal|sixth, 77th, last|
		|PNI|indefinite pronoun|none, everything|
		|PNP|personal pronoun|you, them, ours|
		|PNQ|wh-pronoun|who, whoever|
		|PNX|reflexive pronoun|itself, ourselves|
		|POS|possessive ’s or ’|
		|PRF|the preposition of|
		|PRP|preposition (except of)|for, above, to|
		|PUL|punctuation – left bracket|( or [|
		|PUN|punctuation – general mark|. ! , : ; - ? ...|
		|PUQ|punctuation – quotation mark|‘ ’ ”|
		|PUR|punctuation – right bracket|) or ]|
		|TO0|infinitive marker to|
		|UNC|unclassified items (not English)|
		|VBB|base forms of be (except infinitive)|am, are|
		|VBD|past form of be|was, were|
		|VBG|-ing form of be|being|
		|VBI|infinitive of be|
		|VBN|past participle of be|been|
		|VBZ|-s form of be|is, ’s|
		|VDB/D/G/I/N/Z|form of do|do, does, did, doing, to do, etc.|
		|VHB/D/G/I/N/Z|form of have|have, had, having, to have, etc.|
		|VM0|modal auxiliary verb|can, could, will, ’ll|
		|VVB|base form of lexical verb (except infin.)|take, live|
		|VVD|past tense form of lexical verb|took, lived|
		|VVG|-ing form of lexical verb|taking, living|
		|VVI|infinitive of lexical verb|take, live|
		|VVN|past participle form of lex. verb|taken, lived|
		|VVZ|-s form of lexical verb|takes, lives|
		|XX0|the negative not or n’t|
		|ZZ0|alphabetical symbol|A, B, c, d|
	* Getting help about POS in NLTK
		```python
		>>> nltk.help.upenn_tagset('MD')
			MD: modal auxiliary
			    can cannot could couldn't dare may might must need ought shall should
			    shouldn't will would
		```
	* How to do POS Tagging using NLTK
		1. Splitting a sentence into words/tokens
			```python
			>>> output_of_word_tokenize = nltk.word_tokenize(<STRING_TO_TOKENIZE>) # SYNTAX OF COMMAND
			>>> output_of_word_tokenize = nltk.word_tokenize("Children shouldn't drink a sugary drink before bed.")
			>>> output_of_word_tokenize
				['Children',
				 'should',
				 "n't",
				 'drink',
				 'a',
				 'sugary',
				 'drink',
				 'before',
				 'bed',
				 '.']
			```
		1. NLTK's Tokensizer
			```python
			>>> nltk.pos_tag(output_of_word_tokenize) # SYNTAX OF COMMAND
			# This is just a sample Command and String.
			>>> output_of_word_tokenize = nltk.word_tokenize("Children shouldn't drink a sugary drink before bed.")
			>>> nltk.pos_tag(output_of_word_tokenize)
				[('Children', 'NNP'),
				 ('should', 'MD'),
				 ("n't", 'RB'),
				 ('drink', 'VB'),
				 ('a', 'DT'),
				 ('sugary', 'JJ'),
				 ('drink', 'NN'),
				 ('before', 'IN'),
				 ('bed', 'NN'),
				 ('.', '.')]
			```


##### Review

1. POS tagging provides insights into the word classes/types in a sentence
1. Parsing the grammatical structures helps derive meaning
1. Both tasks are difficult, linguistic ambiguity increases the difficulty even more
1. Better models could be learned with supervised training
1. NLTK provides access to tools and data for training

