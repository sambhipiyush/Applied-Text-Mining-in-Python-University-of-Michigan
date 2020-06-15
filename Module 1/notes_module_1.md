Applied Text Mining in Python
==============================

by University of Michigan

# Module 1
#
## Title: Working with Text in Python

#### Introduction to Text Mining

* What can be done with Text?
	1. Parse Text --> try to understand what it says
	1. Find / Identify / Extract relevant information from Text
	1. Classify Text Documents
	1. Search for relevant text documents --> this is information retrieval
	1. Sentiment analysis --> You could see whether something is positive or negative. Something is happy, something is sad, something's angry
	1. Topic Modelling --> identify what is a topic that is being discussed

#### Handling Text in Python

* Nothing much to write down

#### Regular Expressions

* Meta Characters : Character Matches
	* `.` --> wildcard, matches a single character, Any character, but just once
	* `^` --> indicates the start of the string
	* `$` --> represents the end of the string
		* if your string has a backslash end, the dollar comes after the backslash end.
	* `[]` --> matched one of the characters that are within the square bracket
		* `[A-Z]` --> You have a to z, for example, within square bracket that rematch one of the range of characters a, b up to z
		* `[^abc]` --> It means it matches a character that is not a or b, or c (over here ^ is inverse of match, when used inside square backets)
	* `a|b` --> matches either a or b, where a and b are strings
	* `()` --> Scoping for operators
	* `\` --> Escape character for special characters like `\n, \b \t`
	* `\b` --> Matches word boundary
	* `\d` --> Any digit ( any single digit ), equivalent to `[0-9]`
	* `\D` --> Any non-digit, equivalent to `[^0-9]`
	* `\s` --> Any whitespace character, equivalent to `[\t\n\r\f\v]`
	* `\S` --> Any non-whitespace character, equivalent to `[^\t\n\r\f\v]`
	* `\w` --> Any alphanumeric character, equivalen to `[A-Za-z0-9_]`
	* `\W` --> Any non-alphanumeric character, equivalen to `[^A-Za-z0-9_]`

* Meta Characters : Repetitions
	* `*` --> matches zero or more occurences
	* `+` --> matches one or more occurences
	* `?` --> matches zero or one occurences
	* `{n}` --> exactly **n** repetitions, **n** >= 0
	* `{n,}` --> atleast **n** repetitions
	* `{,n}` --> atmost **n** repetitions
	* `{m,n}` --> atleast **m** and atmost **n** repetitions

#### Internationalization and Issues with Non-ASCII Characters

* English and ASCII
	* ASCII - American Standard Code for Information Interchange
		- 7-bit character encoding standard: 128 valid codes
		- Range: 0x00 - 0x7F [(0000 0000)<sub>2</sub>] to (0111 1111)<sub>2</sub>]
			- it takes the seven bits out of eight bits and uses this lower half of the eight bit encoding
		- It includes alphabets both uppercase and lowercase, it has all ten digits, all punctuations
			- All common symbols like brackets for example or percentage sign and the dollar symbol, and the hash symbol
		- It has some control characters, some characters to describe let's say end of the line or the tab or some other control characters that are needed to see. A paragraph ends, for example, is different from a line end
		- Things which are not encoded by ASCII schema:
			- **Diacritics** are not ready encoded in the ASCII schema things
			- **International Languages** are also not encoded in the ASCII schema, like chinese, hindi, greek, russian
			- **Musical Symbols** are also not encoded in the ASCII schema
			- **Emoticon Symbols** are also not encoded in the ASCII schema
* Other Character Encoding
	1. **IBM EBCDIC** --> which is an 8-bit encoding
	1. **Latin-1** encoding --> slightly different from the ASCII encoding
	1. **JIS** ( **J**apanese **I**ndustrial **S**tandards )
	1. **CCCII** ( **C**hinese **C**haracter **C**ode for **I**nformation **I**nterchanges ) like ASCII
	1. **EUC** ( **E**xtended **U**nix **C**ode )


* **Unicode** and **UTF-8** encoding --> This encoding was introduced to standardize all the encoding methodologies into one

* **Unicode**
	1. **Unicode** is an industry standard for encoding and representing text.
	1. It has over 128,000 characters from 130 odd scripts and symbol sets
		1. symbol sets, include Greek symbols, for example, or symbols for the four suits in a card deck and so on.
	1. It can be implemented using different character endings and **UTF-8** is one of them
	1. **UTF-8** is an extendable encoding set
		1. It goes from one byte up to four bytes
		1. **UTF-16** which uses one or two 16-bit codes
		1. **UTF 16** is a 16 bit sets one or two of them
		1. **UTF-8** is 8 bit sets, so 8 bits is a byte, so you have 1 byte as a minimum and then it goes up to 4 bytes
		1. **UTF-8** kind of uses one big 32-bit encoding for all characters
		1. **UTF-32** is one 32 bit encoding, so even though you have all of these kind of using up to 32 bits

* **UTF-8**
	* **UTF-8** stands for **U**nicode **T**ransformational **F**ormat-**8**-bits, to distinguish it from 16 bit and 32 bit UTF formats
	* Have variable length encoding, it goes from one byte to four bytes
	* it is also **backward compatible** with ASCII
		* For example, all ASCII codes that were seven bit codes and used the leading zero in ASCII are the same codes in UTF-8
			* So all those that where encoded using ASCII use one byte of information in UTF-8 and uses one byte that is similar to what ASCII says
	* **UTF-8** is the dominant character encoding for the Web
