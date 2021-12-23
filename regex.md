## Questions
### Question: What is a regular expression? How do you create it?

Answer: Today, a regular expression (aka RE, regex, regex pattern, regexp) is a search pattern that can be used for advanced "find", "find and replace", or "match" operations on strings. An example regular expression is `r"[0-9]"`. It matches 0, 1, ..., 9. Regular expression get its name from regular languages, the languages that can be recognized by a state machine (a concise way to build rule-based AI). It has sense since evolved from its theoretical computer science roots. Modern programming language such as python made it easy to use regexes. In python, you first "compile" a regex using a string of characters and "metacharacters" using a python library called `re`. This creates a regex object that has in built operations for "matching" and "searching" for strings. 

```
>>> import re
>>> p = re.compile(r"[0-9]")
>>> m = p.match("abc")
>>> print(m)
None
>>> m = p.match("122")
>>> print(m)
<re.Match object; span=(0, 1), match='1'>
```

However, the full power of regular expressions is not required for straightforward patterns. For example, to replace a single fixed string with another one, use replace() method:
```
>>> sentence = "I can use regex"
>>> sentence.replace("regex", "string replacement")
'I can use string replacement'
```
To delete every occurrence of a few characters or replace a set of characters with a corresponding set, use translate() or maketrans().
```
>>> sentence = "î 423424wàñt tô rèmô3243243242vè àł34234234ł dîàçrîtîçś 4234234234and delete234234 nu423mbers." 
>>> sentence.translate(sentence.maketrans("ñôèłàîçś", "noelaics", "0123456789"))
'i want to remove all diacritics and delete numbers.'
```

### Question: What is the purpose of the different metacharacters in python regexes?

Answer: Metacharacters give special powers to the characters in their scope. The complete list of metacharacters is `\` `[ ]` `| ^ $ . + * { } ? ( )`. To treat any of these metacharacters as a regular character, you would "escape" it by adding a backslash to the beginning. For example, the python regular expression to match the backslash itself is re"\\".

The `\` itself is used to match predefined sequences, some of which are listed below:

* "\d" matches any decimal digit; same as "[0-9]" or "0|1|2|3|4|5|6|7|8|9"

* "\D" matches any non-digit; same as "[^0-9]"

* "\s" matches any whitespace character; same as "[ \t\n\r\f\v]"

* "\S" matches any non-whitespace character; same as "[^ \t\n\r\f\v]"

* "\w" matches any alphanumeric character; same as "[a-zA-Z0-9_]"

* "\W" matches any non-alphanumeric character; same as "[^a-zA-Z0-9_]"

`[ ]` are used to definition new sequences (aka character classes) from scratch as you can see in the equivalent versions of the above examples

As you can see in the first example above, `|` is used as an OR operator

As you can see in the alternative examples above, a `^` at the beginning of character class complements (aka negates) the definition of the character class

A `^` is also used to match the beginning of the line, while a $ is used to match the end of a line. 

`.` matches anything except a newline character. If you compile with re.DOTALL instead of just re, it matches any character (even a newline).

`+` at the end of a sequence of characters means that part is matched one or more times. For example, r"\d+" matches 7, 17, 552, etc.

`*` at the end of a sequence of characters means that part is matched zero or more times. For example, r"a\d*b" matches ab, a7b, a17b, a552b, etc.

`?` at the end of a sequence of characters means that part is matched optionally; that is, zero or one times. For example, r"a\d?b" matches ab, a7b, etc. `?` following one of the repeater metacharacters, such as `+`, `*` or the one below, means the least number of characters are matched as opposed to the default "greedy matching". For example,
```
>>> p = re.compile(r"\d+")
>>> m = p.match("123a")
>>> print(m)
<re.Match object; span=(0, 3), match='123'>
>>> p = re.compile(r"\d+?")
>>> m = p.match("123a")
>>> print(m)
<re.Match object; span=(0, 1), match='1'>
``` 

`{m, n}`, where m and n are integers, at the end of a sequence of characters means that part is matched at least m times and at most n times. Thus `+` is same as {1, }; `*` is same as {0, }; `?` is same as {0, 1}.

`( )` are used to group sequence of characters so it is easy to apply other metacharacters for the entire scope. For example, r"ab+" matches ab and abb, but to match abab we need r"(ab)+" as shown below:
```
>>> p = re.compile(r"ab+")
>>> m = p.match("abb")
>>> print(m)
<re.Match object; span=(0, 3), match='abb'>
>>> m = p.match("abab")
>>> print(m)
<re.Match object; span=(0, 2), match='ab'>
>>> p = re.compile(r"(ab)+")
>>> m = p.match("abab")
>>> print(m)
<re.Match object; span=(0, 4), match='abab'>
```

`( )` are used to match groups of characters within a string. For example:
```
>>> p = re.compile(r"([a-zA-Z0-9-\.]+)@([a-zA-Z0-9]+).([a-zA-Z]+)")
>>> m = p.match("servants@ubf.works")
>>> print(m)
<re.Match object; span=(0, 18), match='servants@ubf.works'>
>>> print(m.groups())
('servants', 'ubf', 'works')
```

## Practice Questions
Using the documentation for `re` and `str` in the python docs, try the below in terminal (`$ python`) and insert the code and output following the above example.
### Practice: Compile a regex for the pattern _u*bf_. What will it match and not match? Demonstrate that your understanding is correct.
```



```

### Practice: Do the above with _u[abcd]*f_.
```



```

### Practice: Read what arguments `re.compile()` could accept besides the regex. Demonstrate your understanding by compiling a regex to match _ubf_ by ignoring the case; that is it should match _Ubf_ and _UBF_ also.
```



```

### Practice: Demonstrate how `re.match()` is used to match the beginning of a string.
```



```

### Practice: Demonstrate how `re.search()` is used to scan a string for a pattern.
```



```

### Practice: Demonstrate how `re.findall()` and `re.finditer()` are used to find all matches within a string? How do these methods differ?
```



```

### Practice: How do you get the starting position and ending position of a match? How do you get the exact span matched?
```



```

### Practice: How do you match in one line without needing to compile? When is it okay to use this approach?
```



```

### Practice: Demonstrate how you would use grouping to extract the optional international code (+1), 3-digit area code and the remaining 7-digit in a ten-digit US phone number.
```



```

### Practice: Read about non-capturing and named groups, and demonstrate your understanding.
```



```

### Practice: Read about lookahead assertions and demonstrate your understanding.
```



```

### Practice: Implement a simple regular expression based tokenizer. A tokenizer is a method that takes a sentence and splits the sentence into individual tokens. A simple version is to split on whitespace (space, tab, newline). An example implementation is at https://www.nltk.org/_modules/nltk/tokenize/regexp.html. Use the split() function in `re`.
```



```

### Practice: Implement a tokenizer that splits a sentence into a sequence of alphabetic a non-alphabetic characters
```



```

### Practice: Implement a tokenizer that forms tokens out of alphabetic sequences, money expressions, and any other non-whitespace sequences
```



```

### Practice: Implement a tokenizer that selects just the capitalized words.
```



```

### Practice: Read about [named entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) in wikipedia or some other introductory source. Implement a method to replace all currency named entities (e.g., $20.54) in a sentence with a mask (e.g., MONEY). Use sub() method after compiling the regular expression.
```



```

### Practice: Implement the below string methods as regular expressions. This is just for practice and the original string functions would be more efficient.
| Method         | Description                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------|
| capitalize()   | Converts the first character to upper case                                                    |
| endswith()     | Returns true if the string ends with the specified value                                      |
| find()         | Searches the string for a specified value and returns the position of where it was found      |
| isalnum()      | Returns True if all characters in the string are alphanumeric                                 |
| isalpha()      | Returns True if all characters in the string are in the alphabet                              |
| isascii()      | Returns True if all characters in the string are ascii characters                             |
| isdecimal()    | Returns True if all characters in the string are decimals                                     |
| isdigit()      | Returns True if all characters in the string are digits                                       |
| isidentifier() | Returns True if the string is an identifier                                                   |
| islower()      | Returns True if all characters in the string are lower case                                   |
| isnumeric()    | Returns True if all characters in the string are numeric                                      |
| isprintable()  | Returns True if all characters in the string are printable                                    |
| isspace()      | Returns True if all characters in the string are whitespaces                                  |
| istitle()      | Returns True if the string follows the rules of a title                                       |
| isupper()      | Returns True if all characters in the string are upper case                                   |
| lower()        | Converts a string into lower case                                                             |
| lstrip()       | Returns a left trim version of the string                                                     |
| replace()      | Returns a string where a specified value is replaced with a specified value                   |
| rfind()        | Searches the string for a specified value and returns the last position of where it was found |
| rstrip()       | Returns a right trim version of the string                                                    |
| split()        | Splits the string at the specified separator, and returns a list                              |
| splitlines()   | Splits the string at line breaks and returns a list                                           |
| startswith()   | Returns true if the string starts with the specified value                                    |
| strip()        | Returns a trimmed version of the string                                                       |
| swapcase()     | Swaps cases, lower case becomes upper case and vice versa                                     |
| title()        | Converts the first character of each word to upper case                                       |
| upper()        | Converts a string into upper case                                                             |
```



```

### Practice: what does regular expression correspond to - `[a‐zA‐Z0‐9!#$%&''*+,/=?^_`{|}~‐])+(\\.[a‐zA‐Z0‐9!#$%&''*+/=?^_`{|}~‐]+)*@([A‐Za‐z0‐9]([A‐Za‐z0‐9‐]*[A‐Za‐z0‐9])?\\.)+[A‐Za‐z0‐9]([A‐Za‐z0‐9‐]*[A‐Za‐z0‐9])?`? Implement it in verbose mode and comment each individual part.
```



```

## Homework
### Below is the description of ELIZA from [Wikipedia](https://en.wikipedia.org/wiki/ELIZA). Implement a version of ELIZA in Python in an IDE like VS Code and insert the code at the end of the description.
In 1966, he published a comparatively simple program called ELIZA, named after the ingenue in George Bernard Shaw's Pygmalion, which performed natural language processing. Driven by a script named DOCTOR, it was capable of engaging humans in a conversation which bore a striking resemblance to one with an empathic psychologist. Weizenbaum modeled its conversational style after Carl Rogers, who introduced the use of open-ended questions to encourage patients to communicate more effectively with therapists. The program applied pattern matching rules to statements to figure out its replies. (Programs like this are now called chatbots.) It is considered the forerunner of thinking machines. Weizenbaum was shocked that his program was taken seriously by many users, who would open their hearts to it. Famously, when observing his secretary using the software - who was aware that it was a simulation - she asked Weizenbaum: "would you mind leaving the room please?". ELIZA itself examined the text for keywords, applied values to said keywords, and transformed the input into an output; the script that ELIZA ran determined the keywords, set the values of keywords, and set the rules of transformation for the output. Weizenbaum chose to make the DOCTOR script in the context of psychotherapy to "sidestep the problem of giving the program a data base of real-world knowledge", as in a Rogerian therapeutic situation, the program had only to reflect back the patient's statements. The algorithms of DOCTOR allowed for a deceptively intelligent response, which deceived many individuals when first using the program.
```



```