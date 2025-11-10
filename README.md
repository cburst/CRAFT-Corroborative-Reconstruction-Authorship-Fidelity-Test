🧾 these python scripts can verify text authorship by examining claimed authors' familiarity with the text content.  
🧾 each script starts with a students.tsv file that contains student numbers, student names, and text columns.
  
here is the initial batch:
  
&nbsp;&nbsp;&nbsp;&nbsp;📜 longest-words.py  
this script identifies the 10 longest words in each text sample, replaces them with blanks, and the claimed author should fill-in-the-blanks.  
*requires weasyprint

&nbsp;&nbsp;&nbsp;&nbsp;📜 rarest-words.py  
this script identifies the 10 rarest words in each text sample using the wikipedia word frequency list, and the claimed author should be able to make a new sentence with these words.  
*requires weasyprint and wiki_freq.txt (included)

&nbsp;&nbsp;&nbsp;&nbsp;📜 authorship-recognition.py  
this script uses an LLM to create two plausible decoy sentences for 5 sentences in the original text sample, and the claimed author should be able to identify the sentence they created.  
*requires weasyprint, nltk, and a deepseek API key (compatible with other OpenAI format LLM APIs)

&nbsp;&nbsp;&nbsp;&nbsp;📜 error-repair.py  
this script creates intentional verb tense or preposition errors in the 5 longest sentences in the original text sample, and the claimed author should be able to identify and resolve the errors.   
*requires weasyprint, nltk
