# Filtering Heuristics

Content-based rules for junk filtering.

My subjective opinion:
 - C4's text-modification rules are good
 - C4's document-removal rules are too aggressive
 - Gopher's document-removal rules are good
   - But they are often triggered by lines that C4 would remove

## Gopher

From https://arxiv.org/abs/2112.11446

```
python3 gopher.py documents/cc_en_head/cc_en_head-0055.json.gz attributes/gopher/cc_en_head/cc_en_head-0055.json
```

These filters remove about 28% of the content when applied to `cc_en_head`

```
cat attributes/gopher/cc_en_head/cc_en_head-0055.json | \
jq --compact-output \
'[
if .attributes.word_count < 50 or .attributes.word_count > 100000 then "word_count" else "" end,
if .attributes.median_word_length < 3 or .attributes.median_word_length > 10  then "median_word_length" else "" end,
if .attributes.symbol_to_word_ratio > 0.1 then "symbol_to_word_ratio" else "" end,
if .attributes.fraction_of_words_with_alpha_character < 0.8 then "fraction_of_words_with_alpha_character" else "" end,
if .attributes.required_word_count < 2 then "required_word_count" else "" end,
if .attributes.fraction_of_lines_starting_with_bullet_point > 0.9 then "fraction_of_lines_starting_with_bullet_point" else "" end,
if .attributes.fraction_of_lines_ending_with_ellipsis > 0.3 then "fraction_of_lines_ending_with_ellipsis" else "" end,
if .attributes.fraction_of_duplicate_lines > 0.3 then "fraction_of_duplicate_lines" else "" end,
if .attributes.fraction_of_characters_in_duplicate_lines > 0.3 then "fraction_of_characters_in_duplicate_lines" else "" end,
if .attributes.fraction_of_characters_in_most_common_2gram > 0.20 then "fraction_of_characters_in_most_common_2gram" else "" end,
if .attributes.fraction_of_characters_in_most_common_3gram > 0.18 then "fraction_of_characters_in_most_common_3gram" else "" end,
if .attributes.fraction_of_characters_in_most_common_4gram > 0.16 then "fraction_of_characters_in_most_common_4gram" else "" end,
if .attributes.fraction_of_characters_in_duplicate_5grams > 0.15 then "fraction_of_characters_in_duplicate_5grams" else "" end,
if .attributes.fraction_of_characters_in_duplicate_6grams > 0.14 then "fraction_of_characters_in_duplicate_6grams" else "" end,
if .attributes.fraction_of_characters_in_duplicate_7grams > 0.13 then "fraction_of_characters_in_duplicate_7grams" else "" end,
if .attributes.fraction_of_characters_in_duplicate_8grams > 0.12 then "fraction_of_characters_in_duplicate_8grams" else "" end,
if .attributes.fraction_of_characters_in_duplicate_9grams > 0.11 then "fraction_of_characters_in_duplicate_9grams" else "" end,
if .attributes.fraction_of_characters_in_duplicate_10grams > 0.10 then "fraction_of_characters_in_duplicate_10grams" else "" end
]
' > gopher_filter_counts.txt

$ wc -l gopher_filter_counts.txt 
536095 gopher_filter_counts.txt

$ grep -c '"","","","","","","","","","","","","","","","","",""' gopher_filter_counts.txt
387617
```

Number of documents dropped due to each rule:
```
$ for filter in fraction_of_characters_in_duplicate_lines \
fraction_of_characters_in_duplicate_10grams fraction_of_characters_in_duplicate_5grams fraction_of_characters_in_duplicate_6grams fraction_of_characters_in_duplicate_7grams fraction_of_characters_in_duplicate_8grams fraction_of_characters_in_duplicate_9grams fraction_of_characters_in_most_common_2gram fraction_of_characters_in_most_common_3gram fraction_of_characters_in_most_common_4gram fraction_of_duplicate_lines fraction_of_lines_ending_with_ellipsis fraction_of_lines_starting_with_bullet_point fraction_of_words_with_alpha_character median_word_length required_word_count symbol_to_word_ratio word_count
do
echo $filter `grep -c \"$filter\" gopher_filter_counts.txt`
done

fraction_of_characters_in_duplicate_lines 0
fraction_of_characters_in_duplicate_10grams 21670
fraction_of_characters_in_duplicate_5grams 32865
fraction_of_characters_in_duplicate_6grams 28702
fraction_of_characters_in_duplicate_7grams 25774
fraction_of_characters_in_duplicate_8grams 23830
fraction_of_characters_in_duplicate_9grams 22557
fraction_of_characters_in_most_common_2gram 29600
fraction_of_characters_in_most_common_3gram 46504
fraction_of_characters_in_most_common_4gram 60444
fraction_of_duplicate_lines 0
fraction_of_lines_ending_with_ellipsis 7496
fraction_of_lines_starting_with_bullet_point 139
fraction_of_words_with_alpha_character 35390
median_word_length 5387
required_word_count 63713
symbol_to_word_ratio 594
word_count 104182
```

The rules do a pretty good job of identifying junk, IMO. Examples of documents dropped due to each rule:
```
$ for filter in fraction_of_characters_in_duplicate_lines \
fraction_of_characters_in_duplicate_10grams fraction_of_characters_in_duplicate_5grams fraction_of_characters_in_duplicate_6grams fraction_of_characters_in_duplicate_7grams fraction_of_characters_in_duplicate_8grams fraction_of_characters_in_duplicate_9grams fraction_of_characters_in_most_common_2gram fraction_of_characters_in_most_common_3gram fraction_of_characters_in_most_common_4gram fraction_of_duplicate_lines fraction_of_lines_ending_with_ellipsis fraction_of_lines_starting_with_bullet_point fraction_of_words_with_alpha_character median_word_length required_word_count symbol_to_word_ratio word_count
do
echo "----------" >> gopher_samples.txt
echo $filter >> gopher_samples.txt
echo "--" >> gopher_samples.txt
line=`grep -n \"$filter\" gopher_filter_counts.txt | head -10 | sort -R | head -1 | cut -d: -f 1`
echo "line $line"
text=`gunzip --stdout documents/cc_en_head/cc_en_head-0055.json.gz | head -$line | tail -1 | jq '.text'`
echo $text | jq '.' --raw-output >> gopher_samples.txt
done

$ grep -A 8 -- '--------' gopher_samples.txt | cut -b -150
----------
fraction_of_characters_in_duplicate_lines
--
Tag Archive for Tag: Bonita
National Bank posts huge beat as earnings surge to record
MONTREAL — National Bank beat expectations as a strong performance across its business contributed to record adjusted profits in the quarter.[np_sto
Categories: ugcmbirc | Key Tags: Birdie, Bonita, Cate, Edmund, Lovella, Odelia
----------
fraction_of_characters_in_duplicate_10grams
--
bbc judaism rosh hashanah
Explore our worksheets for children online at iChild judaism (originally from hebrew יהודה ‎, yehudah, judah ; via latin and greek) is an ancie
BBC - Schools - Religion - Judaism
BBC - Religions - Judaism: Rosh Hashanah
Judaism 101: Rosh Hashanah
The link is to the [excellent] bbc featuring activies that go well with the powerpoint bbc: ‘typically british‘ fish & chips introduced by jews ne
--
----------
fraction_of_characters_in_duplicate_5grams
--
Nacido en: Paris, France
From Wikipedia, the free encyclopedia Gérard Jugnot (born 4 May 1951 in Paris) is a French actor, film director, screenwriter and producer. Jugnot wa
From Wikipedia, the free encyclopedia Gérard Jugnot (born 4 May 1951 in Paris) is a French actor, film director, screenwriter and producer. Jugnot wa
Benito Sansón y los taxis rojos
----------
fraction_of_characters_in_duplicate_6grams
--
Light can carry both orbital and spin angular momentum. Orbital angular momentum (OAM) of light is a consequence of its spacial distribution, such as 
Light can carry both orbital and spin angular momentum. Orbital angular momentum (OAM) of light is a consequence of its spacial distribution, such as 
Light can carry both orbital and spin angular momentum. Orbital angular momentum (OAM) of light is a consequence of its spacial distribution, such as 
Diagram of wavefront helicity and associated topological charge
----------
fraction_of_characters_in_duplicate_7grams
--
Formerly Wilson Estes Police Architects, we recently changed our name to more clearly. This basic layout can be used on ranches, in feedlots for the m
----------
fraction_of_characters_in_duplicate_8grams
--
10. Dennis Martinez
Published in Top 50 Washington Nationals
Born: May 14, 1954 in Granada, Gr Nicaragua
The greatest Pitcher to come out of Nicaragua, Dennis Martinez had the best years of his long career with the Montreal Expos where he would be a three
Acquired: Traded form the Baltimore Orioles with a player to be named later for a player to be named later 6/16/86.
Departed: Signed as a Free Agent with the Cleveland Indians 12/2/93.
--
----------
fraction_of_characters_in_duplicate_9grams
--
图书 共有 186 册关于“Congress by less than two nor by more than seven members ; and no person shall be...”的图书，以下是第 51 - 60 
Congress by less than two nor by more than seven members ; and no person shall be capable of being a delegate for more than three years in any term of
The Juvenile Mentor, Or Select Readings: Being American School Class Book No ... - 第253页
作者：Albert Picket - 1820 - 282 页
The Family Library (Harper)., 第 160 卷
...shall be represented in Congress by less than two, nor more than seven members ; and no person shall be capable of being a delegate for more than t
--
----------
fraction_of_characters_in_most_common_2gram
--
Conturelle ($19 to $56)
Millesia ($20 to $65)
----------
fraction_of_characters_in_most_common_3gram
--
Revision as of 14:12, 3 March 2016 by Greg (Talk | contribs)
----------
fraction_of_characters_in_most_common_4gram
--
GEDCOM starting from: MOORE Erma Ernestine
Number of generations: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
----------
fraction_of_duplicate_lines
--
Tag Archive for Tag: Bonita
National Bank posts huge beat as earnings surge to record
MONTREAL — National Bank beat expectations as a strong performance across its business contributed to record adjusted profits in the quarter.[np_sto
Categories: ugcmbirc | Key Tags: Birdie, Bonita, Cate, Edmund, Lovella, Odelia
----------
fraction_of_lines_ending_with_ellipsis
--
Jane Salee
Jane is the creator of Rock Meets Soil, a community-based blog and gallery sharing stories and artwork to broaden the circle of inspiration in the wor
https://www.rockmeetssoil.com/
New Van, New Plan
With spring right around the corner, transformation seems to be a common theme. In late…
Jane SaleeMarch 23 2017
--
----------
fraction_of_lines_starting_with_bullet_point
--
- DVP Food security and agriculture (participant: Saudi Arabia)
----------
fraction_of_words_with_alpha_character
--
Thursday, October 6, 2011 from 4 – 6:30pm
Tuesday, August 30, 2011 at 11:33am and last updated
Thursday, April 19, 2012 at 1:50pm.
entrepreneurs, intellectual property, oen, workshop
----------
median_word_length
--
GEDCOM starting from: MOORE Erma Ernestine
Number of generations: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
----------
required_word_count
--
Prima Donna ($18 to $35)
----------
symbol_to_word_ratio
--
My "Daughter&#039;s" Project https://rickladd.com/2019/12/04/my-daughters-project/
----------
word_count
--
Fell Guide: Broom Fell
Baysdale
The Arctic Tern (Sterna Paradisaea) migrates further than any other bird, spending the breeding season in the Arctic and the rest of the year in the A

```

## C4

From https://arxiv.org/abs/1910.10683

```
python3 c4.py /data/scratch/documents/cc_en_head/cc_en_head-0055.json.gz /data/scratch/attributes/c4/cc_en_head/cc_en_head-0055.json
```

These rules remove ~42% of documents, and ~20% of characters from each retained document, primarily
due to discarding lines that do not end with a terminal punctuation mark. 

```
cat attributes/c4/cc_en_head/cc_en_head-0055.json | \
jq --compact-output \
'[
if .attributes.has_naughty_word then "has_naughty_word" else "" end,
if .attributes.has_javascript then "has_javascript" else "" end,
if .attributes.has_lorem_ipsum then "has_lorem_ipsum" else "" end,
if .attributes.has_curly_brace then "has_curly_brace" else "" end,
if .attributes.line_count < 5 then "line_count" else "" end
]
' > c4_filter_counts.txt

$ wc -l c4_filter_counts.txt
536095 c4_filter_counts.txt

$ grep -c '"","","","",""' c4_filter_counts.txt
310961

$ gunzip --stdout documents/cc_en_head/cc_en_head-0055.json.gz | jq '.text' --raw-output | wc -c
2173684482
$ cat attributes/c4/cc_en_head/cc_en_head-0055.json | jq '.modified_text' --raw-output | wc -c
1624173439
```

Number of documents dropped due to each rule:
```
$ for filter in has_naughty_word \
has_javascript has_lorem_ipsum has_curly_brace line_count
do
echo $filter `grep -c \"$filter\" c4_filter_counts.txt`
done

has_naughty_word 28336
has_javascript 551
has_lorem_ipsum 62
has_curly_brace 1025
line_count 197355
```

The blocklist rules throw out too many good documents, IMO. Line count rule is probably okay.

Examples of documents dropped due to each rule:
```
$ for filter in has_naughty_word \
has_javascript has_lorem_ipsum has_curly_brace line_count
do
echo "----------" >> c4_samples.txt
echo $filter >> c4_samples.txt
echo "--" >> c4_samples.txt
line=`grep -n \"$filter\" c4_filter_counts.txt | head -10 | sort -R | head -1 | cut -d: -f 1`
echo "line $line"
text=`gunzip --stdout documents/cc_en_head/cc_en_head-0055.json.gz | head -$line | tail -1 | jq '.text'`
echo $text | jq '.' --raw-output >> c4_c4_samples.txt
done

$ grep -A 8 -- '--------' c4_samples.txt | cut -b -150
----------
has_naughty_word
--
Something Instead of Nothing
Re: Something Instead of Nothing
by Meno_ » Thu Aug 22, 2019 6:27 am
Meno_ wrote: Nihilism, being a product of phenomenal reduction, works the opposite way, and hence the elementary contraindicated method works against 
Again, this is only my own personal reaction -- the embodiment of dasein -- but explanations of this sort are just intellectual gibberish to me.
Provide me with a particular example in which you are interacting with someone and the conversation shifts from what you are doing to an understanding
--
----------
has_javascript
--
Planning successful projects: The User Story
Wed, 05/22/2013 - 10:08 By John Locke
Hey, that's not what I was thinking!
That's a very common complaint customers have with developers, when they receive the result of weeks or months of hard work. And it indicates a failur
We've found nothing that works better to avoid this result than to write up and discuss user stories in detail.
What's a user story? It's a description of the process a person goes through to get a specific result, and what happens along the way.
--
----------
has_lorem_ipsum
--
Fringilla Proin suscipit luctus orci placerat Donec.
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500
I believe that a simple and unassuming manner of life is best for everyone, best both for the body and the mind.
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500
----------
has_curly_brace
--
Luke 7 25 Meaning
A centurion's servant is healed. From breaking news and entertainment to sports and politics, get the full story with all the live commentary. Among t
----------
line_count
--
Ava, United States

``` 

On the other hand, the rules do a good job of stripping junk text out of the documents it retains.

```shell
$ gunzip --stdout documents/cc_en_head/cc_en_head-0055.json.gz | head -3 | jq '.text' --raw-output > original.txt
$ cat attributes/c4/cc_en_head/cc_en_head-0055.json | head -3 | jq '.modified_text' --raw-output > modified.txt
$ diff --context original.txt modified.txt

- Tag: Kathryn Burns
- Dancing With Comedic Timing ~ Kathryn Burns
- Kathryn Burns Emmy’s Creative Arts 2016 Red Carpet
  Kathryn Burns (Kat) joins us on the red carpet at the Emmys® Creative Arts Awards. She danced her way to an Emmys® for Outstanding Choreography in My Crazy Ex-Girlfriend. She is the first choreographer nominated working with a scripted series.
  Joining her for the evening was her gorgeous mother and father. It is an exciting evening and red carpet for the family. Her family plays a large role in her dance career. She began dancing from her older sister’s influence.
  Burns dance training includes tap, ballet, contemporary, but her passion is hip-hop. After college, she moved to Hollywood. She was her dream was a backup dancer but released she was too tall and white.
  Kathryn turned her attention to comedy. This was not as successful as she envisioned. This left her feeling a little in between dance and comedy. One day someone asked her about dancing when she was doing comedy. She said, “I am a trained dancer.” This question changed her world.
  This opens doors to use her comedic training and dance to choreographing for the scripted comedy. She continues to work at Upright Citizens Brigade Theater, presenting live comedy. Improv, sketch, and stand-up. She has enjoyed being here for over 10 years.
- The challenges with choreography with a written script are working with people who may not have formal training, working within the set, characters, script, and camera angles, while meeting comedic timing. Kathryn said, “It is such an honor and humbling experience being recognized by my peers in dance.”
  Crazy Ex-Girlfriendawarded one Emmys® Creative Arts for Outstanding Choreography 2016.
- Connect with the Emmys here:
- Booth, Doris Regina (1895–1970)
- by Susan Gardner
  Doris Regina Booth (1895-1970), nursing volunteer and goldminer, was born at South Brisbane on 1 October 1895, daughter of Henry Wilde, clerk, and his wife Minna Christina, née Gerler. After a state school education, she enrolled as a trainee nurse at Brisbane General Hospital but met Captain Charles Booth, a shell-shocked soldier who had prospected in Papua before the war; she discontinued her training when they married on 14 May 1919. After twelve penurious months at Mitchell in western Queensland, Booth became a plantation manager for the New Guinea Expropriation Board at Raniolo near Kokopo. When he was discharged late in 1923, his wife took a share in four trade-stores and broke local convention by becoming a licensed recruiter of labour. Financed by Burns Philp & Co. Ltd, they went surreptitiously to Salamaua in 1924 following rumour of gold. Booth went ahead to the Bulolo valley while his wife secured her own miner's right, refused earlier in Rabaul; single-handed, she then spent five weeks taking a line of carriers from Salamaua to Bulolo. There, while her husband prospected, she was employed by William ('Sharkeye') Park to 'man' his lease.
  Booth pegged a lease in his wife's name, then left her as the only resident white woman at Bulolo to work it while he prospected at Edie Creek. From September 1926 to January 1927 she also organized and managed a racially segregated bush hospital to control a dysentery epidemic, treating over 32 patients at one time and more than 130 all told. For this work she received the O.B.E. in 1928 and became known locally as 'the Angel of Bulolo'.
  One of their leases was sold in April 1927 to Morobe Guinea Gold Ltd and Mrs Booth became a director of the firm.
  While Mrs Booth went to Australia for her health and for business between 1927 and 1930, the marriage began to collapse. Nevertheless, the couple travelled to the United States of America and England and while in London in 1928, with M. O'Dwyer as ghost-writer, she published Mountain Gold and Cannibals, a popularized version of her experiences. After her return to New Guinea in March 1929 she slowly wrested control over the family business affairs from her husband, whom she left early in 1932.
  Booth sued in the Central Court of the Territory of New Guinea in August 1933 for restitution of property. Since no Mandated Territory law explicitly safeguarded married women's property rights, it was a test case. Judge F. B. Phillips held that British and Australian Acts passed before 1921 superseded the common law notion of male control of joint property and gave Mrs Booth the verdict. When Booth appealed this particularly acrimonious case to the High Court of Australia, the judgment was upheld and territorial law was amended by the Status of Married Women Ordinance 1935-36.
  There is no evidence that the couple were ever formally divorced; Booth returned to prospecting while his wife became a successful mine-manager and company director. Settling in Brisbane in July 1938, she worked for the Mothercraft Association until after 1945 when she was involved in rebuilding her business from war-damage insurance. Appointed as the sole woman member of the first and second Legislative Councils of Papua-New Guinea in 1951-57, she supported mining interests, public health, secondary education for black and white, land and housing loans for Europeans and the sexual protection of native women. Doris Booth was a strong opponent of the liquor (natives) bill of 1955, and of a section in the public service bill (1953) restricting married women to temporary or exempt positions. She represented the women of Papua-New Guinea at the Pan-Pacific Women's Conference of 1955 in Manila. She retired to Brisbane in 1960, did volunteer work with the Methodist Blue Nursing Service, and died of coronary vascular disease at St Andrew's War Memorial Hospital on 4 November 1970.
- L. Rhys, High Lights and Flights in New Guinea (Lond, 1942)
- Commonwealth Law Reports, 53 (1935), 1-32
- Pacific Islands Monthly, June 1954, Dec 1970
- Rabaul Times, 8 Dec 1933
- High Court of Australia, Transcript of proceedings, 1934, annotated by C. Booth, manuscript 5669 (National Library of Australia)
- A518: AJ/824/1, AC836/3, A846/l/66, 81 (National Archives of Australia)
  Susan Gardner, 'Booth, Doris Regina (1895–1970)', Australian Dictionary of Biography, National Centre of Biography, Australian National University, http://adb.anu.edu.au/biography/booth-doris-regina-5289/text8921, published first in hardcopy 1979, accessed online 28 January 2020.
- Wilde, Doris Regina
- nurse (general)
- Joseph Lelyveld on Gandhi
  Joseph Lelyveld's new biography of Mahatma Gandhi, "Great Soul," has sparked controversy from India to California. For some, it has raised questions about Gandhi's sexual orientation. The Pulitzer Prize-winning author and former New York Times executive editor joins Michael Krasny to discuss his book, and the debate that swirls around it.
- For more podcast issues on this subject see Joseph Lelyveld about Gandhi on Anne is a Man
- The Neutrino
  Melvyn Bragg and guests discuss the neutrino, the so-called 'ghost particle'. With Frank Close, Susan Cartwright and David Wark.
- Hidden Heroes of the Belfast Blitz
  70 yrs ago, on Apr 15th 1941, Germany rained down bombs on Belfast - part of their WW2 offensive on Britain. The hidden story of that night is how the Republic of Ireland put its neutrality at risk by sending its firemen to help their northern brethern.
- Leonard Lopate Show
- The Eichmann Trial
  Deborah Lipstadt, Dorot Professor of Modern Jewish History and Holocaust Studies at Emory University, talks about the capture of SS Lieutenant Colonel Adolf Eichmann by Israeli agents in Argentina in May of 1960, and how his subsequent trial in Jerusalem by an Israeli court electrified the world and sparked a public debate on where, how, and by whom Nazi war criminals should be brought to justice. The Eichmann Trial gives an overview of the trial and analyzes the dramatic effect that the survivors’ courtroom testimony had on the world.
- Tina Fey Reveals All (And Then Some) In 'Bossypants'
  Story: Tina Fey's new memoir Bossypants contains her thoughts on juggling motherhood, acting, writing and executive producing 30 Rock. Fey joins Fresh Air's Terry Gross for a wide-ranging conversation about her years in comedy, her childhood and her 2008 portrayal of Sarah Palin on Saturday Night Live.
```
