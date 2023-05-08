# Filtering Heuristics

Basic character-dsitribution statustics taken from Gopher for junk filtering.

```
python3 gopher.py documents/cc_en_head/cc_en_head-0055.json.gz attributes/cc_en_head/cc_en_head-0055.json
```

These filters remove about 28% of the content when applied to `cc_en_head`

```
cat attributes/cc_en_head/cc_en_head-0055.json | \
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
' > filter_counts.txt

$ wc -l filter_counts.txt 
536095 filter_counts.txt

$ grep -c '"","","","","","","","","","","","","","","","","",""' filter_counts.txt
387617
```

Number of documents dropped due to each rule:
```
$ for filter in fraction_of_characters_in_duplicate_lines \
fraction_of_characters_in_duplicate_10grams \
fraction_of_characters_in_duplicate_5grams \
fraction_of_characters_in_duplicate_6grams \
fraction_of_characters_in_duplicate_7grams \
fraction_of_characters_in_duplicate_8grams \
fraction_of_characters_in_duplicate_9grams \
fraction_of_characters_in_most_common_2gram \
fraction_of_characters_in_most_common_3gram \
fraction_of_characters_in_most_common_4gram \
fraction_of_duplicate_lines \
fraction_of_lines_ending_with_ellipsis \
fraction_of_lines_starting_with_bullet_point \
fraction_of_words_with_alpha_character \
median_word_length \
required_word_count \
symbol_to_word_ratio \
word_count
do
echo $filter `grep -c \"$filter\" filter_counts.txt`
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

Examples of text dropped due to each rule:
```
$ for filter in fraction_of_characters_in_duplicate_lines \
fraction_of_characters_in_duplicate_10grams \
fraction_of_characters_in_duplicate_5grams \
fraction_of_characters_in_duplicate_6grams \
fraction_of_characters_in_duplicate_7grams \
fraction_of_characters_in_duplicate_8grams \
fraction_of_characters_in_duplicate_9grams \
fraction_of_characters_in_most_common_2gram \
fraction_of_characters_in_most_common_3gram \
fraction_of_characters_in_most_common_4gram \
fraction_of_duplicate_lines \
fraction_of_lines_ending_with_ellipsis \
fraction_of_lines_starting_with_bullet_point \
fraction_of_words_with_alpha_character \
median_word_length \
required_word_count \
symbol_to_word_ratio \
word_count
do
echo "----------" >> samples.txt
echo $filter >> samples.txt
echo "--" >> samples.txt
line=`grep -n \"$filter\" filter_counts.txt | head -10 | sort -R | head -1 | cut -d: -f 1`
echo "line $line"
text=`gunzip --stdout documents/cc_en_head/cc_en_head-0055.json.gz | head -$line | tail -1 | jq '.text'`
echo $text | jq '.' --raw-output >> samples.txt
done
```
