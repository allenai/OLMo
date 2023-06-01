Author: Akshita Bhagia @akshitab

# Version: v2

* Starting from v1.
* Removed documents matching the following criteria (Reference: RedPajama code filtering heuristics):
	* Maximum line length > 1000 characters
	* Average line length > 100 characters
	* Proportion of alphanumeric characters < 0.25
	* Ratio of alphabetical characters to number of tokens < 1.5
* Number of unicode tokens: 212 billion.
