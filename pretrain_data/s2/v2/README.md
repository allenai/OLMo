Author: Luca Soldaini [@soldni](github.com/soldni)

# TODO

- [x] Get list of full-text papers that are included in V1 (S2ORC) subset
- [ ] Run new mmda recipe on the full-text papers


# Get IDs of full-text papers

We want to get IDs of training papers so that we can re-run mmda on them
instead of relying on the grobid parses. To do so, we run the following
queries:

1. Load papers with full text: `full_text_ids/step1_load_s2orc_data.sql`
2. Get IDs of training papers: `full_text_ids/step2_save_train_ids.sql`
3. Get IDs of validation papers: `full_text_ids/step3_save_valid_ids.sql`
