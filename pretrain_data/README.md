# pretrain-data

working place for pretraining data curation. please contact:
- common crawl: rodneyk@allenai.org
- semantic scholar data: lucas@allenai.org
- the stack (code): akshitab@allenai.org

### requirements

1. Each dataset should exist as a JSONLines file. 
```python
{"text": "This is the text"}
``` 


### open questions

1. Where should "shared" utilities live? What are they?
2. Shared dependencies or treat each source as own project?
3. Comfortable pushing directly, or pull requests w/ reviews? Who approving?
4. Keeping developer logs. GitHub likely better than Google Docs. Maybe define a minimal convention/cadence for this? Can be rough. 
