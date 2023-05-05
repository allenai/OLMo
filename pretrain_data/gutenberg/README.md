# Gutenberg

Curator: @soldni

## Download

We use a script adapted from here to download https://www.exratione.com/2014/11/how-to-politely-download-all-english-language-text-format-files-from-project-gutenberg/

## Organization

```
books/
|-- raw/
    |-- doab/
        |-- data/
        |-- metadata/
    |-- gutenberg/
        |-- files/
            |-- 1219.txt
            |-- 12190.txt
            |-- ...
        |-- zipfiles/
            |-- 1219.zip
            |-- 12190.zip
            |-- ...
        |-- zipfileLinks.txt
```

## Statistics


## Summary

### `raw/`


The **start** of each file looks like:
```
The Project Gutenberg EBook of Juana, by Honore de Balzac

This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.org


Title: Juana

Author: Honore de Balzac

Translator: Katharine Prescott Wormeley

Release Date: August, 1998  [Etext #1437]
Posting Date: February 25, 2010

Language: English

Character set encoding: ASCII

*** START OF THIS PROJECT GUTENBERG EBOOK JUANA ***
```

and the **end** of each file looks like:
```
End of the Project Gutenberg EBook of Juana, by Honore de Balzac

*** END OF THIS PROJECT GUTENBERG EBOOK JUANA ***

***** This file should be named 1437.txt or 1437.zip *****
This and all associated files of various formats will be found in:
        http://www.gutenberg.org/1/4/3/1437/

Produced by John Bickers, and Dagny

Updated editions will replace the previous one--the old editions
will be renamed.
```

The **middle** of each file looks like the book text:

```
Produced by John Bickers, and Dagny


JUANA


BY HONORE DE BALZAC


Translated By Katharine Prescott Wormeley



                             DEDICATION

                  To Madame la Comtesse Merlin.



JUANA

(THE MARANAS)


CHAPTER I. EXPOSITION

Notwithstanding the discipline which Marechal Suchet had introduced into
his army corps, he was unable to prevent a short period of trouble and
disorder at the taking of Tarragona. 
```

#### Processing Considerations

* Whitespaces are preserved in the `raw` dump. That is, multiple consecutive newlines, many tabs and spaces for formatting, etc.

* We'll want to remove these boilerplates before & after each book. 

* There's a lot more frequent newline splitting in the books corpus than expected. 