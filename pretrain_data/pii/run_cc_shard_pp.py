''' Run CC shard PII extraction with post processing rules over regexes'''

import argparse
import re
import jsonlines
import time

from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

start_time = time.time()


def read_json_file(filename):
    '''
    :param filename: JSON file with data
    :return: list with data
    '''
    data = []

    with jsonlines.open(filename) as reader:
        for obj in reader:
            data.append(obj)

    return data


pattern_dict = None

def contains_url(text):
    '''
    Function to check if text contains URL
    '''

    # findall() has been used
    # with valid conditions for urls in string

    # regrex for url
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, text)
    return len(url) > 0
 
    



def postprocess_email(text_input, match, pii_start, pii_end):
    '''
    Function to post process email addresses
    Rules:
    (1) The email address besides the domain, cannot be only "("
    (2) There must be a "." in the domain
    '''
    addressee=match.split("@")[0]
    domain=match.split("@")[1]


    if addressee.strip()=="(" or "." not in domain:
        return False
    return True

def postprocess_phone_numbers(text_input, match, pii_start, pii_end):
    '''
    Function to post process email addresses
    Rules:
    (1) ISBN, DOI, or "#" cannot appear in a context window of 50 characters from the match
    (2) Cannot contain URL
    '''
    context_window = text_input[max(0, pii_start - 50): min(len(text_input), pii_end + 50)].lower()
    if "isbn" in context_window or "doi" in context_window or "#" in context_window or contains_url(context_window):
        return False
    return True


def postprocess_ip_addresses(text_input, match, pii_start, pii_end):
    '''
    Function to post process email addresses
    Rules:
    (1) ISBN, DOI, or "#" cannot appear in a context window of 50 characters from the match
    '''
    context_window = text_input[max(0, pii_start - 50): min(len(text_input), pii_end + 50)].lower()
    if "isbn" in context_window or "doi" in context_window or "#" in context_window:
        return False
    return True
    

def postprocess_pass(text_input, match, pii_type):
    match = str("".join(match))
    pii_start = text_input.find(match)
    pii_end = pii_start + len(match)

    if pii_type=="email":
        return postprocess_email(text_input, match, pii_start, pii_end)
    elif pii_type=="phone_numbers":
        return postprocess_phone_numbers(text_input, match, pii_start, pii_end)
    elif pii_type=="IP_addresses":
        return postprocess_ip_addresses(text_input, match, pii_start, pii_end)




def extract_pii_regex(batched_inputs: list[str],
                      context_window_one_side: int = 100):
    '''
    Function to identify PII using regular expressions
    :param batched_inputs: list of text inputs to extact PII
    :param context_window_one_side: how much additional context around the PII span to include in the output
    :return: PII in text
    '''
    pii = []

    for text_input in batched_inputs:
        for pii_type in pattern_dict:
            pattern = pattern_dict[pii_type]
            # search for the pattern in the string
            matches = pattern.findall(text_input.lower())
            # loop through the matches and print corresponding values from the dictionary
            for match in matches:
                if postprocess_pass(text_input, match, pii_type):
                    match = str("".join(match))
                    pii_start = text_input.find(match)
                    pii_end = pii_start + len(match)

                    pii.append(pii_type + " | " + match + " | " + text_input[
                                                                  max(0, pii_start - context_window_one_side): min(
                                                                      len(text_input), pii_end + context_window_one_side
                                                                  )
                                                                  ].replace("\n", " "))

    return pii


def extract_pii_presidio(batched_inputs: list[str],
                         context_window_one_side: int = 200):
    '''
    Function to identify PII using regular expressions
    :param batched_inputs: list of text inputs to extact PII
    :param context_window_one_side: how much additional context around the PII span to include in the output
    :return: PII in text
    '''
    pii = []

    for text_input in batched_inputs:
        analyzer_results = analyzer.analyze(text=text_input.lower(),
                                            entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "IBAN_CODE"],
                                            language='en')

        if len(analyzer_results) > 0:
            for res in analyzer_results:
                pii_start = res.start
                pii_end = res.end
                pii_type = res.entity_type
                match = text_input[pii_start:pii_end]

                pii.append(pii_type + " | " + match + " | " + text_input[
                                                              max(0, pii_start - context_window_one_side): min(
                                                                  len(text_input), pii_end + context_window_one_side
                                                              )
                                                              ].replace("\n", " "))

    return pii




def main():
    parse = argparse.ArgumentParser("")

    parse.add_argument("--in_file", type=str, help="file to analyze")
    parse.add_argument("--bs", type=int, default=100, help="batch size for inputs")
    parse.add_argument("--classifier", type=str, default="regex", help="regex or presidio")

    args = parse.parse_args()

    data = read_json_file(args.in_file)
    bs = args.bs
    inputs = []

    # Regular expressions for different types of PII
    global pattern_dict
    pattern_dict = {"email": re.compile("[.\s@,?!;:)(]*([^\s@]+@[^\s@,?!;:)(]+?)[.\s@,?!;:)(]?[\s\n\r]"),
                    "phone_numbers": re.compile("\s+\(?(\d{3})\)?[-\. ]*(\d{3})[-. ]?(\d{4})"),
                    "IP_addresses": re.compile(
                        "(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"),
                    }


    g = open("./analysis_results/" + args.classifier + "_" + args.in_file.split("/")[-1].split(".")[0] + "_pp.txt", "w")

    for row in data:

        # When we hit the correct batch size
        if len(inputs) == bs:
            if args.classifier == "regex":
                preds = extract_pii_regex(inputs)
            else:
                preds = extract_pii_presidio(inputs)
            for p in preds:
                g.write(p + "\n")
                # print(p)
            inputs = []
        if not row['text'] or row['text'].strip() == '': continue
        inputs.append(row['text'])

    if len(inputs) > 0:
        preds = extract_pii_regex(inputs)
        for p in preds:
            print(p)

    print(args)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
