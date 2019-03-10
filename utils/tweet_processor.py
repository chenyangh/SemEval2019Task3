import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import multiprocessing
import numpy as np
from multiprocessing import Pool
import emoji
import string
printable = set(string.printable)

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    # annotate={"hashtag", "allcaps", "elongated", "repeated",
    #           'emphasis', 'censored'},

    annotate={"repeated", "emphasis", "elongated"},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def process_tweet(s):
    return ' '.join(text_processor.pre_process_doc(remove_tags(s)))


def tweet_process(text):
    text = ' '.join(text_processor.pre_process_doc(remove_tags(text)))
    text = emoji.demojize(text, delimiters=(' ', ' '))
    tokens = text.split()
    ret_list = []
    for token in tokens:
        if len(token) > 3 and '_' in token:
            token = token.replace('_', ' ')

        if token[0] == '<' and token[-1] == '>':
            token = token[1:-1]

        ret_list.append(token)
    text = ' '.join(ret_list)
    return text


def remove_dup_emoji(sent):
    ret = []
    for word in sent.split():
        emo_found = [char for char in word if char in UNICODE_EMOJI]
        if len(emo_found) > 1:
            word = emo_found[0]
        ret.append(word)
    return ' '.join(ret)


def remove_underscope_for_emoji(text):
    tokens = text.split()
    ret_list = []
    for token in tokens:
        if len(token) > 3 and '_' in token:
            token = token.replace('_', ' ')

        if token[0] == '<' and token[-1] == '>':
            token = token[1:-1]

        ret_list.append(token)
    return ' '.join(ret_list)


def only_printable(text):
    """
    Usage Warning, for the sake of efficient, this method did not rejoin the string with space
    Therefore, in the 'processing_pipeline', I put it before 'remove_underscope_for_emoji'
    """

    text = ''.join([x for x in text if x in printable])
    return text


def processing_pipeline(text):
    text = text.lower().strip()
    # text = remove_dup_emoji(text)
    text = ' '.join(text_processor.pre_process_doc(text))
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = remove_underscope_for_emoji(text)
    return text



# print(processing_pipelie('e day હત ા શ ા ર ો ગ મ ા ટ ે હ ો મ ી ય ો પ ે થ ી homeop'))