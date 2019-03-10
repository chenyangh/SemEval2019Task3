import emoji
import nltk
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


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


def processing_pipelie(text):
    text = text.lower().strip()
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = ' '.join(text_processor.pre_process_doc(text))
    text = remove_underscope_for_emoji(text)
    return text


def load_data_context(data_path='data/train.txt', is_train=True):
    EMOS_DIC = {'happy': 0,
                'angry': 1,
                'sad': 2,
                'others': 3}

    # data_path = 'data/train.txt'

    data_list = []
    target_list = []
    f_data = open(data_path, 'r')
    data_lines = f_data.readlines()
    f_data.close()
    for i, text in enumerate(data_lines):
        # skip the first line
        if i == 0:
            continue

        tokens = text.split('\t')

        convers = tokens[1:4]

        # normal preprocessing
        a = convers[0]
        b = convers[1]
        c = convers[2]

        a = processing_pipelie(a)
        b = processing_pipelie(b)
        c = processing_pipelie(c)

        data_list.append(a + b + c)
        if is_train:
            emo = tokens[3 + 1].strip()
            target_list.append(EMOS_DIC[emo])

    if is_train:
        return data_list, target_list
    else:
        return data_list

