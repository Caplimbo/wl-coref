""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""

import argparse
from contextlib import contextmanager
import datetime
import random
import sys
import time

import numpy as np  # type: ignore
import torch  # type: ignore

from coref import CorefModel
from transformers import RobertaTokenizerFast
import spacy
from spacy.tokens import Doc


class RbaTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def __call__(self, text):
        words = self.tokenizer.tokenize(text)
        # spacy.vocab.Vocab(self.tokenizer.get_vocab())
        return Doc(self.vocab, words=words)


# tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
NLP = spacy.load("en_core_web_sm", disable=["ner"])
# NLP.tokenizer = RbaTokenizer(NLP.vocab)


@contextmanager
def output_running_time():
    """Prints the time elapsed in the context"""
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """Seed random number generators to get reproducible results"""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def preprocess_doc(doc):
    token_map = {
        ".": ["."],
        ",": [","],
        "!": ["!"],
        "?": ["?"],
        ":": [":"],
        ";": [";"],
        "'s": ["'s"],
    }
    filter_func = lambda _: True
    # doc["span_clusters"] = [[tuple(mention) for mention in cluster]
    #                         for cluster in doc["span_clusters"]]
    word2subword = []
    subwords = []
    word_id = []
    for i, word in enumerate(doc["cased_words"]):
        tokenized_word = (
            token_map[word] if word in token_map else model.tokenizer.tokenize(word)
        )
        tokenized_word = list(filter(filter_func, tokenized_word))
        word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
        subwords.extend(tokenized_word)
        word_id.extend([i] * len(tokenized_word))
    doc["word2subword"] = word2subword
    doc["subwords"] = subwords
    doc["word_id"] = word_id
    return doc


def process_article(article: str):
    doc = {}
    doc["document_id"] = "nw001"
    processed = NLP(article)
    doc["cased_words"] = [token.text for token in processed]
    sents = [sent for sent in processed.sents]
    # print([sent for sent in sents])
    doc["sent_id"] = [sents.index(token.sent) for token in processed]
    doc["speaker"] = [None] * len(doc["cased_words"])
    doc["span_clusters"] = [[1, 2], [2, 3]]
    return doc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=("train", "eval"))
    argparser.add_argument("experiment")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument(
        "--data-split",
        choices=("train", "dev", "test"),
        default="test",
        help="Data split to be used for evaluation."
        " Defaults to 'test'."
        " Ignored in 'train' mode.",
    )
    argparser.add_argument(
        "--batch-size",
        type=int,
        help="Adjust to override the config value if you're"
        " experiencing out-of-memory issues",
    )
    argparser.add_argument(
        "--warm-start",
        action="store_true",
        help="If set, the training will resume from the"
        " last checkpoint saved if any. Ignored in"
        " evaluation modes."
        " Incompatible with '--weights'.",
    )
    argparser.add_argument(
        "--weights",
        help="Path to file with weights to load."
        " If not supplied, in 'eval' mode the latest"
        " weights of the experiment will be loaded;"
        " in 'train' mode no weights will be loaded.",
    )
    argparser.add_argument(
        "--word-level",
        action="store_true",
        help="If set, output word-level conll-formatted"
        " files in evaluation modes. Ignored in"
        " 'train' mode.",
    )
    args = argparser.parse_args()

    if args.warm_start and args.weights is not None:
        print(
            "The following options are incompatible:" " '--warm_start' and '--weights'",
            file=sys.stderr,
        )
        sys.exit(1)

    seed(2020)
    model = CorefModel(args.config_file, args.experiment)

    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size

    if args.mode == "train":
        if args.weights is not None or args.warm_start:
            model.load_weights(
                path=args.weights, map_location="cpu", noexception=args.warm_start
            )
        with output_running_time():
            model.train()
    else:
        model.load_weights(
            path=args.weights,
            map_location="cpu",
            ignore={
                "bert_optimizer",
                "general_optimizer",
                "bert_scheduler",
                "general_scheduler",
            },
        )
        # model.evaluate(data_split=args.data_split, word_level_conll=args.word_level)

        # doc = {
        #     "document_id": "bc/msnbc/00/msnbc_0004",
        #     "cased_words": [
        #         "Both",
        #         "vehicles",
        #         "which",
        #         "are",
        #         "armored",
        #         "can",
        #         "house",
        #         "up",
        #         "to",
        #         "twenty",
        #         "five",
        #         "marines",
        #         "/.",
        #         "This",
        #         "time",
        #         "there",
        #         "were",
        #         "fifteen",
        #         "inside",
        #         "the",
        #         "AAV",
        #         "/.",
        #         "Fourteen",
        #         "died",
        #         "/.",
        #         "and",
        #         "one",
        #         "civilian",
        #         "translator",
        #         "was",
        #         "also",
        #         "killed",
        #         "/.",
        #         "Now",
        #         "this",
        #         "occured",
        #         "in",
        #         "basically",
        #         "the",
        #         "same",
        #         "area",
        #         "where",
        #         "six",
        #         "marines",
        #         "were",
        #         "killed",
        #         "yesterday",
        #         "in",
        #         "an",
        #         "ambush",
        #         "launched",
        #         "by",
        #         "insurgents",
        #         "/.",
        #         "The",
        #         "six",
        #         "marines",
        #         "were",
        #         "snipers",
        #         "located",
        #         "on",
        #         "a",
        #         "roof",
        #         "looking",
        #         "for",
        #         "insurgents",
        #         "travelling",
        #         "down",
        #         "a",
        #         "smuggling",
        #         "route",
        #         "when",
        #         "they",
        #         "themselves",
        #         "were",
        #         "surrounded",
        #         "and",
        #         "killed",
        #         "by",
        #         "insurgent",
        #         "snipers",
        #         "/.",
        #         "Now",
        #         "all",
        #         "of",
        #         "this",
        #         "violence",
        #         "has",
        #         "been",
        #         "located",
        #         "in",
        #         "an",
        #         "area",
        #         "known",
        #         "as",
        #         "the",
        #         "Euphrates",
        #         "river",
        #         "valley",
        #         "/.",
        #         "This",
        #         "is",
        #         "the",
        #         "main",
        #         "smuggling",
        #         "route",
        #         "for",
        #         "insurgents",
        #         "from",
        #         "the",
        #         "Syrian",
        #         "border",
        #         "into",
        #         "central",
        #         "Iraq",
        #         "/.",
        #         "There",
        #         "have",
        #         "been",
        #         "several",
        #         "marine",
        #         "operations",
        #         "in",
        #         "that",
        #         "area",
        #         "since",
        #         "May",
        #         "to",
        #         "try",
        #         "to",
        #         "shut",
        #         "down",
        #         "those",
        #         "smuggling",
        #         "routes",
        #         "/.",
        #         "But",
        #         "today",
        #         "and",
        #         "yesterday",
        #         "'s",
        #         "attacks",
        #         "show",
        #         "that",
        #         "the",
        #         "insurgents",
        #         "are",
        #         "still",
        #         "very",
        #         "resilient",
        #         "/.",
        #         "Chris",
        #         "/?",
        #     ],
        #     "sent_id": [
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         1,
        #         1,
        #         1,
        #         1,
        #         1,
        #         1,
        #         1,
        #         1,
        #         1,
        #         2,
        #         2,
        #         2,
        #         3,
        #         3,
        #         3,
        #         3,
        #         3,
        #         3,
        #         3,
        #         3,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         4,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         5,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         6,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         7,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         8,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         9,
        #         10,
        #         10,
        #     ],
        #     "speaker": [
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #         "speaker1",
        #     ],
        #     "word_clusters": [[43, 56, 73, 72], [46, 139], [92, 124], [107, 145]],
        #     "span_clusters": [
        #         [[42, 44], [54, 57], [73, 74], [72, 74]],
        #         [[46, 47], [139, 140]],
        #         [[91, 99], [123, 125]],
        #         [[107, 108], [144, 146]],
        #     ],
        # }
        text1 = "When set up, the tuner will tell you whether to turn it up or down to hit the nearest note. There’s also a visual indicator that gives you details on the sharpness or the flatness of your guitar. It is recommended to use the built-in microphone on your phone for this purpose. Be advised that you may need to bring your phone close to the guitar to get an accurate representation."
        text2 = """
            The attorney for Brian Laundrie's family said Tuesday that all three members of the family camped at Fort De Soto Park in Pinellas County, Florida, one week before the 23-year-old went missing in early September.

        According to a report from WFLA, Attorney Steven Bertolino said the Laundrie family went camping from September 6 to September 8, before leaving the park together.

        JUST IN. #Laundrie family attorney confirms all three family members camped at Fort De Soto campground from September 6-7th and left park together. #BrianLaundrie #GabbyPetito (via @WFLAJustin) pic.twitter.com/bdvTzoBmlP

        — Josh Benson (@WFLAJosh) September 28, 2021
        The report comes one day after reality TV star Duane Chapman, otherwise known as "Dog the Bounty Hunter," told Fox News that he believes Laundrie may have been hiding in Fort De Soto Park with assistance from his parents after he returned to Florida without his fiancée Gabby Petito.

        Laundrie, 23, has been named a person of interest in the disappearance of Petito, whose remains were found in Wyoming on September 19 after the two went on a cross-country road trip. Days after Petito's body was found, the FBI ruled her death a homicide.

        Laundrie returned home to Florida without Petito on September 1, and was last seen on September 14. Three days later, his family reported him missing and told police they believed he went to Carlton Reserve, a nearly 25,000-acre preserve in Sarasota County.

        Brian Laundrie Search
        A North Port Police officer stands in the driveway of the family home of Brian Laundrie, who is a person of interest after his fiancé Gabby Petito went missing on September 20, 2021 in North Port, Florida. Octavio Jones/Getty Images
        On Monday night, Chapman alleged that he received information that Laundrie instead fled to Fort De Soto park with his parents after he returned home. The reality TV star claimed that Laundrie and his parents, Roberta and Chris, entered the park on September 6, but that only two people left the park on September 8—leaving room for theories that Laundrie could still be located in the area.

        "They were registered, went through the gate. They're on camera. They were here," Chapman told Fox News on Monday evening. "We think at least if he's not here right now, we are sure he was caught on camera as he went in the gate—that he was here for sure. Not over in the swamp."

        "Allegedly, what we're hearing, is two people left on the 8th. Three people came in on the 6th, and two people left on the 8th. I think he's been here for sure," Chapman added.

        Fort De Soto Park is a vast wilderness area located about 75 miles away from the Laundries' home on Wabasso Avenue in North Port. With more than 1,130-acres, the park is the largest in the Pinellas County Park System and contains five interconnected islands.

        Laundrie and Petito have both previously visited the park in February, with Petito posting a photo of their trip on social media. The couple later posted a review of the park on the travel website The Dyrt, describing it as a "really nice campground, beautiful area with many hikes and easy walks, the beach, historic sites, really nice camp store and well maintained sites!."

        Following Chapman's allegations, a spokesperson from the Pinellas Sheriff's Office told the Tampa Bay Times that police were not "not aware of any confirmed sightings of Brian" and were not conducting an investigation in Fort De Soto.

        Laundrie's parents have vehemently denied any knowledge of or involvement in their son's disappearance.

        "The speculation by the public and some in the press that the parents assisted Brian in leaving the family home or in avoiding arrest on a warrant that was issued after Brian had already been missing for several days is just wrong," Bertolino previously said in a statement.

        Anyone with information on Laundrie's whereabouts is asked to contact the FBI at 1-800-CALL-FBI or 303-629-7171.

        Newsweek has contacted the FBI for an update on the Petito case.
        """
        text3 = """Patent sale could bring in any where from 1 to 2 billion, they already have 800 million cashLet’s just say it’s
            2 billion for now + 800 million 2.8 billionBB market cap this morning was 4.6 billion.
            4.6 minus 2.8 billion cash on hand = 1.8 billion.
            They are giving blackberry the WHOLE company, with 68 ev market, spark security, Amazon partnership with ivy,
            qnx in 175 million vehicles, 700 - 900 million revenue per year a valuation of 1.8 billion.
            Let me say it again, the whole company for only 1.8 billion valuation (some people here say 3 billion when factoring in debt, either way it’s low).
            After the covid crash last year BB market cap hit 1.7 billion, so after including potential cash on hand BB is trading at the same market cap today right now as it was at the bottom of covid crash.
            Edit: people in here talking about debt, blackberry has always had debt and is an ongoing part of their valuation,
            it doesn’t change the fact of what the valuation was 12 months ago to what it is now when you include the patent sale,
            they have less debt now than they use to and they are also going to have more cash than they use to.
            The market cap is still extremely low valuation to give the company after you factor in the cash they will have soon.
            From what I was able to find it seems debt was only 459 million in November."""
        texts = [text1, text2, text3]
        docs = [process_article(text) for text in texts]

        docs = [preprocess_doc(doc) for doc in docs]
        doc = docs[0]
        res = model.run(doc)
        all_span_clusters = res.span_clusters
        processed = NLP(texts[0])
        for cluster in all_span_clusters:
            for span in cluster:
                print(f"Span: {span}")
                print(f"word: {processed[span[0]: span[1]]}")
            print("----------------------------------")
