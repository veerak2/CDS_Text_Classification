from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORE_DIR = Path(BASE_DIR, "stores")
#Union of all stopwords from nltk and Spacy
STOPWORDS = ['three',
 'never',
 'thence',
 'until',
 'over',
 'eight',
 "needn't",
 'therefore',
 'aren',
 'on',
 'whole',
 'd',
 'such',
 'everywhere',
 '‘ve',
 'wherein',
 'call',
 'well',
 'but',
 'be',
 'less',
 "she's",
 'go',
 "you'd",
 'anyhow',
 'nobody',
 "'re",
 'get',
 'than',
 'somewhere',
 'bottom',
 'forty',
 'whereas',
 'only',
 'doesn',
 'and',
 'me',
 'shouldn',
 'keep',
 'whose',
 "aren't",
 'back',
 'onto',
 'five',
 'top',
 'to',
 'various',
 'very',
 'up',
 'always',
 'am',
 'part',
 "shouldn't",
 'himself',
 'across',
 '‘re',
 'toward',
 'won',
 'thus',
 "weren't",
 'been',
 "wouldn't",
 "'s",
 'all',
 'ever',
 'now',
 'really',
 "mustn't",
 'theirs',
 'meanwhile',
 'she',
 'or',
 'before',
 'wouldn',
 'seems',
 'o',
 'hadn',
 'except',
 'empty',
 'hers',
 'own',
 'at',
 'seem',
 'are',
 'two',
 'most',
 'through',
 'serious',
 'beside',
 'whenever',
 'hereupon',
 'another',
 'between',
 'where',
 'them',
 '’d',
 'then',
 'itself',
 'thereafter',
 'themselves',
 'you',
 'ten',
 "'m",
 'too',
 'might',
 '‘d',
 'anywhere',
 'as',
 'm',
 'anyway',
 'us',
 "'d",
 'among',
 'thereupon',
 'ours',
 'third',
 'next',
 'became',
 'was',
 'isn',
 'beforehand',
 'side',
 'what',
 '’ve',
 's',
 'ain',
 'nine',
 'eleven',
 'name',
 'afterwards',
 'out',
 'still',
 '’s',
 't',
 'else',
 'somehow',
 'yet',
 'amongst',
 'latterly',
 'n‘t',
 'him',
 'few',
 'twelve',
 'four',
 'first',
 'll',
 'wherever',
 'whereby',
 'formerly',
 'how',
 'did',
 'haven',
 'same',
 'others',
 'myself',
 'latter',
 'don',
 'several',
 'my',
 '‘s',
 'due',
 'often',
 'there',
 'above',
 "don't",
 'neither',
 'seeming',
 'see',
 "doesn't",
 'again',
 'is',
 'towards',
 'other',
 'whereupon',
 'front',
 'put',
 'however',
 'something',
 'herein',
 'move',
 'though',
 'noone',
 'while',
 'yourselves',
 'they',
 'i',
 'enough',
 'fifty',
 'therein',
 'sometimes',
 'doing',
 'were',
 'without',
 'whither',
 'nevertheless',
 'couldn',
 'alone',
 'nor',
 'much',
 'both',
 'these',
 'below',
 'six',
 'everyone',
 'for',
 'along',
 "it's",
 '’re',
 'off',
 'already',
 "wasn't",
 'becomes',
 'should',
 'since',
 'could',
 'here',
 'can',
 'using',
 'mustn',
 'either',
 'please',
 'together',
 'does',
 'last',
 'shan',
 'no',
 'whoever',
 'just',
 'must',
 'seemed',
 'this',
 'the',
 're',
 'ca',
 'down',
 'give',
 'your',
 'more',
 'whence',
 'whatever',
 'would',
 'some',
 'done',
 'full',
 'into',
 'herself',
 'hasn',
 'after',
 'being',
 'fifteen',
 'twenty',
 'thru',
 'cannot',
 'unless',
 'each',
 'against',
 'has',
 'if',
 'that',
 'becoming',
 'yours',
 'so',
 'otherwise',
 'whom',
 'hereby',
 'behind',
 'elsewhere',
 'nowhere',
 'many',
 'say',
 'about',
 'even',
 'upon',
 'sixty',
 "hadn't",
 'with',
 'ma',
 'someone',
 'anything',
 'of',
 'within',
 "hasn't",
 'needn',
 'namely',
 'become',
 'having',
 "mightn't",
 '‘ll',
 'around',
 'hereafter',
 've',
 "you'll",
 'we',
 'from',
 '‘m',
 'via',
 'anyone',
 'it',
 'her',
 'a',
 'do',
 'which',
 'besides',
 'indeed',
 'an',
 'once',
 'wasn',
 'mightn',
 "that'll",
 "'ll",
 'mostly',
 'those',
 'thereby',
 'yourself',
 'y',
 "haven't",
 "'ve",
 'amount',
 'had',
 'least',
 'show',
 'when',
 'every',
 'its',
 'whereafter',
 'ourselves',
 'former',
 'none',
 'nothing',
 "you're",
 'one',
 'his',
 'any',
 'almost',
 'he',
 'throughout',
 'also',
 'everything',
 'per',
 'their',
 'have',
 "shan't",
 'sometime',
 'further',
 "should've",
 "couldn't",
 'mine',
 'during',
 'by',
 "won't",
 'who',
 'rather',
 'moreover',
 'take',
 'whether',
 'under',
 'regarding',
 'quite',
 'made',
 'because',
 'n’t',
 'our',
 'used',
 'perhaps',
 'not',
 'didn',
 'may',
 '’ll',
 'why',
 'in',
 'hence',
 'will',
 '’m',
 'weren',
 "n't",
 'make',
 "didn't",
 'hundred',
 "you've",
 'beyond',
 "isn't",
 'although',
 'i',
 "'m"]
