#Import our custom classes for Preprocess, Naive Bayes, and Evaluation

from preprocess import TextDataProcessor
from modified_nb import MultiLabelNaiveBayes
from evaluation import EmotionEvaluation

train = '../data/ssec/train.csv'
test = '../data/ssec/test.csv' 
dev = '../data/val/val.csv'
