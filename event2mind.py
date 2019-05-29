# pylint: disable=no-self-use,invalid-name,unused-import
import sys
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

# required so that our custom model + predictor + dataset reader
# will be registered by name
import library


class Event2MindClassifierPredictor:

    def test_sentence(self, sentence):
        # Load pre-trained model
        archive = load_archive('model.tar.gz')
        # Load predictor and predict the language of the name
        predictor = Predictor.from_archive(archive, 'event2mind_predictor')
        result = predictor.predict(sentence)
        print(result)

if __name__ == "__main__":
    sentence = "PersonX drops a hint"
    test = Event2MindClassifierPredictor()
    test.test_sentence(sentence)
