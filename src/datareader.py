import csv

SEPARAT0R = '\t'


class DataReader():
    """Reads the text from input file.

    """

    def __init__(self, input_file):
        super(DataReader, self).__init__()
        self._input_file = input_file

    def read_data(self):
        texts = []
        scores = []
        with open(self._input_file, 'r') as tsv:
            reader = csv.reader(tsv, delimiter=SEPARAT0R)
            for row in reader:
                parse_successful, score, text1, text2 = self._parse_row(row)
                if parse_successful:
                    texts.append(text1)
                    texts.append(text2)
                    scores.append(score)
        return texts, scores

    def _parse_row(self, row):
        try:
            score = float(row[4])
            text1 = row[5]
            text2 = row[6]
            return True, score, text1, text2
        except ValueError:
            return False, None, None, None
        except IndexError:
            return False, None, None, None
