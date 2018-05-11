import pandas


class DataReader():
    """Reads the text from input file.

    """

    def __init__(self, input_file):
        super(DataReader, self).__init__()
        self._input_file = input_file

    def read_data(self):
        texts = []
        scores = []
        df = pandas.read_csv(self._input_file, header=None, sep='\t', error_bad_lines=False)
        for _, score, text1, text2 in df.filter([4, 5, 6]).itertuples():
            if type(text1) == str and type(text2) == str:
                texts.append(text1)
                texts.append(text2)
                scores.append(score)
        return texts, scores
