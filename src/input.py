import pandas


class DataReader():
    """Reads the text from input file.

    """

    def __init__(self, input_file):
        super(DataReader, self).__init__()
        self.input_file = input_file

    def read_text(self):
        for text1, text2, _ in self._read_data(self.input_file):
            if type(text1) == str and type(text2) == str:
                yield text1
                yield text2

    def read_scores(self):
        for _, __, score in self._read_data(self.input_file):
            yield score

    def _read_data(self, file_path):
        df = pandas.read_csv(file_path, header=None, sep='\t', error_bad_lines=False)
        for _, score, text1, text2 in df.filter([4, 5, 6]).itertuples():
            yield (text1, text2, score)
