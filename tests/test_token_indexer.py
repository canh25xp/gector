"""Tests for the PretrainedBertIndexer module."""

from allennlp.data.tokenizers import WordTokenizer
from allennlp.common.testing import ModelTestCase
from gector.tokenizer_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary


class TestPretrainedTransformerIndexer(ModelTestCase):
    """A test case that tests PretrainedBertIndexer methods."""

    def setUp(self):
        """Set up tokenizer and indexer."""

        super().setUp()

        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())
        sentence = "the Quick brown fox jumped over the laziest lazy elmo"
        vocab_path = "data/output_vocabulary"
        self.tokens = tokenizer.tokenize(sentence)
        self.vocab = Vocabulary.from_files(vocab_path)
        self.model_name = "roberta-base"

    def test_do_lowercase(self):
        """Test tokenizer to handle setting do_lowercase to be True"""
        token_indexer = PretrainedBertIndexer(
            pretrained_model=self.model_name,
            max_pieces_per_token=5,
            do_lowercase=True,
            max_pieces=512,
            special_tokens_fix=1,
        )
        indexed_tokens = token_indexer.tokens_to_indices(
            self.tokens, self.vocab, self.model_name
        )
        assert indexed_tokens["bert"] == [
            627,
            2119,
            6219,
            23602,
            4262,
            81,
            5,
            40154,
            7098,
            22414,
            1615,
            4992,
        ]
        assert indexed_tokens["bert-offsets"] == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
        assert indexed_tokens["mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_toggle_special_tokens_fix(self):
        """Test togging special_tokens_fix to be False"""
        token_indexer = PretrainedBertIndexer(
            pretrained_model=self.model_name,
            max_pieces_per_token=5,
            do_lowercase=True,
            max_pieces=512,
            special_tokens_fix=0,
        )
        indexed_tokens = token_indexer.tokens_to_indices(
            self.tokens, self.vocab, self.model_name
        )

        assert indexed_tokens["bert"] == [
            627,
            2119,
            6219,
            23602,
            4262,
            81,
            5,
            40154,
            7098,
            22414,
            1615,
            4992,
        ]
        assert indexed_tokens["bert-offsets"] == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
        assert indexed_tokens["mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_truncate_window(self):
        """Test the functionality of truncating word pieces"""
        token_indexer = PretrainedBertIndexer(
            pretrained_model=self.model_name,
            max_pieces_per_token=5,
            do_lowercase=True,
            max_pieces=5,
            special_tokens_fix=1,
        )
        indexed_tokens = token_indexer.tokens_to_indices(
            self.tokens, self.vocab, self.model_name
        )

        assert indexed_tokens["bert"] == []
        assert indexed_tokens["bert-offsets"] == [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
        assert indexed_tokens["mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_as_padded_tensor_dict(self):
        """Test the method as_padded_tensor_dict"""
        tokens = {
            "bert": [
                50265,
                37158,
                15,
                1012,
                2156,
                89,
                32,
                460,
                5,
                12043,
                268,
                8,
                4131,
                22761,
                13659,
                49,
                1351,
                8,
                11360,
                7,
                12043,
                7768,
                479,
            ],
            "bert-offsets": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                11,
                12,
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ],
            "mask": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        }
        padding_lengths = {
            "bert": 42,
            "bert-offsets": 41,
            "mask": 41,
            "num_tokens": 42,
        }
        desired_num_tokens = {"bert": 42, "bert-offsets": 41, "mask": 41}

        token_indexer = PretrainedBertIndexer(
            pretrained_model=self.model_name,
            max_pieces_per_token=5,
            do_lowercase=True,
            max_pieces=512,
            special_tokens_fix=1,
        )

        padded_tensor = token_indexer.pad_token_sequence(
            tokens, desired_num_tokens, padding_lengths
        )
        assert padded_tensor["bert"] == [
            50265,
            37158,
            15,
            1012,
            2156,
            89,
            32,
            460,
            5,
            12043,
            268,
            8,
            4131,
            22761,
            13659,
            49,
            1351,
            8,
            11360,
            7,
            12043,
            7768,
            479,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
