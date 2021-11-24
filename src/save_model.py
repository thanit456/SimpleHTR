import argparse 
from path import Path

from dataloader_mybank import DataLoaderMyBank
from model import Model, DecoderType


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'

def main():
    """Main function."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--output', help='Ouput path to save tf model')

    args = parser.parse_args()

    # set chosen CTC decoder
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # load training data, create TF model
    loader = DataLoaderMyBank(args.data_dir, args.batch_size, fast=args.fast)
    char_list = loader.char_list

    # when in line mode, take care to have a whitespace in the char list
    if args.line_mode and ' ' not in char_list:
        char_list = [' '] + char_list

    # save characters of model for inference mode
    open(FilePaths.fn_char_list, 'w').write(''.join(char_list))

    # save words contained in dataset into file
    open(FilePaths.fn_corpus, 'w').write(' '.join(loader.train_words + loader.validation_words))

    model = Model(char_list, decoder_type, must_restore=True)
    model.export_tf_model(args.output)


if __name__ == '__main__':
    main()
