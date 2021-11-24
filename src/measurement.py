import jiwer 
import pandas as pd 
import argparse 

def evaluate(df): 
    wer_default = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords()
    ])

    wer = jiwer.wer(
        df['gt'].tolist(), 
        df['pred'].tolist(), 
        truth_transform=wer_default, 
        hypothesis_transform=wer_default
    )
    
    cer_default = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])

    cer = jiwer.cer(
        cer_default(df['gt'].tolist()), 
        cer_default(df['pred'].tolist())
    )
    print(f'word accuracy = {(1 - wer) * 100:.2f}%')
    print(f'character error rate = {cer * 100:.2f}%')
    return wer, cer 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help="Path to predictions", type=str, default='../predictions/predictions.csv')
    args = parser.parse_args()

    annot_df = pd.read_csv(args.csv_path)
    print(annot_df)
    evaluate(annot_df)

if __name__ == '__main__':
    main()
