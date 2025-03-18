# # NLP Project



import pandas as pd
import numpy as np



def main():

    # Suppress specific warnings from pandas
    pd.options.mode.chained_assignment = None

    # Suppress warnings general warnings from pandas
    import warnings
    warnings.filterwarnings('ignore')

    # Load the data from the Georgia Animal Shelter Database
    from data_loader import load_text_data


    ## Load the data
    directory = "raw/scraped_chemistry_texts/scraped_chemistry_texts"
    texts=load_text_data(directory)

    # Train model
    from word2vec_train import train_word2vec
    word2vec_model = train_word2vec(texts)

    # Output vector
    from text_vectorization import vectorize_text
    example_text = "oxidation reduction".split()
    output_vector = vectorize_text(example_text, word2vec_model)
    print(output_vector)
    

if __name__=='__main__':

    main()