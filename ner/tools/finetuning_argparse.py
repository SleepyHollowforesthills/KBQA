import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
    #['--task_name','cluener']
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument("--do_evaluate", default=False, action='store_true')
    parser.add_argument("--do_predict", default=False, action='store_true')
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
    parser.add_argument("--max_length", default=125, type=int, required=False,
                        help="setting the max length of the inputs", )
    parser.add_argument("--batch_size", default=30, type=int, required=False,
                        help="setting the batch_size", )
    parser.add_argument("--epoch", default=15, type=int, required=False,
                        help="setting the training times", )

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    return parser