import pandas as pd
import numpy as np
from simpletransformers.classification import MultiLabelClassificationModel,MultiLabelClassificationArgs
from sklearn.metrics import classification_report,precision_score
from .metrics import macro_accuracy
from .utils import prepare_data,text_cleaner,get_proba_df
from .TSDAEModel import TSDAEModelArgs,train_TSDAE
from dataclasses import asdict, dataclass, field, field
import datetime

@dataclass
class MultiLabelArgs(MultiLabelClassificationArgs):
    """
    :param threshold: default 0.5. The threshold for the emotion probability to set the specific emotion
        as True (relevant to the 'predict' function.
    :param reprocess_input_data: default True. If True, the input data will be
        reprocessed even if a cached file of the input data exists in
        the cache_dir.
    :param overwrite_output_dir: default True. If True, the trained model will
        be saved to the output_dir and will overwrite existing saved
        models in the same directory.
    :param num_train_epochs: default 1. The number of epochs the model will be
        trained for.
    :param train_batch_size: default 32. The training batch size.
    :param learning_rate: default 4e-6. The learning rate for training.
    :param evaluate_during_training: default True. Set to True to perform evaluation
        while training models. Make sure eval data is passed to the training
        method if enabled.
    :param evaluate_during_training_steps: default 380. Perform evaluation at every
        specified number of steps. A checkpoint model and the evaluation
        results will be saved.
    :param save_model_every_epoch: default False. Save a model checkpoint at the
        end of every epoch.
    :param save_eval_checkpoints: default False. Save a model checkpoint for every
        evaluation performed.
    :param evaluate_during_training_verbose: default True. Print results from
        evaluation during training.
    :param use_cached_eval_features: default True. Evaluation during training uses
        cached features. Setting this to False will cause features to be
        recomputed at every evaluation step.
    :param save_optimizer_and_scheduler: default False. Save optimizer and scheduler
        whenever they are available..
    :param fp16: default True. Whether or not fp16 mode should be used. Requires Nvidia
        Apex library.
    :param verbose: type: bool. default: True. Enable verbose.


    """
    threshold : float = 0.5
    num_train_epochs: int = 1
    train_batch_size: int = 32
    eval_batch_size: int = 128
    learning_rate: float = 4e-6
    max_seq_length: int = 128
    do_lower_case: bool = True
    evaluate_during_training: bool = True
    dynamic_quantize: bool = False
    early_stopping_consider_epochs: bool = False
    early_stopping_delta: float = 0
    early_stopping_metric: str = "macro_acc"
    early_stopping_metric_minimize: bool = True
    early_stopping_patience: int = 3
    encoding: str = None
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    local_rank: int = -1
    logging_steps: int = 50
    loss_type: str = None
    loss_args: dict = field(default_factory=dict)
    manual_seed: int = None
    max_grad_norm: float = 1.0
    model_name: str = None
    model_type: str = None
    multiprocessing_chunksize: int = -1
    n_gpu: int = 1
    no_cache: bool = True
    no_save: bool = False
    optimizer: str = "AdamW"
    overwrite_output_dir: bool = True
    polynomial_decay_schedule_lr_end: float = 1e-7
    polynomial_decay_schedule_power: float = 1.0
    quantized_model: bool = False
    reprocess_input_data: bool = True
    save_best_model: bool = True
    save_eval_checkpoints: bool = False
    save_model_every_epoch: bool = False
    save_optimizer_and_scheduler: bool = False
    scheduler: str = "linear_schedule_with_warmup"
    silent: bool = False
    skip_special_tokens: bool = True
    tensorboard_dir: str = None
    thread_count: int = None
    tokenizer_name: str = None
    tokenizer_type: str = None
    train_custom_parameters_only: bool = False
    use_cached_eval_features: bool = False
    use_early_stopping: bool = False
    use_hf_datasets: bool = False
    use_multiprocessing: bool = True
    use_multiprocessing_for_evaluation: bool = True
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: float = 0.01    

        
class EmotionMultilabelClassificationModel(MultiLabelClassificationModel):

    def __init__(self,
                 name_or_path = None,
                 model_type = "bert",
                 emotion_list: list = None,
                 emotion_weights: list = None,
                 verbose = True,
                 use_cuda: bool = True,
                 cuda_device = -1,
                 args: MultiLabelArgs = None):
        """
        **Initiate a Multilabel Classification model based on the pre-trained language model.**

        parameters:
        ----------
            :param model_type: should be one of the model types from the supported models (e.g. bert, electra, roberta), or a pre-trained TSDEA model.
            :param name_or_path: pecifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained
            model, or the path to a directory containing model files
            :param emotion_list: type: list. default: None. List of emotion names. If None all the emotion weights will
            be set as specified on the "config.json" file under 'emotion_list' parameter.
            :param emotion_weights: type: list. default: None. The emotion weight by label. If None all the emotion weights
            will be set to 1.
            :param use_cuda: type: bool. default: False. If True, will use GPU.
            :param verbose: type: bool. default: True. Enable verbose.
            :param args: type: MultiLabelArgs. default: None. Model arguments. If None, the arguments will be set as default in MultiLabelArgs class.
            """
        
        if emotion_list is None:
            self.emotion_list = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'neutral']
        else:
            self.emotion_list = emotion_list
            
        self.num_labels = len(self.emotion_list)
        
        if emotion_weights is None:
            self.emotion_weights = [1 for x in range(self.num_labels)]
        
        if args is None:
            args = MultiLabelArgs()
            
        self.args = args
        self.use_cuda = use_cuda
        self.model_type = model_type
        self.name_or_path = name_or_path
        self.model = None
        
        
    def from_pretrained(self,model_path):
        print(f"loading model from {model_path}...")
        self.model = MultiLabelClassificationModel(self.model_type,
                                                   model_path,
                                                   num_labels=self.num_labels,
                                                   pos_weight=self.emotion_weights,
                                                   use_cuda=self.use_cuda,
                                                   args=self.args)
        
        
    def calculate_label_weights(self, train_data: pd.DataFrame, emotion_list: list):
        """
        **calculate the label weights.**

        parameters:
        ----------
            :param train_data:
                type: pd.DataFrame.
                A train DataFrame with two columns:
                "text" column (containing the input text after farasa_tokenization).
                "labels" column (list type object of binary labels).
            :param emotion_list: type: list. default: None. List of emotion names. If None all the emotion weights will
                be set as specified on the "config.json" file under 'emotion_list' parameter.

        returns:
        --------
            :return: emotion_weights: type: list. default: None. The emotion weight by label.
        """
        values_list = train_data[emotion_list].sum(axis=0,skipna=True).values
        emotion_weight = ((len(train_data)-values_list)/values_list).tolist()
        print(pd.DataFrame(emotion_weight,index=emotion_list,columns=['weight']))
        return emotion_weight
         
        
    def train(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
              calculate_weights: bool = True, emotion_weights: list = None,
              num_epochs: int = None, train_batch_size: int = None,
              eval_batch_size: int = None, save_model_dir: str ="multilabel",
              TSDAE_pretrainig: bool = False,TSDAE_args = None,
              TSDAE_all_data: bool = True):
        """
        **Train the model.**

        parameters:
        ----------
            :param train_data:
                type: pd.DataFrame.
                A train DataFrame with two columns:
                "text" column (containing the input text after farasa_tokenization).
                "labels" column (list type object of binary labels).
            :param eval_data:
                type: pd.DataFrame.
                An eval DataFrame with two columns:
                "text" column (containing the input text after farasa_tokenization).
                "labels" column (list type object of binary labels).
            :param calculate_weights: type: bool. default: True. To calculate the emotion weight by label in the train_data.
            :param emotion_weights: type: list. default: None. The emotion weight by label. If None all the emotion weights
            will be set to 1.
            :param save_model_dir: default “multilabel/”. The directory where all outputs will be stored.
            This includes model checkpoints and evaluation results.
            :param TSDAE_pretrainig: default False. if True, pre-training an unsupervised TSDAE model before the classification fine-tuning.
            :param TSDAE_args: trainig arguments for the TSDAE model.
            :param TSDAE_all_data: default True. pre-training an unsupervised TSDAE model on both train and test data.

        returns:
        --------
            :return: type: object. Trained BERT model. The trained model will be saved locally in the output_dir folder as
                set in the config file ("outputs\" as Default).
        """
        TRAIN_DF = prepare_data(train_data,self.emotion_list)
        TEST_DF = prepare_data(test_data,self.emotion_list)
        
        if TSDAE_pretrainig:
            if TSDAE_args is None:
                TSDAE_args = TSDAEModelArgs()
                TSDAE_args.train_batch_size = self.args.train_batch_size
                TSDAE_args.max_seq_length = self.args.max_seq_length
            if TSDAE_all_data:
                TSDAE_sentences = list(TRAIN_DF["text"]) + list(TEST_DF["text"])
            else:
                TSDAE_sentences = list(TRAIN_DF["text"])         
            TSDAE_save_to = save_model_dir
            print("pre-training an unsupervised TSDAE model...")
            train_TSDAE(model_name_or_path=self.name_or_path,train_sentences=TSDAE_sentences,save_model_dir=save_model_dir,tsdae_model_args=TSDAE_args)
            trained_TSDAE_path = f"{save_model_dir}/TSDAE/0_Transformer"
            basic_model_path = trained_TSDAE_path
            print(f"finished unsupervised training, starting supervised training from {basic_model_path}...")
        else:
            basic_model_path = self.name_or_path
        
        if num_epochs:
            self.args.num_train_epochs = num_epochs
        if train_batch_size:
            self.args.train_batch_size = train_batch_size
        if eval_batch_size:
            self.args.eval_batch_size = eval_batch_size 
        self.args.model_name = save_model_dir
        self.args.output_dir = save_model_dir  
        best_model_dir = f"{save_model_dir}/best_model_{datetime.datetime.now().strftime('%m%d')}"
        self.args.best_model_dir = best_model_dir
        
        if self.model is None:
            self.from_pretrained(basic_model_path)
        
        if calculate_weights:
            print("Calculating weights...")
            self.model.pos_weight = self.calculate_label_weights(train_data,self.emotion_list)
        if emotion_weights:
            self.model.pos_weight = emotion_weights

        
        self.model.args.evaluate_during_training_steps = int(((len(TRAIN_DF)/self.model.args.train_batch_size))/1.5)
        macro_acc = macro_accuracy(self.model.args.threshold,self.emotion_list)
        
        self.model.train_model(TRAIN_DF.sample(frac=1),macro_acc=macro_acc,eval_df=TEST_DF)
    
    
    def evaluate(self, data: pd.DataFrame, args: dict = None, threshold: float = None,print_report=True):
        """
        **Evaluate the model performance with classification_report from sklearn.
         (for multi-label classification).**

        parameters:
        ----------
            :param data: A Pandas DataFrame with minimum two columns.
                type: pd.DataFrame.
                "text" column (containing the input text after farasa_tokenization).
                "labels" column (list type object of binary labels) - the ground truth.
            :param args: model args (see main class for more info)
            :param threshold: default None. The threshold for the emotion probability to set the specific emotion
                as True (relevant to the 'predict' function. if None will be set to the model default (0.5)
            :param print_report: print an evaluation_report.

        returns:
        --------
            :return: proba_df
        """
        if threshold:
            self.args.threshold = threshold
        if args:
            self.args = args
        if self.model is None:
            self.from_pretrained(self.name_or_path)
        self.model.args = self.args
        DATA = prepare_data(data,self.emotion_list)
        ground_truth = list(DATA['labels'])
        text_input = list(DATA['text'])
        predictions, raw_outputs = self.model.predict(text_input)
        proba_df = get_proba_df(predictions, raw_outputs, self.emotion_list)
        proba_df['ground_truth'] = ground_truth
        proba_df['clean_text'] = DATA["text"]
        proba_df['text'] = data["text"]

        if print_report:
            evaluation_report = classification_report(ground_truth,predictions,target_names=self.emotion_list,zero_division=0)
            print(evaluation_report)
            
        return proba_df
    
    
    def predict(self, data, args: dict = None, threshold: float = None):
        """
        **Preprocess new data and predict the emotions**.

        parameters:
        ----------
            :param data: A Pandas DataFrame, list or str.

            :param args: model args (see main class for more info)
            :param threshold: default None. The threshold for the emotion probability to set the specific emotion
                as True (relevant to the 'predict' function. if None will be set to the model default (0.5)

        returns:
        --------
            :return: proba_df
        """
        if threshold:
            self.args.threshold = threshold
        if args:
            self.args = args
        if self.model is None:
            self.from_pretrained(self.name_or_path)
        text_input = text_cleaner(data)
        self.model.args = self.args

        if type(text_input) is str:
            text_input = [text_input]
        predictions, raw_outputs = self.model.predict(text_input)
        proba_df = get_proba_df(predictions, raw_outputs, self.emotion_list)
        proba_df['clean_text'] = text_input            
        return proba_df
