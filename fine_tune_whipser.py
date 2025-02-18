import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import GenerationConfig

#######################     ARGUMENT PARSING        #########################

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
parser.add_argument(
    '--model_name', 
    type=str, 
    required=False, 
    default='openai/whisper-base', 
    help='Huggingface model name to fine-tune. Eg: openai/whisper-base'
)
parser.add_argument(
    '--language', 
    type=str, 
    required=False, 
    default='Pashto', 
    help='Language the model is being adapted to in Camel case.'
)
parser.add_argument(
    '--sampling_rate', 
    type=int, 
    required=False, 
    default=16000, 
    help='Sampling rate of audios.'
)
parser.add_argument(
    '--num_proc', 
    type=int, 
    required=False, 
    default=2, 
    help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
)
parser.add_argument(
    '--train_strategy', 
    type=str, 
    required=False, 
    default='steps', 
    help='Training strategy. Choose between steps and epoch.'
)
parser.add_argument(
    '--learning_rate', 
    type=float, 
    required=False, 
    default=1.75e-5, 
    help='Learning rate for the fine-tuning process.'
)
parser.add_argument(
    '--warmup', 
    type=int, 
    required=False, 
    default=20000, 
    help='Number of warmup steps.'
)
parser.add_argument(
    '--train_batchsize', 
    type=int, 
    required=False, 
    default=48, 
    help='Batch size during the training phase.'
)
parser.add_argument(
    '--eval_batchsize', 
    type=int, 
    required=False, 
    default=32, 
    help='Batch size during the evaluation phase.'
)
parser.add_argument(
    '--num_epochs', 
    type=int, 
    required=False, 
    default=20, 
    help='Number of epochs to train for.'
)
parser.add_argument(
    '--num_steps', 
    type=int, 
    required=False, 
    default=100000, 
    help='Number of steps to train for.'
)
parser.add_argument(
    '--resume_from_ckpt', 
    type=str, 
    required=False, 
    default=None, 
    help='Path to a trained checkpoint to resume training from.'
)
parser.add_argument(
    '--output_dir', 
    type=str, 
    required=False, 
    default='output_model_dir', 
    help='Output directory for the checkpoints generated.'
)
parser.add_argument(
    '--train_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help='List of datasets to be used for training.'
)
parser.add_argument(
    '--train_dataset_configs', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of training dataset configs. Eg. 'hi' for the Pashto part of Common Voice",
)
parser.add_argument(
    '--train_dataset_splits', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of training dataset splits. Eg. 'train' for the train split of Common Voice",
)
parser.add_argument(
    '--train_dataset_text_columns', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="Text column name of each training dataset. Eg. 'sentence' for Common Voice",
)
parser.add_argument(
    '--eval_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help='List of datasets to be used for evaluation.'
)
parser.add_argument(
    '--eval_dataset_configs', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of evaluation dataset configs. Eg. 'ps_af' for the Pashto part of Google Fleurs",
)
parser.add_argument(
    '--eval_dataset_splits', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of evaluation dataset splits. Eg. 'test' for the test split of Common Voice",
)
parser.add_argument(
    '--eval_dataset_text_columns', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="Text column name of each evaluation dataset. Eg. 'transcription' for Google Fleurs",
)
parser.add_argument(
    '--use_augmented_data', 
    type=bool, 
    required=False, 
    default=True, 
    help='Whether to use augmented data for training.'
)
parser.add_argument(
    '--use_processed_data', 
    type=bool, 
    required=False, 
    default=False, 
    help='Whether to use processed data for training.'
)
parser.add_argument(
    '--model_index_name',
    type=str,
    required=False,
    default='Whisper Base Model',
    help='Name of the model for the model hub'
)
parser.add_argument(
    '--per_device_train_batch_size',
    type=int,
    required=False,
    default=32,
    help='Training batch size per device'
)
parser.add_argument(
    '--per_device_eval_batch_size',
    type=int,
    required=False,
    default=16,
    help='Evaluation batch size per device'
)
parser.add_argument(
    '--gradient_accumulation_steps',
    type=int,
    required=False,
    default=1,
    help='Number of steps for gradient accumulation'
)
parser.add_argument(
    '--logging_steps',
    type=int,
    required=False,
    default=25,
    help='Number of steps between logging'
)
parser.add_argument(
    '--eval_strategy',
    type=str,
    required=False,
    default='steps',
    help='Evaluation strategy (steps or epoch)'
)
parser.add_argument(
    '--eval_steps',
    type=int,
    required=False,
    default=1,
    help='Number of steps between evaluations'
)
parser.add_argument(
    '--save_strategy',
    type=str,
    required=False,
    default='steps',
    help='Save strategy (steps or epoch)'
)
parser.add_argument(
    '--save_steps',
    type=int,
    required=False,
    default=1,
    help='Number of steps between saves'
)
parser.add_argument(
    '--generation_max_length',
    type=int,
    required=False,
    default=225,
    help='Maximum length for generation'
)
parser.add_argument(
    '--length_column_name',
    type=str,
    required=False,
    default='input_length',
    help='Name of the length column'
)
parser.add_argument(
    '--max_duration_in_seconds',
    type=int,
    required=False,
    default=30,
    help='Maximum duration in seconds'
)
parser.add_argument(
    '--text_column_name',
    type=str,
    required=False,
    default='transcription',
    help='Name of the text column'
)
parser.add_argument(
    '--freeze_feature_encoder',
    type=bool,
    required=False,
    default=False,
    help='Whether to freeze the feature encoder'
)
parser.add_argument(
    '--report_to',
    type=str,
    required=False,
    default='tensorboard',
    help='Where to report training metrics'
)
parser.add_argument(
    '--metric_for_best_model',
    type=str,
    required=False,
    default='wer',
    help='Metric to use for best model selection'
)
parser.add_argument(
    '--greater_is_better',
    type=bool,
    required=False,
    default=False,
    help='Whether higher metric is better'
)
parser.add_argument(
    '--load_best_model_at_end',
    type=bool,
    required=False,
    default=True,
    help='Whether to load the best model at the end'
)
parser.add_argument(
    '--gradient_checkpointing',
    action='store_true',
    help='Enable gradient checkpointing'
)
parser.add_argument(
    '--fp16',
    action='store_true',
    help='Enable fp16 training'
)
parser.add_argument(
    '--overwrite_output_dir',
    action='store_true',
    help='Overwrite the output directory'
)
parser.add_argument(
    '--do_train',
    action='store_true',
    help='Whether to run training'
)
parser.add_argument(
    '--do_eval',
    action='store_true',
    help='Whether to run evaluation'
)
parser.add_argument(
    '--predict_with_generate',
    action='store_true',
    help='Whether to use generation for prediction'
)
parser.add_argument(
    '--do_normalize_eval',
    action='store_true',
    help='Whether to normalize evaluation'
)
parser.add_argument(
    '--streaming',
    type=bool,
    required=False,
    default=False,
    help='Whether to use streaming'
)
parser.add_argument(
    '--use_auth_token',
    action='store_true',
    help='Use Hugging Face auth token'
)
parser.add_argument(
    '--push_to_hub',
    action='store_true',
    help='Push model to Hugging Face Hub'
)
args = parser.parse_args()

if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('The train strategy should be either steps and epoch.')

if len(args.train_datasets) == 0:
    raise ValueError('No train dataset has been passed')
if len(args.eval_datasets) == 0:
    raise ValueError('No evaluation dataset has been passed')

if len(args.train_datasets) != len(args.train_dataset_configs):
    raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_configs. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_configs)} for train_dataset_configs.")
if len(args.eval_datasets) != len(args.eval_dataset_configs):
    raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_configs. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_configs)} for eval_dataset_configs.")

if len(args.train_datasets) != len(args.train_dataset_splits):
    raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_splits. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_splits)} for train_dataset_splits.")
if len(args.eval_datasets) != len(args.eval_dataset_splits):
    raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_splits. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_splits)} for eval_dataset_splits.")

if len(args.train_datasets) != len(args.train_dataset_text_columns):
    raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_text_columns. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_text_columns)} for train_dataset_text_columns.")
if len(args.eval_datasets) != len(args.eval_dataset_text_columns):
    raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_text_columns. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_text_columns)} for eval_dataset_text_columns.")

print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
print('ARGUMENTS OF INTEREST:')
print(vars(args))
print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()


#############################       MODEL LOADING       #####################################

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task="transcribe")
processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that config.decoder_start_token_id is correctly defined")

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

if gradient_checkpointing:
    model.config.use_cache = False
#******************************* Custom Code ******************************************
AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "sentence"

def normalize_dataset(ds, audio_column_name=None, text_column_name=None):
    if audio_column_name is not None and audio_column_name != AUDIO_COLUMN_NAME:
        ds = ds.rename_column(audio_column_name, AUDIO_COLUMN_NAME)
    if text_column_name is not None and text_column_name != TEXT_COLUMN_NAME:
        ds = ds.rename_column(text_column_name, TEXT_COLUMN_NAME)
    # resample to the same sampling rate
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    # normalise columns to ["audio", "sentence"]
    ds = ds.remove_columns(set(ds.features.keys()) - set([AUDIO_COLUMN_NAME, TEXT_COLUMN_NAME]))
    return ds

# The use_auth_token and trust_remote_code parameters should be passed to load_dataset(), not as part of the config
# cv = load_dataset("fsicoli/common_voice_18_0", "ps", split="train+validation+test", trust_remote_code=True)
# cv = load_dataset("fsicoli/common_voice_19_0", "ps", split="train+validation+other", trust_remote_code=True)

def load_datasets(split="train+validated", use_augmented_data=False, use_processed_data=False):
    """Load datasets based on configuration"""
    try:
        if use_processed_data:
            ds = load_dataset("ihanif/pashto_speech_ds")
        elif use_augmented_data:
            ds = load_dataset("ihanif/augmented_ds")
        else:
            ds = load_dataset("ihanif/common_voice_ps_20_0", split=split, trust_remote_code=True)
            ds = normalize_dataset(ds)
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

common_voice = load_datasets(split="train+validated", use_augmented_data=args.use_augmented_data, use_processed_data=args.use_processed_data)
print(common_voice)


############################        DATASET LOADING AND PREP        ##########################

def load_all_datasets(split):    
    combined_dataset = []
    if split == 'train':
        for i, ds in enumerate(args.train_datasets):
            dataset = load_dataset(ds, args.train_dataset_configs[i], split=args.train_dataset_splits[i])
            dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
            if args.train_dataset_text_columns[i] != "sentence":
                dataset = dataset.rename_column(args.train_dataset_text_columns[i], "sentence")
            dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
            combined_dataset.append(dataset)
    elif split == 'eval':
        for i, ds in enumerate(args.eval_datasets):
            dataset = load_dataset(ds, args.eval_dataset_configs[i], split=args.eval_dataset_splits[i])
            dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
            if args.eval_dataset_text_columns[i] != "sentence":
                dataset = dataset.rename_column(args.eval_dataset_text_columns[i], "sentence")
            dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
            combined_dataset.append(dataset)
    
    ds_to_return = concatenate_datasets(combined_dataset)
    ds_to_return = ds_to_return.shuffle(seed=22)
    return ds_to_return

def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

max_label_length = model.config.max_length
min_input_length = 0.0
max_input_length = 30.0
def is_in_length_range(length, labels):
    return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length


print('DATASET PREPARATION IN PROGRESS...')
try:
    raw_dataset = load_datasets(
        split=args.train_split_name if hasattr(args, 'train_split_name') else "train+validated",
        use_augmented_data=args.use_augmented_data,
        use_processed_data=args.use_processed_data
    )
except Exception as e:
    print(f"Failed to load dataset: {e}")
    raise

if not args.use_processed_data:
    raw_dataset = raw_dataset.map(prepare_dataset, num_proc=args.num_proc)

    raw_dataset = raw_dataset.filter(
        is_in_length_range,
        input_columns=["input_length", "labels"],
        num_proc=args.num_proc,
    ) 

###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print('DATASET PREPARATION COMPLETED')


metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


###############################     TRAINING ARGS AND TRAINING      ############################
generation_config = GenerationConfig.from_pretrained(args.model_name)
generation_config.save_pretrained(args.output_dir)

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=1,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        resume_from_checkpoint=args.resume_from_ckpt,
        push_to_hub=True,
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=False,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        max_steps=args.num_steps,
        save_total_limit=1,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        resume_from_checkpoint=args.resume_from_ckpt,
        push_to_hub=True,
    )

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

processor.save_pretrained(training_args.output_dir)

print('TRAINING IN PROGRESS...')
trainer.train()
print('DONE TRAINING')