nvidia-smi
sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg
sudo apt install git-lfs
env_name=whisper
python3 -m venv $env_name
echo "source ~/$env_name/bin/activate" >> ~/.bashrc
bash
git clone https://github.com/huggingface/community-events.git
pip install -r community-events/whisper-fine-tuning-event/requirements.txt

python -c "import torch; print(torch.cuda.is_available())"
python

import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

common_voice = load_dataset("common_voice", "en", split="validation", streaming=True)

inputs = feature_extractor(next(iter(common_voice))["audio"]["array"], sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
logits = model(input_features, decoder_input_ids=decoder_input_ids).logits

print("Environment set up successful?", logits.shape[-1] == 51865)


git config --global credential.helper store
huggingface-cli login
------------------ Large ------------------------
huggingface-cli repo create whisper-large-pashto
git clone https://huggingface.co/ihanif/whisper-large-pashto
cd whisper-large-pashto/
cp ~/community-events/whisper-fine-tuning-event/run_speech_recognition_seq2seq_streaming.py .

echo 'python fine-tune_whipser.py \
	--model_name_or_path="openai/whisper-large-v2" \
	--dataset_name="google/fleurs" \
	--dataset_config_name="ps_af" \
	--language="pashto" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Large Pashto" \
	--max_steps="1000" \
	--output_dir="./" \
	--per_device_train_batch_size="8" \
  --per_device_eval_batch_size="4" \
  --gradient_accumulation_steps="8" \
	--logging_steps="10" \
	--learning_rate="3e-7" \
	--warmup_steps="50" \
	--evaluation_strategy="steps" \
	--eval_steps="100" \
	--save_strategy="steps" \
	--save_steps="100" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="raw_transcription" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
  --load_best_model_at_end="True" \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--use_auth_token \
	--streaming="False" \
	--push_to_hub' >> run.sh

tmux new -s mysession
bash run.sh
tmux a -t mysession

Ctrl–b–d

------------------ Medium ------------------------
{'eval_loss': 1.158418893814087, 'eval_wer': 69.97055176689399, 'eval_runtime': 1566.2048, 'eval_samples_per_second': 0.327, 
'eval_steps_per_second': 0.02, 'epoch': 28.57}

{'eval_loss': 1.4428775310516357, 'eval_wer': 83.59423434593924, 'eval_runtime': 1406.9013, 'eval_samples_per_second': 0.364, 
'eval_steps_per_second': 0.023, 'epoch': 14.29} 

huggingface-cli repo create whisper-medium-pashto-3e-7
git clone https://huggingface.co/ihanif/whisper-medium-ps whisper-medium-pashto-3e-7
cd whisper-medium-pashto-3e-7/
cp ~/community-events/whisper-fine-tuning-event/run_speech_recognition_seq2seq_streaming.py .

echo 'python run_speech_recognition_seq2seq_streaming.py \
--model_name_or_path="openai/whisper-medium" \
--dataset_name="google/fleurs" \
--dataset_config_name="ps_af" \
--language="pashto" \
--train_split_name="train+validation" \
--eval_split_name="test" \
--model_index_name="Whisper Medium Pashto" \
--max_steps="300" \
--output_dir="./" \
--per_device_train_batch_size="32" \
--per_device_eval_batch_size="16" \
--gradient_accumulation_steps="2" \
--logging_steps="25" \
--learning_rate="3e-7" \
--warmup_steps="10" \
--evaluation_strategy="steps" \
--eval_steps="100" \
--save_strategy="steps" \
--save_steps="100" \
--generation_max_length="225" \
--length_column_name="input_length" \
--max_duration_in_seconds="30" \
--text_column_name="raw_transcription" \
--freeze_feature_encoder="False" \
--report_to="tensorboard" \
--metric_for_best_model="wer" \
--greater_is_better="False" \
--load_best_model_at_end \
--gradient_checkpointing \
--fp16 \
--overwrite_output_dir="True" \
--do_train \
--do_eval \
--predict_with_generate \
--do_normalize_eval \
--use_auth_token \
--streaming="False" \
--push_to_hub' >> run.sh



tmux new -s mysession
bash run.sh
tmux a -t mysession

------------------ Small ------------------------
huggingface-cli repo create whisper-small-pashto
git clone https://huggingface.co/ihanif/whisper-small-pashto
cd whisper-small-pashto/
cp ~/community-events/whisper-fine-tuning-event/run_speech_recognition_seq2seq_streaming.py .
echo 'python run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="openai/whisper-small" \
	--dataset_name="google/fleurs" \
	--dataset_config_name="ps_af" \
	--language="pashto" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Small Pashto" \
	--max_steps="400" \
	--output_dir="./" \
	--per_device_train_batch_size="32" \
  --per_device_eval_batch_size="16" \
  --gradient_accumulation_steps="2" \
	--logging_steps="25" \
	--learning_rate="3e-7" \
	--warmup_steps="10" \
	--evaluation_strategy="steps" \
	--eval_steps="100" \
	--save_strategy="steps" \
	--save_steps="100" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="transcription" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
  --greater_is_better="False" \
  --load_best_model_at_end \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir="False" \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--use_auth_token \
	--streaming="False" \
	--push_to_hub' >> run.sh

	echo 'python run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="openai/whisper-small" \
	--dataset_name="google/fleurs" \
	--dataset_config_name="ps_af" \
	--language="pashto" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Small Pashto" \
	--max_steps="1000" \
	--output_dir="./" \
	--per_device_train_batch_size="32" \
  --per_device_eval_batch_size="16" \
  --gradient_accumulation_steps="2" \
	--logging_steps="25" \
	--learning_rate="3e-7" \
	--warmup_steps="50" \
	--evaluation_strategy="steps" \
	--eval_steps="100" \
	--save_strategy="steps" \
	--save_steps="100" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="transcription" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--use_auth_token \
	--streaming="False" \
	--push_to_hub' >> run-1.sh

bash run.sh
tmux a -t mysession

ctrl + b + [ 

------------------------------------------------------------------
echo 'python run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="openai/whisper-medium" \
	--dataset_name="google/fleurs" \
	--dataset_config_name="ps_af" \
	--language="pashto" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Medium Pashto" \
	--max_steps="5000" \
	--output_dir="./" \
	--per_device_train_batch_size="32" \
  --per_device_eval_batch_size="16" \
  --gradient_accumulation_steps="2" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="transcription" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--use_auth_token \
	--optim="adamw_bnb_8bit" \
	----streaming="False" 
	--push_to_hub' >> run.sh

----------------------------------------------------------
echo 'python run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="openai/whisper-small" \
	--dataset_name="google/fleurs" \
	--dataset_config_name="ps_af" \
	--language="pashto" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Small Pashto" \
	--max_steps="5000" \
	--output_dir="./" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="32" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="transcription" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--use_auth_token \
	--push_to_hub' >> run.sh

  ssh ubuntu@129.213.91.60

------------------ Base ------------------------
nvidia-smi
sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg
git-lfs -v
sudo apt install git-lfs
env_name=whisper
python3 -m venv $env_name
echo "source ~/$env_name/bin/activate" >> ~/.bashrc
bash
git clone https://github.com/huggingface/community-events.git
pip install -r community-events/whisper-fine-tuning-event/requirements.txt
pip install bitsandbytes
python -c "import torch; print(torch.cuda.is_available())"

echo 'import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

common_voice = load_dataset("common_voice", "en", split="validation", streaming=True)

inputs = feature_extractor(next(iter(common_voice))["audio"]["array"], sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
logits = model(input_features, decoder_input_ids=decoder_input_ids).logits

print("Environment set up successful?", logits.shape[-1] == 51865)
' > test_python.py

python test_python.py

git config --global credential.helper store
huggingface-cli login
git lfs install
git clone https://huggingface.co/ihanif/whisper-base-ps whisper-base-pashto
cd whisper-base-pashto/
cp ~/community-events/whisper-fine-tuning-event/run_speech_recognition_seq2seq_streaming.py .

echo 'python fine_tune_whipser.py \
	--model_name="openai/whisper-base" \
	--language="Pashto" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Base Pashto" \
	--max_steps="2" \
	--output_dir="./ps_base_v1" \
	--per_device_train_batch_size="64" \
  	--per_device_eval_batch_size="32" \
  	--gradient_accumulation_steps="1" \
	--logging_steps="25" \
	--learning_rate="3e-7" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1" \
	--save_strategy="steps" \
	--save_steps="1" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="transcription" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
  	--load_best_model_at_end="True" \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--streaming="False" \
	--use_auth_token \
	--push_to_hub' >> run_base.sh

tmux new -s mysession
bash run.sh
tmux a -t mysession
	------------------ Tiny ------------------------
nvidia-smi
sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg
git-lfs -v
sudo apt install git-lfs
env_name=whisper
python3 -m venv $env_name
echo "source ~/$env_name/bin/activate" >> ~/.bashrc
bash
git clone https://github.com/huggingface/community-events.git
pip install -r community-events/whisper-fine-tuning-event/requirements.txt
pip install bitsandbytes
python -c "import torch; print(torch.cuda.is_available())"

echo 'import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

common_voice = load_dataset("common_voice", "en", split="validation", streaming=True)

inputs = feature_extractor(next(iter(common_voice))["audio"]["array"], sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
logits = model(input_features, decoder_input_ids=decoder_input_ids).logits

print("Environment set up successful?", logits.shape[-1] == 51865)
' > test_python.py

python test_python.py

git config --global credential.helper store
huggingface-cli login
git lfs install
git clone https://huggingface.co/ihanif/whisper-tiny-ps whisper-tiny-pashto
cd whisper-tiny-pashto/
cp ~/community-events/whisper-fine-tuning-event/run_speech_recognition_seq2seq_streaming.py .

echo 'python run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="openai/whisper-tiny" \
	--dataset_name="google/fleurs" \
	--dataset_config_name="ps_af" \
	--language="pashto" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Tiny Pashto" \
	--max_steps="2" \
	--output_dir="./" \
	--per_device_train_batch_size="64" \
  --per_device_eval_batch_size="32" \
  --gradient_accumulation_steps="1" \
	--logging_steps="25" \
	--learning_rate="3e-7" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1" \
	--save_strategy="steps" \
	--save_steps="1" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="transcription" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
  --load_best_model_at_end="True" \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--streaming="False" \
	--use_auth_token \
	--push_to_hub' >> run.sh

tmux new -s mysession
bash run.sh
tmux a -t mysession
	------------------ Medium Urdu ------------------------
echo 'python run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="openai/whisper-medium" \
	--dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="ur" \
	--language="urdu" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Medium Urdu" \
	--max_steps="200" \
	--output_dir="./" \
	--per_device_train_batch_size="32" \
  --per_device_eval_batch_size="16" \
  --gradient_accumulation_steps="1" \
	--logging_steps="10" \
	--learning_rate="1e-5" \
	--warmup_steps="40" \
	--evaluation_strategy="steps" \
	--eval_steps="100" \
	--save_strategy="steps" \
	--save_steps="100" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
  --load_best_model_at_end="True" \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir="False" \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--use_auth_token \
	--streaming="False" \
	--push_to_hub' >> run.sh