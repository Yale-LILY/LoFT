export EXP_NAME=verifier_exp

python run_verifier.py \
  --do_train \
  --do_eval \
  --output_dir $EXP_NAME \
  --model_name_or_path roberta-base \
  --overwrite_output_dir \
  --per_device_train_batch_size 24 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 32 \
  --eval_accumulation_steps 8 \
  --logging_steps 50 \
  --learning_rate 3e-5 \
  --eval_steps 250 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --weight_decay 1e-2 \
  --max_steps 5000 \
  --max_grad_norm 0.1 \
  --save_total_limit 2
