python -m table.experiments test \
    --cuda \
    --eval-batch-size=128 \
    --eval-beam-size=4 \
    --model=./runs/demo_run/model.best.bin \
    --extra-config='{}' 