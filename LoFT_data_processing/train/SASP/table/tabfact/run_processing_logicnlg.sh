python preprocess_logicnlg.py

export CLASSPATH="$CLASSPATH:/root/CoreNLP/javanlp-core.jar:/root/CoreNLP/stanford-corenlp-4.5.2-models.jar";
for file in `find /root/CoreNLP/lib -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done

echo "==> Finish preparing LogicNLG training data for SASP Pre-processing"

python preprocess_example.py \
    --min_frac_for_props=0.25 \
    --corenlp_path=/root/CoreNLP

echo "==> Finish Pre-Processing LogicNLG training data for SASP"

python gen_with_func.py 