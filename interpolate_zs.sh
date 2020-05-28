for epoch in 1 3 5 10;
do
  for theta in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
  do
    ckpt_dir="$1--$2--$theta"
    python3 tools/interp_model.py --start ckpts/$1/model_state_epoch_$epoch.th --end ckpts/$2/model_state_epoch_$epoch.th --theta $theta --out-dir ckpts/$ckpt_dir || exit 1;
    UD_GT="/home/nlpmaster/ssd-1t/corpus/ud/ud-treebanks-v2.2/**/" UD_ROOT="/home/nlpmaster/ssd-1t/corpus/ud/ud-v2.2-preped/" bash run-adapter-eval-all.sh data/ori_test_langs.txt zs $ckpt_dir $epoch || exit 1 ;
    UD_GT="/home/nlpmaster/ssd-1t/corpus/ud/ud-treebanks-v2.5/**/" UD_ROOT="/home/nlpmaster/ssd-1t/corpus/ud/ud-v2.5-preped/" bash run-adapter-eval-all.sh data/test_langs-v2.5.txt zs $ckpt_dir $epoch || exit 1 ;
    rm ckpts/$ckpt_dir/*.th ;
  done
done

