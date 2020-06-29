export WANDB_MODE=""
while IFS=" " read -a words; do
  FT_LANG="${words[0]}";
  NUM_EPOCHS=0;
  if ! ls ${UD_GT}${FT_LANG}**-test.conllu 1> /dev/null 2>&1;
  then
    echo "no train; skipping"
  else
    for method in $3;
    do
      CKPT_DIR="${method}_$4_${FT_LANG}_$2"
      echo $CKPT_DIR
      if [[ ! ( -s "ckpts/$CKPT_DIR/train-result.txt" ) && -f "ckpts/${method}/model_state_epoch_$4.th" ]];
      then
        rm -r ckpts/$CKPT_DIR;
        bash zs-eval.sh $method $4 $FT_LANG $NUM_EPOCHS $2 || exit 1;
      else
        echo "no ckpts/${method}/model_state_epoch_$4.th, or already runned; continue"
      fi
    done
  fi
done<$1

