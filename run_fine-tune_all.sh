export WANDB_MODE=""
while IFS=" " read -a words; do
  FT_LANG="${words[0]}";
  for i in 1 0 3 5 10 ; do
    NUM_EPOCHS=$3;
    if ! ls ${UD_ROOT}${FT_LANG}**-train.conllu 1> /dev/null 2>&1;
    then
      # NUM_EPOCHS=1;
      echo "no train; skipping"
    else
      #for method in fomaml-adapter-crf-fixedlr-inherit multi-adapter-crf-fixedlr reptile-adapter-crf-fixedlr-inherit ;
      #for method in reptile-adapter-maxlen256-v2.2-inner5e5 multi-adapter-maxlen256-v2.2 reptile-adapter-maxlen256-v2.2-inner5e5-531 multi-adapter-maxlen256-v2.2-531;
      #for method in reptile-adapter-lr1e-4-noinherit reptile-adapter-lr3e-4-noinherit multi-adapter-lr1e-4 multi-adapter-lr3e-4 reptile-adapter-lr1e-4-noinherit-t531 reptile-adapter-lr3e-4-noinherit-t531 multi-adapter-lr1e-4-t531 multi-adapter-lr3e-4-t531 fomaml-adapter-lr1e-4-noinherit fomaml-adapter-lr3e-4-noinherit fomaml-adapter-lr1e-4-noinherit-t531 fomaml-adapter-lr3e-4-noinherit-t531 ; 
      #do
      #  if grep -w $FT_LANG data/ensemble_langs.txt;
      #  then
      #    CKPT_DIR="${method}_${i}_${FT_LANG}_ens_$2"
      #  else
      #    CKPT_DIR="${method}_${i}_${FT_LANG}_$2"
      #  fi
      #  echo $CKPT_DIR
      #  if [[ ! ( -s "ckpts/$CKPT_DIR/result.txt" ) && -f "ckpts/${method}/model_state_epoch_${i}.th" ]];
      #  then
      #    rm -r ckpts/$CKPT_DIR;
      #    BASE_MODEL=$method bash fine-tune.sh $method $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
      #  else
      #    echo "no ckpts/${method}/model_state_epoch_${i}.th, or already runned; continue"
      #  fi
      #done
      for method in multi-fixed-lstm-mix-shift-langnorm ;
      do
        if grep $FT_LANG data/ensemble_langs.txt;
        then
          CKPT_DIR="${method}_${i}_${FT_LANG}_ens_$2"
        else
          CKPT_DIR="${method}_${i}_${FT_LANG}_$2"
        fi
        echo $CKPT_DIR
        if [[ ! ( -s "ckpts/$CKPT_DIR/result.txt" ) && -f "ckpts/${method}/model_state_epoch_${i}.th" ]];
        then
          rm -r ckpts/$CKPT_DIR;
          BASE_MODEL="multi-fixed-lstm-mix-ft-base" bash fine-tune.sh $method $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
        else
          echo "no ckpts/${method}/model_state_epoch_${i}.th, or already runned; continue"
        fi
      done
      #for method in fomaml-fixed-lstm-inherit-shift fomaml-fixed-lstm-shift multi-fixed-lstm-shift fomaml-fixed-lstm-inherit-lex-shift multi-fixed-lstm-lex-shift multi-fixed-lstm-lex-shift-nomem_1e-3 fomaml-fixed-lstm-inherit-lex-shift-nomem_1e-3;
      #do
      #  if grep $FT_LANG data/ensemble_langs.txt;
      #  then
      #    CKPT_DIR="${method}_${i}_${FT_LANG}_ens_$2"
      #  else
      #    CKPT_DIR="${method}_${i}_${FT_LANG}_$2"
      #  fi
      #  echo $CKPT_DIR
      #  if [[ ! ( -s "ckpts/$CKPT_DIR/result.txt" ) && -f "ckpts/${method}/model_state_epoch_${i}.th" ]];
      #  then
      #    rm -r ckpts/$CKPT_DIR;
      #    BASE_MODEL="multi-fixed-vib-lstm-base" SHIFT=1 bash fine-tune.sh $method $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
      #  else
      #    echo "no ckpts/${method}/model_state_epoch_${i}.th, or already runned; continue"
      #  fi
      #done
      #for method in fomaml-fixed-lstm-s4 fomaml-fixed-lstm-s4-nomem_1e-3 multi-fixed-lstm-nomem_1e-3 fomaml-fixed-lstm-nomem_1e-3 multi-fixed-lstm fomaml-fixed-lstm ; #fomaml-fixed-lstm-adv-0.1 multi-fixed-lstm-adv-0.1 ;
      #do
      #  if grep $FT_LANG data/ensemble_langs.txt;
      #  then
      #    CKPT_DIR="${method}_${i}_${FT_LANG}_ens_$2"
      #  else
      #    CKPT_DIR="${method}_${i}_${FT_LANG}_$2"
      #  fi
      #  echo $CKPT_DIR
      #  if [[ ! ( -s "ckpts/$CKPT_DIR/result.txt" ) && -f "ckpts/${method}/model_state_epoch_${i}.th" ]];
      #  then
      #    rm -r ckpts/$CKPT_DIR;
      #    BASE_MODEL="multi-fixed-vib-lstm-base" bash fine-tune.sh $method $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
      #  else
      #    echo "no ckpts/${method}/model_state_epoch_${i}.th, or already runned; continue"
      #  fi
      #done
    fi
  done
done<$1

