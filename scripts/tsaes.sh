#!/bin/bash
#$ -N train_tsaes_gym
#$ -cwd
#$ -pe threads 64
#$ -l m_mem_free=1G
#$ -t 1-1
#$ -o logs/$JOB_ID_$TASK_ID/stdout.txt
#$ -e logs/$JOB_ID_$TASK_ID/stderr.txt

python train_tsaes.py \
--epochs 5000 \
--population_size 256 \
--population_top_best 128 \
--learning_rate 0.05 \
--momentum 0.0 \
--lookahead_scaling 0.0 \
--environment "gym.make('SwimmerSwimmer6-v0')" \
--model "wriggly_train.tsaes_model.ActorOnly(
  actor=wriggly_train.tsaes_model.Actor(
    encoder=wriggly_train.tsaes_model.ObservationEncoder(),
    torso=None,
    head=tonic.torch.models.DeterministicPolicyHead(
      activation=torch.nn.Identity,
      fn=wriggly_train.tsaes_model.init_linear_zeros_,
    )
  ),
  observation_normalizer=wriggly_train.tsaes_model.MeanStd()
)" \
--test_size 5 \
--num_workers 32
# --job_id ${JOB_ID}_${SGE_TASK_ID} \
# --seed ${SGE_TASK_ID}
# try momentum on and off
# lookahead scaling off = ARS, on = TSAES