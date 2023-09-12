#!/bin/bash
#$ -N train_openaies_gym
#$ -cwd
#$ -pe threads 18
#$ -l m_mem_free=1G
#$ -t 1-1
#$ -o logs/$JOB_ID_$TASK_ID/stdout.txt
#$ -e logs/$JOB_ID_$TASK_ID/stderr.txt

python train_openai_es.py \
--epochs 1000 \
--population_size 64 \
--population_top_best 64 \
--learning_rate 0.02 \
--sigma 0.02 \
--weight_decay 0.0 \
--task_sync 0 \
--environment "gym.make('SwimmerSwimmer6-v0')" \
--model "wriggly_train.tsaes_model.ActorOnly(
  actor=wriggly_train.tsaes_model.Actor(
    encoder=tonic.torch.models.ObservationEncoder(),
    torso=None,
    head=tonic.torch.models.DeterministicPolicyHead(
      activation=torch.nn.Identity,
      fn=wriggly_train.tsaes_model.init_linear_zeros_,
    )
  ),
  observation_normalizer=tonic.torch.normalizers.MeanStd()
)" \
--test_size 5 \
--num_workers 20
# --job_id ${JOB_ID}_${SGE_TASK_ID} \
# --seed ${SGE_TASK_ID}