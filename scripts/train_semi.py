# coding=utf-8

from sh import git, sbatch
import uuid
import time

train_data = "full"
unlabeled_data = ""
lr_decay = 0.5
unsup_loss_weight = 0.1
encoder = ""
decoder = ""
exp_id = time.strftime("%Y%m%d-%H%M%S") + '-' + str(uuid.uuid4())[:4]

job_name = 'model.semisup.label_{train_data}.unlabeled_{unlabeled_data}.lr_decay{lr_decay}' \
           '.unsup_w{unsup_loss_weight}.encoder_{encoder}.decoder_{decoder}.exp_id'.format(train_data=train_data,
                                                                                           unlabeled_data=unlabeled_data,
                                                                                           lr_decay=lr_decay,
                                                                                           unsup_loss_weight=0.1,
                                                                                           encoder=encoder[:10],
                                                                                           decoder=decoder[:10],
                                                                                           exp_id=exp_id)
job_script = '.job_scripts/%s.sh' % job_name
with open(job_script, 'w') as f:
    f.write("""#!/bin/sh
    
echo "{job_name}" > logs/{job_name}.log

python exp.py \
    --cuda \
    --mode train \
    --batch_size 10 \
    --train_file ../data/django/{train_data} \
    --unlabeled_file ../data/django/{unlabeled_data} \
    --dev_file ../data/django/dev.bin \
    --load_model saved_models/{encoder} \
    --load_decoder saved_models/{decoder} \
    --patience 5 \
    --max_num_trial 3 \
    --lr_decay {lr_decay} \
    --save_to saved_models/{job_name} 2>>logs/{job_name}.log

python exp.py \
	--cuda \
    --mode test \
    --load_model saved_models/{job_name}.bin \
    --beam_size 15 \
    --test_file ../data/django/test.bin \
    --decode_max_time_step 100 2>>logs/{job_name}.log

""".format(train_data=train_data, unlabeled_data=unlabeled_data,
           encoder=encoder, decoder=decoder,
           lr_decay=lr_decay,
           job_name=job_name))

cmd = sbatch('--gres', 'gpu:1',
             '--job-name', job_name,
             '--mem', 25000,  # memory
             '--cpus-per-task', 8,  # number of cpus
             '--time', 0,  # wait time: unlimited
             '--output', 'logs/%s.out' % job_name,  # redirect stdout to file
             job_script)

print cmd.stdout
