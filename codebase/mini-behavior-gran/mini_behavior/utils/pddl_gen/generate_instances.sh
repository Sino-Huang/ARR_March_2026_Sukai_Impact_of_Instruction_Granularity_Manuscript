#!/bin/bash

touch /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log

dm="laying_wood_floors"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi



dm="organizing_file_cabinet"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="thawing_frozen_food"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="making_tea"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="opening_packages"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="boxing_books_up_for_storage"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="collect_misplaced_items"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="putting_away_dishes_after_cleaning"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="washing_pots_and_pans"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="cleaning_shoes"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="installing_a_printer"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="setting_up_candles"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="watering_houseplants"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="cleaning_a_car"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="storing_food"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="throwing_away_leftovers"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="moving_boxes_to_storage"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="sorting_books"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="preparing_salad"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi

dm="cleaning_up_the_kitchen_only"
python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --domain_name "${dm}" --num_output 500 --timeout 12m  --max_processes 49

# check return code 
rcode=$?
if [ $rcode -ne 0 ]; then
    echo "Error for domain ${dm} with return code ${rcode}" >> /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/error.log
fi 


