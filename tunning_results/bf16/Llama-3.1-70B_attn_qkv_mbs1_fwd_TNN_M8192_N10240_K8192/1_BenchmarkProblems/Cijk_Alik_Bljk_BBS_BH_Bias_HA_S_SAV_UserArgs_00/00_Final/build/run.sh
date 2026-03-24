#!/bin/bash

set -ex
set +e
ERR1=0
/workspace/lorri/hipblaslt-tunning-helper/tmp_rebuild/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client --config-file /workspace/lorri/hipblaslt-tunning-helper/tunning_results/bf16/Llama-3.1-70B_attn_qkv_mbs1_fwd_TNN_M8192_N10240_K8192/1_BenchmarkProblems/Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_00/00_Final/build/../source/ClientParameters.ini
ERR2=$?


ERR=0
if [[ $ERR1 -ne 0 ]]
then
    echo one
    ERR=$ERR1
fi
if [[ $ERR2 -ne 0 ]]
then
    echo two
    ERR=$ERR2
fi
exit $ERR
