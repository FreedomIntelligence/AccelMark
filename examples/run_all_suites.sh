runner_id='nvidia_vllm_47f5d58e'
PYTHON_BIN="${1:-python}"
$PYTHON_BIN run.py --runner $runner_id --suite suite_A --tier verified --scenario all
$PYTHON_BIN run.py --runner $runner_id --suite suite_B --tier verified --scenario all
$PYTHON_BIN run.py --runner $runner_id --suite suite_C --tier verified --scenario all
$PYTHON_BIN run.py --runner $runner_id --suite suite_D --tier verified --scenario all
$PYTHON_BIN run.py --runner $runner_id --suite suite_E --tier verified --scenario all
$PYTHON_BIN run.py --runner $runner_id --suite suite_F --tier verified --scenario all
$PYTHON_BIN run.py --runner $runner_id --suite suite_G --tier verified --scenario all