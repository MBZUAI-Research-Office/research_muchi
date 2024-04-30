#!/bin/bash
osascript - "$1" "$2" "$3" "$4" <<EOF
    on run argv -- argv is a list of strings
        tell application "Terminal"
            do script ("conda activate dbrx_poc && cd ~/research_muchi/dbrx && python moe_shard_batch2.py --port " & item 1 of argv & " --model-path ~/dbrx-base/distributable/batch2 --config-filename moe_shard_config_0.json")
            do script ("conda activate dbrx_poc && cd ~/research_muchi/dbrx && python moe_shard_batch2.py --port " & item 2 of argv & " --model-path ~/dbrx-base/distributable/batch2 --config-filename moe_shard_config_1.json")
            do script ("conda activate dbrx_poc && cd ~/research_muchi/dbrx && python moe_shard_batch2.py --port " & item 3 of argv & " --model-path ~/dbrx-base/distributable/batch2 --config-filename moe_shard_config_2.json")
            do script ("conda activate dbrx_poc && cd ~/research_muchi/dbrx && python moe_shard_batch2.py --port " & item 4 of argv & " --model-path ~/dbrx-base/distributable/batch2 --config-filename moe_shard_config_3.json")
        end tell
    end run
EOF
