#!/bin/bash
conda activate polimi
while read -p 'Retry build?' -r RESPONSE
do
  echo "Response: '$RESPONSE'"
  if [ "$RESPONSE" == "" ]; then
    echo 'Exiting'
    break
  fi
  echo 'Building source'
  python setup.py build_ext > build_log.txt 2>&1
  echo 'Errors:'
  cat build_log.txt | grep -C 3 "/gpp_domain." | grep -C 3 error:
done