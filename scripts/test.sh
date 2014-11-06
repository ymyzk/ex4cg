#!/bin/bash
cd `dirname $0`

echo 'Starting tests...'
echo

vrml_files='../samples/*.wrl'

echo '===== task1.py ====='
python task1.py > task1.py.log 2>&1
if [ $? -eq 0 ]; then
  echo 'Success'
else
  echo 'Error'
  cat task1.py.log
  exit 1
fi
echo

echo '===== task2.py ====='
for vrml in $vrml_files; do
  echo $vrml
  python task2.py $vrml > task2.py.log 2>&1
  if [ $? -eq 0 ]; then
    echo 'Success'
  else
    echo 'Error'
    cat task2.py.log
    exit 1
  fi
done

exit 0
