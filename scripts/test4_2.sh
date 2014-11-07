#!/bin/bash
cd `dirname $0`

echo 'Starting tests...'
echo

vrml_files='../samples/*.wrl'

echo '===== task4_2.py ====='
start_time=`date +%s`
for vrml in $vrml_files; do
  echo $vrml
  ppm=${vrml%.wrl}_4_2_f.ppm
  profile=${vrml%.wrl}_4_2_f.json
  python task4_2.py -s flat -p $profile -o $ppm $vrml > task4_2.py.log 2>&1
  if [ $? -eq 0 ]; then
    echo 'Success (Flat)'
  else
    echo 'Error (Flat)'
    cat task4_2.py.log
    exit 1
  fi
  ppm=${vrml%.wrl}_4_2_g.ppm
  profile=${vrml%.wrl}_4_2_g.json
  python task4_2.py -s gouraud -p $profile -o $ppm $vrml > task4_2.py.log 2>&1
  if [ $? -eq 0 ]; then
    echo 'Success (Gouraud)'
  else
    echo 'Error (Gouraud)'
    cat task4_2.py.log
    exit 1
  fi
  ppm=${vrml%.wrl}_4_2_p.ppm
  profile=${vrml%.wrl}_4_2_p.json
  python task4_2.py -s phong -p $profile -o $ppm $vrml > task4_2.py.log 2>&1
  if [ $? -eq 0 ]; then
    echo 'Success (Phong)'
  else
    echo 'Error (Phong)'
    cat task4_2.py.log
    exit 1
  fi
done
echo
end_time=`date +%s`
seconds=`expr $end_time - $start_time`
echo "Finished $seconds [sec]"
echo

echo 'Analyzing performance profiles...'
echo

echo '===== task4_2.py (Flat) ====='
python profile.py ../samples/*4_2_f.json
echo

echo '===== task4_2.py (Gouraud) ====='
python profile.py ../samples/*4_2_g.json
echo

echo '===== task4_2.py (Phong) ====='
python profile.py ../samples/*4_2_p.json
echo

exit 0
