#!/bin/bash

cd `dirname $0`

FILES=(
  'av1_r'
  'av2_r'
  'av3_r'
  'av4_r'
  'av5_r'
  'head_r'
  'iiyama1997_r'
  'aa053_r'
  'av007_r'
  'av020_r')

for (( I = 0; I < ${#FILES[@]}; ++I )); do
  FILE=${FILES[$I]}
  wget "http://www.mm.media.kyoto-u.ac.jp/old/education/le4cg/${FILE}.txt" -O ${FILE}.wrl
done
