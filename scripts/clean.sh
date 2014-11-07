#!/bin/bash
cd `dirname $0`

echo 'Removing files...'
rm ./*.log
rm ../samples/*.ppm
rm ../samples/*.json
