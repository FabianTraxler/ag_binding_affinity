#!/bin/bash

PDB_PATH=$1;
ROSETTA_DIR=$3;

rm -f list1.txt list2.txt list3.txt
for i in `cat $2`
do
  IFS=, read -r file ab_chains ag_chains <<< $i
  python ../scripts/split_complex.py $PDB_PATH/$file $ab_chains $ag_chains
  cp $PDB_PATH/$file.pdb $file.comp.pdb
  echo $file.rec.pdb >> list1.txt
  echo $file.lig.pdb >> list2.txt
  echo $file.comp.pdb >> list3.txt
done

#rm list1.txt.ros.scores.out list2.txt.ros.scores.out list3.txt.ros.scores.out
../scripts/run_ros_score.pl list1.txt $ROSETTA_DIR  > list1.log
../scripts/run_ros_score.pl list2.txt $ROSETTA_DIR > list2.log
../scripts/run_ros_score.pl list3.txt $ROSETTA_DIR > list3.log

echo ""
echo "Antibody"
awk '{print $2}' list1.txt.ros.scores.out
echo ""
echo "Antigen"
awk '{print $2}' list2.txt.ros.scores.out
echo ""
echo "Complex"
awk '{print $2}' list3.txt.ros.scores.out

# clean up
rm -f *.rec.pdb *.comp.pdb *.lig.pdb list*.txt.ros.scores.out list*.log *.pdb
