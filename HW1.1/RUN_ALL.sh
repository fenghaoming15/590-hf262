ScriptLoc=${PWD}
cd LectureCodes
for i in *.py; do echo $i; python $i; done
grep "I HAVE WORKED" *
cd $ScriptLoc
for i in *.py; do echo $i; python $i; done





