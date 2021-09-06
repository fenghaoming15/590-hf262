
#------------------------------------------------------------------------------------
#DESCRIPTION
#------------------------------------------------------------------------------------

#ASSIGNMENT: 
      #0) OPEN THIS FILE AND A COMMAND LINE SIDE BY SIDE
      #1) ONE BY ONE, UNCOMMENT EACH BLOCK OF CODE, 
            #RUN THE SHELL SCRIPT, OBSERVE THE OUTPUT 
      #2) ADD SHORT COMMENTS AS NEEDED, THEN COMMENT THE BLOCK OUT AGAIN
            #GOOGLE THE COMMANDS IF YOU NEED TO 
            #OR USE "man" FOR DETAILS 
            #OR COME SEE ME 
      #3) MOVE ONTO THE NEXT BLOCK  
      #4) IF A COMMAND HAS ALREADY BEEN COMMENTED ON THEN YOU DONT NEED TO COMMENT A SECOND TIME 

      #NOTE: BLOCK COMMENT IN GEDIT IS cntl-m IN SUBL TEXT ITS cntl-/
      #NOTE --> THE CHARACTER "#" COMMENTS OUT A LINE IN A SHELL SCRIPT


#------------------------------------------------------------------------------------

echo "-------------------------------"
echo "PART-1: BASICS"
echo "-------------------------------"

echo "0) echo: one of the simpliest commands is 'echo'. It prints whatever is fed to it to the screen"
echo;       #print blank line
sleep 5     #pause script execution for 5 seconds 

echo "0) The command 'sleep 5' pauses the script execution for 5 seconds"
sleep 5  

printf "\n1) Any command that can be excuted from the command line can also be excuted sequentially in an executable file known as a shell script, usually with the extension '.sh'. To make the file executable, you need to change the file permission using the command 'chmod a+x file_name.sh' \n"
sleep 10

echo "
2) we can define variables 'sleep_time=5'
"

sleep_time=5 #define variable with time to sleep in seconds 

echo "3) variables are referenced using a $ at the beginning, for example sleep_time="$sleep_time

sleep $sleep_time

printf "\n 4) In linux, folders are called directories, the entire linux system is stored in a hieracrhical directory tree, to see where you are use pwd which stands for print working directory \n\n"

pwd

sleep $sleep_time

SCRIPT_LOC=${PWD} #SAVE ABSOLUTE PATH TO L1.1.4-LinuxBasics.sh SHELL SCRIPT TO FILE 


DATE=$(date -Is) #save date as variable $(date +"%Y-%m-%d")
printf "\n DATE="$DATE"\n"

sleep $sleep_time




#---------------------------
#FILESYSTEM: 
#---------------------------

echo "------------------------"
echo "EXPLORE THE LINUX FILE SYSTEM"
echo "------------------------"
cd / 
pwd
ls 

echo "A------------------------"; sleep $sleep_time
cd /home/
pwd
ls 

echo "B------------------------"; sleep $sleep_time
cd ~/; 
pwd
echo "----------"
ls -ltr 
echo "----------"
ls *;
echo "----------"
ls -a
echo "----------"
du -csh *


#---------------------------
#BLOCK: MAKING FOLDERS AND FILES
#---------------------------


echo "A------------------------"; sleep $sleep_time
cd ~/Documents;
ls

echo "B------------------------"; sleep $sleep_time

rm -rf example_directory
mkdir example_directory
ls


echo "WARNING: BE EXTREMELY CAREFUL WITH 'rm -rf'"
echo "      ESPECIALLY COUPLED THE WILDCARD VARIABLE * OR SUDO (SUPERUSER STATUS)"
echo "      FOR EXAMPLE RUNNING  'cd /; sudo rm -rf *'"
echo "      WOULD DELETE THE ENTIRE OPERATING SYSTEM AND EVERY FILE ON THE LINUX MACHINE"


echo "C------------------------"; sleep $sleep_time

cd example_directory
ls 
echo "im writing to a file" > file1.dat
echo "hello computer" > file2.dat
echo "hello human" >> file2.dat
ls 
more file*.dat
nano file*.dat
rm file1.dat
ls
more file2.dat
> file2.dat
more file2.dat



#---------------------------
#FOR LOOPS AND WILDCARD *: 
#---------------------------

#RETURN TO SCRIPT LOCATION
cd $SCRIPT_LOC

echo "----HERE-A----"
ls *.py
echo "----HERE-B----"
ls *.sh 

echo "----HERE-C----"
for i in *.py
do
      echo "FILE=" $i
done
sleep 5

echo "----HERE-D----"
for i in *.py
do
      echo "-----------" $i "-----------" 
      grep "np" $i  #print everywhere you see string np in file
done
sleep 5


echo "----HERE-E----"
for i in *.py
do
      echo "-----------" $i "-----------" 
      python $i  #run all python scripts
done
sleep 5


#---------------------------
#BLOCK: MANUALS AND ALIAS
#---------------------------

man pwd
man ls
man echo


# CREATING AN ALIAS 
# alias cd590="cd ~/590-CODES/"
# cd590

exit #STOP THE SCRIPT 


#------------------------
#NOTES
#-------------------------

##IMPORTANT LOCATIONS 
## /
## IS THE BOTTOME OF THE DIRECTORY TREE, SHOULD ALMOST NEVER EDIT THINGS HERE
      ## IT IS WHERE THE OPERATING SYSTEM AND 

##GROUPS

##SUPERUSERS --> FULL PERMISSION (CANT DO VERY BAD THINGS)
#     # sudo rm -rf /  WOULD DELETE THE OPERATING SYSTEM AND ALL FILES
##USERS --> LIMITED FILE PERMISSIONS (CANT BREAK THINGS TOO BADLY)


##NAVIGATING THE DIRECTORY TREE