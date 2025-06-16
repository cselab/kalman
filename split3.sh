n=`nproc`
i=0
while test $i -lt $n
do d=`printf %03d $i`
   mkdir -p $d
   i=$((i+1))
done

ls -d [0-9][0-9][0-9]/ | xargs -n 1 -P $n sh -xc \
			       'cd $1 && python ../split3.py ${1%/}' run
