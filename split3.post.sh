good () {
    local x
    for x in bad/*.gv
    do if cmp -s $i $x || cmp -s $j $x
       then
	   return 1
       fi
    done
    return 0
}

k=0
for d in [0-9][0-9][0-9]/
do d=${d%/}
   for x in `awk '$2 == $3 { print $1}' $d/cost`
   do i=${d}/$x.forward.gv
      j=${d}/$x.backward.gv
      if good
      then
	  p=`printf %03d $k`.png
	  gvpack 2>/dev/null -u $i $j | dot -Tpng -o $p
	  echo $p $i $j
	  k=$((k+1))
      fi
   done
done
montage 0*.png -bordercolor black -geometry +0+0 -border 3 x.png
