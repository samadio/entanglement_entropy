for ifile in "$@"
do
ofile="copy_${ifile}"
tr -s '\r\n' ',' < $ifile | sed -e 's/,$/\n/' > $ofile
numb=$(echo $ifile | sed -e 's/[^(0-9|)]//g' | sed -e 's/|/,/g')
prefix="results_${numb}"
sed -i -e "s/^/$prefix = [/" $ofile
sed -i -e 's/$/]/' $ofile

echo "\nfrom numpy import array\n$prefix = array($prefix)\nY_${numb} = $prefix.T[0]\nS_${numb} = $prefix.T[1]" >>$ofile
done
