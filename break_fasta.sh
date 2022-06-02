fasta=$1
while read line
do
    if [[ ${line:0:1} == '>' ]]
    then
        tmp_outfile=${line#>}
        tmp_outfile2=${tmp_outfile%% *}
        outfile=${tmp_outfile2##*|}.fa
        
        echo $line > "$outfile"
    else
        echo $line >> "$outfile"
    fi
done < $f
