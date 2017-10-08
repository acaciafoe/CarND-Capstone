j=0;for i in *.jpg; do mv "$i" frame"$(printf "%03d" j)".jpg; let j=j+1;done
