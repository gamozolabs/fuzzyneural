clear
reset
set border 3

set terminal wxt size 506,253

set xrange [0:2500]

# Add a vertical dotted line at x=0 to show centre (mean) of distribution.
set yzeroaxis

# Each bar is half the (visual) width of its x-range.
set boxwidth 5 absolute
set style fill solid 1.0 noborder

set xlabel "Number of mutations until 0 error rate"
set ylabel "Frequency"

bin_width = 10;

bin_number(x) = floor(x/bin_width)

rounded(x) = bin_width * ( bin_number(x) + 0.5 )

plot 'output_nofeedback.txt' using (rounded($1)):(1) smooth frequency with boxes t "No feedback", \
    'output_feedback.txt' using (rounded($1)):(1) smooth frequency with boxes t "Feedback"
