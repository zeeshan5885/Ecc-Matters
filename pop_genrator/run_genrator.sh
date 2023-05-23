rm combine_pop.txt ; rm -r txt_events ; rm -r dat_events ; rm -r __pycache__ ;
python population_gen.py ; mkdir txt_events ; mkdir dat_events; mv eve* ./txt_events ; cd txt_events ; cp eve* ../dat_events ; cd .. ; cd dat_events;  rename txt dat *.txt
