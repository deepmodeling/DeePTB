# band plot
# dptb run band.json -sk -i ./1.start/checkpoint/best_nnsk_b1.600_c1.600_w0.300.pth -o 1.start
# dptb run band.json -sk -i ./2.strain/checkpoint/best_nnsk_b1.600_c1.600_w0.300.pth -o 2.strain
# dptb run band.json -sk -i ./3.varycutoff/checkpoint/latest_nnsk_b3.600_c3.599_w0.300.pth -o 3.varycutoff
# dptb run band.json -sk -i ./4.longtrain/checkpoint/latest_nnsk_b3.600_c3.600_w0.300.pth -o 4.longtrain

# dos plot
dptb run dos.json -sk -i ./1.start/checkpoint/best_nnsk_b1.600_c1.600_w0.300.pth -o 1.start
dptb run dos.json -sk -i ./2.strain/checkpoint/best_nnsk_b1.600_c1.600_w0.300.pth -o 2.strain
dptb run dos.json -sk -i ./3.varycutoff/checkpoint/latest_nnsk_b3.600_c3.599_w0.300.pth -o 3.varycutoff
dptb run dos.json -sk -i ./4.longtrain/checkpoint/latest_nnsk_b3.600_c3.600_w0.300.pth -o 4.longtrain