help:
.PHONY: help

run:
	python test_vivado_accelerator.py
.PHONY: run	

clean:
	rm -f classes.npy
	rm -rf training_dir
	rm -rf __pycache__
	rm -rf test_backend_with_tb_axi_stream
	rm -f test_backend_with_tb_axi_stream.tar.gz
	rm -rf test_backend_with_tb_axi_lite
	rm -f test_backend_with_tb_axi_lite.tar.gz
	rm -rf test_backend_with_tb_axi_master
	rm -f test_backend_with_tb_axi_master.tar.gz
	rm -f X_test.npy
	rm -f y_qkeras.npy
	rm -f y_test.npy
	rm -f *.log
	rm -f *.jou
	rm -f *.str
	rm -rf NA
.PHONY: clean
