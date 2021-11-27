help:
.PHONY: help

run:
	python test_vivado_accelerator.py
.PHONY: run	

run-profile:
	python test_vivado_accelerator.py profile
.PHONY: run-profile	

vivado-gui:
	vivado ./test_backend_with_tb_axi_master/myproject_vivado_accelerator/project_1.xpr
.PHONY: vivado-gui

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
	make -C sdk clean
.PHONY: clean

ultraclean: clean
	make -C sdk ultraclean
.PHONY: ultraclean
