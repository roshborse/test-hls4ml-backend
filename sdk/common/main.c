/**
 *
 * Set Heap Size in ldscript.ld to 0x1000000 (16MB)
 *
 */

#include "xmyproject_axi.h"  /* accelerator */
#include "stdio.h"       /* PRINTF */
#include "unistd.h"      /* sleep */
#include "stdlib.h"
#include "malloc.h"
#include "assert.h"
#include "xil_io.h"      /* peripheral read/write wrappers */
#include "xtime_l.h"     /* to measure performance of the system */
#include "platform.h"    /* platform init/cleanup functions */
#include "xil_cache.h"   /* enable/disable caches etc */
#include "xil_printf.h"  /* UART debug print functions */
#include "xparameters.h" /* peripherals base addresses */

/* TODO: move the following code in separate header files */
#include "data.h"

#define ACC_NAME "Myaccelerator"

#define __DEBUG__

#define MAX_PRINT_ELEMENTS (16)

#define PRINTF printf

const unsigned INPUT_N_ELEMENTS = N_SAMPLES * N_X_INPUTS;
const unsigned OUTPUT_N_ELEMENTS = N_SAMPLES * N_Y_OUTPUTS;

#if 0
/* base address for the accelerator */
#define MEM_BASE_ADDR XPAR_PS7_DDR_0_S_AXI_BASEADDR

/* data offsets and pointers */
#define SRC_BUFFER_BASE (MEM_BASE_ADDR + 0x00000000)
float *inputs_mem = (float*)SRC_BUFFER_BASE;

#define GLD_BUFFER_BASE (MEM_BASE_ADDR + 0x00010000)
float *reference_mem = (float*)GLD_BUFFER_BASE;

#define DST_BUFFER_BASE (MEM_BASE_ADDR + 0x00020000)
float *outputs_mem = (float*)DST_BUFFER_BASE;
#else
float *inputs_mem;
float *outputs_mem;
float *reference_mem;
#endif

/* accelerator configuration */
XMyproject_axi accelerator;
XMyproject_axi_Config *accelerator_cfg;

/* accelerator initialization routine */
void init_accelerators() {
    PRINTF("INFO: Initializing accelerator\n\r");
    accelerator_cfg = XMyproject_axi_LookupConfig(XPAR_MYPROJECT_AXI_0_DEVICE_ID);
    if (accelerator_cfg) {
        int status  = XMyproject_axi_CfgInitialize(&accelerator, accelerator_cfg);
        if (status != XST_SUCCESS) {
            PRINTF("ERROR: Initializing accelerator\n\r");
        }
    }
}

//#if defined(__HPC_ACCELERATOR__) || defined(__ACP_ACCELERATOR__)
///*
// *  TODO: remember to edit core_baremetal_polling_bsp/psu_cortexa53_0/libsrc/standalon_v6_5/src/bspconfig.h
// *
// *  #define EL1_NONSECURE 1
// *
// */
//void init_accelerator_coherency(UINTPTR base_addr)
//{
//    /* Enable snooping of APU caches from CCI */
//    Xil_Out32(0xFD6E4000, 0x1);
//
//    /* Configure AxCACHE for write-back read and write-allocate (ARCACHE is [7:4], AWCACHE is [11:8]) */
//    /* Configure AxPROT[2:0] for data access [2], secure access [1], unprivileged access [0] */
//    Xil_Out32(base_addr, 0xFF0);
//}
//#endif

/* golden model of the accelerator in software */
int accelerator_sw(float *src, float *dst, unsigned input_n_elements, unsigned output_n_elements) {
	PRINTF("INFO: Golden results are pre-compiled. It would be nice to run a software model here.\r\n");
    // See src.h and dst.h for input and golden output respectively.
    return 0;
}

/* profiling function */
double get_elapsed_time(XTime start, XTime stop) {
    return 1.0 * (stop - start) / (COUNTS_PER_SECOND);
}

/* dump data to the console */
void dump_data(const char* label, float* data, unsigned n_samples, unsigned feature_count) {
	PRINTF("INFO:   %s[%u][%u]:\n\r", label, n_samples, feature_count);
    /* print at most MAX_PRINT_ELEMENTS */
    for (unsigned i = 0; i < n_samples && i < MAX_PRINT_ELEMENTS; i++) {
    	PRINTF("INFO:     [%u] ", i);
        for (unsigned j = 0; j < feature_count; j++) {
        	unsigned index = i * feature_count + j;
        	PRINTF("%f ", data[index]);
        }
        PRINTF("\r\n");
    }
}

/* the top of the hill :-) */
int main(int argc, char** argv) {

    XTime start, stop;
    double calibration_time;
    double sw_elapsed = 0;
    double hw_elapsed = 0;
    double cache_elapsed = 0;
    unsigned hw_errors;

    char __attribute__ ((unused)) dummy; /* dummy input */

    /* initialize platform (uart and caches) */
    init_platform();

    PRINTF("\n\r");
    PRINTF("INFO: ===============================================\n\r");
    PRINTF("INFO: "ACC_NAME" (w/ polling)\n\r");
    PRINTF("INFO: ===============================================\n\r");

    init_accelerators();

    inputs_mem = malloc(INPUT_N_ELEMENTS * sizeof(float));
    outputs_mem = malloc(OUTPUT_N_ELEMENTS * sizeof(float));
    reference_mem = malloc(OUTPUT_N_ELEMENTS * sizeof(float));

    /* calibration */
    XTime_GetTime(&start);
    sleep(1);
    XTime_GetTime(&stop);
    calibration_time = get_elapsed_time(start, stop);
    PRINTF("INFO: Time calibration for one second (%lf sec)\n\r", calibration_time);

    /* initialize memory */
    PRINTF("INFO: Initialize memory\n\r");
    PRINTF("INFO:   - Samples count: %u\n\r", N_SAMPLES); /* Same as dst_SAMPLE_COUNT */
    PRINTF("INFO:   - Inputs count: %u\n\r", N_X_INPUTS);
    PRINTF("INFO:   - Outputs count: %u\n\r", N_Y_OUTPUTS);
    PRINTF("INFO:   - Data size: %u B\n\r", sizeof(float));
    PRINTF("INFO:   - Total input size: %u B, %.2f KB, %.2f MB\n\r", N_X_INPUTS * N_SAMPLES * sizeof(float), (N_X_INPUTS * N_SAMPLES * sizeof(float)) / (float)1024, (N_X_INPUTS * N_SAMPLES * sizeof(float)) / (float)(1024*1024));
    PRINTF("INFO:   - Total output size: %u B, %.2f KB, %.2f MB\n\r", N_Y_OUTPUTS * N_SAMPLES * sizeof(float), (N_Y_OUTPUTS * N_SAMPLES * sizeof(float)) / (float)1024, (N_Y_OUTPUTS * N_SAMPLES * sizeof(float)) / (float)(1024*1024));

    // Set Heap Size in ldscript.ld to 0x1000000 (16MB)
    //malloc_stats();

    for (int i = 0; i < INPUT_N_ELEMENTS; i++) {
        inputs_mem[i] = src_data[i];
    }
    for (int i = 0; i < OUTPUT_N_ELEMENTS; i++) {
        reference_mem[i] = dst_data[i];
        outputs_mem[i] = 0x0;
    }

    /* ****** SOFTWARE REFERENCE ****** */
#ifdef __DEBUG__
    PRINTF("INFO: Start SW accelerator\n\r");
#endif
    XTime_GetTime(&start);

    accelerator_sw(inputs_mem, reference_mem, INPUT_N_ELEMENTS, OUTPUT_N_ELEMENTS);
    XTime_GetTime(&stop);
    sw_elapsed = get_elapsed_time(start, stop);

#ifdef __DEBUG__
    PRINTF("INFO: Number of accelerator invocations: %u\n\r", N_SAMPLES);
#endif
#if 1
    /* ****** ACCELERATOR ****** */
    PRINTF("INFO: Press any key to start the accelerator: ");
    dummy = inbyte();
    PRINTF("\n\rINFO: \n\r");

#ifdef __DEBUG__
    PRINTF("INFO: Configure and start accelerator\n\r");
#endif

    XTime_GetTime(&start);
    Xil_DCacheFlushRange((UINTPTR)inputs_mem, INPUT_N_ELEMENTS * sizeof(float));
    Xil_DCacheFlushRange((UINTPTR)outputs_mem, OUTPUT_N_ELEMENTS * sizeof(float));
    Xil_DCacheFlushRange((UINTPTR)reference_mem, OUTPUT_N_ELEMENTS * sizeof(float));
    XTime_GetTime(&stop);
    cache_elapsed = get_elapsed_time(start, stop);

    for (unsigned j = 0; j < N_SAMPLES; j++) {
    	float *inputs_mem_i = inputs_mem + j * N_X_INPUTS;
    	float *outputs_mem_i = outputs_mem + j * N_Y_OUTPUTS;

    	/* Configure the accelerator */
    	XTime_GetTime(&start);
        XMyproject_axi_Set_in_r(&accelerator, (unsigned)inputs_mem_i);
    	XMyproject_axi_Set_out_r(&accelerator, (unsigned)outputs_mem_i);

    	XMyproject_axi_Start(&accelerator);

    	/* polling */
    	while (!XMyproject_axi_IsDone(&accelerator));

    	/* get error status */
    	//hw_flags = XMyproject_axi_Get_return(&accelerator);
    	XTime_GetTime(&stop);
    	hw_elapsed += get_elapsed_time(start, stop);
    }

    XTime_GetTime(&start);
    Xil_DCacheFlushRange((UINTPTR)outputs_mem, OUTPUT_N_ELEMENTS * sizeof(float));
    XTime_GetTime(&stop);
    cache_elapsed += get_elapsed_time(start, stop);

    PRINTF("INFO: Accelerator done!");

    /* ****** VALIDATION ****** */

#ifdef __DEBUG__
    PRINTF("INFO: ================== Validation =================\n\r");
    PRINTF("INFO: Dump data\n\r");
    dump_data("inputs_mem", inputs_mem, N_SAMPLES, N_X_INPUTS);
    dump_data("outputs_mem", outputs_mem, N_SAMPLES, N_Y_OUTPUTS);
    dump_data("reference_mem", reference_mem, N_SAMPLES, N_Y_OUTPUTS);
#endif

    PRINTF("INFO: Software execution time: %f sec\n\r", sw_elapsed);

    PRINTF("INFO: Total acceleration execution time (%d inferences): %f sec\n\r", N_SAMPLES, hw_elapsed);
    PRINTF("INFO: Per-inference execution time (average): %.12f sec (%f ns)\n\r", hw_elapsed / (N_SAMPLES), (hw_elapsed*1000.0) / (N_SAMPLES));
    PRINTF("INFO: Cache flush time: %f sec\n\r", cache_elapsed);
    PRINTF("INFO: Accelerator/software speedup (the software is fake so this does not count...): %.2f X\n\r", (sw_elapsed >= (hw_elapsed+cache_elapsed))?(sw_elapsed/(hw_elapsed+cache_elapsed)):-((hw_elapsed+cache_elapsed)/sw_elapsed));

    /* Accelerator validation */
    hw_errors = 0;
#if 1
    for (int i = 0; i < OUTPUT_N_ELEMENTS; i++) {
        if (outputs_mem[i] != reference_mem[i]) {
            PRINTF("ERROR: [%d]: Accelerator hw %f != sw %f\n\r", i, outputs_mem[i], reference_mem[i]);
            hw_errors++;
        }
    }
    PRINTF("INFO: Total errors = %d (out of %d elements)\n\r", hw_errors, OUTPUT_N_ELEMENTS);
    if (hw_errors > 0)
        PRINTF("INFO: Accelerator validation: FAIL\n\r");
    else
        PRINTF("INFO: Accelerator validation: PASS!\n\r");
#endif
    PRINTF("INFO: Validation done!\n\r");
#endif
    PRINTF("INFO: ===============================================\n\r");

    cleanup_platform();

    return 0;
}


