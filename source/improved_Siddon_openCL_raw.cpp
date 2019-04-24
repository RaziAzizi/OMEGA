#include "arrayfire.h"
#include "mex.h"
#include <algorithm>
#include <vector>
#include <af/opencl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <CL/cl.h>
#include <cstdint>

using namespace af;
using std::vector;

// this error code was taken from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char *getErrorString(cl_int error)
{
	switch (error) {
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
	case -69: return "CL_INVALID_PIPE_SIZE";
	case -70: return "CL_INVALID_DEVICE_QUEUE";
	case -71: return "CL_INVALID_SPEC_ID";
	case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
	case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
	case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
	case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
	case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
	case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
	case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
	case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
	case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
	case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
	case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
	case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
	case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
	case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
	case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
	case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
	case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
	case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
	case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
	case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
	default: return "Unknown OpenCL error";
	}
}

void reconstruction(const size_t koko, const uint16_t* lor1, const float* z_det, const float* x, const float* y, const float* Sin, const int Nx, const int Ny, 
	const int Nz, const int Niter, const float* x0, const float d, const float dz, const float bxf, const float byf, const float bzf, const float bxb, 
	const float byb, const float bzb, const float bx, const float by, const float bz, const float maxxx, const float maxyy, const float minxx, 
	const float minyy, const float zmax, const float xxv, const float yyv, const float xxa, const float yya, const float NSlices, const uint32_t* pituus, 
	const uint64_t* summa, const size_t koko_l, const uint16_t* L, const int size_x, const int TotSinos, float* ele, const bool verbose, 
	const int attenuation_correction, const float* atten, const int size_atten, const int subsets, const float epps, const int* pseudos, 
	const int det_per_ring, const int prows, const char* k_path) {


	array lor(koko_l, lor1, afHost);

	array LL(koko, L, afHost);

	array Sino(koko_l, Sin, afHost);

	array pz_osem = constant(1, Ny*Nx*Nz, Niter + 1);

	array x00(Nx*Ny*Nz, x0, afHost);


	pz_osem(span, 0) = x00;

	array pz_osem_apu;



	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	int status = CL_SUCCESS;

	std::string kernel_path;

	kernel_path = k_path;
	std::fstream sourceFile(kernel_path.c_str());
	std::string content((std::istreambuf_iterator<char>(sourceFile)),std::istreambuf_iterator<char>());
	const char* sourceCode = new char[content.size()];
	sourceCode = content.c_str();
	cl_program program = clCreateProgramWithSource(af_context, 1, (const char **)&sourceCode, NULL, &status);


	if (status != CL_SUCCESS) {
		std::cerr << getErrorString(status) << std::endl;
		mexPrintf("Failed to create OpenCL program\n");
		return;
	}
	else if (verbose) {
		mexPrintf("OpenCL program successfully created\n");
		mexEvalString("pause(.0001);");
	}


	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);

	if (status != CL_SUCCESS) {
		std::cerr << getErrorString(status) << std::endl;
		mexPrintf("Failed to build OpenCL program. Build log: \n");
		size_t len;
		char *buffer;
		clGetProgramBuildInfo(program, af_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		buffer = (char*)calloc(len, sizeof(size_t));
		clGetProgramBuildInfo(program, af_device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		mexPrintf("%s\n", buffer);
		return;
	}
	else if (verbose) {
		mexPrintf("OpenCL program successfully built\n");
		mexEvalString("pause(.0001);");
	}

	cl_kernel kernel = clCreateKernel(program, "siddon", &status);

	if (status != CL_SUCCESS) {
		std::cerr << getErrorString(status) << std::endl;
		mexPrintf("Failed to create OpenCL kernel\n");
		return;
	}
	else if (verbose) {
		mexPrintf("OpenCL kernel successfully created\n");
		mexEvalString("pause(.0001);");
	}

	cl_mem d_Nx = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
	cl_mem d_Ny = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
	cl_mem d_Nz = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
	cl_mem d_dz = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_d = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_bz = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_bx = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_by = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_bxf = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_byf = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_bzf = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_bxb = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_byb = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_bzb = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_maxxx = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_maxyy = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_minxx = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_minyy = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_zmax = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_NSlices = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_xxv = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_yyv = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_xxa = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_yya = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);
	cl_mem d_z = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float) * TotSinos * 2, NULL, &status);
	cl_mem d_x = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float) * size_x * 2, NULL, &status);
	cl_mem d_y = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float) * size_x * 2, NULL, &status);
	cl_mem d_atten = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(float) * size_atten, NULL, &status);
	cl_mem d_det_per_ring = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
	cl_mem d_pseudos = clCreateBuffer(af_context, CL_MEM_READ_ONLY, sizeof(int) * prows, NULL, &status);

	if (status != CL_SUCCESS) {
		std::cerr << getErrorString(status) << std::endl;
		mexPrintf("Buffer creation failed\n");
	}
	else if (verbose){
		mexPrintf("Buffers created\n");
		mexEvalString("pause(.0001);");
	}

	status = clEnqueueWriteBuffer(af_queue, d_Nx, CL_FALSE, 0, sizeof(int), &Nx, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_Ny, CL_FALSE, 0, sizeof(int), &Ny, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_Nz, CL_FALSE, 0, sizeof(int), &Nz, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_dz, CL_FALSE, 0, sizeof(float), &dz, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_d, CL_FALSE, 0, sizeof(float), &d, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_bx, CL_FALSE, 0, sizeof(float), &bx, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_by, CL_FALSE, 0, sizeof(float), &by, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_bz, CL_FALSE, 0, sizeof(float), &bz, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_bxf, CL_FALSE, 0, sizeof(float), &bxf, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_byf, CL_FALSE, 0, sizeof(float), &byf, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_bzf, CL_FALSE, 0, sizeof(float), &bzf, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_bxb, CL_FALSE, 0, sizeof(float), &bxb, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_byb, CL_FALSE, 0, sizeof(float), &byb, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_bzb, CL_FALSE, 0, sizeof(float), &bzb, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_maxxx, CL_FALSE, 0, sizeof(float), &maxxx, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_maxyy, CL_FALSE, 0, sizeof(float), &maxyy, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_minxx, CL_FALSE, 0, sizeof(float), &minxx, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_minyy, CL_FALSE, 0, sizeof(float), &minyy, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_zmax, CL_FALSE, 0, sizeof(float), &zmax, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_NSlices, CL_FALSE, 0, sizeof(float), &NSlices, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_xxv, CL_FALSE, 0, sizeof(float), &xxv, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_yyv, CL_FALSE, 0, sizeof(float), &yyv, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_xxa, CL_FALSE, 0, sizeof(float), &xxa, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_yya, CL_FALSE, 0, sizeof(float), &yya, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_x, CL_FALSE, 0, sizeof(float) * size_x * 2, x, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_y, CL_FALSE, 0, sizeof(float) * size_x * 2, y, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_z, CL_FALSE, 0, sizeof(float) * TotSinos * 2, z_det, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_atten, CL_FALSE, 0, sizeof(float) * size_atten, atten, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_det_per_ring, CL_FALSE, 0, sizeof(int), &det_per_ring, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(af_queue, d_pseudos, CL_FALSE, 0, sizeof(int) * prows, pseudos, 0, NULL, NULL);


	int im_dim = Nx * Ny * Nz;

	for (int iter = 0; iter < Niter; iter++) {

		pz_osem_apu = pz_osem(span, iter);

		for (int osa_iter = 0; osa_iter < subsets; osa_iter++) {

			size_t length = pituus[osa_iter + 1] - pituus[osa_iter];

			size_t outputSize1 = sizeof(int) * summa[osa_iter];
			size_t outputSize2 = sizeof(float) * summa[osa_iter];


			array sub_index_array = range(dim4(length), 0, u32) + pituus[osa_iter];

			array apu_lor = lor(sub_index_array);

			array row_csr = accum(apu_lor.as(s32));

			row_csr = join(0, constant(0, 1, s32), row_csr);

			array Lo = LL(range(dim4(length * 2), 0, u32) + 2 * pituus[osa_iter]);

			cl_mem d_indices = clCreateBuffer(af_context, CL_MEM_READ_WRITE, outputSize1, NULL, &status);

			if (status != CL_SUCCESS) {
				std::cerr << getErrorString(status) << std::endl;
				mexPrintf("Failed to create buffer for system matrix indices. Check that you have sufficient memory available\n");
			}

			cl_mem d_elements = clCreateBuffer(af_context, CL_MEM_READ_WRITE, outputSize2, NULL, &status);

			if (status != CL_SUCCESS) {
				std::cerr << getErrorString(status) << std::endl;
				mexPrintf("Failed to create buffer for system matrix elements. Check that you have sufficient memory available\n");
			}

			cl_mem * d_lor = apu_lor.device<cl_mem>();
			cl_mem * d_row = row_csr.device<cl_mem>();

			cl_mem * d_L = Lo.device<cl_mem>();

			clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_indices);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_elements);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), d_lor);
			clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_Nx);
			clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_Ny);
			clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_Nz);
			clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dz);
			clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_d);
			clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_bz);
			clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_bx);
			clSetKernelArg(kernel, 10, sizeof(cl_mem), &d_by);
			clSetKernelArg(kernel, 11, sizeof(cl_mem), &d_bxf);
			clSetKernelArg(kernel, 12, sizeof(cl_mem), &d_byf);
			clSetKernelArg(kernel, 13, sizeof(cl_mem), &d_bzf);
			clSetKernelArg(kernel, 14, sizeof(cl_mem), &d_bxb);
			clSetKernelArg(kernel, 15, sizeof(cl_mem), &d_byb);
			clSetKernelArg(kernel, 16, sizeof(cl_mem), &d_bzb);
			clSetKernelArg(kernel, 17, sizeof(cl_mem), &d_maxxx);
			clSetKernelArg(kernel, 18, sizeof(cl_mem), &d_maxyy);
			clSetKernelArg(kernel, 19, sizeof(cl_mem), &d_minxx);
			clSetKernelArg(kernel, 20, sizeof(cl_mem), &d_minyy);
			clSetKernelArg(kernel, 21, sizeof(cl_mem), &d_zmax);
			clSetKernelArg(kernel, 22, sizeof(cl_mem), &d_NSlices);
			clSetKernelArg(kernel, 23, sizeof(cl_mem), &d_x);
			clSetKernelArg(kernel, 24, sizeof(cl_mem), &d_y);
			clSetKernelArg(kernel, 25, sizeof(cl_mem), &d_z);
			clSetKernelArg(kernel, 26, sizeof(int), &size_x);
			clSetKernelArg(kernel, 27, sizeof(cl_mem), d_row);
			clSetKernelArg(kernel, 28, sizeof(cl_mem), &d_xxv);
			clSetKernelArg(kernel, 29, sizeof(cl_mem), &d_yyv);
			clSetKernelArg(kernel, 30, sizeof(cl_mem), &d_xxa);
			clSetKernelArg(kernel, 31, sizeof(cl_mem), &d_yya);
			clSetKernelArg(kernel, 32, sizeof(cl_mem), d_L);
			clSetKernelArg(kernel, 33, sizeof(cl_mem), &d_det_per_ring);
			clSetKernelArg(kernel, 34, sizeof(int), &TotSinos);
			clSetKernelArg(kernel, 35, sizeof(int), &attenuation_correction);
			clSetKernelArg(kernel, 36, sizeof(cl_mem), &d_atten);
			clSetKernelArg(kernel, 37, sizeof(cl_mem), &d_pseudos);
			clSetKernelArg(kernel, 38, sizeof(int), &prows);
			status = clEnqueueNDRangeKernel(af_queue, kernel, 1, NULL, &length, NULL, 0, NULL, NULL);


			if (status != CL_SUCCESS) {
				std::cerr << getErrorString(status) << std::endl;
				mexPrintf("Failed to execute the OpenCL kernel\n");
			}
			else if (verbose) {
				mexPrintf("OpenCL kernel executed successfully\n");
				mexEvalString("pause(.0001);");
			}

			Lo.unlock();
			apu_lor.unlock();
			row_csr.unlock();



			array s_elements = afcl::array(summa[osa_iter], d_elements, f32, true);
			array s_indices = afcl::array(summa[osa_iter], d_indices, s32, true);


			array H = sparse(row_csr.dims(0) - 1, im_dim, s_elements, row_csr, s_indices);

			array Summ = matmul(H, constant(1, H.dims(0), 1, f32), AF_MAT_TRANS);

			pz_osem_apu = (pz_osem_apu / (Summ + epps)) * (matmul(H, (Sino(sub_index_array) / (matmul(H, pz_osem_apu) + epps)), AF_MAT_TRANS));


			mexPrintf("OSEM sub-iteration %d complete\n", osa_iter + 1);
			mexEvalString("pause(.0001);");
			
			clFinish(af_queue);

			status = clReleaseMemObject(d_indices);
			if (status != CL_SUCCESS) {
				std::cerr << getErrorString(status) << std::endl;
				mexPrintf("Failed to release the memory of indices\n");
			}
			status = clReleaseMemObject(d_elements);
			if (status != CL_SUCCESS) {
				std::cerr << getErrorString(status) << std::endl;
				mexPrintf("Failed to release the memory of elements\n");
			}

		}

		pz_osem(span, iter + 1) = pz_osem_apu;
		mexPrintf("Iteration %d complete\n", iter + 1);
		mexEvalString("pause(.0001);");
	}

	pz_osem.host(ele);

	clReleaseMemObject(d_Nx);
	clReleaseMemObject(d_Ny);
	clReleaseMemObject(d_Nz);
	clReleaseMemObject(d_dz);
	clReleaseMemObject(d_d);
	clReleaseMemObject(d_bz);
	clReleaseMemObject(d_bx);
	clReleaseMemObject(d_by);
	clReleaseMemObject(d_bxb);
	clReleaseMemObject(d_byb);
	clReleaseMemObject(d_bzb);
	clReleaseMemObject(d_bxf);
	clReleaseMemObject(d_byf);
	clReleaseMemObject(d_bzf);
	clReleaseMemObject(d_maxxx);
	clReleaseMemObject(d_minxx);
	clReleaseMemObject(d_maxyy);
	clReleaseMemObject(d_minyy);
	clReleaseMemObject(d_zmax);
	clReleaseMemObject(d_NSlices);
	clReleaseMemObject(d_xxv);
	clReleaseMemObject(d_yyv);
	clReleaseMemObject(d_xxa);
	clReleaseMemObject(d_yya);
	clReleaseMemObject(d_z);
	clReleaseMemObject(d_x);
	clReleaseMemObject(d_y);
	clReleaseMemObject(d_atten);
	clReleaseMemObject(d_det_per_ring);
	clReleaseMemObject(d_pseudos);

	status = clReleaseProgram(program);
	if (status != CL_SUCCESS) {
		std::cerr << getErrorString(status) << std::endl;
		mexPrintf("Failed to release program\n");
	}
	status = clReleaseKernel(kernel);
	if (status != CL_SUCCESS) {
		std::cerr << getErrorString(status) << std::endl;
		mexPrintf("Failed to release kernel\n");
	}
}


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray*prhs[])

{
	if (nrhs != 40)
		mexErrMsgTxt("Invalid number of input arguments.  There must be exactly 40.");

	if (nlhs != 1)
		mexErrMsgTxt("Invalid number of output arguments.  There must be exactly one.");

	int Ny = (int)mxGetScalar(prhs[1]);

	int Nx = (int)mxGetScalar(prhs[2]);

	int Nz = (int)mxGetScalar(prhs[3]);

	float d = (float)mxGetScalar(prhs[4]);

	float dz = (float)mxGetScalar(prhs[5]);

	float by = (float)mxGetScalar(prhs[6]);

	float bx = (float)mxGetScalar(prhs[7]);

	float bz = (float)mxGetScalar(prhs[8]);

	float *z_det = (float*)mxGetData(prhs[9]);

	float *x = (float*)mxGetData(prhs[10]);

	float *y = (float*)mxGetData(prhs[11]);

	float *iij = (float*)mxGetData(prhs[12]);

	float *jji = (float*)mxGetData(prhs[13]);

	float *kkj = (float*)mxGetData(prhs[14]);

	float *yy = (float*)mxGetData(prhs[15]);

	float *xx = (float*)mxGetData(prhs[16]);

	int NSinos = (int)mxGetScalar(prhs[17]);

	float NSlices = (float)mxGetScalar(prhs[18]);

	float yyv = yy[1] - yy[0];

	float xxv = xx[1] - xx[0];

	float yya = yy[0];

	float xxa = xx[0];

	int size_x = (int)mxGetScalar(prhs[19]);

	float zmax = (float)mxGetScalar(prhs[20]);

	int TotSinos = (int)mxGetScalar(prhs[21]);

	int32_t ind_size = (int32_t)mxGetScalar(prhs[22]);

	float *atten = (float*)mxGetData(prhs[23]);
	int size_atten = mxGetNumberOfElements(prhs[23]);

	int *index = (int*)mxGetData(prhs[24]);

	uint32_t *pituus = (uint32_t*)mxGetData(prhs[25]);

	int attenuation_correction = (int)mxGetScalar(prhs[26]);

	int Niter = (int)mxGetScalar(prhs[27]);

	int subsets = (int)mxGetScalar(prhs[28]);

	bool* rekot = (bool*)mxGetData(prhs[29]);

	float epps = (float)mxGetScalar(prhs[30]);

	float* Sin = (float*)mxGetData(prhs[31]);

	float* x0 = (float*)mxGetData(prhs[32]);

	uint16_t* lor1 = (uint16_t*)mxGetData(prhs[33]);

	uint64_t* summa = (uint64_t*)mxGetData(prhs[34]);

	uint16_t *L = (uint16_t*)mxGetData(prhs[35]);
	int numRows = (int)mxGetM(prhs[35]);

	int *pseudos = (int*)mxGetData(prhs[36]);
	int pRows = (int)mxGetM(prhs[36]);

	int det_per_ring = (int)mxGetScalar(prhs[37]);

	bool verbose = (bool)mxGetScalar(prhs[38]);

	int device = (int)mxGetScalar(prhs[39]);
	
	af::setDevice(device);

	char *k_path = mxArrayToString(prhs[0]);

	float maxyy = *std::max_element(yy, yy + Ny + 1);
	float minyy = *std::min_element(yy, yy + Ny + 1);

	float maxxx = *std::max_element(xx, xx + Nx + 1);
	float minxx = *std::min_element(xx, xx + Nx + 1);

	float bxf = bx + iij[0] * d;
	float byf = by + jji[0] * d;
	float bzf = bz + kkj[0] * dz;
	float bxb = bx + iij[mxGetNumberOfElements(prhs[12]) - 1] * d;
	float byb = by + jji[mxGetNumberOfElements(prhs[13]) - 1] * d;
	float bzb = bz + kkj[mxGetNumberOfElements(prhs[14]) - 1] * dz;

	size_t koko = mxGetNumberOfElements(prhs[35]);
	size_t koko_l = mxGetNumberOfElements(prhs[33]);
	
	size_t outSize = Nx * Ny * Nz;
	size_t outSize2 = Niter + 1;

	plhs[0] = mxCreateNumericMatrix(outSize, outSize2, mxSINGLE_CLASS, mxREAL);

	float* elements = (float*)mxGetData(plhs[0]);

	reconstruction(koko, lor1, z_det, x, y, Sin, Nx, Ny, Nz, Niter, x0, d, dz, bxf, byf, bzf, bxb, byb, bzb, bx, by, bz, maxxx, maxyy, minxx, minyy, zmax, 
		xxv, yyv, xxa, yya, NSlices, pituus, summa, koko_l, L, size_x, TotSinos, elements, verbose, attenuation_correction, atten, size_atten, subsets, 
		epps, pseudos, det_per_ring, pRows, k_path);
	

	sync();
	deviceGC();
	
	return;
}
