/* neuromatrix.c - NeuroMatrix pocl device driver layer implementation

   TODO: add copyright

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "neuromatrix.h"
// TODO
#include "common.h"
#include "config.h"
#include "config2.h"
#include "cpuinfo.h"
#include "devices.h"
#include "pocl_local_size.h"
#include "pocl_util.h"
#include "topology/pocl_topology.h"
#include "utlist.h"

#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>

#include "pocl_cache.h"
#include "pocl_timing.h"
#include "pocl_file_util.h"
#include "pocl_workgroup_func.h"

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

#include <mc12101load.h>

typedef struct pocl_neuromatrix_device_data_s
{
  PL_Board *board;
  PL_Access *access;
} pocl_neuromatrix_device_data_t;

void
pocl_neuromatrix_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "NeuroMatrix";
  ops->probe = pocl_neuromatrix_probe;
  ops->init = pocl_neuromatrix_init;
  ops->build_hash = pocl_neuromatrix_build_hash;

  ops->uninit = pocl_neuromatrix_uninit;
  ops->reinit = NULL;
  ops->alloc_mem_obj = pocl_neuromatrix_alloc_mem_obj;

  // TODO
  //ops->free = pocl_basic_free;
  //ops->read = pocl_basic_read;
  //ops->read_rect = pocl_basic_read_rect;
  //ops->write = pocl_basic_write;
  //ops->write_rect = pocl_basic_write_rect;
  //ops->copy = pocl_basic_copy;
  //ops->copy_rect = pocl_basic_copy_rect;
  //ops->memfill = pocl_basic_memfill;
  //ops->map_mem = pocl_basic_map_mem;
  //ops->compile_kernel = pocl_basic_compile_kernel;
  //ops->unmap_mem = pocl_basic_unmap_mem;
  //ops->run = pocl_basic_run;
  //ops->run_native = pocl_basic_run_native;
  //ops->join = pocl_basic_join;
  //ops->submit = pocl_basic_submit;
  //ops->broadcast = pocl_broadcast;
  //ops->notify = pocl_basic_notify;
  //ops->flush = pocl_basic_flush;
  //ops->compute_local_size = pocl_default_local_size_optimizer;

  //ops->get_device_info_ext = NULL;

  //ops->svm_free = pocl_basic_svm_free;
  //ops->svm_alloc = pocl_basic_svm_alloc;
  ///* no need to implement these two as they're noop
  // * and pocl_exec_command takes care of it */
  //ops->svm_map = NULL;
  //ops->svm_unmap = NULL;
  //ops->svm_copy = pocl_basic_svm_copy;
  //ops->svm_fill = pocl_basic_svm_fill;

  //ops->create_image = NULL;
  //ops->free_image = NULL;
  //ops->create_sampler = NULL;
  //ops->free_sampler = NULL;
  //ops->copy_image_rect = pocl_basic_copy_image_rect;
  //ops->write_image_rect = pocl_basic_write_image_rect;
  //ops->read_image_rect = pocl_basic_read_image_rect;
  //ops->map_image = pocl_basic_map_image;
  //ops->unmap_image = pocl_basic_unmap_image;
  //ops->fill_image = pocl_basic_fill_image;
}

char *
pocl_neuromatrix_build_hash (cl_device_id device)
{
  char* res = calloc(100, sizeof(char));
  snprintf(res, 100, "NMC-%s", device->llvm_target_triplet); // TODO
  return res;
}

unsigned int
pocl_neuromatrix_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count (ops->device_name);
  unsigned boards_count = 0;

  if (PL_GetBoardCount (&boards_count) != PL_OK)
    boards_count = 0;

  if (env_count >= 0)
  {
    if (env_count > boards_count)
      POCL_ABORT ("[NeuroMatrix] %d devices requested, but only %d are available\n",
                  env_count, boards_count);
    boards_count = env_count;
  }

  return boards_count;
}

cl_int
pocl_neuromatrix_uninit (unsigned j, cl_device_id device)
{
  pocl_neuromatrix_device_data_t *data = device->data;

  if (device->available) {
    PL_CloseAccess(data->access);
    PL_CloseBoardDesc(data->board);
  }

  POCL_MEM_FREE (data);
  device->data = NULL;

  POCL_MEM_FREE (device->long_name);
  return CL_SUCCESS;
}

cl_int
pocl_cuda_alloc_mem_obj (cl_device_id device, cl_mem mem_obj, void *host_ptr)
{
  /* If memory for this global memory is not yet allocated -> do it */
  if (mem_obj->device_ptrs[device->global_mem_id].mem_ptr == NULL)
  {
    cl_mem_flags flags = mem_obj->flags;

    assert((flags & CL_MEM_READ_WRITE ||) && "CL_MEM_READ_WRITE and CL_MEM_WRITE_ONLY only are supported");

    if (flags & CL_MEM_USE_HOST_PTR)
    {
#if defined __arm__
      /* cuMemHostRegister is not supported on ARM.
           * Allocate device memory and perform explicit copies
           * before and after running a kernel */
          result = cuMemAlloc ((CUdeviceptr *)&b, mem_obj->size);
          CUDA_CHECK (result, "cuMemAlloc");
#else
      result = cuMemHostRegister (host_ptr, mem_obj->size,
                                  CU_MEMHOSTREGISTER_DEVICEMAP);
      if (result != CUDA_SUCCESS
          && result != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
        CUDA_CHECK (result, "cuMemHostRegister");
      result = cuMemHostGetDevicePointer ((CUdeviceptr *)&b, host_ptr, 0);
      CUDA_CHECK (result, "cuMemHostGetDevicePointer");
#endif
    }
    else if (flags & CL_MEM_ALLOC_HOST_PTR)
    {
      result = cuMemHostAlloc (&mem_obj->mem_host_ptr, mem_obj->size,
                               CU_MEMHOSTREGISTER_DEVICEMAP);
      CUDA_CHECK (result, "cuMemHostAlloc");
      result = cuMemHostGetDevicePointer ((CUdeviceptr *)&b,
                                          mem_obj->mem_host_ptr, 0);
      CUDA_CHECK (result, "cuMemHostGetDevicePointer");
    }
    else
    {
      result = cuMemAlloc ((CUdeviceptr *)&b, mem_obj->size);
      if (result != CUDA_SUCCESS)
      {
        const char *err;
        cuGetErrorName (result, &err);
        POCL_MSG_PRINT2 (CUDA, __FUNCTION__, __LINE__,
                         "-> Failed to allocate memory: %s\n", err);
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
    }

    if (flags & CL_MEM_COPY_HOST_PTR)
    {
      result = cuMemcpyHtoD ((CUdeviceptr)b, host_ptr, mem_obj->size);
      CUDA_CHECK (result, "cuMemcpyHtoD");

      result = cuStreamSynchronize (0);
      CUDA_CHECK (result, "cuStreamSynchronize");
    }

    mem_obj->device_ptrs[device->global_mem_id].mem_ptr = b;
    mem_obj->device_ptrs[device->global_mem_id].global_mem_id
        = device->global_mem_id;
  }

  /* Copy allocated global mem info to devices own slot */
  mem_obj->device_ptrs[device->dev_id]
      = mem_obj->device_ptrs[device->global_mem_id];

  return CL_SUCCESS;
}

cl_int
pocl_neuromatrix_init (unsigned j, cl_device_id dev, const char* parameters)
{
  cl_int ret = CL_SUCCESS;

  if (dev->data)
    return ret;

  pocl_init_default_device_infos (dev);

  dev->svm_caps = 0; // no support of SVM
  dev->vendor = "NTC Module";
  dev->vendor_id = 0x2c6a;
  dev->type = CL_DEVICE_TYPE_ACCELERATOR;
  dev->llvm_target_triplet = "neuromatrix";
  dev->address_bits = (sizeof (PL_Addr)* 8);
  dev->image_support = CL_FALSE;

  dev->max_compute_units = 1; // TODO

  /* Get specific device name TODO */
  dev->long_name = dev->short_name = calloc (256, sizeof (char));
  snprintf (dev->long_name, 255, "TODO"); // TODO

  SETUP_DEVICE_CL_VERSION (NEUROMATRIX_DEVICE_CL_VERSION_MAJOR,
                           NEUROMATRIX_DEVICE_CL_VERSION_MINOR);

  dev->llvm_cpu = NULL;
  dev->extensions = NEUROMATRIX_DEVICE_EXTENSIONS;

  dev->max_mem_alloc_size = 100 * 1024 * 1024; // TODO

  pocl_neuromatrix_device_data_t *data = calloc (1, sizeof (pocl_neuromatrix_device_data_t));

  if (PL_GetBoardDesc(0, &data->board) != PL_OK)
    POCL_ABORT ("[NeuroMatrix] Failed to open board\n");

  if (PL_GetAccess(data->board, 0, &data->access) != PL_OK)
    POCL_ABORT ("[NeuroMatrix] Failed to open board\n");

  return ret;
}
