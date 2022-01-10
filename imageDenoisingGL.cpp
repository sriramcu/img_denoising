/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample demonstrates two adaptive image denoising techniques:
 * KNN and NLM, based on computation of both geometric and color distance
 * between texels. While both techniques are already implemented in the
 * DirectX SDK using shaders, massively speeded up variation
 * of the latter technique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imageDenoising.h"

// includes, project
#include <helper_functions.h>  // includes for helper utility functions
#include <helper_cuda.h>  // includes for cuda error checking and initialization

const char *sSDKsample = "CUDA ImageDenoising";

const char *filterMode[] = {"Passthrough", "KNN method", "NLM method",
                            "Quick NLM(NLM2) method", NULL};

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] = {"image_passthru.ppm", "image_knn.ppm",
                           "image_nlm.ppm", "image_nlm2.ppm", NULL};

const char *sReference[] = {"ref_passthru.ppm", "ref_knn.ppm", "ref_nlm.ppm",
                            "ref_nlm2.ppm", NULL};

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
// OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
// Source image on the host side
uchar4 *h_Src;
int imageW, imageH;
GLuint shader;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int g_Kernel = 0;
bool g_FPS = false;
bool g_Diag = false;
StopWatchInterface *timer = NULL;

// Algorithms global parameters
const float noiseStep = 0.025f;
const float lerpStep = 0.025f;
static float knnNoise = 0.32f;
static float nlmNoise = 1.45f;
static float lerpC = 0.2f;

const int frameN = 24;
int frameCounter = 0;

#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 1;  // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX_EPSILON_ERROR 5
#define REFRESH_DELAY 10  // ms

void cleanup();

void runImageFilters(TColor *d_dst) {
  switch (g_Kernel) {
    case 0:
      cuda_Copy(d_dst, imageW, imageH, texImage);
      break;

    case 1:
      if (!g_Diag) {
        cuda_KNN(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC,
                 texImage);
      } else {
        cuda_KNNdiag(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC,
                     texImage);
      }

      break;

    case 2:
      if (!g_Diag) {
        cuda_NLM(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC,
                 texImage);
      } else {
        cuda_NLMdiag(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC,
                     texImage);
      }

      break;

    case 3:
      if (!g_Diag) {
        cuda_NLM2(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC,
                  texImage);
      } else {
        cuda_NLM2diag(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise),
                      lerpC, texImage);
      }

      break;
  }

  getLastCudaError("Filtering kernel execution failed.\n");
}

void displayFunc(void) {
  TColor *d_dst = NULL;
  size_t num_bytes;

  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  getLastCudaError("cudaGraphicsMapResources failed");
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&d_dst, &num_bytes, cuda_pbo_resource));
  getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");
  g_Kernel=2;
  runImageFilters(d_dst);
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

void cleanup() {
  free(h_Src);
  checkCudaErrors(CUDA_FreeArray());
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
  glDeleteProgramsARB(1, &shader);
}



int main(int argc, char **argv) {
  if(argc!=3)
  {
    printf("Format: ./imageDenoising filename")
    exit(-1);
  }
  char *dump_file = NULL;

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", sSDKsample);

  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    getCmdLineArgumentString(argc, (const char **)argv, "file",
                             (char **)&dump_file);

    int kernel = 1;

    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
      kernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    }

    runAutoTest(argc, argv, dump_file, kernel);
  } else {
    printf("[%s]\n", sSDKsample);

    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
      printf("[%s]\n", argv[0]);
      printf("   Does not explicitly support -device=n in OpenGL mode\n");
      printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
      printf(" > %s -device=n -qatest\n", argv[0]);
      printf("exiting...\n");
      exit(EXIT_SUCCESS);
    }

    // First load the image, so we know what the size of the image (imageW and
    // imageH)
    printf("Allocating host and CUDA memory and loading image file...\n");
    const char *image_path = sdkFindFilePath(argv[1], argv[0]); 
    //pass input image file as argument


    if (image_path == NULL) {
      printf(
          "imageDenoisingGL was unable to find and load image file "
          "<portrait_noise.bmp>.\nExiting...\n");
      exit(EXIT_FAILURE);
    }

    LoadBMPFile(&h_Src, &imageW, &imageH, image_path);  
    // calculate image width and height, initialise img pointer
    
    printf("Data init done.\n");


    findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

  }

}
