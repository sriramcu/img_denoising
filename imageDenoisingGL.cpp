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


GLuint gl_PBO, gl_Tex;

struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
// Source image on the host side
uchar4 *h_Src;
int imageW, imageH;
GLuint shader;


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

#define BUFFER_DATA(i) ((char *)0 + i)



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



void cleanup() 
{
  free(h_Src);
  checkCudaErrors(CUDA_FreeArray());
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

  glDeleteProgramsARB(1, &shader);

  sdkDeleteTimer(&timer);
}



void runDenoising(int argc, char **argv, const char *filename,
                 int kernel_param) {
  printf("[%s] - (automated testing w/ readback)\n", sSDKsample);

  int devID = findCudaDevice(argc, (const char **)argv);

  // First load the image, so we know what the size of the image (imageW and
  // imageH)
  printf("Allocating host and CUDA memory and loading image file...\n");
  //const char *image_path = sdkFindFilePath("portrait_noise.bmp", argv[0]);
  const char *image_path = sdkFindFilePath(argv[2], argv[0]);


  if (image_path == NULL) {
    printf(
        "imageDenoisingGL was unable to find and load image file "
        "<portrait_noise.bmp>.\nExiting...\n");
    exit(EXIT_FAILURE);
  }

  LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
  printf("Data init done.\n");

  checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

  TColor *d_dst = NULL;
  unsigned char *h_dst = NULL;
  checkCudaErrors(
      cudaMalloc((void **)&d_dst, imageW * imageH * sizeof(TColor)));
  h_dst = (unsigned char *)malloc(imageH * imageW * 4);

  {
    g_Kernel = kernel_param;
    printf("[AutoTest]: %s <%s>\n", sSDKsample, filterMode[g_Kernel]);

    runImageFilters(d_dst);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW * imageH * sizeof(TColor),
                               cudaMemcpyDeviceToHost));
    

    char final_filename[100] = "custom_output/";
    strcat(final_filename,filename);
    printf("Saving to %s\n",final_filename);
    sdkSavePPM4ub(final_filename, h_dst, imageW, imageH);
  }

  checkCudaErrors(CUDA_FreeArray());
  free(h_Src);

  checkCudaErrors(cudaFree(d_dst));
  free(h_dst);

  printf("\n[%s] -> Kernel %d, Saved: %s\n", sSDKsample, kernel_param,
         filename);

  exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}



int main(int argc, char **argv) {
  char *ptr;
  g_Kernel = strtol(argv[1], &ptr, 10);
  char dump_file[100] = "";
  strcpy(dump_file, argv[2]);
  int i;
  for(i=0;i<100;i++)
  {
    if(dump_file[i]=='.')
      dump_file[i] = '_';
  }
  strcat(dump_file, "_");
  strcat(dump_file, sReference[g_Kernel]);


  printf("%s\n",dump_file);
#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", sSDKsample);
  runDenoising(argc, argv, dump_file, 1);  

}
