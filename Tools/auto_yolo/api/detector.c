#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "../src/parser.h"
#include "../src/utils.h"
#include "../src/cuda.h"
#include "../src/blas.h"
#include "../src/connected_layer.h"

extern image load_image_color(char *filename, int w, int h);
extern void detect(image im, const char* outfile);
extern void delete_net();
extern void init_net(  const char *cfgfile
                     , const char *weightfile
                     , const char* namelist
                     , const char* labeldir
                     , float thresh
                     , float hier_thresh);

int main(int argc, char **argv)
{
    if(argc != 8){
        fprintf(stderr, "usage: %s gpuid cfgfile weightfile namelist labeldir imagefile outputfile!\n", argv[0]);
        return 0;
    }

    int gpu_index = atoi(argv[1]);
    char* cfgfile = argv[2];
    char* weightfile = argv[3];
    char* namelist = argv[4];
    char* labeldir = argv[5];
    char* imagefile = argv[6];
    char* outputfile = argv[7];
    float thresh = 0.24;
    float hier_thresh = 0.5;

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif

    /// step 1: init net
    init_net(  cfgfile
             , weightfile
             , namelist
             , labeldir
             , thresh
             , hier_thresh);

    /// step 2: predict
    image im = load_image_color(imagefile ,0 ,0);

    detect(im, outputfile);

    /// step 3: free all
    delete_net();
    free_image(im);

    return 0;
}

