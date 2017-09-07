#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "blas.h"

#ifndef MAX_FILENM_LENGTH
#define MAX_FILENM_LENGTH 1024
#endif

static float   nms_=.4;
static network net_;
static char**  names_;
static image** alphabet_;
static float   thresh_;
static float   hier_thresh_;

image **load_labels(char* lable_dir)
{
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[MAX_FILENM_LENGTH];
            sprintf(buff, "%s/%d_%d.png", lable_dir, i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}

void init_net(  const char *cfgfile
              , const char *weightfile
              , const char* namelist
              , const char* labeldir
              , float thresh
              , float hier_thresh){
    /// name list    
    names_ = get_labels(namelist);

    /// label
    alphabet_ = load_labels(labeldir);
    thresh_ = thresh;
    hier_thresh_ = hier_thresh;

    /// net
    net_ = parse_network_cfg(cfgfile);
    if(weightfile)
        load_weights(&net_, weightfile);
    set_batch_network(&net_, 1);

    srand(2222222);

}
void delete_net(){
    ///free_net(net_);

    const int nsize = 8;
    int i, j;
    for(i = 0; i < nsize; i++){
        for(j = 0; j < 127; j++){
            free_image(alphabet_[i][j]);
        }
        free(alphabet_[i]);
    }
    free(alphabet_);
}
void detect(image im, const char* outfile){
    int j;
    image sized = letterbox_image(im, net_.w, net_.h);

    layer l = net_.layers[net_.n - 1];

    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

    float *X = sized.data;
    float time=clock();
    network_predict(net_, X);
    printf("Predicted in %f seconds.\n", sec(clock()-time));

    get_region_boxes(l, im.w, im.h, net_.w, net_.h, thresh_, probs, boxes, 0, 0, hier_thresh_, 1);
    if (nms_) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms_);

    draw_detections(im, l.w*l.h*l.n, thresh_, boxes, probs, names_, alphabet_, l.classes);
    save_image(im, outfile);

    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);
}
