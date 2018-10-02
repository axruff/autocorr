/**
* @file    2D Autocorrelation flow using NVIDIA CUDA
* @author  Institute for Photon Science and Synchrotron Radiation, Karlsruhe Institute of Technology
*          
* @date    2018
* @version 0.5.0
*
*
* @section LICENSE
*
* This program is copyrighted by the author and Institute for Photon Science and Synchrotron Radiation,
* Karlsruhe Institute of Technology, Karlsruhe, Germany;
*
* The current implemetation contains the following licenses:
*
* 1. TinyXml package:
*      Original code (2.0 and earlier )copyright (c) 2000-2006 Lee Thomason (www.grinninglizard.com). <www.sourceforge.net/projects/tinyxml>.
*      See src/utils/tinyxml.h for details.
*
*/

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

#include <cuda.h>

#include "src/data_types/data2d.h"
#include "src/data_types/data_structs.h"
#include "src/data_types/operation_parameters.h"
#include "src/utils/cuda_utils.h"
#include "src/utils/io_utils.h"
#include "src/utils/settings.h"

#include "src/correlation/correlation_flow_2d.h"

const bool key_press = true;
const bool use_visualization = false;
const bool silent_mode = true;


int main(int argc, char** argv)
{


    /* Initialize CUDA */
    CUcontext cu_context;
    if (!InitCudaContextWithFirstAvailableDevice(&cu_context)) {
        return 1;
    }

    std::printf("//----------------------------------------------------------------------//\n");
    std::printf("//        2D Correlation flow (Test) using NVIDIA CUDA. Version 0.5	   \n");
    std::printf("//                                                                        \n");
    std::printf("//           Author: Alexey Ershov. <ershov.alexey@gmail.com>             \n");
    std::printf("//            Karlsruhe Institute of Technology. 2018                     \n");
    std::printf("//----------------------------------------------------------------------//\n");

    

    size_t width = 128;
    size_t height = 100;

    /* Correlation flow variables */
    size_t  correlation_window_size = 18;


    string  file_name1              = "real_frame-128-100.raw";
    string  input_path              =  "c:\\Users\\fe0968\\Documents\\gpuflow3d\\gpuflow2d\\data\\";
    string  output_path             =  "c:\\Users\\fe0968\\Documents\\gpuflow3d\\gpuflow2d\\data\\output\\";
    string  counter                 =  "";

    /*------------------------------------------------------*/
    /*               Correlation algorithm                  */
    /*------------------------------------------------------*/

    if (argc == 5 || argc == 6)  {
        file_name1 = argv[1];
        width = atoi(argv[2]);
        height = atoi(argv[3]);
        output_path = string(argv[5]);

    if (argc == 6)
        counter = argv[4];
    }
    else if (argc != 1) {
        cout<<"Usage: "<< argv[0] <<" <settings file>. Otherwise settings.xml in the current directory is used"<<endl;
    return 0;

    }

    /* Correlation flow computation class */
    CorrelationFlow2D correlation_flow;

    Data2D image;
    DataSize3 image_size ={ width, height, 1 };

    /* Load input data */
    if (!image.ReadRAWFromFileF32((input_path + file_name1).c_str(), image_size.width, image_size.height)) {
        //if (!image.ReadRAWFromFileU8("./data/squares_many.raw", image_size.width, image_size.height)) {
        //if (!image.ReadRAWFromFileF32("./data/73_flat_corr.raw", image_size.width, image_size.height)) {
        return 2;
    }


    if (correlation_flow.Initialize(image_size, correlation_window_size)) {



        Data2D flow_x(image_size.width, image_size.height);
        Data2D flow_y(image_size.width, image_size.height);
        Data2D corr(image_size.width, image_size.height);

        Data2D corr_temp(image_size.width*correlation_window_size, image_size.height*correlation_window_size);

        correlation_flow.silent = silent_mode;

        OperationParameters params;
        // params.PushValuePtr("correlation_window_size", &correlation_window_size);
        //params.PushValuePtr("warp_scale_factor", &warp_scale_factor);


        correlation_flow.ComputeFlow(image, flow_x, flow_y, corr, corr_temp, params);

        std::string filename =
            "-" + std::to_string(width) +
            "-" + std::to_string(height) + ".raw";

        std::string filename_ext =
            "-" + std::to_string(width*correlation_window_size) +
            "-" + std::to_string(height*correlation_window_size) + ".raw";

        flow_x.WriteRAWToFileF32(std::string(output_path + counter + "corr-flow-x" + filename).c_str());
        flow_y.WriteRAWToFileF32(std::string(output_path + counter + "corr-flow-y" + filename).c_str());
        corr.WriteRAWToFileF32(std::string(output_path + counter + "corr-coeff" + filename).c_str());

        corr_temp.WriteRAWToFileF32(std::string(output_path + "corr-temp" + filename_ext).c_str());

        IOUtils::WriteFlowToImageRGB(flow_x, flow_y, 3, output_path + counter + "corr-res.pgm");

        IOUtils::WriteMagnitudeToFileF32(flow_x, flow_y, std::string(output_path + counter + "corr-amp" + filename).c_str());



        correlation_flow.Destroy();
    }

    if (key_press) {
        std::printf("Press enter to continue...");
        std::getchar();
    }


    /* Release resources */
    cuCtxDestroy(cu_context);

    return 0;
}
