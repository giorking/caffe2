/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Based on https://github.com/daijifeng001/caffe-rfcn/blob/r-fcn/src/caffe/layers/psroi_pooling_layer.cu
//
// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li, Tairui Chen
// ------------------------------------------------------------------
//
// COPYRIGHT
//
// All contributions by the University of California:
// Copyright (c) 2014, 2015, The Regents of the University of California
// (Regents)
// All rights reserved.
//
// All other contributions:
// Copyright (c) 2014, 2015, the respective contributors
// All rights reserved.
//
// Caffe uses a shared copyright model: each contributor holds copyright over
// their contributions to Caffe. The project versioning records all such
// contribution and copyright details. If a contributor wants to further mark
// their specific copyright on a particular contribution, they should indicate
// their copyright solely in the commit message of the change when it is
// committed.
//
// LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// CONTRIBUTION AGREEMENT
//
// By contributing to the BVLC/caffe repository through pull-request, comment,
// or otherwise, the contributor releases their content to the
// license and copyright terms herein.

#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "ps_roi_align_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__
float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
    const int height, const int width,
    T y, T x,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void PSRoIAlignForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const T* bottom_rois,
    const int output_dim,
    const int group_size,
    T* top_data,
    int* mapping_channel) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
 
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;
 
    // T roi_start_w = static_cast<T>(
    //   round(offset_bottom_rois[1])) * spatial_scale;
    // T roi_start_h = static_cast<T>(
    //   round(offset_bottom_rois[2])) * spatial_scale;
    // T roi_end_w = static_cast<T>(
    //   round(offset_bottom_rois[3]) + 1.) * spatial_scale;
    // T roi_end_h = static_cast<T>(
    //   round(offset_bottom_rois[4]) + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
    T roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    T wstart = static_cast<T>(pw)* bin_size_w + roi_start_w;
    T hend = static_cast<T>(ph + 1) * bin_size_h + roi_start_h;
    T wend = static_cast<T>(pw + 1) * bin_size_w + roi_start_w;

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, (T)0), (T)height);
    hend = min(max(hend, (T)0), (T)height);
    wstart = min(max(wstart, (T)0),(T)width);
    wend = min(max(wend, (T)0), (T)width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gw = pw;
    int gh = ph;
    int c = (ctop * group_size + gh) * group_size + gw;

    const T* offset_bottom_data =
      bottom_data + (roi_batch_ind * channels + c) * height * width;
    
    int sampling_ratio = 2;
    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = is_empty ? 0. : output_val;


    // T out_sum = 0;
    // for (int h = hstart; h < hend; ++h){
    //  for (int w = wstart; w < wend; ++w){
    //    int bottom_index = h*width + w;
    //    out_sum += offset_bottom_data[bottom_index];
    //  }
    // }

    // T bin_area = (hend - hstart) * (wend - wstart);
    // top_data[index] = is_empty ? 0. : out_sum / bin_area;
    mapping_channel[index] = c;
  }
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}


template <typename T>
__global__ void PSRoIAlignBackward(
    const int nthreads,
    const T* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int output_dim,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    // int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;
 
    // T roi_start_w = static_cast<T>(
    //   round(offset_bottom_rois[1])) * spatial_scale;
    // T roi_start_h = static_cast<T>(
    //   round(offset_bottom_rois[2])) * spatial_scale;
    // T roi_end_w = static_cast<T>(
    //   round(offset_bottom_rois[3]) + 1.) * spatial_scale;
    // T roi_end_h = static_cast<T>(
    //   round(offset_bottom_rois[4]) + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    T roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    T wstart = static_cast<T>(pw)* bin_size_w + roi_start_w;
    T hend = static_cast<T>(ph + 1) * bin_size_h + roi_start_h;
    T wend = static_cast<T>(pw + 1) * bin_size_w + roi_start_w;

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, (T)0), (T)height);
    hend = min(max(hend, (T)0), (T)height);
    wstart = min(max(wstart, (T)0), (T)width);
    wend = min(max(wend, (T)0), (T)width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Compute c at bottom
    int c = mapping_channel[index];
    T* offset_bottom_diff =
      bottom_diff + (roi_batch_ind * channels + c) * height * width;

    // int top_offset    = (n * channels + ctop) * pooled_height * pooled_width;
    // const T* offset_top_diff = top_diff + top_offset;
    // const T top_diff_this_bin = is_empty ? 0. : offset_top_diff[ph * pooled_width + pw];
    const T top_diff_this_bin = is_empty ? 0. : top_diff[index];

    int sampling_ratio = 2;
    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height, width, y, x,
            w1, w2, w3, w4,
            x_low, x_high, y_low, y_high,
            index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
        {
          gpu_atomic_add(static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
          gpu_atomic_add(static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
          gpu_atomic_add(static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
          gpu_atomic_add(static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
        } // if
      } // ix
    } // iy



    // T bin_area = (hend - hstart) * (wend - wstart);
    // T diff_val = is_empty ? 0. : top_diff[index] / bin_area;
    // for (int h = hstart; h < hend; ++h){
    //   for (int w = wstart; w < wend; ++w){
    //     int bottom_index = h * width + w;
    //     gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
    //   }
    // }
  }
}

} // namespace

template<>
bool PSRoIAlignOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data to pool
  auto& R = Input(1);  // RoIs
  auto* Y = Output(0); // PSRoI pooled data
  auto* A = Output(1); // mapping_channel

  Y->Resize(R.dim32(0), output_dim_, pooled_height_, pooled_width_);
  A->Resize(Y->dims());
  int output_size = Y->size();
  PSRoIAlignForward<float><<<CAFFE_GET_BLOCKS(output_size),
                            CAFFE_CUDA_NUM_THREADS,
                            0, context_.cuda_stream()>>>(
      output_size, X.data<float>(), spatial_scale_, X.dim32(1), X.dim32(2),
      X.dim32(3), pooled_height_, pooled_width_, R.data<float>(), output_dim_,
      group_size_, Y->mutable_data<float>(), A->mutable_data<int>());
  return true;
}


template<>
bool PSRoIAlignGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X  = Input(0);  // Input data to pool
  auto& R  = Input(1);  // RoIs
  auto& A  = Input(2);  // mapping channels
  auto& dY = Input(3);  // Gradient of net w.r.t. output of "forward" op
                        // (aka "gradOutput")
  auto* dX = Output(0); // Gradient of net w.r.t. input to "forward" op
                        // (aka "gradInput")

  dX->ResizeLike(X);
  // Must zero-out dX before accumulating gradients
  math::Set<float, CUDAContext>(
      dX->size(), 0.f, dX->mutable_data<float>(), &context_);
  PSRoIAlignBackward<float><<<CAFFE_GET_BLOCKS(dY.size()),
                             CAFFE_CUDA_NUM_THREADS,
                             0, context_.cuda_stream()>>>(
      dY.size(), dY.data<float>(), A.data<int>(), R.dim32(0), spatial_scale_,
      X.dim32(1), X.dim32(2), X.dim32(3), pooled_height_, pooled_width_,
      output_dim_, dX->mutable_data<float>(), R.data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(PSRoIAlign,
                       PSRoIAlignOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(PSRoIAlignGradient,
                       PSRoIAlignGradientOp<float, CUDAContext>);
} // namespace caffe2
