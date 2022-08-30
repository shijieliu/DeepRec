/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "common/check.h"
#include "lookup_adapter.hpp"
#include "tensorflow/core/framework/embedding/embedding_var.h"

namespace tensorflow {

template <typename KeyType, typename DType>
TFAdapter<KeyType, DType>::TFAdapter()
    : d_data_(nullptr), d_dimensions_(nullptr),
      d_id_space_to_local_index_(nullptr), d_scale_(nullptr), stream_(0) {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  CUDACHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount,
                                   device));
  // tensorflow::core::RefCountPtr<tensorflow::EmbeddingVar<KeyType, DType>> var;
  auto storage_manager = new embedding::StorageManager<KeyType, float>(
                 "name", embedding::StorageConfig());
  TF_CHECK_OK(storage_manager->Init());
  EmbeddingVar<int64, float>* emb_var
    = new EmbeddingVar<KeyType, float>("name",
        storage_manager, EmbeddingConfig(0, 0, 1, 1, "", -1, 0, 99999, 14.0));
  emb_var ->Init(value, 1);
  // emb_var->BatchCommit();
}

// template <typename KeyType, typename DType>
// void TFAdapter<KeyType, DType>::set(
//     std::vector<tensorflow::core::RefCountPtr<tensorflow::EmbeddingVar<KeyType,
//     DType>> &vars, std::vector<tensorflow::tf_shared_lock> &locks,
//     std::vector<int> &dimensions, std::vector<int> &scale,
//     cudaStream_t stream) {
// }

// template <typename KeyType, typename DType>
// void TFAdapter<KeyType, DType>::free() {
// }

template <typename KeyType, typename DType>
TFAdapter<KeyType, DType>::~TFAdapter() {
  //   free();
}

template <typename KeyType, typename DType>
void TFAdapter<KeyType, DType>::lookup(const ::core::Tensor &keys,
                                       size_t num_keys,
                                       const ::core::Tensor &id_space_offset,
                                       size_t num_id_space_offset,
                                       const ::core::Tensor &id_space,
                                       ::core::TensorList &embedding_vec) {
  // TFAdapterKernel<KeyType, DType><<<2 * sm_count_, 1024ul, 0, stream_>>>(
  //     d_data_, d_dimensions_, d_scale_, d_id_space_to_local_index_,
  //     keys.get<KeyType>(), num_keys, id_space_offset.get<uint32_t>(),
  //     num_id_space_offset - 1, id_space.get<int>(),
  //     embedding_vec.get<DType>());
  // // CUDACHECK(cudaStreamSynchronize(stream_));
  // // CUDACHECK(cudaGetLastError());
  // // clang-format off
  // id_space_offset_.resize(num_id_space_offset);
  // CUDACHECK(cudaMemcpyAsync(id_space_offset_.data(),
  //                           id_space_offset.get<uint32_t>(),
  //                           sizeof(uint32_t) * (num_id_space_offset),
  //                           cudaMemcpyDeviceToHost, stream_));
  // id_space_.resize(num_id_space_offset - 1);
  // CUDACHECK(cudaMemcpyAsync(id_space_.data(),
  //                           id_space.get<int>(),
  //                           sizeof(int) * (num_id_space_offset - 1),
  //                           cudaMemcpyDeviceToHost, stream_));
  // // clang-format on

  // CUDACHECK(cudaStreamSynchronize(stream_));

  // DType** output = embedding_vec.get<DType>();
  // const KeyType* input = keys.get<KeyType>();
  // for (int i = 0; i < num_id_space_offset - 1; ++i) {
  //   size_t num = id_space_offset_[i + 1] - id_space_offset_[i];
  //   auto var = vars_[id_space_[i]];
  //   var->lookup(input, output, num, stream_);
  //   input += num;
  //   output += num;
  // }
}

template class TFAdapter<int32_t, float>;
// template class TFAdapter<int32_t, __half>;
template class TFAdapter<int64_t, float>;
// template class TFAdapter<int64_t, __half>;
} // namespace deeprec
