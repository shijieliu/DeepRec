/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef INTEL_MKL

#include "mkldnn.hpp"
#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

// Compare performance of default Tensorflow convolution kernels (Eigen) with
// MKL kernels on CPU.
// Before running these benchmarks configure OpenMP environment variables:
//   export KMP_BLOCKTIME=0
//   export OMP_NUM_THREADS=${num_threads}

namespace tensorflow {

template <typename T>
static Graph* Activation(const string& op_name, const string& kind,
                         const TensorShape& shape) {
  auto* graph = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const string node_name = kind + "_" + op_name;
  const bool isForwardOp = !tensorflow::str_util::EndsWith(op_name, "Grad");
  const bool isDefault = (kind == "Default");

  Tensor input_t(type, shape);
  input_t.flat<T>().setRandom();
  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  if (isForwardOp) {
    auto nodeBuilder = NodeBuilder(graph->NewName(node_name), isDefault ? op_name : "_Mkl" + op_name)
                           .Input(input)
                           .Attr("T", type);
    isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                         .Attr("_kernel", "MklLayoutDependentOp");
    TF_CHECK_OK(nodeBuilder.Finalize(graph, nullptr));
    return graph;
  }

  Tensor grad_t(type, shape);
  grad_t.flat<T>().setRandom();
  Node* grad = test::graph::Constant(graph, grad_t, "grad");

  auto nodeBuilder = NodeBuilder(graph->NewName(node_name), isDefault ? op_name : "_Mkl" + op_name)
                         .Input(grad)
                         .Input(input)
                         .Attr("T", type);
  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");

  TF_CHECK_OK(nodeBuilder.Finalize(graph, nullptr));
  return graph;
}

#define BM_Activation_Base(op, kind, A, B, C, D, T, device, NTH)                             \
  static void BM_##op##_##kind##_##T##_##A##_##B##_##C##_##D##_##device##_##NTH(int iters) { \
    testing::UseRealTime();                                                                  \
    testing::ItemsProcessed(static_cast<int64>(iters) * A * B * C * D);                      \
    SessionOptions opts;                                                                     \
    opts.config.set_intra_op_parallelism_threads(NTH);                                       \
    test::Benchmark(#device, Activation<T>(#op, #kind, {A, B, C, D}), &opts).Run(iters);     \
  }                                                                                          \
  BENCHMARK(BM_##op##_##kind##_##T##_##A##_##B##_##C##_##D##_##device##_##NTH)               \

#define BM_Activation_NTH(op, kind, A, B, C, D, T, device) \
  BM_Activation_Base(op, kind, A, B, C, D, T, device, 1);  \
  BM_Activation_Base(op, kind, A, B, C, D, T, device, 4);  \
  BM_Activation_Base(op, kind, A, B, C, D, T, device, 8);  \

#define BM_Activation_Kind(op, A, B, C, D, T, device)    \
  BM_Activation_NTH(op, Default, A, B, C, D, T, device); \
  BM_Activation_NTH(op, Mkl, A, B, C, D, T, device);     \

#define BM_Activation(op, A, B, C, D)                \
  BM_Activation_Kind(op, A, B, C, D, float, cpu);    \
  BM_Activation_Kind(op, A, B, C, D, bfloat16, cpu); \

#define TEST_Activation_ALL(OP)        \
  BM_Activation(OP, 2, 4, 8, 16);      \
  BM_Activation(OP, 3, 5, 9, 17);      \
  BM_Activation(OP, 32, 64, 128, 256); \
  BM_Activation(OP, 33, 65, 129, 257); \

TEST_Activation_ALL(Tanh)
TEST_Activation_ALL(TanhGrad)
TEST_Activation_ALL(Elu)
TEST_Activation_ALL(EluGrad)
TEST_Activation_ALL(Relu)
TEST_Activation_ALL(ReluGrad)
TEST_Activation_ALL(Relu6)
TEST_Activation_ALL(Relu6Grad)
TEST_Activation_ALL(LeakyRelu)
TEST_Activation_ALL(LeakyReluGrad)

}  // namespace tensorflow

#endif  // INTEL_MKL
