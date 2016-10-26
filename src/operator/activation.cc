/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu
*/
#include "./activation-inl.h"
#include "./mshadow_op.h"
#if MXNET_USE_MKL2017 == 1
#include <mxnet/mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_relu-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ActivationParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  if (param.act_type == activation::kReLU) {
      switch (dtype) {
      case mshadow::kFloat32:
          return new MKLReluOp<cpu, float>();
      case mshadow::kFloat64:
          return new MKLReluOp<cpu, double>();
      default:
          break;
      }
  }

#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.act_type) {
      case activation::kReLU:
        op = new ActivationOp<cpu, mshadow_op::relu, mshadow_op::relu_grad, DType>();
        break;
      case activation::kSigmoid:
        op = new ActivationOp<cpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad, DType>();
        break;
      case activation::kTanh:
        op = new ActivationOp<cpu, mshadow_op::tanh, mshadow_op::tanh_grad, DType>();
        break;
      case activation::kSoftReLU:
        op = new ActivationOp<cpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>();
        break;
      default:
        LOG(FATAL) << "unknown activation type";
    }
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ActivationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(ActivationParam);

MXNET_REGISTER_OP_PROPERTY(Activation, ActivationProp)
.describe(R"(Elementwise activation function.

The following activation types are supported (operations are applied elementwisely to each
scalar of the input tensor):

- `relu`: Rectified Linear Unit, `y = max(x, 0)`
- `sigmoid`: `y = 1 / (1 + exp(-x))`
- `tanh`: Hyperbolic tangent, `y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
- `softrelu`: Soft ReLU, or SoftPlus, `y = log(1 + exp(x))`

See `LeakyReLU` for other activations with parameters.
)")
.add_arguments(ActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

