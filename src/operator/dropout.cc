/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout.cc
 * \brief
 * \author Bing Xu
*/

#include "./dropout-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(DropoutParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new DropoutOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *DropoutProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(DropoutParam);

MXNET_REGISTER_OP_PROPERTY(Dropout, DropoutProp)
.describe(R"(Apply dropout to input.
During training, each element of the input is randomly set to zero with probability p.
And then the whole tensor is rescaled by 1/(1-p) to keep the expectation the same as
before applying dropout. During the test time, this behaves as an identity map.
)")
.add_argument("data", "Symbol", "Input data to dropout.")
.add_arguments(DropoutParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


