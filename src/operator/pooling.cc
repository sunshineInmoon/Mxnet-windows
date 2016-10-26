/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cc
 * \brief
 * \author Bing Xu
*/
#include "./pooling-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mxnet/mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_pooling-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mxnet {
namespace op {


template<>
Operator* CreateOp<cpu>(PoolingParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
    if ((param.pool_type == pool_enum::kMaxPooling
      || param.pool_type == pool_enum::kAvgPooling)
      && UseMKLPooling(param, in_shape, out_shape)) {
      switch (dtype) {
      case mshadow::kFloat32:
        return new MKLPoolingOp<cpu, float>(param);
      case mshadow::kFloat64:
        return new MKLPoolingOp<cpu, double>(param);
      default:
        break;
      }
    }
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.pool_type) {
      case pool_enum::kMaxPooling:
        op = new PoolingOp<cpu, mshadow::red::maximum, DType>(param);
        break;
      case pool_enum::kAvgPooling:
        op = new PoolingOp<cpu, mshadow::red::sum, DType>(param);
        break;
      case pool_enum::kSumPooling:
        op = new PoolingOp<cpu, mshadow::red::sum, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown pooling type";
        return NULL;
    }
  })

  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* PoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape);
}

DMLC_REGISTER_PARAMETER(PoolingParam);

MXNET_REGISTER_OP_PROPERTY(Pooling, PoolingProp)
.describe("Perform spatial pooling on inputs.")
.add_argument("data", "Symbol", "Input data to the pooling operator.")
.add_arguments(PoolingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

