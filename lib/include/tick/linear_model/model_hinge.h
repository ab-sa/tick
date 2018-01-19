#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_HINGE_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_HINGE_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"

#include <cereal/types/base_class.hpp>

class DLL_PUBLIC ModelHinge : public virtual ModelGeneralizedLinear {
 public:
  ModelHinge(const SBaseArrayDouble2dPtr features,
             const SArrayDoublePtr labels,
             const bool fit_intercept,
             const int n_threads = 1);

  const char *get_class_name() const override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear", cereal::base_class<ModelGeneralizedLinear>(this)));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHinge, cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_HINGE_H_
