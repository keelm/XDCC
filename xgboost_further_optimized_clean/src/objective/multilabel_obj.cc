/*!
 * Copyright 2015 by Contributors
 * \file multi_class.cc
 * \brief Definition of multi-class classification objectives.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../common/math.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(multilabel_obj);

struct MultilabelParam : public dmlc::Parameter<MultilabelParam> {
  int num_label;
  // declare parameters
  DMLC_DECLARE_PARAMETER(MultilabelParam) {
    DMLC_DECLARE_FIELD(num_label).set_lower_bound(1)
        .describe("Number of output labels in the multilabel classification.");
  }
};

class MultilabelObj : public ObjFunction {
 public:
  explicit MultilabelObj(bool output_prob)
      : output_prob_(output_prob) {
  }
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(HostDeviceVector<bst_float>* preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<bst_gpair>* out_gpair) override {
    CHECK_NE(info.labels.size(), 0U) << "label set cannot be empty";
    CHECK(preds->size() == info.labels.size()) << "MultilabelObj: label size and pred size does not match";
    const int nlabel = param_.num_label;
    std::vector<bst_float>& preds_h = preds->data_h();
    out_gpair->resize(preds_h.size()/nlabel);
    std::vector<bst_gpair>& gpair = out_gpair->data_h();
    const omp_ulong ndata = static_cast<omp_ulong>(preds_h.size() / nlabel);

    int label_error = 0;
    #pragma omp parallel
    {
      #pragma omp for schedule(static)
      for (omp_ulong i = 0; i < ndata; ++i) {
        std::vector<double> grad_vec(nlabel);
        std::vector<double> hess_vec(nlabel);
        for (int k = 0; k < nlabel; ++k) {
          int label = static_cast<int>(info.labels[i * nlabel + k]);
          if (label != 0 && label != 1)  {
            label_error = label; label = 0;
          }
          bst_float wt = info.GetWeight(i * nlabel + k);
          bst_float p = common::Sigmoid(preds_h[i * nlabel + k]);
          bst_float y = info.labels[i * nlabel + k];
          grad_vec[k] = (p - y) * wt;
          hess_vec[k] = (p * (1-p)) * wt;
        }
        gpair[i] = bst_gpair(grad_vec, hess_vec);
      }

    }
    CHECK(label_error >= 0 && label_error < nlabel)
        << "MultilabelObj: label must be 0 or 1,"
        << " num_label=" << nlabel
        << " but found " << label_error << " in label.";
  }
  void PredTransform(HostDeviceVector<bst_float>* io_preds) override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(HostDeviceVector<bst_float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override {
    return "ml_error";
  }

 private:
  inline void Transform(HostDeviceVector<bst_float> *io_preds, bool prob) {
    std::vector<bst_float> &preds = io_preds->data_h();
    std::vector<bst_float> tmp;
    const int nlabel = param_.num_label;
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size() / nlabel);

    #pragma omp parallel
    {
      std::vector<bst_float> rec(nlabel);
      #pragma omp for schedule(static)
      for (omp_ulong i = 0; i < io_preds->size(); ++i) {
          preds[i] = common::Sigmoid(preds[i]);
      }
    }
  }
  // output probability
  bool output_prob_;
  // parameter
  MultilabelParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(MultilabelParam);

XGBOOST_REGISTER_OBJECTIVE(MultilabelObj, "multilabel")
.describe("Softmax for multilabel classification, output class index.")
.set_body([]() { return new MultilabelObj(false); });

}  // namespace obj
}  // namespace xgboost
