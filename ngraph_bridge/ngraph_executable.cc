//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <sstream>

#include "ngraph/ngraph.hpp"

#include "ngraph_executable.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

Executable::Executable() {}

Executable::~Executable() {}

bool Executable::call_with_validate(
    const vector<shared_ptr<runtime::Tensor>>& outputs,
    const vector<shared_ptr<runtime::Tensor>>& inputs) {
  validate(outputs, inputs);
  return call(outputs, inputs);
}

void Executable::validate(
    const vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const vector<std::shared_ptr<runtime::Tensor>>& inputs) {
  const ParameterVector& parameters = get_parameters();
  const ResultVector& results = get_results();
  if (parameters.size() != inputs.size()) {
    stringstream ss;
    ss << "Call input count " << inputs.size()
       << " does not match Function's Parameter count " << parameters.size();
    throw runtime_error(ss.str());
  }
  if (results.size() != outputs.size()) {
    stringstream ss;
    ss << "Call output count " << outputs.size()
       << " does not match Function's Result count " << results.size();
    throw runtime_error(ss.str());
  }

  for (size_t i = 0; i < parameters.size(); i++) {
    if (parameters[i]->get_element_type().is_static() &&
        parameters[i]->get_element_type() != inputs[i]->get_element_type()) {
      stringstream ss;
      ss << "Input " << i << " type '" << inputs[i]->get_element_type()
         << "' does not match Parameter type '"
         << parameters[i]->get_element_type() << "'";
      throw runtime_error(ss.str());
    }
    if (!(parameters[i]->get_output_partial_shape(0).relaxes(
            inputs[i]->get_partial_shape()))) {
      stringstream ss;
      ss << "Input " << i << " shape " << inputs[i]->get_partial_shape()
         << " does not match Parameter shape "
         << parameters[i]->get_output_partial_shape(0);
      throw runtime_error(ss.str());
    }
  }

  for (size_t i = 0; i < results.size(); i++) {
    if (outputs[i]->get_element_type().is_static() &&
        results[i]->get_element_type().is_static() &&
        results[i]->get_element_type() != outputs[i]->get_element_type()) {
      stringstream ss;
      ss << "Output " << i << " type '" << outputs[i]->get_element_type()
         << "' does not match Result type '" << results[i]->get_element_type()
         << "'";
      throw runtime_error(ss.str());
    }
    if (!outputs[i]->get_partial_shape().relaxes(
            results[i]->get_output_partial_shape(0))) {
      stringstream ss;
      ss << "Output " << i << " shape " << outputs[i]->get_partial_shape()
         << " does not match max Result shape "
         << results[i]->get_output_partial_shape(0).get_max_shape();
      throw runtime_error(ss.str());
    }
  }
}

const ngraph::ParameterVector& Executable::get_parameters() const {
  return m_parameters;
}

const ngraph::ResultVector& Executable::get_results() const {
  return m_results;
}

size_t Executable::get_preferred_pipeline_depth() const { return 2; }

void Executable::set_parameters_and_results(const ngraph::Function& func) {
  m_parameters = func.get_parameters();
  m_results = func.get_results();
}

void Executable::save(std::ostream& /* output_stream */) {
  throw runtime_error("save operation unimplemented.");
}

shared_ptr<runtime::Tensor> Executable::create_input_tensor(
    size_t /* input_index */) {
  throw runtime_error("create_input_tensor unimplemented");
}

shared_ptr<runtime::Tensor> Executable::create_input_tensor(
    size_t /* input_index */, void* /* memory_pointer */) {
  throw runtime_error("create_input_tensor unimplemented");
}

shared_ptr<runtime::Tensor> Executable::create_output_tensor(
    size_t /* output_index */) {
  throw runtime_error("create_output_tensor unimplemented");
}

shared_ptr<runtime::Tensor> Executable::create_output_tensor(
    size_t /* output_index */, void* /* memory_pointer */) {
  throw runtime_error("create_output_tensor unimplemented");
}

vector<shared_ptr<runtime::Tensor>> Executable::create_input_tensor(
    size_t /* input_index */, size_t /* pipeline_depth */) {
  throw runtime_error("create_input_tensor unimplemented");
}

vector<shared_ptr<runtime::Tensor>> Executable::create_input_tensor(
    size_t /* input_index */, size_t /* pipeline_depth */,
    std::vector<void*> /* memory_pointer */) {
  throw runtime_error("create_input_tensor unimplemented");
}

vector<shared_ptr<runtime::Tensor>> Executable::create_output_tensor(
    size_t /* output_index */, size_t /* pipeline_depth */) {
  throw runtime_error("create_output_tensor unimplemented");
}

vector<shared_ptr<runtime::Tensor>> Executable::create_output_tensor(
    size_t /* output_index */, size_t /* pipeline_depth */,
    std::vector<void*> /* memory_pointer */) {
  throw runtime_error("create_output_tensor unimplemented");
}

}  // namespace ngraph_bridge
}  // namespace tensorflow