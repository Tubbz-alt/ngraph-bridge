/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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
 *******************************************************************************/

#include "gtest/gtest.h"

#include "tensorflow/core/graph/node_builder.h"

#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Set using C API, get using C API
TEST(DisableOps, GettingSettingTest) {
  ASSERT_EQ(config::IsDynamic(), false);
  config::UseDynamic();
  ASSERT_EQ(config::IsDynamic(), true);
    config::UseStatic();
  ASSERT_EQ(config::IsDynamic(), false);
}

}
}
}