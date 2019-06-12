# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dev_tools import (
    check_incremental_coverage,
    check_pylint,
    check_pytest_with_coverage,
    check_typecheck,
)


pylint = check_pylint.LintCheck()
typecheck = check_typecheck.TypeCheck()
pytest = check_pytest_with_coverage.TestAndPrepareCoverageCheck()
incremental_coverage = check_incremental_coverage.IncrementalCoverageCheck(
    pytest)

ALL_CHECKS = [
    pylint,
    typecheck,
    pytest,
    incremental_coverage,
]
