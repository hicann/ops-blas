#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_full_test_common.py"
SPEC = importlib.util.spec_from_file_location("run_full_test_common", MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TestLoadCannEnv(unittest.TestCase):
    def test_nonzero_set_env_status_is_not_treated_as_success(self):
        if shutil.which("bash") is None:
            self.skipTest("bash is required to source set_env.sh")

        with tempfile.TemporaryDirectory() as temp_dir:
            setenv_path = Path(temp_dir) / "set_env.sh"
            setenv_path.write_text(
                "export PARTIAL_CANN_ENV=1\nreturn 19\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "exit code 19"):
                RUNNER.load_cann_env(temp_dir)


if __name__ == "__main__":
    unittest.main()
