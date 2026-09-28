#!/usr/bin/env bash
#
# logging: server log files (amadeo_utils.logging_utils) and the 'log_file' setting in the server configs.
# Plain Python - no models, no conda env.

source "$(dirname "$0")/../testlib/suite_lib.sh"
cd "$(dirname "$0")" || exit 1

run "log files: screen + file, colours stripped, 0660, daily, 90 kept, config loaders" python3 check_log_file.py

finish
