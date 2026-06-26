"""
Copyright (C) 2026 Bitan Majumder

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import os

# Add gnn_services directory to path
GNN_SERVICE_DIR = os.path.join(
    os.path.dirname(__file__),  # tests/
    '..',                        # graphml-studio/
    'gnn_services'               # gnn_services/
)
sys.path.insert(0, os.path.abspath(GNN_SERVICE_DIR))